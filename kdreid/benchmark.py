"""Deployment metrics: parameters, MACs, latency, ONNX export.

"Deploy" = what runs at inference: backbone + pooling + [embed] + BNNeck, returning the
retrieval feature. The identity classifier is training-only and is excluded.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


class DeployWrapper(nn.Module):
    """Eval-mode model that returns only the retrieval embedding (BNNeck output)."""

    def __init__(self, reid_net: nn.Module):
        super().__init__()
        self.net = reid_net
        self.eval()

    def train(self, mode: bool = True):
        # Always stay in eval mode: torch.onnx.export restores the caller's original mode
        # afterwards, which would otherwise flip BatchNorm to batch statistics.
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["bn_feat"]


@dataclass
class BenchResult:
    name: str
    deploy_params_m: float
    classifier_params_m: float
    gmacs: float
    feat_dim: int
    latency_ms: dict[str, float]
    onnx_path: str | None = None
    onnx_max_abs_diff: float | None = None


def count_params(model: nn.Module) -> tuple[int, int]:
    """(deploy params, classifier params)."""
    deploy = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("classifier"))
    clf = sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier"))
    return deploy, clf


@torch.no_grad()
def count_macs(model: nn.Module, size: tuple[int, int] = (256, 128)) -> float:
    """Multiply-adds for one image (FLOPs / 2), comparable to the OSNet paper's 'Mult-Adds'."""
    wrapper = DeployWrapper(model)
    with FlopCounterMode(display=False) as fc:
        wrapper(torch.randn(1, 3, *size))
    return fc.get_total_flops() / 2


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    batch: int,
    size: tuple[int, int] = (256, 128),
    warmup: int = 10,
    runs: int = 50,
    half: bool = False,
) -> float:
    """Median milliseconds per forward pass of `batch` images."""
    wrapper = DeployWrapper(model).to(device)
    x = torch.randn(batch, 3, *size, device=device)
    if half:
        wrapper, x = wrapper.half(), x.half()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        wrapper(x)
    sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        wrapper(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def export_onnx(
    model: nn.Module, path: str | Path, size: tuple[int, int] = (256, 128)
) -> tuple[Path, float | None]:
    """Export with a dynamic batch axis; if onnxruntime is installed, return the max abs
    difference between ONNX and PyTorch outputs on a random batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DeployWrapper(model).cpu()
    x = torch.randn(2, 3, *size)
    torch.onnx.export(
        wrapper,
        (x,),
        str(path),
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    try:
        import onnxruntime as ort
    except ImportError:
        return path, None
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(3, 3, *size)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    out = sess.run(None, {"images": x.numpy()})[0]
    return path, float(abs(out - ref).max())


def measure_ort_latency(
    path: str | Path,
    batch: int,
    size: tuple[int, int] = (256, 128),
    warmup: int = 10,
    runs: int = 50,
    threads: int = 0,
) -> float:
    """Median ms per run with ONNX Runtime on CPU (the usual edge / server deployment path)."""
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, 3, *size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"images": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"images": x})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def benchmark(
    name: str,
    model: nn.Module,
    devices: list[torch.device],
    batches: list[int],
    onnx_dir: str | Path | None = None,
    runs: int = 50,
    cpu_threads: int = 0,
) -> BenchResult:
    model = model.eval()
    deploy, clf = count_params(model)
    lat: dict[str, float] = {}
    for dev in devices:
        for b in batches:
            lat[f"{dev.type}_b{b}"] = measure_latency(model, dev, b, runs=runs)
            if dev.type == "cuda":
                lat[f"{dev.type}_fp16_b{b}"] = measure_latency(model, dev, b, runs=runs, half=True)
        model.cpu()
    res = BenchResult(
        name=name,
        deploy_params_m=deploy / 1e6,
        classifier_params_m=clf / 1e6,
        gmacs=count_macs(model) / 1e9,
        feat_dim=model.feat_dim,
        latency_ms=lat,
    )
    if onnx_dir is not None:
        p, diff = export_onnx(model, Path(onnx_dir) / f"{name}.onnx")
        res.onnx_path, res.onnx_max_abs_diff = str(p), diff
        if diff is not None:  # onnxruntime is available
            for b in batches:
                lat[f"ort_cpu_b{b}"] = measure_ort_latency(p, b, runs=runs, threads=cpu_threads)
    return res


def to_markdown(results: list[BenchResult]) -> str:
    lat_keys = sorted({k for r in results for k in r.latency_ms})
    head = ["model", "params (M)", "GMACs", "dim", *[f"{k} (ms)" for k in lat_keys], "ONNX Δ"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        cells = [
            r.name,
            f"{r.deploy_params_m:.3f}",
            f"{r.gmacs:.3f}",
            str(r.feat_dim),
            *[f"{r.latency_ms[k]:.2f}" if k in r.latency_ms else "–" for k in lat_keys],
            "–" if r.onnx_max_abs_diff is None else f"{r.onnx_max_abs_diff:.1e}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def results_to_dicts(results: list[BenchResult]) -> list[dict]:
    return [asdict(r) for r in results]
