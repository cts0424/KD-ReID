"""Deployment benchmark: params, MACs, latency (CPU / GPU / GPU fp16), optional ONNX export.

Compare architectures (random weights are fine for size and speed):
    python tools/benchmark.py --models resnet50 resnet18 osnet_x1_0:512 osnet_x0_25:512

Benchmark trained checkpoints (each config's model block is used to build the network):
    python tools/benchmark.py --runs-dir /content/drive/MyDrive/KD-ReID/outputs \
        --onnx-dir /content/drive/MyDrive/KD-ReID/onnx

`name:512` means embed_dim=512. Results are printed as a Markdown table and saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kdreid.benchmark import benchmark, results_to_dicts, to_markdown  # noqa: E402
from kdreid.config import load_config  # noqa: E402
from kdreid.models import ReIDNet, build_model  # noqa: E402
from kdreid.utils import load_model_weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="backbone[:embed_dim] ...")
    ap.add_argument("--runs-dir", help="folder of output dirs (each with config.yaml + best.pth)")
    ap.add_argument("--num-classes", type=int, default=751)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--cpu-threads", type=int, default=0, help="0 = PyTorch default")
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--onnx-dir")
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    devices = [] if args.no_cpu else [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    jobs: list[tuple[str, torch.nn.Module]] = []
    for spec in args.models:
        name, _, emb = spec.partition(":")
        m = ReIDNet(name, args.num_classes, pretrained=False, embed_dim=int(emb or 0))
        jobs.append((spec.replace(":", "-e"), m))
    if args.runs_dir:
        for cfg_path in sorted(Path(args.runs_dir).glob("*/config.yaml")):
            ckpt = cfg_path.parent / "best.pth"
            if not ckpt.exists():
                continue
            cfg = load_config(cfg_path)
            cfg.model.pretrained = False
            m = build_model(cfg.model, args.num_classes)
            load_model_weights(m, ckpt)
            jobs.append((cfg_path.parent.name, m))
    if not jobs:
        ap.error("nothing to benchmark: pass --models and/or --runs-dir")

    results = []
    for name, m in jobs:
        print(f"benchmarking {name} ...", flush=True)
        results.append(
            benchmark(name, m, devices, args.batches, args.onnx_dir, args.runs, args.cpu_threads)
        )
    print()
    print(to_markdown(results))
    Path(args.out).write_text(json.dumps(results_to_dicts(results), indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
