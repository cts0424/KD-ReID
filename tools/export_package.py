"""Package a trained OSNet run for use outside this repo (ONNX + PyTorch + inference script).

    python tools/export_package.py \
        --run /content/drive/MyDrive/KD-ReID/outputs/osnet/3d_dkd_r50 \
        --out /content/drive/MyDrive/KD-ReID/packages \
        --eval-root /content/data            # optional: re-measure accuracy of the exported weights

Produces <out>/<name>/ and <out>/<name>.zip containing:
    model.onnx        opset 17, input "images" B x 3 x 256 x 128, output "features" B x D
    weights.pt        deploy state_dict (no classifier) for deploy/osnet_reid.py
    osnet_reid.py     standalone PyTorch model definition
    reid_infer.py     selftest / extract / search CLI
    selftest.npz      fixed inputs + reference outputs to verify a new install
    metadata.json     preprocessing, output spec, metrics, provenance
    README.md, requirements.txt

Every artefact is checked before writing: standalone PyTorch vs training model, ONNX vs PyTorch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from osnet_reid import OSNetReID  # noqa: E402

from kdreid.benchmark import count_macs  # noqa: E402
from kdreid.config import load_config  # noqa: E402
from kdreid.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from kdreid.engine.trainer import teacher_specs  # noqa: E402
from kdreid.models import build_model  # noqa: E402
from kdreid.models.osnet import CHANNELS  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except Exception:
        return "unknown"


def final_metrics_from_log(run: Path) -> dict | None:
    log = run / "log.txt"
    if not log.exists():
        return None
    m = None
    for line in log.read_text(encoding="utf-8").splitlines():
        hit = re.search(r"Final \(epoch (\d+)\) mAP ([\d.]+)% \| R1 ([\d.]+)%", line)
        if hit:
            m = {"epoch": int(hit[1]), "mAP": float(hit[2]), "rank1": float(hit[3])}
    return m


def training_summary(cfg) -> dict:
    s = {"dataset": cfg.data.name, "epochs": cfg.train.epochs, "lr": cfg.train.lr}
    if cfg.kd.get("enabled"):
        s["distillation"] = {
            "teachers": [t.backbone for t in teacher_specs(cfg)],
            "losses": dict(cfg.kd.losses),
            "temperature": cfg.kd.get("temperature"),
        }
        if cfg.kd.losses.get("dkd"):
            s["distillation"]["dkd"] = dict(cfg.kd.get("dkd") or {})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir with config.yaml + weights")
    ap.add_argument("--weights", default="last.pth", help="last.pth = the reported final epoch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="package name (default: kdreid-<run name>)")
    ap.add_argument("--eval-root", help="Market-1501 root: re-measure accuracy with/without flip")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    if cfg.model.backbone not in CHANNELS:
        sys.exit(f"only OSNet models can be packaged (got {cfg.model.backbone})")
    name = args.name or f"kdreid-{run.name}"
    pkg = Path(args.out) / name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    # 1. training model with the trained weights
    state = torch.load(run / args.weights, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    cfg.model.pretrained = False
    train_model = build_model(cfg.model, state["classifier.weight"].shape[0]).eval()
    train_model.load_state_dict(state)

    # 2. standalone deploy model: same weights minus the training-only classifier
    deploy_state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    embed_dim = int(cfg.model.get("embed_dim", 0) or 0)
    model = OSNetReID(cfg.model.backbone, embed_dim).eval()
    model.load_state_dict(deploy_state, strict=True)
    torch.save(deploy_state, pkg / "weights.pt")

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, *cfg.data.size, generator=gen)
    with torch.no_grad():
        ref = train_model(x)["bn_feat"]
        out = model(x)
    d_torch = float((ref - out).abs().max())
    assert d_torch < 1e-5, f"standalone model differs from training model by {d_torch}"
    np.savez(pkg / "selftest.npz", inputs=x.numpy(), outputs=out.numpy())

    # 3. ONNX, checked against PyTorch
    torch.onnx.export(
        model, (x[:2],), str(pkg / "model.onnx"),
        input_names=["images"], output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17, dynamo=False,
    )  # fmt: skip
    model.eval()
    d_onnx = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(pkg / "model.onnx"), providers=["CPUExecutionProvider"])
        d_onnx = float(np.abs(sess.run(None, {"images": x.numpy()})[0] - out.numpy()).max())
        assert d_onnx < 1e-3, f"ONNX differs from PyTorch by {d_onnx}"
    except ImportError:
        print("onnxruntime not installed: ONNX file written but not verified")

    # 4. accuracy of exactly these weights
    metrics = {"source": "log.txt Final line", **(final_metrics_from_log(run) or {})}
    if args.eval_root:
        from kdreid.data import build_test_loader
        from kdreid.engine import evaluate

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader, ds = build_test_loader(cfg.data.name, args.eval_root, cfg.data.size)
        train_model.to(dev)
        metrics = {"source": f"re-evaluated on {cfg.data.name} at export"}
        for flip in (True, False):
            r = evaluate(train_model, loader, len(ds.query), dev, flip=flip)
            metrics["flip" if flip else "no_flip"] = {k: round(100 * v, 2) for k, v in r.items()}
        print("accuracy:", json.dumps(metrics, ensure_ascii=False))

    params = sum(v.numel() for v in deploy_state.values() if v.dtype.is_floating_point)
    meta = {
        "name": name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {"run": str(run), "weights": args.weights, "git_commit": git_commit()},
        "model": {
            "backbone": cfg.model.backbone,
            "embed_dim": embed_dim,
            "feature_dim": model.feat_dim,
            "parameters": params,
            "gmacs": round(count_macs(train_model.cpu(), tuple(cfg.data.size)) / 1e9, 4),
        },
        "preprocess": {
            "color": "RGB",
            "size_hw": list(cfg.data.size),
            "resize": "bilinear (PIL), aspect ratio not preserved",
            "scale": "pixel / 255",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "layout": "NCHW float32",
        },
        "output": {
            "name": "features",
            "dim": model.feat_dim,
            "postprocess": "L2-normalise; similarity = cosine (dot product)",
            "flip_tta": "reported accuracy adds the embedding of the horizontally flipped image",
        },
        "training": training_summary(cfg),
        "metrics": metrics,
        "evaluation_protocol": "Market-1501 standard, cosine distance, no re-ranking",
        "verification": {
            "standalone_vs_training_max_abs": d_torch,
            "onnx_vs_torch_max_abs": d_onnx,
        },
    }
    (pkg / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")

    for f in ("osnet_reid.py", "reid_infer.py"):
        shutil.copy(ROOT / "deploy" / f, pkg / f)
    (pkg / "requirements.txt").write_text(
        "numpy\npillow\nonnxruntime>=1.16      # default backend (onnxruntime-gpu for CUDA)\n"
        "# torch>=2.1           # only for --backend torch / weights.pt\n",
        "utf-8",
    )
    readme = (ROOT / "deploy" / "README_template.md").read_text(encoding="utf-8")
    (pkg / "README.md").write_text(render_readme(readme, meta), "utf-8")

    archive = shutil.make_archive(str(pkg), "zip", root_dir=pkg.parent, base_dir=pkg.name)
    print(f"package: {pkg}\narchive: {archive}")
    print(f"checks : standalone vs training {d_torch:.1e} | ONNX vs PyTorch {d_onnx}")


def render_readme(tpl: str, meta: dict) -> str:
    m = meta["metrics"]
    if "flip" in m:
        acc = (
            f"| flip TTA（預設） | {m['flip']['mAP']:.1f} | {m['flip']['rank1']:.1f} | {m['flip']['rank5']:.1f} |\n"
            f"| 單次前向（`--no-flip`） | {m['no_flip']['mAP']:.1f} | {m['no_flip']['rank1']:.1f} | {m['no_flip']['rank5']:.1f} |"
        )
    elif "mAP" in m:
        acc = f"| flip TTA（預設） | {m['mAP']:.1f} | {m['rank1']:.1f} | – |"
    else:
        acc = "| （未提供） | – | – | – |"
    kd = meta["training"].get("distillation", {})
    vals = {
        "name": meta["name"],
        "created": meta["created"][:10],
        "backbone": meta["model"]["backbone"],
        "params": f"{meta['model']['parameters'] / 1e6:.3f}M",
        "gmacs": f"{meta['model']['gmacs']:.3f}",
        "dim": str(meta["output"]["dim"]),
        "accuracy_rows": acc,
        "teachers": " + ".join(kd.get("teachers", [])) or "（無蒸餾）",
        "kd_losses": ", ".join(f"{k} {v}" for k, v in kd.get("losses", {}).items() if v) or "–",
        "commit": meta["source"]["git_commit"],
        "run": Path(meta["source"]["run"]).name,
    }
    for k, v in vals.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl


if __name__ == "__main__":
    main()
