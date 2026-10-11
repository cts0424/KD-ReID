"""Evaluate trained runs on several datasets: in-domain and cross-domain (zero-shot).

Each run folder needs config.yaml and last.pth (the final-epoch weights, i.e. the reported
model). Every model is evaluated with cosine distance and NO re-ranking, with and without
flip test-time augmentation.

    python tools/eval_matrix.py \
        --runs /content/drive/MyDrive/KD-ReID/outputs/teacher_r50 \
               /content/drive/MyDrive/KD-ReID/outputs/osnet/3d_dkd_r50 ... \
        --datasets market1501:/content/data dukemtmc:/content/data \
        --out /content/drive/MyDrive/KD-ReID/eval_matrix.json

A model trained on Market and tested on Duke never saw a Duke image or identity (zero-shot).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kdreid.config import load_config  # noqa: E402
from kdreid.data import build_test_loader  # noqa: E402
from kdreid.engine import evaluate  # noqa: E402
from kdreid.models import build_model  # noqa: E402
from kdreid.utils import get_device  # noqa: E402


def load_run(run_dir: Path, weights: str):
    cfg = load_config(run_dir / "config.yaml")
    state = torch.load(run_dir / weights, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    cfg.model.pretrained = False  # weights come from the checkpoint
    model = build_model(cfg.model, num_classes=state["classifier.weight"].shape[0])
    model.load_state_dict(state)
    return cfg, model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="output dirs of trained runs")
    ap.add_argument("--datasets", nargs="+", required=True, help="name:root, e.g. dukemtmc:/data")
    ap.add_argument("--weights", default="last.pth", help="last.pth (reported) or best.pth")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="eval_matrix.json")
    args = ap.parse_args()

    device = get_device()
    loaders = {}
    rows = []
    for run in map(Path, args.runs):
        if not (run / args.weights).exists():
            print(f"skip {run}: no {args.weights}")
            continue
        cfg, model = load_run(run, args.weights)
        model.to(device)
        train_set = cfg.data.name
        for spec in args.datasets:
            name, _, root = spec.partition(":")
            if (name, tuple(cfg.data.size)) not in loaders:
                loaders[(name, tuple(cfg.data.size))] = build_test_loader(
                    name, root, cfg.data.size, num_workers=args.workers
                )
            loader, ds = loaders[(name, tuple(cfg.data.size))]
            for flip in (True, False):
                r = evaluate(model, loader, len(ds.query), device, flip=flip)
                rows.append(
                    {
                        "run": run.name,
                        "backbone": cfg.model.backbone,
                        "train": train_set,
                        "test": name,
                        "zero_shot": name != train_set,
                        "flip": flip,
                        **{k: round(100 * v, 2) for k, v in r.items()},
                    }
                )
                tag = "zero-shot" if name != train_set else "in-domain"
                print(
                    f"{run.name:28s} {train_set}->{name:10s} ({tag}, flip={flip!s:5s}) "
                    f"mAP {r['mAP']:.1%} R1 {r['rank1']:.1%}",
                    flush=True,
                )
        model.cpu()

    print(
        "\n| run | backbone | train → test | flip | mAP | R1 | R5 |\n|---|---|---|---|---|---|---|"
    )
    for r in rows:
        print(
            f"| {r['run']} | {r['backbone']} | {r['train']} → {r['test']} | "
            f"{'yes' if r['flip'] else 'no'} | {r['mAP']:.1f} | {r['rank1']:.1f} | {r['rank5']:.1f} |"
        )
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}  (re-ranking: OFF for every row)")


if __name__ == "__main__":
    main()
