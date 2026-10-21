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

--bn-recal adds rows where every BatchNorm's running statistics are first re-estimated on the
*test dataset's training split* (images only, no labels; AdaBN), then the model is restored.
Those rows use unlabeled target-domain images, i.e. they are UDA rows, not zero-shot; on the
training dataset itself they show what pure source-domain statistics give.

External reference models (e.g. the official CLIP-ReID checkpoint) are evaluated the same way
from a teacher config, without a training run:

    python tools/eval_matrix.py --teacher-configs configs/clip/teacher_eval.yaml \
        --datasets market1501:/content/data cuhk03np:/content/data
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
from kdreid.data.build import ImageDataset  # noqa: E402
from kdreid.data.transforms import build_transforms  # noqa: E402
from kdreid.engine import evaluate, recalibrate_bn  # noqa: E402
from kdreid.models import build_model  # noqa: E402
from kdreid.utils import check_drive_mounted, get_device  # noqa: E402


def load_run(run_dir: Path, weights: str):
    cfg = load_config(run_dir / "config.yaml")
    state = torch.load(run_dir / weights, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    cfg.model.pretrained = False  # weights come from the checkpoint
    model = build_model(cfg.model, num_classes=state["classifier.weight"].shape[0])
    model.load_state_dict(state)
    return cfg, model.eval()


def num_classes_from_checkpoint(path) -> int:
    """Identity count of a checkpoint (any key ending in classifier.weight, DataParallel or not)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    for k, v in state.items():
        if k.endswith("classifier.weight"):
            return v.shape[0]
    raise KeyError(f"{path}: no classifier.weight to infer the number of identities")


def load_teacher_config(path: str):
    """A frozen reference model from a teacher config (cfg.teacher), trained on cfg.data.name."""
    from kdreid.engine import load_teachers

    cfg = load_config(path)
    model = load_teachers(cfg, num_classes_from_checkpoint(cfg.teacher.checkpoint))[0]
    return cfg, model.eval(), cfg.teacher.backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="output dirs of trained runs")
    ap.add_argument(
        "--teacher-configs",
        nargs="*",
        default=[],
        help="configs whose `teacher:` is a reference model to evaluate (e.g. official CLIP-ReID)",
    )
    ap.add_argument("--datasets", nargs="+", required=True, help="name:root, e.g. dukemtmc:/data")
    ap.add_argument("--weights", default="last.pth", help="last.pth (reported) or best.pth")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="eval_matrix.json")
    ap.add_argument(
        "--bn-recal",
        action="store_true",
        help="also evaluate after re-estimating BN statistics on each test dataset's train split",
    )
    args = ap.parse_args()
    if not args.runs and not args.teacher_configs:
        ap.error("give --runs and/or --teacher-configs")
    check_drive_mounted(*args.runs, args.out)

    device = get_device()
    loaders = {}
    rows = []

    def models():
        for path in args.teacher_configs:
            cfg, model, backbone = load_teacher_config(path)
            check_drive_mounted(cfg.teacher.checkpoint)
            yield f"ref:{Path(path).stem}", cfg, model, backbone
        for run in map(Path, args.runs):
            if not (run / args.weights).exists():
                print(f"skip {run}: no {args.weights}")
                continue
            cfg, model = load_run(run, args.weights)
            yield run.name, cfg, model, cfg.model.backbone

    for name_run, cfg, model, backbone in models():
        model.to(device)
        train_set = cfg.data.name
        for spec in args.datasets:
            name, _, root = spec.partition(":")
            if (name, tuple(cfg.data.size)) not in loaders:
                loaders[(name, tuple(cfg.data.size))] = build_test_loader(
                    name, root, cfg.data.size, num_workers=args.workers
                )
            loader, ds = loaders[(name, tuple(cfg.data.size))]
            for recal in (False, True) if args.bn_recal else (False,):
                original = None
                if recal:  # BN statistics from this dataset's *training* images (no labels)
                    rl = torch.utils.data.DataLoader(
                        ImageDataset(ds.train, build_transforms(cfg.data.size, False)),
                        batch_size=64,
                        shuffle=True,  # mixed identities per batch: unbiased variance estimate
                        generator=torch.Generator().manual_seed(0),
                        num_workers=args.workers,
                        drop_last=True,
                    )
                    original = recalibrate_bn(model, rl, device)
                for flip in (True, False):
                    r = evaluate(model, loader, len(ds.query), device, flip=flip)
                    rows.append(
                        {
                            "run": name_run,
                            "backbone": backbone,
                            "train": train_set,
                            "test": name,
                            # target-domain BN statistics use target images: not zero-shot
                            "zero_shot": name != train_set and not recal,
                            "bn_recal": recal,
                            "flip": flip,
                            **{k: round(100 * v, 2) for k, v in r.items()},
                        }
                    )
                    tag = "in-domain" if name == train_set else "zero-shot"
                    tag += ", BN re-estimated" if recal else ""
                    print(
                        f"{name_run:28s} {train_set}->{name:10s} ({tag}, flip={flip!s:5s}) "
                        f"mAP {r['mAP']:.1%} R1 {r['rank1']:.1%}",
                        flush=True,
                    )
                if original is not None:
                    model.load_state_dict(original)
        model.cpu()

    print(
        "\n| run | backbone | train → test | BN stats | flip | mAP | R1 | R5 |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        print(
            f"| {r['run']} | {r['backbone']} | {r['train']} → {r['test']} | "
            f"{'re-est. on ' + r['test'] + ' train' if r['bn_recal'] else 'trained'} | "
            f"{'yes' if r['flip'] else 'no'} | {r['mAP']:.1f} | {r['rank1']:.1f} | {r['rank5']:.1f} |"
        )
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}  (re-ranking: OFF for every row)")


if __name__ == "__main__":
    main()
