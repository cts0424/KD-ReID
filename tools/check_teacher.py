"""Sanity-check a teacher before distilling from it.

    python tools/check_teacher.py --config configs/clip/teacher_eval.yaml

1. retrieval accuracy of the teacher itself with our evaluator (compare with its paper / log);
2. identity-label alignment: top-1 accuracy of the teacher's logits against *our* training labels.
   Logit KD / DKD assume class i means the same person for teacher and student; a teacher trained
   with a different label order scores ~0% here even if its retrieval accuracy is perfect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kdreid.config import load_config  # noqa: E402
from kdreid.data import build_loaders  # noqa: E402
from kdreid.engine import evaluate, load_teachers  # noqa: E402
from kdreid.utils import get_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=50, help="training batches for label check")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = get_device()
    train_loader, test_loader, ds = build_loaders(cfg)
    teacher = load_teachers(cfg, ds.num_train_pids)[0].to(dev).eval()

    correct = total = 0
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for i, (imgs, pids, _) in enumerate(train_loader):
            if i == args.batches:
                break
            out = teacher(imgs.to(dev))
            logits = (
                out["logits"]
                if out.get("logits") is not None
                else teacher.classifier(out["bn_feat"])
            )
            correct += (logits.argmax(1).cpu() == pids).sum().item()
            total += len(pids)
    acc = correct / total
    print(f"label alignment: teacher top-1 on our training labels = {acc:.1%} ({total} images)")
    if acc < 0.5:
        print("!! labels do not line up with this dataset's relabelling: do NOT use logit KD / DKD")

    for flip in (True, False):
        r = evaluate(teacher, test_loader, len(ds.query), dev, flip=flip)
        print(
            f"retrieval (flip={flip}): mAP {r['mAP']:.1%} | R1 {r['rank1']:.1%} | R5 {r['rank5']:.1%}"
        )


if __name__ == "__main__":
    main()
