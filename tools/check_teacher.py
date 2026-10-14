"""Sanity-check a teacher before distilling from it.

    python tools/check_teacher.py --config configs/clip/teacher_eval.yaml
    python tools/check_teacher.py --config configs/osnet/3d_dkd_r50.yaml     # R50 reference

1. retrieval accuracy of the teacher itself with our evaluator (compare with its paper / log);
2. identity-label alignment of the teacher's logits against *our* training labels.
   Logit KD / DKD assume class i means the same person for teacher and student; a teacher trained
   with a different label order scores ~0% here even if its retrieval accuracy is perfect.

The label check runs on *clean* training images (test transform, whole training set) so that the
number does not depend on augmentation, and separates two causes of a low score:
  - label order mismatch: predictions for one true class consistently land on another class, so
    re-mapping each class to its most frequent prediction ("remapped top-1") recovers accuracy;
  - weak classifier head: errors are scattered, remapping does not help, and confidence is low.
For CLIP-ReID every logit head (768-d CLS, 512-d projected, mean) is reported.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from kdreid.config import load_config  # noqa: E402
from kdreid.data import build_loaders  # noqa: E402
from kdreid.data.build import ImageDataset  # noqa: E402
from kdreid.data.transforms import build_transforms  # noqa: E402
from kdreid.engine import evaluate, load_teachers  # noqa: E402
from kdreid.utils import get_device  # noqa: E402


def _logits(teacher, imgs):
    out = teacher(imgs)
    return out["logits"] if out.get("logits") is not None else teacher.classifier(out["bn_feat"])


def label_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Top-1 / top-5, re-mapped top-1 (each true class -> its most frequent prediction) and
    confidence statistics. Pure function so it can be unit-tested."""
    prob = logits.float().softmax(1)
    pred = prob.argmax(1)
    top5 = prob.topk(min(5, prob.shape[1]), 1).indices
    by_class: dict[int, Counter] = {}
    for y, p in zip(labels.tolist(), pred.tolist(), strict=True):
        by_class.setdefault(y, Counter())[p] += 1
    remapped = sum(c.most_common(1)[0][1] for c in by_class.values())
    return {
        "top1": (pred == labels).float().mean().item(),
        "top5": (top5 == labels[:, None]).any(1).float().mean().item(),
        "remapped_top1": remapped / len(labels),
        "max_prob": prob.max(1).values.mean().item(),
        "true_prob": prob.gather(1, labels[:, None]).mean().item(),
    }


def fit_logit_scale(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling (Guo et al., ICML 2017): the single factor s minimising the NLL of
    softmax(s * logits) on the given labels. Ranking (and top-1) is unchanged by s."""
    logits, labels = logits.float(), labels.long()
    log_s = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_s], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits * log_s.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_s.exp().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=50, help="augmented training batches")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = get_device()
    train_loader, test_loader, ds = build_loaders(cfg)
    teacher = load_teachers(cfg, ds.num_train_pids)[0].to(dev).eval()
    heads = ["mean", "cls", "proj"] if hasattr(teacher, "logit_head") else [None]
    used_head = getattr(teacher, "logit_head", None)

    def collect(loader, max_batches=None):
        out = {h: [] for h in heads}
        labels = []
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for i, (imgs, pids, _) in enumerate(loader):
                if max_batches is not None and i == max_batches:
                    break
                imgs = imgs.to(dev)
                for h in heads:
                    if h is not None:
                        teacher.logit_head = h
                    out[h].append(_logits(teacher, imgs).float().cpu())
                labels.append(pids)
        if used_head is not None:
            teacher.logit_head = used_head
        return {h: torch.cat(v) for h, v in out.items()}, torch.cat(labels)

    clean = DataLoader(
        ImageDataset(ds.train, build_transforms(cfg.data.size, False)),
        batch_size=cfg.data.test_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    worst, gap, fitted = 1.0, 0.0, None
    for name, (logits, labels) in [
        ("clean, full train set", collect(clean)),
        (f"augmented, {args.batches} batches", collect(train_loader, args.batches)),
    ]:
        print(f"label alignment ({name}, {len(labels)} images):")
        for h in heads:
            s = label_stats(logits[h], labels)
            tag = (
                "" if h is None else f"  head={h:<4}" + (" (used for KD)" if h == used_head else "")
            )
            print(
                f"  top-1 {s['top1']:.1%} | top-5 {s['top5']:.1%} | remapped top-1 "
                f"{s['remapped_top1']:.1%} | mean max-prob {s['max_prob']:.3f} | "
                f"mean p(true) {s['true_prob']:.3f}{tag}"
            )
            if name.startswith("clean") and h == used_head:
                worst, gap = s["top1"], s["remapped_top1"] - s["top1"]
                fitted = (logits[h], labels)
    if worst < 0.5 and gap > 0.2:
        print("!! label ORDER mismatch (remapping recovers accuracy): do NOT use logit KD / DKD")
    elif worst < 0.9:
        print("!! low top-1 without an order mismatch: the classifier head is weak, not misaligned")

    # Soft-target sharpness that logit KD / DKD actually see, before and after temperature scaling.
    lg, lb = fitted
    temp = float(cfg.kd.get("temperature", 4.0))
    scale = fit_logit_scale(lg, lb)
    print(f"logit spread: mean per-image std of logits {lg.float().std(1).mean():.3f}")
    for tag, f in (("raw", 1.0), (f"x{scale:.2f} (fitted)", scale)):
        a, b = label_stats(lg * f, lb), label_stats(lg * f / temp, lb)
        print(
            f"  {tag:<18} T=1: max-prob {a['max_prob']:.3f}, p(true) {a['true_prob']:.3f} | "
            f"T={temp:g}: max-prob {b['max_prob']:.4f}, p(true) {b['true_prob']:.4f}"
        )
    print(
        f"fitted teacher.logit_scale = {scale:.2f}  (put this in the teacher config if the raw "
        "T=1 max-prob is far below an ordinary teacher's)"
    )

    for flip in (True, False):
        r = evaluate(teacher, test_loader, len(ds.query), dev, flip=flip)
        print(
            f"retrieval (flip={flip}): mAP {r['mAP']:.1%} | R1 {r['rank1']:.1%} | R5 {r['rank5']:.1%}"
        )


if __name__ == "__main__":
    main()
