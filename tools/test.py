"""Evaluate a checkpoint.

python tools/test.py --config outputs/student_r18_kd/config.yaml --ckpt outputs/student_r18_kd/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdreid.config import load_config  # noqa: E402
from kdreid.data import build_loaders  # noqa: E402
from kdreid.engine import evaluate  # noqa: E402
from kdreid.models import build_model  # noqa: E402
from kdreid.utils import get_device, load_model_weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = get_device()
    _, test_loader, ds = build_loaders(cfg)
    model = build_model(cfg.model, ds.num_train_pids)
    load_model_weights(model, args.ckpt)
    r = evaluate(model.to(device), test_loader, len(ds.query), device, flip=not args.no_flip)
    print(f"mAP {r['mAP']:.1%} | R1 {r['rank1']:.1%} | R5 {r['rank5']:.1%} | R10 {r['rank10']:.1%}")


if __name__ == "__main__":
    main()
