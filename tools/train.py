"""Train a baseline/teacher or a KD student.

python tools/train.py --config configs/teacher_r50.yaml
python tools/train.py --config configs/student_r18_kd.yaml teacher.checkpoint=/path/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdreid.config import load_config  # noqa: E402
from kdreid.data import build_loaders  # noqa: E402
from kdreid.engine import Trainer, load_teachers, teacher_specs  # noqa: E402
from kdreid.models import build_model  # noqa: E402
from kdreid.utils import get_device, get_logger, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.get("seed", 42))
    logger = get_logger(cfg.output_dir)
    device = get_device()
    logger.info(f"device={device} | output_dir={cfg.output_dir}")

    train_loader, test_loader, ds = build_loaders(cfg)
    logger.info(ds.summary())

    student = build_model(cfg.model, ds.num_train_pids)
    teachers = load_teachers(cfg, ds.num_train_pids) if cfg.kd.get("enabled", False) else None
    for spec in teacher_specs(cfg) if teachers else []:
        logger.info(f"teacher={spec.backbone} weight={spec.get('weight', 1.0)} ({spec.checkpoint})")
    logger.info(
        f"student={cfg.model.backbone} | params={sum(p.numel() for p in student.parameters()) / 1e6:.2f}M"
        f" | pretrained: {student.pretrained_info}"
    )

    Trainer(cfg, student, train_loader, test_loader, len(ds.query), device, logger, teachers).fit()


if __name__ == "__main__":
    main()
