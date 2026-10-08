"""CPU smoke tests on a tiny synthetic Market-style dataset. No downloads needed.

pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from kdreid.config import load_config
from kdreid.data import build_loaders
from kdreid.engine import Trainer, eval_market
from kdreid.losses import DistillLoss, TripletLoss, similarity_kd
from kdreid.models import build_model
from kdreid.utils import get_logger, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _make_fake_market(root: Path, n_ids: int = 6, per_cam: int = 2) -> Path:
    base = root / "Market-1501-v15.09.15"
    rng = np.random.default_rng(0)
    for split, offset in [("bounding_box_train", 0), ("query", 100), ("bounding_box_test", 100)]:
        d = base / split
        d.mkdir(parents=True)
        for pid in range(1, n_ids + 1):
            for cam in (1, 2):
                if split == "query" and cam == 2:
                    continue
                for k in range(per_cam):
                    arr = rng.integers(0, 255, (32, 16, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(d / f"{pid + offset:04d}_c{cam}s1_{k:06d}_00.jpg")
    return root


def _cfg(tmp: Path, overrides: list[str]):
    base = [
        f"data.root={tmp / 'data'}",
        "data.size=[64,32]",
        "data.batch_size=8",
        "data.num_instances=2",
        "data.test_batch_size=16",
        "data.num_workers=0",
        "model.pretrained=false",
        "train.epochs=1",
        "train.eval_period=1",
        "train.warmup_epochs=1",
        "train.log_period=1",
    ]
    return load_config(ROOT / "configs/base.yaml", base + overrides)


@pytest.fixture()
def fake_data(tmp_path: Path) -> Path:
    _make_fake_market(tmp_path / "data")
    return tmp_path


def test_config_inheritance_and_overrides():
    cfg = load_config(
        ROOT / "configs/student_r18_kd.yaml", ["train.lr=1e-3", "kd.losses.feature=0.5"]
    )
    assert cfg.model.backbone == "resnet18"
    assert cfg.data.name == "market1501"  # inherited from base.yaml
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.kd.losses.feature == 0.5


def test_eval_market_perfect_ranking():
    # 2 queries, 4 gallery; correct match is always the nearest and from another camera
    dist = np.array([[0.1, 0.9, 0.8, 0.7], [0.9, 0.1, 0.8, 0.7]])
    q_pids, g_pids = np.array([1, 2]), np.array([1, 2, 3, 4])
    cmc, mAP = eval_market(dist, q_pids, g_pids, np.array([0, 0]), np.array([1, 1, 1, 1]))
    assert cmc[0] == 1.0 and mAP == pytest.approx(1.0)


def test_triplet_and_kd_losses_are_finite():
    feats = torch.randn(8, 16, requires_grad=True)
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    TripletLoss(0.3)(feats, pids).backward()
    assert torch.isfinite(feats.grad).all()
    assert similarity_kd(torch.randn(8, 16), torch.randn(8, 32)).item() >= 0


def test_resnet18_last_stride_1_map_size():
    from kdreid.models import build_backbone

    body, dim = build_backbone("resnet18", pretrained=False, last_stride=1)
    assert dim == 512
    assert body(torch.randn(1, 3, 256, 128)).shape[-2:] == (16, 8)


def test_teacher_then_kd_student_end_to_end(fake_data: Path):
    # stage 1: teacher
    t_cfg = _cfg(fake_data, ["model.backbone=resnet18", f"output_dir={fake_data / 'teacher'}"])
    train_loader, test_loader, ds = build_loaders(t_cfg)
    assert ds.num_train_pids == 6
    teacher = build_model(t_cfg.model, ds.num_train_pids)
    logger = get_logger(t_cfg.output_dir, "test_teacher")
    Trainer(
        t_cfg, teacher, train_loader, test_loader, len(ds.query), torch.device("cpu"), logger
    ).fit()
    ckpt = fake_data / "teacher" / "best.pth"
    if not ckpt.exists():  # random data may give mAP 0 -> no "best"; save explicitly
        save_checkpoint({"model": teacher.state_dict()}, ckpt)
    assert (fake_data / "teacher" / "last.pth").exists()

    # stage 2: KD student with all three KD terms
    s_cfg = _cfg(
        fake_data,
        [
            "model.backbone=mobilenet_v3_small",
            f"output_dir={fake_data / 'student'}",
            "kd.enabled=true",
            "kd.losses={logit: 1.0, feature: 1.0, similarity: 1.0}",
            "teacher={backbone: resnet18, pretrained: false, last_stride: 1, neck: bnneck}",
            f"teacher.checkpoint={ckpt}",
        ],
    )
    from kdreid.engine import load_teacher

    t = load_teacher(s_cfg, ds.num_train_pids)
    student = build_model(s_cfg.model, ds.num_train_pids)
    trainer = Trainer(
        s_cfg,
        student,
        train_loader,
        test_loader,
        len(ds.query),
        torch.device("cpu"),
        get_logger(s_cfg.output_dir, "test_student"),
        t,
    )
    assert isinstance(trainer.kd, DistillLoss)
    stats = trainer.train_one_epoch(0)
    assert {"ce", "tri", "kd_logit", "kd_feat", "kd_sim"} <= stats.keys()
    assert all(np.isfinite(v) for v in stats.values())


def test_resume_continues_from_last(fake_data: Path):
    cfg = _cfg(
        fake_data, ["model.backbone=resnet18", f"output_dir={fake_data / 'r'}", "train.epochs=1"]
    )
    tl, vl, ds = build_loaders(cfg)
    log = get_logger(cfg.output_dir, "test_resume")
    Trainer(
        cfg,
        build_model(cfg.model, ds.num_train_pids),
        tl,
        vl,
        len(ds.query),
        torch.device("cpu"),
        log,
    ).fit()
    cfg.train.epochs = 2
    tr = Trainer(
        cfg,
        build_model(cfg.model, ds.num_train_pids),
        tl,
        vl,
        len(ds.query),
        torch.device("cpu"),
        log,
    )
    tr.try_resume()
    assert tr.start_epoch == 1
