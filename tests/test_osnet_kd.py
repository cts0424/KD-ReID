"""OSNet + extended KD losses + multi-teacher training. CPU, synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_smoke import _cfg, _make_fake_market

from kdreid.data import build_loaders
from kdreid.engine import Trainer, load_teachers
from kdreid.losses import attention_kd, dkd_loss, rkd_loss, simdist_kd
from kdreid.models import ReIDNet, build_model
from kdreid.utils import get_logger


@pytest.fixture()
def fake_data(tmp_path: Path) -> Path:
    _make_fake_market(tmp_path / "data")
    return tmp_path


def test_osnet_x025_shapes_and_size():
    m = ReIDNet("osnet_x0_25", num_classes=751, pretrained=False, embed_dim=512).eval()
    out = m(torch.randn(2, 3, 256, 128))
    assert out["feat_map"].shape == (2, 128, 16, 8)  # same 16x8 grid as R50 with last_stride=1
    assert out["feat"].shape == out["bn_feat"].shape == (2, 512)
    deploy = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("classifier"))
    assert 0.15e6 < deploy < 0.25e6  # paper: 0.2M


def test_osnet_without_embed_uses_native_width():
    m = ReIDNet("osnet_x0_25", num_classes=10, pretrained=False, embed_dim=0)
    assert m.feat_dim == 128 and m.embed is None


@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_relational_losses_vanish_for_identical_structure(scale):
    f = torch.randn(16, 32)
    assert simdist_kd(f, f * scale).item() == pytest.approx(0, abs=1e-5)
    assert rkd_loss(f, f * scale).item() == pytest.approx(0, abs=1e-5)  # scale-invariant
    fm = torch.randn(4, 8, 16, 8)
    # attention transfer ignores channel count: compare against a teacher with 4x the channels
    assert attention_kd(fm, fm.repeat(1, 4, 1, 1) * scale).item() == pytest.approx(0, abs=1e-6)


def test_dkd_zero_when_equal_and_positive_otherwise():
    logits, y = torch.randn(8, 20), torch.randint(0, 20, (8,))
    assert dkd_loss(logits, logits, y).item() == pytest.approx(0, abs=1e-5)
    s = torch.randn(8, 20, requires_grad=True)
    loss = dkd_loss(s, logits, y)
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(s.grad).all()


def test_unknown_kd_key_is_rejected():
    from kdreid.config import Config
    from kdreid.losses import DistillLoss

    with pytest.raises(KeyError):
        DistillLoss(Config.wrap({"losses": {"logits": 1.0}}), 8, 8)


def test_multi_teacher_osnet_student_all_losses(fake_data: Path):
    """R18 'big teacher' + OSNet x1.0 'assistant' -> OSNet x0.25 student, every KD term on,
    backbone frozen in epoch 1, then resumed."""
    ckpts = {}
    for name, extra in [("resnet18", {}), ("osnet_x1_0", {"embed_dim": 512})]:
        m = ReIDNet(name, num_classes=6, pretrained=False, **extra)
        ckpts[name] = fake_data / f"{name}.pth"
        torch.save({"model": m.state_dict()}, ckpts[name])

    cfg = _cfg(
        fake_data,
        [
            "model={backbone: osnet_x0_25, pretrained: false, embed_dim: 512}",
            f"output_dir={fake_data / 'mt'}",
            "train.freeze_backbone_epochs=1",
            "train.epochs=2",
            "data.aug.color_jitter=0.2",
            "kd.enabled=true",
            "kd.losses={logit: 1.0, similarity: 1.0, attention: 1.0}",
            "teachers=["
            f"{{backbone: resnet18, checkpoint: {ckpts['resnet18']}, weight: 0.5}},"
            f"{{backbone: osnet_x1_0, embed_dim: 512, checkpoint: {ckpts['osnet_x1_0']},"
            " weight: 1.0, kd: {losses: {dkd: 1.0, feature: 1.0, simdist: 1.0, rkd: 1.0}}}"
            "]",
        ],
    )
    tl, vl, ds = build_loaders(cfg)
    teachers = load_teachers(cfg, ds.num_train_pids)
    log = get_logger(cfg.output_dir, "test_mt")
    student = build_model(cfg.model, ds.num_train_pids)
    tr = Trainer(cfg, student, tl, vl, len(ds.query), torch.device("cpu"), log, teachers)

    before = [p.clone() for p in student.backbone.parameters()]
    stats = tr.train_one_epoch(0)  # frozen epoch
    assert all(
        torch.equal(a, b) for a, b in zip(before, student.backbone.parameters(), strict=True)
    )
    assert all(p.requires_grad for p in student.backbone.parameters())  # unfrozen afterwards
    expected = {"t0_kd_logit", "t0_kd_sim", "t0_kd_at"}
    expected |= {"t1_kd_dkd", "t1_kd_feat", "t1_kd_simdist", "t1_kd_rkd"}
    expected |= {"t1_kd_logit", "t1_kd_sim", "t1_kd_at"}  # global terms inherited
    assert expected <= stats.keys()
    assert all(np.isfinite(v) for v in stats.values())

    tr.train_one_epoch(1)
    assert not all(
        torch.equal(a, b) for a, b in zip(before, student.backbone.parameters(), strict=True)
    )

    # full fit + resume round-trip with the multi-teacher KD state (feature projector)
    tr.fit()
    cfg.train.epochs = 3
    tr2 = Trainer(
        cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
        torch.device("cpu"), log, load_teachers(cfg, ds.num_train_pids),
    )  # fmt: skip
    tr2.try_resume()
    assert tr2.start_epoch == 2


def test_benchmark_counts_match_osnet_paper(tmp_path: Path):
    from kdreid.benchmark import benchmark

    m = ReIDNet("osnet_x0_25", num_classes=751, pretrained=False, embed_dim=512)
    onnx_dir = tmp_path if _has_onnx() else None
    r = benchmark("x025", m, [torch.device("cpu")], [1], onnx_dir=onnx_dir, runs=2)
    assert r.deploy_params_m == pytest.approx(0.2, abs=0.01)  # paper: 0.2M
    assert r.gmacs == pytest.approx(0.0823, abs=0.002)  # paper: 82.3M Mult-Adds
    assert r.latency_ms["cpu_b1"] > 0
    if onnx_dir is not None:
        assert r.onnx_max_abs_diff < 1e-4
        assert m.training is False  # export must not flip BN back to batch statistics


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize(
    "cfg_path",
    sorted(p for p in (Path(__file__).parents[1] / "configs").rglob("*.yaml") if p.name[0] != "_"),
    ids=lambda p: p.name,
)
def test_every_config_builds(cfg_path: Path):
    from kdreid.config import load_config
    from kdreid.engine import teacher_specs

    cfg = load_config(cfg_path, ["model.pretrained=false"])
    m = build_model(cfg.model, 751)
    assert m(torch.randn(2, 3, 256, 128).float())["bn_feat"].shape[0] == 2
    if cfg.kd.get("enabled"):
        assert teacher_specs(cfg) and cfg.kd.get("losses")
