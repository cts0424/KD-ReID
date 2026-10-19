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
            " weight: 1.0, kd: {losses: {dkd: 1.0, feature: 1.0, cosine: 1.0, simdist: 1.0,"
            " rkd: 1.0}}}"
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
    expected |= {"t1_kd_dkd", "t1_kd_feat", "t1_kd_cos", "t1_kd_simdist", "t1_kd_rkd"}
    assert expected <= stats.keys()
    # a teacher's own kd.losses replaces the global list (losses are never merged)
    assert not {"t1_kd_logit", "t1_kd_sim", "t1_kd_at"} & stats.keys()
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


def test_relation_losses_run_in_fp32_under_autocast():
    """Under AMP a @ b is low precision; the relation losses must still be fp32 and finite
    (simdist overflowed fp16 with the CLIP-ReID teacher)."""
    from kdreid.losses.kd import rkd_loss, simdist_kd, similarity_kd

    torch.manual_seed(0)
    s, t = torch.randn(16, 512), torch.randn(16, 1280)
    for fn in (similarity_kd, simdist_kd, rkd_loss):
        ref = fn(s, t)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = fn(s.bfloat16(), t.bfloat16())
        assert out.dtype == torch.float32 and torch.isfinite(out)
        assert torch.allclose(out, ref, rtol=0.05, atol=1e-3)


def test_imagenet_download_retries_transient_gdown_errors(tmp_path, monkeypatch):
    import sys
    import types

    from kdreid.models import osnet

    calls = []

    def fake_download(url, out, quiet=False):
        calls.append(url)
        if len(calls) < 3:
            raise RuntimeError("Cannot retrieve the public link of the file")
        Path(out).write_bytes(b"x")

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake_download))
    p = osnet.resolve_imagenet_weights("osnet_x1_0", str(tmp_path / "w.pth"), retry_wait=0)
    assert p.exists() and len(calls) == 3
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        calls.clear()
        osnet.resolve_imagenet_weights("osnet_x1_0", str(tmp_path / "v.pth"), 2, retry_wait=0)


def test_fp16_latency_leaves_the_model_in_fp32():
    """GPU benchmarks time fp32 then fp16 for each batch size; the fp16 run must not leave the
    weights in half precision (the next fp32 run failed with a dtype mismatch on Colab)."""
    from kdreid.benchmark import measure_latency

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 4 * 2, 5)

        def forward(self, x):
            return {"bn_feat": self.fc(x.flatten(1))}

    m = Tiny()
    measure_latency(m, torch.device("cpu"), 2, size=(4, 2), warmup=1, runs=1, half=True)
    assert m.fc.weight.dtype == torch.float32
    measure_latency(m, torch.device("cpu"), 2, size=(4, 2), warmup=1, runs=1)


def test_cosine_kd_is_scale_free_and_learns_a_projector():
    from kdreid.losses import CosineKD

    t = torch.randn(16, 40)
    same = CosineKD(40, 40)  # identity projector
    assert same(t, t * 3).item() == pytest.approx(0, abs=1e-6)
    kd = CosineKD(8, 40)
    s = torch.randn(16, 8, requires_grad=True)
    loss = kd(s, t)
    loss.backward()
    assert 0.5 < loss.item() < 1.5  # O(1) for unrelated features, whatever the dimension
    assert kd.proj.weight.grad is not None and torch.isfinite(s.grad).all()


def _unlabeled_cfg(root: Path, extra: list[str]):
    ckpt = root / "r18.pth"
    torch.save({"model": ReIDNet("resnet18", num_classes=6, pretrained=False).state_dict()}, ckpt)
    return _cfg(
        root,
        [
            "model={backbone: osnet_x0_25, pretrained: false, embed_dim: 256}",  # != teacher 512
            f"output_dir={root / 'unl'}",
            "kd.enabled=true",
            "kd.losses={dkd: 1.0}",
            f"teacher={{backbone: resnet18, checkpoint: {ckpt}}}",
            "unlabeled={enabled: true, dataset: dukemtmc, root: " + str(root / "data") + ","
            " batch_size: 4, weight: 0.5, losses: {cosine: 1.0, simdist: 1.0, logit: 1.0}}",
            *extra,
        ],
    )


def test_unlabeled_distillation_end_to_end(fake_data: Path):
    """Method 4: Market (labeled) + Duke train images with labels discarded; label-free KD terms
    on the extra batch, logged with a u_ prefix; projector state survives a resume."""
    from test_cross_domain import _make_fake_duke

    from kdreid.data import build_unlabeled_loader

    _make_fake_duke(fake_data / "data")
    cfg = _unlabeled_cfg(fake_data, ["train.epochs=2"])
    tl, vl, ds = build_loaders(cfg)
    unl = build_unlabeled_loader(cfg)
    assert len(unl.dataset) == 15 and all(pid == -1 for _, pid, _ in unl.dataset.items)
    log = get_logger(cfg.output_dir, "test_unl")
    mk = lambda: Trainer(  # noqa: E731
        cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
        torch.device("cpu"), log, load_teachers(cfg, ds.num_train_pids), unl,
    )  # fmt: skip
    tr = mk()
    stats = tr.train_one_epoch(0)
    assert {"kd_dkd", "u_kd_cos", "u_kd_simdist", "u_kd_logit"} <= stats.keys()
    assert "u_kd_dkd" not in stats and all(np.isfinite(v) for v in stats.values())
    assert any(p is tr.unl_kds[0].cos_kd.proj.weight for g in tr.optimizer.param_groups
               for p in g["params"])  # fmt: skip
    tr.fit()
    tr2 = mk()
    tr2.try_resume()
    assert tr2.start_epoch == 2
    assert torch.equal(tr2.unl_kds[0].cos_kd.proj.weight, tr.unl_kds[0].cos_kd.proj.weight)


def test_unlabeled_rejects_label_losses_and_missing_loader(fake_data: Path):
    from test_cross_domain import _make_fake_duke

    from kdreid.data import build_unlabeled_loader

    _make_fake_duke(fake_data / "data")
    log = get_logger(fake_data / "x", "test_unl_bad")
    for extra, err in [
        (["unlabeled.losses={dkd: 1.0}"], "need identity labels"),
        ([], "no unlabeled_loader"),
    ]:
        cfg = _unlabeled_cfg(fake_data, extra)
        tl, vl, ds = build_loaders(cfg)
        unl = build_unlabeled_loader(cfg) if extra else None
        with pytest.raises(ValueError, match=err):
            Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
                    torch.device("cpu"), log, load_teachers(cfg, ds.num_train_pids), unl)  # fmt: skip


def test_unlabeled_folder_source(tmp_path: Path):
    from PIL import Image

    from kdreid.config import Config
    from kdreid.data.build import unlabeled_items

    (tmp_path / "a" / "b").mkdir(parents=True)
    for i, ext in enumerate([".jpg", ".png", ".txt"]):
        p = tmp_path / "a" / "b" / f"{i}{ext}"
        if ext == ".txt":
            p.write_text("x")
        else:
            Image.new("RGB", (16, 32)).save(p)
    items = unlabeled_items(Config.wrap({"folder": str(tmp_path / "a")}))
    assert len(items) == 2 and {pid for _, pid, _ in items} == {-1}
