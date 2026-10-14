"""CLIP-ReID teacher: official checkpoint format, normalisation, stride inference, KD end-to-end.

The ViT itself was checked once against the official CLIP-ReID code (identical outputs for
stride 16 and 12); here a tiny width/depth keeps the tests fast."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_smoke import _cfg, _make_fake_market

from kdreid.data import build_loaders
from kdreid.engine import Trainer, load_teachers
from kdreid.models import build_model
from kdreid.models import clipreid as cr
from kdreid.utils import get_logger

TINY = dict(width=64, layers=2, heads=4, patch=16, output_dim=32)


@pytest.fixture(autouse=True)
def tiny_arch(monkeypatch):
    monkeypatch.setitem(cr.ARCHS, "clipreid_tiny", TINY)


def _official_ckpt(path: Path, num_classes: int, size=(64, 32), stride=16, sie=False) -> Path:
    """Save a checkpoint the way the official code does: DataParallel prefix, text branch and
    prompt learner included, fp16 weights."""
    m = cr.CLIPReIDNet(num_classes, "clipreid_tiny", size, stride)
    state = {f"module.{k}": v.half() if v.is_floating_point() else v
             for k, v in m.state_dict().items()}  # fmt: skip
    state["module.prompt_learner.cls_ctx"] = torch.zeros(num_classes, 4, 512)
    state["module.text_encoder.ln_final.weight"] = torch.ones(512)
    if sie:
        state["module.cv_embed"] = torch.zeros(6, TINY["width"])
    torch.save(state, path)
    return path


def test_loads_official_format_and_always_returns_logits(tmp_path):
    ckpt = _official_ckpt(tmp_path / "c.pth", 5)
    m = cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32))
    m.load_checkpoint(str(ckpt))
    m.eval()
    out = m(torch.randn(3, 3, 64, 32))
    assert out["feat"].shape == (3, TINY["width"] + TINY["output_dim"])
    assert torch.equal(out["feat"], out["bn_feat"])  # official test feature: before BN
    assert out["logits"].shape == (3, 5)  # also in eval mode (frozen teacher)
    assert out["feat_map"].shape == (3, TINY["width"], 4, 2)


def test_logit_heads(tmp_path):
    x = torch.randn(2, 3, 64, 32)
    ckpt = _official_ckpt(tmp_path / "c.pth", 5)
    outs = {}
    for head in ("cls", "proj", "mean"):
        m = cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32), logit_head=head)
        m.load_checkpoint(str(ckpt))
        outs[head] = m.eval()(x)["logits"]
    torch.testing.assert_close(outs["mean"], (outs["cls"] + outs["proj"]) / 2)


def test_input_is_renormalised_to_clip_statistics():
    """Student pipeline gives ImageNet-normalised x; the encoder must see (raw - 0.5) / 0.5."""
    m = cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32)).eval()
    raw = torch.rand(2, 3, 64, 32)
    mean = torch.tensor(cr.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(cr.IMAGENET_STD).view(1, 3, 1, 1)
    seen = {}
    m.image_encoder.conv1.register_forward_hook(lambda mod, inp, out: seen.update(x=inp[0]))
    m((raw - mean) / std)
    torch.testing.assert_close(seen["x"], (raw - 0.5) / 0.5, atol=1e-5, rtol=1e-5)


def test_stride_inferred_and_unsupported_checkpoints_rejected(tmp_path):
    assert cr.stride_from_checkpoint(_official_ckpt(tmp_path / "s16.pth", 5), (64, 32)) == 16
    olp = _official_ckpt(tmp_path / "s12.pth", 5, size=(256, 128), stride=12)
    assert cr.stride_from_checkpoint(olp, (256, 128)) == 12
    with pytest.raises(ValueError, match="SIE"):
        cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32)).load_checkpoint(
            str(_official_ckpt(tmp_path / "sie.pth", 5, sie=True))
        )
    with pytest.raises(ValueError, match="identities"):
        cr.CLIPReIDNet(7, "clipreid_tiny", (64, 32)).load_checkpoint(str(tmp_path / "s16.pth"))


def test_dkd_from_clipreid_teacher_end_to_end(tmp_path):
    _make_fake_market(tmp_path / "data")
    ckpt = _official_ckpt(tmp_path / "clip.pth", 6)
    cfg = _cfg(
        tmp_path,
        [
            "model={backbone: osnet_x0_25, pretrained: false}",
            f"output_dir={tmp_path / 'run'}",
            "kd.enabled=true",
            "kd.losses={dkd: 1.0, similarity: 1.0, attention: 1.0}",
            f"teacher={{backbone: clipreid_tiny, checkpoint: {ckpt}, logit_scale: 30.0}}",
        ],
    )
    tl, vl, ds = build_loaders(cfg)
    teachers = load_teachers(cfg, ds.num_train_pids)
    assert teachers[0].image_encoder.h == 4  # token grid follows data.size
    tr = Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
                 torch.device("cpu"), get_logger(cfg.output_dir, "test_clip"), teachers)  # fmt: skip
    stats = tr.train_one_epoch(0)
    assert {"kd_dkd", "kd_sim", "kd_at"} <= stats.keys()
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
