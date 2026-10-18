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


TEXT = dict(n_ctx=4, prefix_len=5, context=16, width=64, layers=2, heads=1)


def _tiny_text(num_classes: int) -> cr.TextBranch:
    """Random tiny text branch (same layout as CLIP-ReID's, smaller sizes)."""
    torch.manual_seed(0)
    t = cr.TextBranch(num_classes, embed_dim=TINY["output_dim"], **TEXT)
    for p in t.parameters():
        torch.nn.init.normal_(p, std=0.2)
    for b in ("token_prefix", "token_suffix"):
        getattr(t, b).normal_(std=0.2)
    return t


def _official_ckpt(path: Path, num_classes: int, size=(64, 32), stride=16, sie=False,
                   text: cr.TextBranch | None = None) -> Path:  # fmt: skip
    """Save a checkpoint the way the official code does: DataParallel prefix, text branch and
    prompt learner included (official key names), fp16 weights."""
    m = cr.CLIPReIDNet(num_classes, "clipreid_tiny", size, stride)
    state = {f"module.{k}": v.half() if v.is_floating_point() else v
             for k, v in m.state_dict().items()}  # fmt: skip
    text = text or _tiny_text(num_classes)
    for k, v in text.state_dict().items():
        pre = (
            "prompt_learner."
            if k in ("cls_ctx", "token_prefix", "token_suffix")
            else "text_encoder."
        )
        state[f"module.{pre}{k}"] = v.half()
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


def test_text_prototype_logits_follow_the_official_i2t_definition(tmp_path):
    """logit_head='text': prototypes come from the checkpoint's prompts + text encoder (official
    key names, fp16), logits = projected CLS feature . prototype (CLIP-ReID stage-2 i2t)."""
    text = _tiny_text(5)
    ckpt = _official_ckpt(tmp_path / "c.pth", 5, text=text)
    m = cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32), logit_head="text")
    m.load_checkpoint(str(ckpt))
    ref = text.half().float().prototypes()  # what the fp16 checkpoint encodes
    torch.testing.assert_close(m.text_prototypes(), ref, atol=1e-4, rtol=1e-4)
    out = m.eval()(torch.randn(3, 3, 64, 32))
    f_proj = out["feat"][:, TINY["width"] :]
    torch.testing.assert_close(out["logits"], f_proj @ ref.t(), atol=1e-4, rtol=1e-4)
    assert m.text.default_eot == TEXT["prefix_len"] + TEXT["n_ctx"] + 2


def test_text_branch_is_causal_and_per_identity():
    """Tokens after the EOT position must not matter (causal mask); changing one identity's
    learned context changes only that identity's prototype."""
    t = _tiny_text(4)
    p0 = t.prototypes()
    with torch.no_grad():
        t.token_suffix[:, 3:] += 5.0  # suffix index 3 = position 12 > EOT 11
    torch.testing.assert_close(t.prototypes(), p0)
    with torch.no_grad():  # random, not a constant shift (LayerNorm removes a constant)
        t.cls_ctx[2] += torch.randn_like(t.cls_ctx[2])
    p1 = t.prototypes()
    assert not torch.allclose(p1[2], p0[2])
    torch.testing.assert_close(p1[[0, 1, 3]], p0[[0, 1, 3]])


def test_text_head_needs_a_text_branch(tmp_path):
    ckpt = tmp_path / "noText.pth"
    m = cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32))
    torch.save({f"module.{k}": v for k, v in m.state_dict().items()}, ckpt)
    with pytest.raises(ValueError, match="text branch"):
        cr.CLIPReIDNet(5, "clipreid_tiny", (64, 32), logit_head="text").load_checkpoint(str(ckpt))


def test_dkd_from_clipreid_text_prototypes_end_to_end(tmp_path):
    _make_fake_market(tmp_path / "data")
    ckpt = _official_ckpt(tmp_path / "clip.pth", 6)
    cfg = _cfg(
        tmp_path,
        [
            "model={backbone: osnet_x0_25, pretrained: false}",
            f"output_dir={tmp_path / 'run'}",
            "kd.enabled=true",
            "kd.losses={dkd: 1.0}",
            f"teacher={{backbone: clipreid_tiny, checkpoint: {ckpt}, logit_head: text,"
            " logit_scale: 2.0}",
        ],
    )
    tl, vl, ds = build_loaders(cfg)
    teachers = load_teachers(cfg, ds.num_train_pids)
    assert teachers[0].text is not None
    tr = Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
                 torch.device("cpu"), get_logger(cfg.output_dir, "test_clip_text"), teachers)  # fmt: skip
    stats = tr.train_one_epoch(0)
    assert "kd_dkd" in stats and all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_check_teacher_reports_text_prototypes_and_eot_sweep(tmp_path, monkeypatch, capsys):
    """tools/check_teacher.py with logit_head=text: text head row, EOT sweep (default index is
    the best one for a checkpoint built with it) and a logit_scale fit."""
    import importlib.util
    import sys

    _make_fake_market(tmp_path / "data")
    ckpt = _official_ckpt(tmp_path / "clip.pth", 6)
    root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "t.yaml"
    cfg.write_text(
        f"_base_: {root / 'configs/base.yaml'}\n"
        f"output_dir: {tmp_path / 'out'}\n"
        "teacher: {backbone: clipreid_tiny, checkpoint: " + str(ckpt) + ", logit_head: text}\n"
    )
    spec = importlib.util.spec_from_file_location("ct", root / "tools/check_teacher.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    overrides = [f"data.root={tmp_path / 'data'}", "data.size=[64,32]", "data.batch_size=8",
                 "data.num_instances=2", "data.test_batch_size=16", "data.num_workers=0"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["ct", "--config", str(cfg), "--batches", "2", *overrides])
    ct.main()
    out = capsys.readouterr().out
    assert "head=text (used for KD)" in out
    assert "end-of-text index sweep (default 11" in out
    assert "fitted teacher.logit_scale" in out and "retrieval (flip=True)" in out


def test_unset_logit_scale_refuses_to_train(tmp_path):
    _make_fake_market(tmp_path / "data")
    ckpt = _official_ckpt(tmp_path / "clip.pth", 6)
    cfg = _cfg(tmp_path, ["model={backbone: osnet_x0_25, pretrained: false}",
                          f"output_dir={tmp_path / 'run'}", "kd.enabled=true", "kd.losses={dkd: 1.0}",
                          f"teacher={{backbone: clipreid_tiny, checkpoint: {ckpt}, logit_scale: null}}"])  # fmt: skip
    tl, vl, ds = build_loaders(cfg)
    with pytest.raises(ValueError, match="logit_scale is unset"):
        Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
                torch.device("cpu"), get_logger(cfg.output_dir, "t"), load_teachers(cfg, 6))  # fmt: skip


def test_eval_matrix_evaluates_a_reference_teacher_config(tmp_path, monkeypatch, capsys):
    """tools/eval_matrix.py --teacher-configs: an official-format CLIP-ReID checkpoint is evaluated
    in-domain and zero-shot without a training run (identity count read from the checkpoint)."""
    import importlib.util
    import json
    import sys

    from test_cross_domain import _make_fake_duke

    _make_fake_market(tmp_path / "data")
    _make_fake_duke(tmp_path / "data")
    ckpt = _official_ckpt(tmp_path / "clip.pth", 6)
    root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "ref.yaml"
    cfg.write_text(
        f"_base_: {root / 'configs/base.yaml'}\n"
        f"output_dir: {tmp_path / 'out'}\n"
        "data: {size: [64, 32]}\n"
        "teacher: {backbone: clipreid_tiny, checkpoint: " + str(ckpt) + "}\n"
    )
    spec = importlib.util.spec_from_file_location("em", root / "tools/eval_matrix.py")
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    assert em.num_classes_from_checkpoint(ckpt) == 6
    out = tmp_path / "m.json"
    d = tmp_path / "data"
    argv = ["em", "--teacher-configs", str(cfg), "--datasets", f"market1501:{d}",
            f"dukemtmc:{d}", "--workers", "0", "--out", str(out)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    em.main()
    rows = json.loads(out.read_text())
    assert {r["run"] for r in rows} == {"ref:ref"}
    assert {r["backbone"] for r in rows} == {"clipreid_tiny"}
    assert {(r["test"], r["zero_shot"]) for r in rows} == {
        ("market1501", False), ("dukemtmc", True)
    }  # fmt: skip
    assert "re-ranking: OFF" in capsys.readouterr().out


def test_benchmark_supports_clipreid():
    """tools/benchmark.py --models clipreid_*: deploy params exclude both identity heads and the
    deploy output is the retrieval feature (768 + 512 for ViT-B/16)."""
    from kdreid.benchmark import benchmark
    from kdreid.config import Config

    m = build_model(Config(backbone="clipreid_tiny", input_size=[256, 128]), 7)
    r = benchmark("clip_tiny", m, [torch.device("cpu")], [1], runs=1)
    heads = sum(p.numel() for n, p in m.named_parameters() if n.startswith("classifier"))
    assert r.classifier_params_m * 1e6 == heads > 0
    assert r.feat_dim == TINY["width"] + TINY["output_dim"]
    assert r.gmacs > 0 and r.latency_ms["cpu_b1"] > 0
    # fp16 timing (GPU benchmark) must run and leave the model in fp32
    from kdreid.benchmark import measure_latency

    measure_latency(m, torch.device("cpu"), 2, warmup=1, runs=1, half=True)
    assert all(p.dtype == torch.float32 for p in m.parameters())
