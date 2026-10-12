"""Export a trained OSNet run as a standalone package and use it like an outside user would."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from test_smoke import _cfg, _make_fake_market

from kdreid.data import build_loaders
from kdreid.engine import Trainer
from kdreid.models import build_model
from kdreid.utils import get_logger

ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pkg")
    _make_fake_market(tmp / "data")
    cfg = _cfg(tmp, ["model={backbone: osnet_x0_25, pretrained: false, embed_dim: 512}",
                     f"output_dir={tmp / 'run'}"])  # fmt: skip
    tl, vl, ds = build_loaders(cfg)
    Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
            torch.device("cpu"), get_logger(cfg.output_dir, "test_pkg")).fit()  # fmt: skip
    out = _run([ROOT / "tools/export_package.py", "--run", tmp / "run", "--out", tmp / "out",
                "--name", "pkg", "--eval-root", tmp / "data"], cwd=tmp)  # fmt: skip
    assert "checks" in out
    return tmp, tmp / "out" / "pkg"


def test_package_contents_and_metadata(package):
    tmp, pkg = package
    names = {p.name for p in pkg.iterdir()}
    assert {"model.onnx", "weights.pt", "osnet_reid.py", "reid_infer.py", "selftest.npz",
            "metadata.json", "README.md", "requirements.txt"} <= names  # fmt: skip
    meta = json.loads((pkg / "metadata.json").read_text())
    assert meta["model"]["feature_dim"] == 512
    assert 0.15e6 < meta["model"]["parameters"] < 0.25e6
    assert {"flip", "no_flip"} <= meta["metrics"].keys()
    assert meta["verification"]["onnx_vs_torch_max_abs"] < 1e-3
    assert "{{" not in (pkg / "README.md").read_text()
    assert not any(k.startswith("classifier") for k in torch.load(pkg / "weights.pt"))
    with zipfile.ZipFile(tmp / "out" / "pkg.zip") as z:
        assert "pkg/model.onnx" in z.namelist()


def test_package_works_standalone(package, tmp_path):
    """Unzip elsewhere and run with the repo NOT importable: the package must be self-contained."""
    tmp, _ = package
    with zipfile.ZipFile(tmp / "out" / "pkg.zip") as z:
        z.extractall(tmp_path)
    pkg = tmp_path / "pkg"
    infer = pkg / "reid_infer.py"
    out = _run(["-I", infer, "--backend", "both", "selftest"], cwd=tmp_path)
    assert out.count("[ok]") == 2
    gallery = tmp / "data" / "Market-1501-v15.09.15" / "bounding_box_test"
    _run(["-I", infer, "extract", "--images", gallery, "--out", tmp_path / "f"], cwd=tmp_path)
    feats = np.load(tmp_path / "f.npy")
    assert feats.shape[1] == 512 and np.allclose(np.linalg.norm(feats, axis=1), 1, atol=1e-4)
    query = sorted((tmp / "data" / "Market-1501-v15.09.15" / "query").glob("*.jpg"))[0]
    out = _run(["-I", infer, "search", "--query", query, "--gallery", gallery, "--topk", "3"],
               cwd=tmp_path)  # fmt: skip
    assert out.count("  1. ") == 1


def test_package_preprocess_matches_training_transform(package):
    """reid_infer.preprocess must equal the evaluation transform used to measure accuracy."""
    tmp, pkg = package
    sys.path.insert(0, str(pkg))
    try:
        import reid_infer
        from PIL import Image

        from kdreid.data import build_transforms

        img = Image.open(next((tmp / "data" / "Market-1501-v15.09.15" / "query").glob("*.jpg")))
        meta = json.loads((pkg / "metadata.json").read_text())
        a = reid_infer.preprocess(img, meta)
        b = build_transforms(meta["preprocess"]["size_hw"], is_train=False)(img).numpy()
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 1e-5
    finally:
        sys.path.remove(str(pkg))


def test_resnet_package(tmp_path):
    """ResNet runs (e.g. the R50 teacher, packaged for system-level comparison) export too:
    last-stride-1 body + BNNeck, ONNX checked against PyTorch, standalone selftest passes."""
    _make_fake_market(tmp_path / "data")
    cfg = _cfg(tmp_path, ["model={backbone: resnet18, pretrained: false, last_stride: 1}",
                          f"output_dir={tmp_path / 'run'}"])  # fmt: skip
    tl, vl, ds = build_loaders(cfg)
    Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
            torch.device("cpu"), get_logger(cfg.output_dir, "test_pkg_r18")).fit()  # fmt: skip
    _run([ROOT / "tools/export_package.py", "--run", tmp_path / "run", "--out", tmp_path / "out",
          "--name", "r18"], cwd=tmp_path)  # fmt: skip
    pkg = tmp_path / "out" / "r18"
    meta = json.loads((pkg / "metadata.json").read_text())
    assert meta["model"]["backbone"] == "resnet18" and meta["model"]["feature_dim"] == 512
    assert meta["model"]["last_stride"] == 1
    assert meta["verification"]["onnx_vs_torch_max_abs"] < 1e-3
    assert "{{" not in (pkg / "README.md").read_text()
    out = _run(["-I", pkg / "reid_infer.py", "--backend", "both", "selftest"], cwd=tmp_path)
    assert out.count("[ok]") == 2
