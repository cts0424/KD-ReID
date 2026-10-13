"""Cross-domain evaluation: several datasets under one root, Market-trained model on Duke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_smoke import _cfg, _make_fake_market

from kdreid.data import build_dataset, build_loaders
from kdreid.engine import Trainer
from kdreid.models import build_model
from kdreid.utils import get_logger

ROOT = Path(__file__).resolve().parents[1]


def _make_fake_duke(root: Path, folder: str = "DukeMTMC-reID", n_ids: int = 5) -> None:
    base = root / folder
    rng = np.random.default_rng(1)
    for split, offset in [("bounding_box_train", 0), ("query", 50), ("bounding_box_test", 50)]:
        (base / split).mkdir(parents=True)
        for pid in range(1, n_ids + 1):
            for cam in (1, 2, 3):
                if split == "query" and cam != 1:
                    continue
                img = rng.integers(0, 255, (32, 16, 3), dtype=np.uint8)
                Image.fromarray(img).save(
                    base / split / f"{pid + offset:04d}_c{cam}_f{pid:07d}.jpg"
                )


def test_two_datasets_under_one_root_resolve_correctly(tmp_path: Path):
    _make_fake_market(tmp_path)
    _make_fake_duke(tmp_path)
    assert build_dataset("market1501", tmp_path).num_train_pids == 6
    assert build_dataset("dukemtmc", tmp_path).num_train_pids == 5
    assert build_dataset("dukemtmc", tmp_path).num_train_cams == 3


def test_ambiguous_folder_names_raise_instead_of_guessing(tmp_path: Path):
    _make_fake_market(tmp_path)
    _make_fake_duke(tmp_path, folder="some_other_dataset")
    with pytest.raises(FileNotFoundError, match="Cannot tell"):
        build_dataset("dukemtmc", tmp_path)


def test_png_dataset(tmp_path: Path):
    """CUHK03-NP as released (cuhk03-np/{detected,labeled}/...) next to Market under one root:
    each name must resolve to its own folder."""
    _make_fake_market(tmp_path)
    rng = np.random.default_rng(2)
    for kind, n_pids in (("detected", 2), ("labeled", 3)):
        base = tmp_path / "cuhk03-np" / kind
        for split in ("bounding_box_train", "query", "bounding_box_test"):
            (base / split).mkdir(parents=True)
            for pid in range(1, n_pids + 1):
                for cam in (1, 2):
                    img = rng.integers(0, 255, (32, 16, 3), dtype=np.uint8)
                    Image.fromarray(img).save(base / split / f"{pid:04d}_c{cam}_{pid}.png")
    det = build_dataset("cuhk03np", tmp_path)
    lab = build_dataset("cuhk03np_labeled", tmp_path)
    assert det.num_train_pids == 2 and lab.num_train_pids == 3
    assert det.num_train_cams == 2 and all("detected" in p for p, _, _ in det.gallery)
    assert build_dataset("market1501", tmp_path).num_train_pids != 2


def test_eval_matrix_market_to_duke_zero_shot(tmp_path: Path):
    _make_fake_market(tmp_path / "data")
    _make_fake_duke(tmp_path / "data")
    cfg = _cfg(tmp_path, ["model.backbone=resnet18", f"output_dir={tmp_path / 'run'}"])
    tl, vl, ds = build_loaders(cfg)
    log = get_logger(cfg.output_dir, "test_xd")
    Trainer(cfg, build_model(cfg.model, ds.num_train_pids), tl, vl, len(ds.query),
            torch.device("cpu"), log).fit()  # fmt: skip
    out = tmp_path / "m.json"
    res = subprocess.run(
        [sys.executable, str(ROOT / "tools/eval_matrix.py"), "--runs", str(tmp_path / "run"),
         "--datasets", f"market1501:{tmp_path / 'data'}", f"dukemtmc:{tmp_path / 'data'}",
         "--workers", "0", "--out", str(out)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    rows = json.loads(out.read_text())
    assert {(r["test"], r["zero_shot"], r["flip"]) for r in rows} == {
        ("market1501", False, True), ("market1501", False, False),
        ("dukemtmc", True, True), ("dukemtmc", True, False),
    }  # fmt: skip
    assert "re-ranking: OFF" in res.stdout


def test_wrong_dataset_folder_is_refused(tmp_path: Path, monkeypatch):
    """A 'cuhk03np' folder whose split sizes are Duke's must stop the run, naming Duke."""
    from kdreid.data import datasets as D

    monkeypatch.delenv(D.SKIP_CHECK_ENV, raising=False)
    fake = D.ReIDDataset(train=[], query=[], gallery=[], name="cuhk03np", root="/x")
    monkeypatch.setattr(D, "dataset_stats", lambda ds: D.STANDARD_STATS["dukemtmc"])
    with pytest.raises(ValueError, match="exactly the dukemtmc numbers"):
        D.check_standard(fake)
    monkeypatch.setattr(D, "dataset_stats", lambda ds: D.STANDARD_STATS["cuhk03np"])
    D.check_standard(fake)  # the real split passes
    _make_fake_market(tmp_path)
    with pytest.raises(ValueError, match="published market1501 split"):
        build_dataset("market1501", tmp_path)  # no folder has 751 ids


def test_split_folders_unpacked_into_root_do_not_shadow_other_datasets(tmp_path: Path):
    """A zip that unpacks bounding_box_train/ straight into root (Duke here) must not be returned
    for another dataset name just because root/<sub> is missing."""
    _make_fake_duke(tmp_path)  # creates tmp_path/<folder>/...; move its splits up to root
    duke = next(p.parent for p in tmp_path.rglob("bounding_box_train"))
    for split in duke.iterdir():
        split.rename(tmp_path / split.name)
    base = tmp_path / "cuhk03-np" / "detected"
    for split in ("bounding_box_train", "query", "bounding_box_test"):
        (base / split).mkdir(parents=True)
        for pid in (1, 2):
            for cam in (1, 2):
                Image.new("RGB", (16, 32)).save(base / split / f"{pid:04d}_c{cam}_{pid}.png")
    ds = build_dataset("cuhk03np", tmp_path)
    assert ds.root == str(base) and ds.num_train_pids == 2
