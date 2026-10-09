"""Dataset indexes. Each dataset produces three lists of (img_path, pid, camid)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

Item = tuple[str, int, int]  # (img_path, pid, camid)

# Matches Market-1501 / DukeMTMC-reID file names, e.g. 0002_c1s1_000451_03.jpg, 0005_c2_f0046985.jpg
_NAME_RE = re.compile(r"([-\d]+)_c(\d+)")


@dataclass
class ReIDDataset:
    train: list[Item]
    query: list[Item]
    gallery: list[Item]
    name: str = ""
    num_train_pids: int = field(init=False)
    num_train_cams: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_train_pids = len({pid for _, pid, _ in self.train})
        self.num_train_cams = len({cam for _, _, cam in self.train})

    def summary(self) -> str:
        def stat(items: list[Item]) -> str:
            pids = {p for _, p, _ in items}
            cams = {c for _, _, c in items}
            return f"{len(pids):>5} ids | {len(items):>6} imgs | {len(cams):>2} cams"

        return (
            f"=> {self.name}\n"
            f"  train   | {stat(self.train)}\n"
            f"  query   | {stat(self.query)}\n"
            f"  gallery | {stat(self.gallery)}"
        )


def _scan(folder: Path, relabel: bool) -> list[Item]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    raw: list[tuple[str, int, int]] = []
    for p in sorted(folder.glob("*.jpg")):
        m = _NAME_RE.search(p.name)
        if m is None:
            continue
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid == -1:  # junk images in Market-1501
            continue
        raw.append((str(p), pid, cam - 1))  # camid is 0-based
    if relabel:
        mapping = {pid: i for i, pid in enumerate(sorted({pid for _, pid, _ in raw}))}
        raw = [(path, mapping[pid], cam) for path, pid, cam in raw]
    return raw


# Sub-folder layout for "Market-style" datasets; root points at the folder containing these.
_LAYOUTS = {
    "market1501": ("Market-1501-v15.09.15", "bounding_box_train", "query", "bounding_box_test"),
    "dukemtmc": ("DukeMTMC-reID", "bounding_box_train", "query", "bounding_box_test"),
}


def _locate(root: Path, sub: str, train_dir: str) -> Path:
    """Find the folder that holds `train_dir`. Tries root/sub, root, then searches up to
    3 levels down — zips from Kaggle etc. often unpack with an extra or renamed top folder."""
    for cand in (root / sub, root):
        if (cand / train_dir).is_dir():
            return cand
    hits = sorted(
        p.parent for depth in ("*", "*/*", "*/*/*") for p in root.glob(f"{depth}/{train_dir}")
    )
    hits = [h for h in hits if "__MACOSX" not in h.parts]
    if not hits:
        raise FileNotFoundError(f"No '{train_dir}' folder found under {root}")
    preferred = [h for h in hits if h.name == sub]
    return (preferred or hits)[0]


def build_dataset(name: str, root: str | Path) -> ReIDDataset:
    name = name.lower()
    if name not in _LAYOUTS:
        raise KeyError(f"Unknown dataset {name!r}; available: {sorted(_LAYOUTS)}")
    sub, tr, q, g = _LAYOUTS[name]
    base = _locate(Path(root), sub, tr)
    return ReIDDataset(
        train=_scan(base / tr, relabel=True),
        query=_scan(base / q, relabel=False),
        gallery=_scan(base / g, relabel=False),
        name=name,
    )
