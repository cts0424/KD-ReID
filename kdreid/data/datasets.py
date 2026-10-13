"""Dataset indexes. Each dataset produces three lists of (img_path, pid, camid)."""

from __future__ import annotations

import os
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
    root: str = ""
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
            f"=> {self.name}  ({self.root})\n"
            f"  train   | {stat(self.train)}\n"
            f"  query   | {stat(self.query)}\n"
            f"  gallery | {stat(self.gallery)}"
        )


def _scan(folder: Path, relabel: bool) -> list[Item]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    raw: list[tuple[str, int, int]] = []
    for p in sorted([*folder.glob("*.jpg"), *folder.glob("*.png")]):
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


# Sub-folder layout for "Market-style" datasets: (preferred folder name, train, query, gallery).
_LAYOUTS = {
    "market1501": ("Market-1501-v15.09.15", "bounding_box_train", "query", "bounding_box_test"),
    "dukemtmc": ("DukeMTMC-reID", "bounding_box_train", "query", "bounding_box_test"),
    # CUHK03 new protocol (767/700 split), detected boxes, Market-style folders (PNG files)
    "cuhk03np": ("detected", "bounding_box_train", "query", "bounding_box_test"),
    # same split, hand-labelled boxes (the easier CUHK03-L setting, reported by some papers)
    "cuhk03np_labeled": ("labeled", "bounding_box_train", "query", "bounding_box_test"),
}
# Path keywords that identify each dataset when several are unpacked under the same root.
_KEYWORDS = {
    "market1501": ("market",),
    "dukemtmc": ("duke",),
    "cuhk03np": ("cuhk03", "detected"),
    "cuhk03np_labeled": ("cuhk03", "labeled"),
}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _all_hits(root: Path, train_dir: str) -> list[Path]:
    """Every folder (root itself and up to 3 levels down) that contains `train_dir`."""
    hits = [root] if (root / train_dir).is_dir() else []
    hits += sorted(
        p.parent for depth in ("*", "*/*", "*/*/*") for p in root.glob(f"{depth}/{train_dir}")
    )
    return [h for h in hits if "__MACOSX" not in h.parts]


def _locate(root: Path, name: str, sub: str, train_dir: str) -> Path:
    """Find the folder that holds `train_dir`: root/sub if it exists, else search root and up to
    3 levels down — zips from Kaggle etc. often unpack with an extra, renamed or missing top folder.

    With several candidates (e.g. Market and Duke both under /content/data) the one whose path
    contains the dataset's keywords is used. root itself never wins that way: a zip that unpacks
    its split folders straight into root must not be mistaken for every other dataset (this is
    how a "cuhk03np" run once trained on DukeMTMC-reID). Still ambiguous -> raise.
    """
    if (root / sub / train_dir).is_dir():
        return root / sub
    hits = _all_hits(root, train_dir)
    if not hits:
        raise FileNotFoundError(f"No '{train_dir}' folder found under {root}")
    if len(hits) == 1:
        return hits[0]
    keys = _KEYWORDS.get(name, (name,))
    matched = [
        h for h in hits if h != root and all(k in _norm(str(h.relative_to(root))) for k in keys)
    ]
    if len(matched) == 1:
        return matched[0]
    raise FileNotFoundError(
        f"Cannot tell which folder under {root} is {name!r}: candidates "
        f"{[str(h) for h in (matched or hits)]}. Set data.root to the exact dataset folder."
    )


# Published split sizes: (train ids, train images, query images, gallery images), with Market's
# pid -1 junk excluded from the gallery. Used to refuse a wrong or partial folder instead of
# silently training on it (e.g. a "cuhk03np" run that actually found DukeMTMC-reID).
STANDARD_STATS = {
    "market1501": (751, 12936, 3368, 15913),
    "dukemtmc": (702, 16522, 2228, 17661),
    "cuhk03np": (767, 7365, 1400, 5332),
    "cuhk03np_labeled": (767, 7368, 1400, 5328),
}
SKIP_CHECK_ENV = "KDREID_SKIP_DATASET_CHECK"  # set by the test-suite (synthetic data)


def dataset_stats(ds: ReIDDataset) -> tuple[int, int, int, int]:
    return (ds.num_train_pids, len(ds.train), len(ds.query), len(ds.gallery))


def check_standard(ds: ReIDDataset) -> None:
    expected = STANDARD_STATS.get(ds.name)
    got = dataset_stats(ds)
    if expected is None or got == expected or os.environ.get(SKIP_CHECK_ENV):
        return
    lookalike = [n for n, st in STANDARD_STATS.items() if st == got]
    hint = f" — these are exactly the {lookalike[0]} numbers" if lookalike else ""
    raise ValueError(
        f"{ds.name} at {ds.root}: got (train ids, train imgs, query, gallery) = {got}, "
        f"expected {expected}{hint}. Wrong or incomplete folder; set data.root to the exact "
        f"dataset folder (or export {SKIP_CHECK_ENV}=1 for a deliberately modified split)."
    )


def _build(name: str, base: Path) -> ReIDDataset:
    _, tr, q, g = _LAYOUTS[name]
    return ReIDDataset(
        train=_scan(base / tr, relabel=True),
        query=_scan(base / q, relabel=False),
        gallery=_scan(base / g, relabel=False),
        name=name,
        root=str(base),
    )


def build_dataset(name: str, root: str | Path) -> ReIDDataset:
    """Index a dataset under `root` and verify its split sizes against the published ones.

    If the folder found by name is ambiguous or has the wrong sizes, every candidate folder is
    tried and the one with exactly the published sizes is used; none -> a clear error."""
    name = name.lower()
    if name not in _LAYOUTS:
        raise KeyError(f"Unknown dataset {name!r}; available: {sorted(_LAYOUTS)}")
    sub, tr, _, _ = _LAYOUTS[name]
    root = Path(root)
    try:
        ds = _build(name, _locate(root, name, sub, tr))
        check_standard(ds)
        return ds
    except (FileNotFoundError, ValueError) as err:
        expected = STANDARD_STATS.get(name)
        hits = _all_hits(root, tr)
        if expected is None or os.environ.get(SKIP_CHECK_ENV) or not hits:
            raise
        found = {}
        for h in hits:
            cand = _build(name, h)
            if dataset_stats(cand) == expected:
                return cand
            found[str(h)] = dataset_stats(cand)
        raise ValueError(
            f"No folder under {root} has the published {name} split {expected} "
            f"(train ids, train imgs, query, gallery). Candidates: {found}"
        ) from err
