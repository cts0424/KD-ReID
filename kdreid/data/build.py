from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .datasets import Item, ReIDDataset, build_dataset
from .sampler import RandomIdentitySampler
from .transforms import build_transforms


class ImageDataset(Dataset):
    def __init__(self, items: list[Item], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, pid, cam = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid, cam


def build_loaders(cfg) -> tuple[DataLoader, DataLoader, ReIDDataset]:
    """Returns (train_loader, test_loader, dataset). test_loader = query + gallery concatenated."""
    ds = build_dataset(cfg.data.name, cfg.data.root)
    d = cfg.data
    train_set = ImageDataset(ds.train, build_transforms(d.size, True, d.get("aug")))
    test_set = ImageDataset(ds.query + ds.gallery, build_transforms(d.size, False))

    train_loader = DataLoader(
        train_set,
        batch_size=d.batch_size,
        sampler=RandomIdentitySampler(ds.train, d.batch_size, d.num_instances),
        num_workers=d.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=d.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=d.test_batch_size,
        shuffle=False,
        num_workers=d.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, ds


def build_test_loader(
    name: str, root, size: list[int], batch_size: int = 256, num_workers: int = 2
) -> tuple[DataLoader, ReIDDataset]:
    """Query + gallery loader for any dataset — used for cross-domain (zero-shot) evaluation,
    where the model was trained on a different dataset."""
    ds = build_dataset(name, root)
    loader = DataLoader(
        ImageDataset(ds.query + ds.gallery, build_transforms(size, False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, ds


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def unlabeled_items(ucfg) -> list[Item]:
    """Images for label-free distillation (method 4). Either the *training split* of a known
    dataset (`dataset: cuhk03np`, labels discarded; its query / gallery identities stay unseen)
    or every image under a folder (`folder: /path`)."""
    if ucfg.get("folder"):
        paths = sorted(
            str(p) for p in Path(ucfg.folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise FileNotFoundError(f"unlabeled.folder {ucfg.folder}: no images")
        return [(p, -1, -1) for p in paths]
    ds = build_dataset(ucfg.dataset, ucfg.root)
    return [(path, -1, -1) for path, _, _ in ds.train]


def build_unlabeled_loader(cfg) -> DataLoader | None:
    """Shuffled loader of unlabeled images with the training augmentation, or None when
    `unlabeled.enabled` is false. Identity labels are replaced by -1 and never used."""
    u = cfg.get("unlabeled") or {}
    if not u.get("enabled", False):
        return None
    d = cfg.data
    items = unlabeled_items(u)
    return DataLoader(
        ImageDataset(items, build_transforms(d.size, True, d.get("aug"))),
        batch_size=int(u.get("batch_size", 32)),
        shuffle=True,
        num_workers=d.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=d.num_workers > 0,
    )
