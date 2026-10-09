from __future__ import annotations

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
