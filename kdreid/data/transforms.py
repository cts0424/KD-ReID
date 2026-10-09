from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(size: list[int], is_train: bool, cfg_aug: dict | None = None):
    """size = [H, W]; standard ReID recipe: flip + pad/crop + random erasing
    (+ optional color jitter, off by default)."""
    cfg_aug = cfg_aug or {}
    h, w = size
    if not is_train:
        return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ops = [T.Resize((h, w))]
    if cfg_aug.get("flip", True):
        ops.append(T.RandomHorizontalFlip(p=0.5))
    pad = cfg_aug.get("pad", 10)
    if pad:
        ops += [T.Pad(pad), T.RandomCrop((h, w))]
    cj = cfg_aug.get("color_jitter", 0)  # strength s -> brightness/contrast/saturation = s
    if cj:
        ops.append(T.RandomApply([T.ColorJitter(cj, cj, cj, 0)], p=0.5))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    erase = cfg_aug.get("random_erasing", 0.5)
    if erase:
        ops.append(T.RandomErasing(p=erase, value="random"))
    return T.Compose(ops)
