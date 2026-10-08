from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(output_dir: str | Path, name: str = "kdreid") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(output_dir) / "log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_checkpoint(state: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a Colab disconnect mid-save won't corrupt the last checkpoint


def load_model_weights(model: torch.nn.Module, path: str | Path, strict: bool = True) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model.load_state_dict(state, strict=strict)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
