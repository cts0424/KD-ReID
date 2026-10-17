from __future__ import annotations

import logging
import os
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


class _AppendCloseHandler(logging.Handler):
    """Open, append, close on every record.

    A plain FileHandler keeps the file open for the whole run, and Google Drive (Colab mount)
    does not upload a file that is still open — so a crash lost the entire log. Closing after
    each write lets Drive sync it continuously. Logging is a few lines per minute, so the
    reopen cost is negligible.
    """

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def get_logger(output_dir: str | Path, name: str = "kdreid") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fh = _AppendCloseHandler(Path(output_dir) / "log.txt")
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


DRIVE_ROOT = "/content/drive"


def check_drive_mounted(*paths) -> None:
    """Fail fast when a path lives under /content/drive but Google Drive is not mounted.

    Without this, reads fail with a confusing FileNotFoundError and, worse, writes (logs,
    checkpoints, downloaded weights) silently land on the VM's local disk and vanish with the
    runtime; a local /content/drive folder then also blocks drive.mount()."""
    on_drive = sorted({str(p) for p in paths if p and str(p).startswith(DRIVE_ROOT + "/")})
    if on_drive and not os.path.ismount(DRIVE_ROOT):
        raise RuntimeError(
            f"Google Drive is not mounted at {DRIVE_ROOT}, but these paths are on it: "
            f"{on_drive}. Run drive.mount('/content/drive') first. If mounting fails because "
            f"{DRIVE_ROOT} is not empty, restart the runtime (Runtime -> Disconnect and delete "
            "runtime) instead of deleting the folder."
        )


def cfg_drive_paths(cfg) -> list[str]:
    """Every path in a config that has to exist on / be written to persistent storage."""
    paths = [cfg.get("output_dir"), cfg.get("model", {}).get("pretrained_path")]
    teachers = cfg.get("teachers") or ([cfg.teacher] if cfg.get("teacher") else [])
    paths += [t.get("checkpoint") for t in teachers]
    return [p for p in paths if p]
