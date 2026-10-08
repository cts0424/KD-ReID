"""YAML config loading with `_base_` inheritance and `key.sub=value` CLI overrides."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access (cfg.model.name)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base is not None:
        data = _merge(_load_yaml((path.parent / base).resolve()), data)
    return data


_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _coerce(obj: Any) -> Any:
    """PyYAML (YAML 1.1) reads `1e-3` as a string; turn such strings into floats."""
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(v) for v in obj]
    if isinstance(obj, str) and _FLOAT_RE.match(obj):
        return float(obj)
    return obj


def _set_by_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, resolving `_base_`, then apply `a.b=value` overrides.

    Override values are parsed with YAML, so `train.lr=3e-4`, `kd.enabled=false`
    and `data.root=/content/data` all get the right type.
    """
    data = _load_yaml(Path(path).resolve())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_by_path(data, key.strip(), yaml.safe_load(raw))
    return Config.wrap(_coerce(data))


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
