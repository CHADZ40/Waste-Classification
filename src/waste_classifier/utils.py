"""Config loading, seeding and device selection."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    return _merge(_read_config(path.parent / parent), cfg) if parent else cfg


def load_config(path: str | Path) -> dict:
    """Load a config, resolving `extends` and adding `data.classes` from the class mapping."""
    cfg = _read_config(Path(path))
    with open(cfg["data"]["class_mapping"]) as f:
        cfg["data"]["classes"] = list(yaml.safe_load(f))
    return cfg


def checkpoint_path(cfg: dict) -> Path:
    return Path(cfg["output"]["models_dir"]) / f"{cfg['experiment']}.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
