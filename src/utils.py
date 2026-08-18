from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # The active Bitaboost baseline is CatBoost-only. Torch is optional so a clean
    # baseline environment does not need a large PyTorch install just for seeding.
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed; device() is only for optional neural experiments") from exc
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    result = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "latents": root / "latents",
        "logs": root / "logs",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def save_json(obj, path: str | Path) -> None:
    if isinstance(obj, (str, Path)) and not isinstance(path, (str, Path)):
        obj, path = path, obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)
