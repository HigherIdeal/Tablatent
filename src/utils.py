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
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


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

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert) + "\n", encoding="utf-8")
