from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError("configuration root must be a mapping")
    validate_config(cfg)
    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(repo_root())
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_repo_root"]) / path


def validate_config(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {})
    runtime = cfg.get("runtime", {})
    model = cfg.get("catboost", {})
    if data.get("target_col") != "control_success":
        raise ValueError("baseline target must be control_success")
    if int(data.get("validation_season", -1)) != 2024:
        raise ValueError("the recovered SAFE baseline is defined on validation season 2024")
    physical = str(runtime.get("physical_gpu", ""))
    logical = str(runtime.get("catboost_device", ""))
    if physical != "2":
        raise ValueError("SAFE training is pinned to physical RTX 4090 GPU 2")
    if logical != "0" or any(token in logical.lower() for token in (",", ":", "all")):
        raise ValueError("CatBoost must use exactly logical device '0' after CUDA_VISIBLE_DEVICES=2")
    if int(model.get("iterations", 0)) < 600:
        raise ValueError("recovered recipe requires at least 600 trees")
    if int(model.get("depth", 0)) != 8:
        raise ValueError("recovered recipe requires depth=8")
