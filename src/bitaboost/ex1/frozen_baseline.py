from __future__ import annotations

from pathlib import Path

import numpy as np


def load_frozen_baseline(path: Path, y: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Load the frozen SAFE prediction vector without requiring pickle.

    The historical SAFE NPZ stores ``gt`` as an object-dtype NumPy array, so touching
    that member with ``allow_pickle=False`` raises.  EX1 does not need to deserialize
    that object array: the exact validation target vector and row count are sufficient
    to verify that the frozen prediction artifact corresponds to the same 2024 split.

    This deliberately keeps ``allow_pickle=False`` and never reads the object payload.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"frozen baseline predictions not found: {path}. "
            "Run the stable baseline once; EX1 never retrains the forward model."
        )

    with np.load(path, allow_pickle=False) as data:
        pred = np.asarray(data["pred"], dtype=np.float64)
        y_ref = np.asarray(data["y"], dtype=np.float64)
        files = set(data.files)

    if "pred" not in files or "y" not in files:
        raise RuntimeError("frozen baseline artifact is missing pred/y arrays")
    if len(pred) != len(y) or not np.array_equal(y_ref, y):
        raise RuntimeError("frozen baseline predictions do not match the 2024 validation rows")
    if len(gt) != len(pred):
        raise RuntimeError("current 2024 game_type vector length does not match frozen predictions")
    if not np.isfinite(pred).all():
        raise RuntimeError("frozen baseline predictions contain non-finite values")

    return pred
