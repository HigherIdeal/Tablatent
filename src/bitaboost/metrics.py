from __future__ import annotations

import numpy as np


def brier(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def local_score(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    ref = float(y.mean() * (1.0 - y.mean()))
    if ref <= 0:
        return 0.0
    return max(0.0, 1e5 * (1.0 - brier(y, p) / ref))


def summary(y, p) -> dict[str, float]:
    p = np.asarray(p, dtype=np.float64)
    return {"brier": brier(y, p), "score": local_score(y, p), "mean": float(p.mean()), "std": float(p.std()), "min": float(p.min()), "max": float(p.max())}
