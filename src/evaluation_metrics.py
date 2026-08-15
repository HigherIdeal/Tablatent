from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Return Brier metrics with both raw and clipped competition-style scores.

    `raw_score` preserves negative skill, which is important for regime-shift
    diagnostics such as the 2023 fold. `competition_score` is kept as a
    backwards-compatible alias of `clipped_score`.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, p={p.shape}")
    if y.size == 0:
        raise ValueError("empty target")

    brier = float(np.mean((p - y) ** 2))
    target_mean = float(y.mean())
    reference_brier = float(target_mean * (1.0 - target_mean))
    if reference_brier <= 0.0:
        brier_skill = float("nan")
        raw_score = float("nan")
        clipped_score = float("nan")
    else:
        brier_skill = float(1.0 - brier / reference_brier)
        raw_score = float(100000.0 * brier_skill)
        clipped_score = float(max(0.0, raw_score))

    auc = float(roc_auc_score(y, p)) if np.unique(y).size >= 2 else float("nan")
    return {
        "brier": brier,
        "brier_skill": brier_skill,
        "raw_score": raw_score,
        "clipped_score": clipped_score,
        "competition_score": clipped_score,
        "auc": auc,
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "target_mean": target_mean,
        "reference_brier": reference_brier,
    }
