from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import resolve_path
from .runtime import log


# Numerical oracles recovered from the original Tablatent workspace.  They are used
# only for local reproduction auditing; training never reads them.
_COMPONENT_REFS = {
    "mixed": ("mixed_gate_conditional.npz", "pred"),
    "offset": ("offset_residual_f2024_full_cross1_recent_of1/predictions.npz", "recent400"),
    "joint": (
        "joint_outcome_d8_r1_reverse-middle-ball-strike_f2024_s42_fw2_anchor_multi_banchor_cross4_match_count_pressure_domain_auxprof_pressure_i600/predictions.npz",
        "pred_t600",
    ),
    "structured": ("structured_multiclass_f2024_s42_ids/predictions.npz", "t600"),
}
_SAFE_REF = "current_best_safe_2024_predictions.npz"


def _delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return (
        float(np.mean(np.abs(d))),
        float(np.sqrt(np.mean(d * d))),
        float(np.max(np.abs(d))),
    )


def audit_if_available(cfg: dict, y: np.ndarray, components: dict[str, np.ndarray]) -> dict:
    ref_cfg = cfg.get("reference", {})
    if not bool(ref_cfg.get("audit_if_available", True)):
        return {"available": False, "reason": "disabled"}
    root_value = ref_cfg.get("artifact_root")
    if not root_value:
        return {"available": False, "reason": "artifact_root_not_configured"}
    root = resolve_path(cfg, root_value)
    safe_path = root / _SAFE_REF
    if not safe_path.is_file():
        log(f"[audit] original artifacts not found at {root}; prediction audit skipped")
        return {"available": False, "reason": "missing_artifact_root", "root": str(root)}

    safe_ref = np.load(safe_path, allow_pickle=True)
    y_ref = np.asarray(safe_ref["y"], np.float64)
    if len(y_ref) != len(y) or not np.array_equal(y_ref, np.asarray(y, np.float64)):
        raise RuntimeError("reference audit row/target alignment mismatch")

    refs: dict[str, np.ndarray] = {
        "safe": np.asarray(safe_ref["components"][:, 0], np.float64),
        "pred": np.asarray(safe_ref["pred"], np.float64),
    }
    for name, (relative, key) in _COMPONENT_REFS.items():
        path = root / relative
        if path.is_file():
            z = np.load(path, allow_pickle=True)
            if "y" in z.files:
                zy = np.asarray(z["y"], np.float64)
                if len(zy) != len(y) or not np.array_equal(zy, np.asarray(y, np.float64)):
                    raise RuntimeError(f"reference audit alignment mismatch: {name}")
            refs[name] = np.asarray(z[key], np.float64)

    rows = {}
    for name in ("mixed", "offset", "joint", "structured", "safe", "pred"):
        if name not in components or name not in refs:
            continue
        mae, rmse, max_abs = _delta(components[name], refs[name])
        rows[name] = {"mae": mae, "rmse": rmse, "max_abs": max_abs}
        log(f"[audit] {name:<10} MAE={mae:.3e} RMSE={rmse:.3e} MAX={max_abs:.3e}")
    return {"available": True, "root": str(root), "components": rows}
