from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bitaboost.baseline import train as train_baseline
from bitaboost.config import load_config


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else _root() / p


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    out: dict[str, np.ndarray] = {}
    # These are trusted project-generated artifacts. gt may be stored as object dtype.
    with np.load(p, allow_pickle=True) as data:
        for key in data.files:
            out[key] = np.asarray(data[key])
    return out


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, np.float64)
    pp = np.asarray(p, np.float64)
    return float(np.mean((yy - pp) ** 2))


def _domain_brier(y: np.ndarray, p: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    g = np.asarray(gt).astype(str)
    for dom in ("R", "F"):
        m = g == dom
        if m.any():
            out[dom] = _brier(y[m], p[m])
    return out


def _frozen_safe_prediction(
    source: dict[str, np.ndarray],
    *,
    metrics: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Recombine source-fold components using the frozen current SAFE weights.

    The source fold's own labels are not used to fit mixed/simplex weights here.
    This is important because the residual map is supposed to be estimated from
    an out-of-time SAFE-family prediction rather than an in-fold optimized blend.
    """

    gt = np.asarray(source["gt"]).astype(str)
    direct = np.asarray(source["direct"], np.float64)
    reverse = np.asarray(source["reverse600"], np.float64)
    middle = np.asarray(source["middle400"], np.float64)
    learned_gate = np.asarray(source["gate600"], np.float64)
    conditional = np.asarray(source["cond400"], np.float64)

    recipe = cfg["recipe"]["mixed"]
    c = float(recipe["interaction_c"])
    wi = float(recipe["independent_gate_weight"])
    wg = float(recipe["learned_gate_weight"])
    independent = np.clip(1.0 - reverse - middle + c * reverse * middle, 0.0, 1.0)
    logic = (wi * independent + wg * learned_gate) * conditional

    mixed = direct.copy()
    frozen_mixed = metrics["mixed_blend"]
    for dom in ("R", "F"):
        m = gt == dom
        w = float(frozen_mixed[dom])
        mixed[m] = direct[m] + w * (logic[m] - direct[m])

    offset = np.asarray(source["offset"], np.float64)
    joint = np.asarray(source["joint"], np.float64)
    structured = np.asarray(source["structured"], np.float64)
    safe = np.empty(len(gt), np.float64)
    final = np.empty(len(gt), np.float64)

    frozen_safe = metrics["safe_weights"]
    frozen_final = metrics["final_weights"]
    matrix = np.column_stack([mixed, offset, joint])
    for dom in ("R", "F"):
        m = gt == dom
        sw = np.asarray(frozen_safe[dom], np.float64)
        fw = np.asarray(frozen_final[dom], np.float64)
        safe[m] = matrix[m] @ sw
        final[m] = safe[m] * fw[0] + structured[m] * fw[1]

    return {
        "mixed": np.clip(mixed, 0.0, 1.0),
        "safe": np.clip(safe, 0.0, 1.0),
        "pred": np.clip(final, 0.0, 1.0),
    }


def _source_residual_maps(y: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    yy = np.asarray(y, np.float64)
    pp = np.asarray(pred, np.float64)
    gg = np.asarray(gt).astype(str)
    residual = yy - pp
    global_mean = float(np.mean(residual))
    by_gt: dict[str, float] = {}
    rows: dict[str, int] = {}
    for dom in ("R", "F"):
        m = gg == dom
        by_gt[dom] = float(np.mean(residual[m]))
        rows[dom] = int(m.sum())
    centered = {dom: float(by_gt[dom] - global_mean) for dom in ("R", "F")}
    return {
        "global": global_mean,
        "game_type": by_gt,
        "game_type_centered": centered,
        "rows": rows,
    }


def _apply_map(
    base: np.ndarray,
    gt: np.ndarray,
    maps: dict[str, Any],
    *,
    kind: str,
    alpha: float,
) -> np.ndarray:
    p = np.asarray(base, np.float64)
    g = np.asarray(gt).astype(str)
    if kind == "global":
        corr = np.full(len(p), float(maps["global"]), np.float64)
    elif kind == "game_type":
        table = maps["game_type"]
        corr = np.asarray([float(table[str(x)]) for x in g], np.float64)
    elif kind == "game_type_centered":
        table = maps["game_type_centered"]
        corr = np.asarray([float(table[str(x)]) for x in g], np.float64)
    else:
        raise ValueError(kind)
    return np.clip(p + float(alpha) * corr, 0.0, 1.0)


def _report(result: dict[str, Any]) -> str:
    base = result["target_2024"]
    src = result["source_2023"]
    lines = [
        "# Cycle 3 — SAFE post-2023 regime bridge",
        "",
        "> Diagnostic gate. A 2023 OOT SAFE-family residual map is transferred unchanged to the frozen 2024 SAFE prediction. No 2024 label is used to fit any correction.",
        "",
        "## Source 2023",
        "",
        f"- frozen SAFE-family Brier: `{src['brier']:.12f}`",
        f"- residual mean: `{src['residual_maps']['global']:+.9f}`",
        f"- game_type residuals: `{src['residual_maps']['game_type']}`",
        f"- centered game_type residuals: `{src['residual_maps']['game_type_centered']}`",
        "",
        "## Target 2024",
        "",
        f"- frozen SAFE982 Brier: `{base['brier']:.12f}`",
        f"- R/F Brier: `{base['domain_brier']}`",
        "",
        "| candidate | alpha | Brier | delta vs SAFE | R delta | F delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["trials"]:
        lines.append(
            f"| {item['kind']} | {item['alpha']:.2f} | {item['brier']:.12f} | "
            f"{item['delta_vs_safe']:+.12f} | {item['domain_delta'].get('R', float('nan')):+.9f} | "
            f"{item['domain_delta'].get('F', float('nan')):+.9f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- predeclared Cycle2 bridge (`game_type`, alpha=0.50): `{result['predeclared']}`",
        f"- best exploratory candidate: `{result['best']}`",
        "",
        "Promotion requires a material improvement of the predeclared transfer, not merely an exploratory 2024-selected winner. If only the centered correction works, interpret that as game_type-specific structural drift rather than global calibration drift.",
    ]
    return "\n".join(lines) + "\n"


def run(
    *,
    baseline_config: str = "configs/baseline_safe_981.yaml",
    baseline_predictions: str = "outputs/baseline/predictions.npz",
    baseline_metrics: str = "outputs/baseline/metrics.json",
    output_dir: str = "outputs/experiments/cycle3_sota_regime_bridge",
    reuse_source: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_config(_resolve(baseline_config))
    out = _resolve(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_dir = out / "source2023_components"
    source_npz = source_dir / "predictions.npz"

    if not (reuse_source and source_npz.is_file()):
        source_cfg = copy.deepcopy(cfg)
        source_cfg["data"]["validation_season"] = 2023
        source_cfg["output"]["dir"] = str(source_dir)
        source_cfg["output"]["save_models"] = False
        source_cfg["output"]["save_components"] = True
        print("[Cycle3] training SAFE-family components: 2019-2022 -> 2023", flush=True)
        train_baseline(source_cfg)
    else:
        print(f"[Cycle3] reusing {source_npz}", flush=True)

    source = _load_npz(source_npz)
    target = _load_npz(baseline_predictions)
    metrics = _load_json(baseline_metrics)

    source_frozen = _frozen_safe_prediction(source, metrics=metrics, cfg=cfg)
    source_y = np.asarray(source["y"], np.float64)
    source_gt = np.asarray(source["gt"]).astype(str)
    maps = _source_residual_maps(source_y, source_frozen["pred"], source_gt)

    target_y = np.asarray(target["y"], np.float64)
    target_gt = np.asarray(target["gt"]).astype(str)
    target_pred = np.asarray(target["pred"], np.float64)
    base_brier = _brier(target_y, target_pred)
    base_domain = _domain_brier(target_y, target_pred, target_gt)

    kinds = ("global", "game_type", "game_type_centered")
    alphas = (0.25, 0.50, 0.75, 1.00)
    trials: list[dict[str, Any]] = []
    for kind in kinds:
        for alpha in alphas:
            pred = _apply_map(target_pred, target_gt, maps, kind=kind, alpha=alpha)
            brier = _brier(target_y, pred)
            dom = _domain_brier(target_y, pred, target_gt)
            item = {
                "kind": kind,
                "alpha": alpha,
                "brier": brier,
                "delta_vs_safe": brier - base_brier,
                "domain_brier": dom,
                "domain_delta": {k: dom[k] - base_domain[k] for k in dom},
            }
            trials.append(item)
            print(
                f"[Cycle3 {kind:18s} a={alpha:.2f}] brier={brier:.12f} "
                f"delta={brier-base_brier:+.12f}",
                flush=True,
            )

    predeclared = next(x for x in trials if x["kind"] == "game_type" and x["alpha"] == 0.50)
    best = min(trials, key=lambda x: float(x["brier"]))
    result = {
        "source_2023": {
            "rows": int(len(source_y)),
            "brier": _brier(source_y, source_frozen["pred"]),
            "domain_brier": _domain_brier(source_y, source_frozen["pred"], source_gt),
            "residual_maps": maps,
        },
        "target_2024": {
            "rows": int(len(target_y)),
            "brier": base_brier,
            "domain_brier": base_domain,
        },
        "trials": trials,
        "predeclared": predeclared,
        "best": best,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    return result
