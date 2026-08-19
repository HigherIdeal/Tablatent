from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.baseline import _params, _prepare_x
from bitaboost.config import load_config
from bitaboost.ex2.hypothesis_backward import ID_FEATURES, _base_features, _prepare_x as _prepare_domain_x
from bitaboost.ex7.stable_trait_injection import (
    _domain_brier,
    _frozen_recombine,
    _load_baseline_vectors,
)
from bitaboost.features import AUX_NAMES, auxiliary_targets, prepare
from bitaboost.night.common import auc


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else _root() / p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("EX9 config root must be a mapping")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, np.float64)
    pp = np.asarray(p, np.float64)
    return float(np.mean((yy - pp) ** 2))


def _domain_features(season_col: str) -> list[str]:
    features = []
    for name in _base_features():
        if name in ID_FEATURES:
            continue
        if name == season_col or name.lower() == "season":
            continue
        if name == "row_id" or name == "control_success":
            continue
        features.append(name)
    if len(features) != len(set(features)):
        raise RuntimeError("duplicate EX9 domain features")
    return features


def _domain_params(exp: dict[str, Any]) -> dict[str, Any]:
    p = dict(exp["domain_model"])
    p.update(
        {
            "loss_function": "Logloss",
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "logging_level": "Silent",
        }
    )
    p["gpu_ram_part"] = float(p.get("gpu_ram_part", 0.82))
    return p


def _fit_domain_model(
    train: pd.DataFrame,
    *,
    season_col: str,
    recent_season: int,
    features: list[str],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    season = pd.to_numeric(train[season_col], errors="raise").astype(int).to_numpy()
    label = (season == int(recent_season)).astype(np.int8)
    if label.min() == label.max():
        raise RuntimeError("EX9 domain target has one class")

    x, cats = _prepare_domain_x(train, features)
    index = np.arange(len(train), dtype=np.int64)
    fit_mask = (index % 5) != 0
    hold_mask = ~fit_mask

    fit_pool = Pool(
        x.loc[fit_mask],
        label[fit_mask],
        cat_features=cats,
        feature_names=features,
    )
    hold_pool = Pool(x.loc[hold_mask], cat_features=cats, feature_names=features)
    probe = CatBoostClassifier(**params).fit(fit_pool)
    hold_pred = np.asarray(probe.predict_proba(hold_pool)[:, 1], np.float64)
    hold_auc = auc(label[hold_mask], hold_pred)

    full_pool = Pool(x, label, cat_features=cats, feature_names=features)
    model = CatBoostClassifier(**params).fit(full_pool)
    full_pred = np.asarray(model.predict_proba(full_pool)[:, 1], np.float64)

    recent_n = int(label.sum())
    old_n = int(len(label) - recent_n)
    prior_recent = recent_n / len(label)
    diagnostics = {
        "features": len(features),
        "rows": int(len(label)),
        "recent_rows": recent_n,
        "old_rows": old_n,
        "recent_prior": float(prior_recent),
        "holdout_auc": None if hold_auc is None else float(hold_auc),
        "prob_mean": float(full_pred.mean()),
        "prob_std": float(full_pred.std()),
    }
    return full_pred, diagnostics


def _density_ratio(
    domain_pred: np.ndarray,
    season: np.ndarray,
    *,
    recent_season: int,
    clip_lo: float,
    clip_hi: float,
) -> tuple[np.ndarray, dict[str, float]]:
    season = np.asarray(season, dtype=np.int16)
    recent = season == int(recent_season)
    old = season < int(recent_season)
    if not old.any() or not recent.any():
        raise RuntimeError("EX9 needs old and recent rows")

    p = np.clip(np.asarray(domain_pred, np.float64), 1e-4, 1.0 - 1e-4)
    prior_correction = float(old.sum() / recent.sum())
    ratio = (p / (1.0 - p)) * prior_correction
    ratio = np.clip(ratio, float(clip_lo), float(clip_hi))

    # Preserve the total contribution of historical rows; EX9 tests covariate
    # composition reweighting rather than a hidden recent-season boost.
    ratio_old = ratio[old]
    ratio_old /= float(np.mean(ratio_old))
    weights = np.ones(len(season), dtype=np.float64)
    weights[old] = ratio_old
    weights[recent] = 1.0

    info = {
        "old_weight_mean": float(weights[old].mean()),
        "old_weight_std": float(weights[old].std()),
        "old_weight_p05": float(np.quantile(weights[old], 0.05)),
        "old_weight_p50": float(np.quantile(weights[old], 0.50)),
        "old_weight_p95": float(np.quantile(weights[old], 0.95)),
        "old_weight_min": float(weights[old].min()),
        "old_weight_max": float(weights[old].max()),
    }
    return weights, info


def _fit_direct_weighted(
    cfg: dict[str, Any],
    tr: pd.DataFrame,
    va: pd.DataFrame,
    aux_train: pd.DataFrame,
    features: list[str],
    domain_weight: np.ndarray,
) -> np.ndarray:
    from catboost import CatBoostRegressor, Pool

    x, cats = _prepare_x(tr, features)
    xv, _ = _prepare_x(va, features)
    keep = aux_train[list(AUX_NAMES)].notna().all(axis=1).to_numpy()
    success = tr.loc[keep, "control_success"].to_numpy(np.float32)
    av = aux_train.loc[keep, list(AUX_NAMES)].to_numpy(np.float32)
    repeats = int(cfg["recipe"]["direct"]["success_repeats"])
    labels = np.column_stack([*[success] * repeats, av])

    f_weight = float(cfg["recipe"]["direct"]["f_weight"])
    base_weight = np.where(
        tr.loc[keep, "game_type"].astype(str).to_numpy() == "F", f_weight, 1.0
    ).astype(np.float64)
    weight = base_weight * np.asarray(domain_weight, np.float64)[keep]
    weight *= float(len(weight) / weight.sum())

    pool = Pool(
        x.loc[keep],
        labels,
        weight=weight.astype(np.float32),
        cat_features=cats,
        feature_names=features,
    )
    vp = Pool(xv, cat_features=cats, feature_names=features)
    model = CatBoostRegressor(**_params(cfg, "MultiRMSE")).fit(pool)
    pred = np.clip(
        model.predict(vp, ntree_end=int(cfg["recipe"]["direct"]["tree"])),
        0.0,
        1.0,
    )[:, 0].astype(np.float64)
    return pred


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# EX9 density-ratio reweighting report",
        "",
        "> The domain model only sees <=2023 input features and excludes season/entity IDs. 2024 is evaluation only. SAFE downstream components and weights are frozen.",
        "",
        f"SAFE982 final Brier: `{result['baseline']['final_brier']:.12f}`",
        f"Domain holdout AUC (2023 vs older): `{result['domain']['holdout_auc']}`",
        "",
        "| alpha | direct | mixed | final | delta final | R delta | F delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = result["baseline"]
    for item in result["trials"]:
        lines.append(
            f"| {item['alpha']:.2f} | {item['direct_brier']:.9f} | {item['mixed_brier']:.9f} | "
            f"{item['final_brier']:.9f} | {item['final_brier'] - base['final_brier']:+.9f} | "
            f"{item['domain_final_brier']['R'] - base['domain_final_brier']['R']:+.9f} | "
            f"{item['domain_final_brier']['F'] - base['domain_final_brier']['F']:+.9f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- best alpha: `{result['best']['alpha']}`",
        f"- best final Brier: `{result['best']['final_brier']:.12f}`",
        f"- delta vs SAFE982: `{result['best']['final_brier'] - base['final_brier']:+.12f}`",
        "",
        "A positive result requires the density-weighted direct head to improve the frozen SAFE final prediction, not merely the standalone direct head. Alpha=0 is the exact retrain control.",
    ]
    return "\n".join(lines) + "\n"


def run(experiment_config: str | Path) -> dict[str, Any]:
    exp = _load_yaml(experiment_config)
    cfg = load_config(_resolve(exp["baseline_config"]))
    baseline_metrics = _load_json(exp["baseline_metrics"])
    base = _load_baseline_vectors(exp["baseline_predictions"])
    out = _resolve(exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = prepare(cfg)
    frame = data.frame.copy()
    season_col = cfg["data"]["season_col"]
    target_col = cfg["data"]["target_col"]
    recent_season = int(exp.get("recent_season", 2023))
    validation_season = int(exp.get("validation_season", 2024))

    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    train_mask = season.lt(validation_season).to_numpy()
    valid_mask = season.eq(validation_season).to_numpy()
    tr = frame.loc[train_mask].reset_index(drop=True)
    va = frame.loc[valid_mask].reset_index(drop=True)
    aux_train = auxiliary_targets(tr)
    y = pd.to_numeric(va[target_col], errors="raise").to_numpy(np.float64)
    gt = va["game_type"].astype(str).to_numpy()
    if len(y) != len(base["pred"]) or not np.allclose(y, base["y"], atol=0.0, rtol=0.0):
        raise RuntimeError("EX9 baseline/frame alignment failed")

    domain_features = _domain_features(season_col)
    domain_pred, domain_diag = _fit_domain_model(
        tr,
        season_col=season_col,
        recent_season=recent_season,
        features=domain_features,
        params=_domain_params(exp),
    )
    clip_lo, clip_hi = [float(x) for x in exp.get("ratio_clip", [0.25, 4.0])]
    full_ratio_weight, ratio_info = _density_ratio(
        domain_pred,
        pd.to_numeric(tr[season_col], errors="raise").astype(int).to_numpy(),
        recent_season=recent_season,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )
    domain_diag["ratio"] = ratio_info

    baseline = {
        "final_brier": _brier(y, base["pred"]),
        "direct_brier": _brier(y, base["direct"]),
        "mixed_brier": _brier(y, base["mixed"]),
        "domain_final_brier": _domain_brier(y, base["pred"], gt),
    }

    trials: list[dict[str, Any]] = []
    rich = data.feature_sets["rich"]
    for alpha in [float(x) for x in exp.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0])]:
        sample_weight = (1.0 - alpha) + alpha * full_ratio_weight
        direct = _fit_direct_weighted(cfg, tr, va, aux_train, rich, sample_weight)
        recombined = _frozen_recombine(
            direct,
            base=base,
            gt=gt,
            baseline_metrics=baseline_metrics,
            recipe=cfg["recipe"]["mixed"],
        )
        item = {
            "alpha": alpha,
            "direct_brier": _brier(y, direct),
            "mixed_brier": _brier(y, recombined["mixed"]),
            "final_brier": _brier(y, recombined["final"]),
            "domain_final_brier": _domain_brier(y, recombined["final"], gt),
        }
        trials.append(item)
        print(
            f"[EX9 alpha={alpha:.2f}] direct={item['direct_brier']:.9f} "
            f"mixed={item['mixed_brier']:.9f} final={item['final_brier']:.9f} "
            f"delta={item['final_brier'] - baseline['final_brier']:+.9f}",
            flush=True,
        )

    best = min(trials, key=lambda x: float(x["final_brier"]))
    result = {
        "baseline": baseline,
        "domain": domain_diag,
        "trials": trials,
        "best": best,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    return result
