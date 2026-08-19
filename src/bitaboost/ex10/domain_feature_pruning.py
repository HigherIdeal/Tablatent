from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.ex2.hypothesis_backward import _prepare_x as _prepare_domain_x
from bitaboost.ex7.stable_trait_injection import (
    _brier,
    _domain_brier,
    _fit_direct,
    _frozen_recombine,
    _load_baseline_vectors,
)
from bitaboost.ex9.density_ratio_reweight import _domain_features
from bitaboost.features import auxiliary_targets, prepare
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
        raise TypeError("EX10 config root must be a mapping")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


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


def _fit_domain_ranker(
    tr: pd.DataFrame,
    *,
    season_col: str,
    recent_season: int,
    features: list[str],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    season = pd.to_numeric(tr[season_col], errors="raise").astype(int).to_numpy()
    label = (season == int(recent_season)).astype(np.int8)
    if label.min() == label.max():
        raise RuntimeError("EX10 domain label has only one class")

    x, cats = _prepare_domain_x(tr, features)
    index = np.arange(len(tr), dtype=np.int64)
    fit_mask = (index % 5) != 0
    hold_mask = ~fit_mask

    fit_pool = Pool(
        x.loc[fit_mask],
        label[fit_mask],
        cat_features=cats,
        feature_names=features,
    )
    hold_pool = Pool(
        x.loc[hold_mask],
        cat_features=cats,
        feature_names=features,
    )
    probe = CatBoostClassifier(**params).fit(fit_pool)
    hold_pred = np.asarray(probe.predict_proba(hold_pool)[:, 1], dtype=np.float64)
    hold_auc = auc(label[hold_mask], hold_pred)

    full_pool = Pool(x, label, cat_features=cats, feature_names=features)
    model = CatBoostClassifier(**params).fit(full_pool)
    importance = np.asarray(model.get_feature_importance(type="FeatureImportance"), dtype=np.float64)
    if len(importance) != len(features):
        raise RuntimeError("EX10 feature-importance length mismatch")

    ranking = [
        {"feature": str(name), "importance": float(score)}
        for name, score in zip(features, importance)
    ]
    ranking.sort(key=lambda item: item["importance"], reverse=True)

    diag = {
        "rows": int(len(tr)),
        "recent_rows": int(label.sum()),
        "old_rows": int((label == 0).sum()),
        "features": int(len(features)),
        "holdout_auc": None if hold_auc is None else float(hold_auc),
    }
    return ranking, diag


def _report(result: dict[str, Any]) -> str:
    base = result["baseline"]
    lines = [
        "# EX10 domain-shift feature pruning report",
        "",
        "> The 2023-vs-older domain classifier is trained only on <=2023 input rows. 2024 is evaluation only. The experiment retrains only the SAFE direct MultiRMSE head; every downstream SAFE component and blend weight is frozen.",
        "",
        f"SAFE982 final Brier: `{base['final_brier']:.12f}`",
        f"Domain holdout AUC: `{result['domain']['holdout_auc']}`",
        "",
        "## Highest domain-shift features",
        "",
        "| rank | feature | importance | protected |",
        "|---:|---|---:|---|",
    ]
    protected = set(result.get("protected_features", []))
    for i, item in enumerate(result["domain_ranking"][:25], start=1):
        lines.append(
            f"| {i} | `{item['feature']}` | {item['importance']:.6f} | "
            f"{'yes' if item['feature'] in protected else 'no'} |"
        )

    lines += [
        "",
        "## Pruning trials",
        "",
        "| drop k | dropped | direct | mixed | final | delta vs SAFE | R delta | F delta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["trials"]:
        dropped = ", ".join(item["dropped_features"]) if item["dropped_features"] else "-"
        lines.append(
            f"| {item['drop_k']} | {dropped} | {item['direct_brier']:.9f} | "
            f"{item['mixed_brier']:.9f} | {item['final_brier']:.9f} | "
            f"{item['final_brier'] - base['final_brier']:+.9f} | "
            f"{item['domain_final_brier']['R'] - base['domain_final_brier']['R']:+.9f} | "
            f"{item['domain_final_brier']['F'] - base['domain_final_brier']['F']:+.9f} |"
        )

    best = result["best"]
    lines += [
        "",
        "## Decision",
        "",
        f"- best drop_k: `{best['drop_k']}`",
        f"- best final Brier: `{best['final_brier']:.12f}`",
        f"- delta vs SAFE982: `{best['final_brier'] - base['final_brier']:+.12f}`",
        f"- dropped features: `{', '.join(best['dropped_features']) or '-'}`",
        "",
        "If drop_k=0 wins, strong 2023 covariate shift is mostly nuisance information that SAFE already handles or useful predictive structure that cannot simply be removed. If a small positive k wins, the top domain-separating variables are candidates for regime-robust pruning or explicit invariant treatment.",
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
        raise RuntimeError("EX10 baseline/frame alignment failed")

    ranking, domain_diag = _fit_domain_ranker(
        tr,
        season_col=season_col,
        recent_season=recent_season,
        features=_domain_features(season_col),
        params=_domain_params(exp),
    )

    protected = {str(x) for x in exp.get("protected_features", [])}
    rich = list(data.feature_sets["rich"])
    rich_set = set(rich)
    droppable = [
        item["feature"]
        for item in ranking
        if item["feature"] in rich_set and item["feature"] not in protected
    ]

    baseline = {
        "final_brier": _brier(y, base["pred"]),
        "direct_brier": _brier(y, base["direct"]),
        "mixed_brier": _brier(y, base["mixed"]),
        "domain_final_brier": _domain_brier(y, base["pred"], gt),
    }

    trials: list[dict[str, Any]] = []
    for requested_k in [int(x) for x in exp.get("drop_ks", [0, 1, 3, 5, 10])]:
        k = min(max(requested_k, 0), len(droppable))
        dropped = droppable[:k]
        features = [name for name in rich if name not in set(dropped)]
        direct = _fit_direct(cfg, tr, va, aux_train, features)
        recombined = _frozen_recombine(
            direct,
            base=base,
            gt=gt,
            baseline_metrics=baseline_metrics,
            recipe=cfg["recipe"]["mixed"],
        )
        item = {
            "drop_k": int(k),
            "requested_drop_k": int(requested_k),
            "dropped_features": dropped,
            "direct_brier": _brier(y, direct),
            "mixed_brier": _brier(y, recombined["mixed"]),
            "final_brier": _brier(y, recombined["final"]),
            "domain_final_brier": _domain_brier(y, recombined["final"], gt),
        }
        trials.append(item)
        print(
            f"[EX10 drop={k:02d}] direct={item['direct_brier']:.9f} "
            f"mixed={item['mixed_brier']:.9f} final={item['final_brier']:.9f} "
            f"delta={item['final_brier'] - baseline['final_brier']:+.9f} "
            f"features={','.join(dropped) if dropped else '-'}",
            flush=True,
        )

    best = min(trials, key=lambda item: float(item["final_brier"]))
    result = {
        "baseline": baseline,
        "domain": domain_diag,
        "domain_ranking": ranking,
        "protected_features": sorted(protected),
        "droppable_ranked": droppable,
        "trials": trials,
        "best": best,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (out / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    return result
