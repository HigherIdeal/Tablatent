from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.ex1.backward_suite import _prepare_minimal_frame
from bitaboost.ex1.pitcher_season_backward import _season_profiles
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage


def _load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[3]
        path = root / path
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError("experiment configuration root must be a mapping")
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(Path(__file__).resolve().parents[3])
    return cfg


def _path(cfg: dict[str, Any], value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_repo_root"]) / p


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(np.float64)
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def _weighted_profile_mean(profile: pd.DataFrame, name: str) -> float:
    values = pd.to_numeric(profile[name], errors="coerce").to_numpy(np.float64)
    weights = pd.to_numeric(profile["pitch_count"], errors="coerce").fillna(0).to_numpy(np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float(np.nanmean(values)) if np.isfinite(values).any() else 0.5
    return float(np.average(values[finite], weights=weights[finite]))


def _attach_frozen_previous_season_traits(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    stable_traits: list[str],
    negative_control_trait: str,
    reliability_k: float,
) -> pd.DataFrame:
    """Attach only season s-1 pitcher state to rows from season s.

    The lookup is frozen by construction: a row from season s never uses any target
    aggregate from season s. Missing previous-season pitchers receive the source
    season league mean and reliability zero.
    """
    all_traits = list(dict.fromkeys([*stable_traits, negative_control_trait]))
    prior = profiles[[season_col, pitcher_col, "pitch_count", *all_traits]].copy()
    prior["current_season"] = pd.to_numeric(prior[season_col], errors="raise").astype(int) + 1
    prior = prior.rename(
        columns={
            "pitch_count": "prior_pitch_count",
            **{name: f"prior_{name}" for name in all_traits},
        }
    )
    prior = prior[["current_season", pitcher_col, "prior_pitch_count", *[f"prior_{x}" for x in all_traits]]]

    keys = frame[[season_col, pitcher_col]].copy()
    keys["current_season"] = pd.to_numeric(keys[season_col], errors="raise").astype(int)
    keys["_row"] = np.arange(len(keys), dtype=np.int64)
    attached = keys[["_row", "current_season", pitcher_col]].merge(
        prior,
        on=["current_season", pitcher_col],
        how="left",
        sort=False,
    )
    attached = attached.sort_values("_row", kind="stable").reset_index(drop=True)

    # Previous-season league means are computed only from the corresponding frozen
    # source season profile and are therefore available before the current season.
    fallback: dict[int, dict[str, float]] = {}
    profile_seasons = sorted(pd.to_numeric(profiles[season_col], errors="raise").astype(int).unique().tolist())
    for source_season in profile_seasons:
        source = profiles[pd.to_numeric(profiles[season_col], errors="raise").astype(int) == source_season]
        current_season = int(source_season) + 1
        fallback[current_season] = {
            name: _weighted_profile_mean(source, name)
            for name in all_traits
        }

    out = frame.copy()
    prior_n = pd.to_numeric(attached["prior_pitch_count"], errors="coerce").fillna(0).to_numpy(np.float64)
    has_prior = prior_n > 0
    out["stable_prior_n"] = prior_n.astype(np.float32)
    out["stable_prior_log_n"] = np.log1p(prior_n).astype(np.float32)
    out["stable_has_prior"] = has_prior.astype(np.float32)
    k = float(reliability_k)
    if k <= 0:
        raise ValueError("reliability_k must be positive")
    reliability = prior_n / (prior_n + k)
    out["stable_reliability"] = reliability.astype(np.float32)

    current_seasons = pd.to_numeric(out[season_col], errors="raise").astype(int).to_numpy()
    for name in all_traits:
        raw = pd.to_numeric(attached[f"prior_{name}"], errors="coerce").to_numpy(np.float64)
        fallback_values = np.array(
            [fallback.get(int(s), {}).get(name, 0.5) for s in current_seasons],
            dtype=np.float64,
        )
        raw = np.where(np.isfinite(raw), raw, fallback_values)
        shrunk = reliability * raw + (1.0 - reliability) * fallback_values
        out[f"stable_raw_{name}"] = np.clip(raw, 0.0, 1.0).astype(np.float32)
        out[f"stable_shrunk_{name}"] = np.clip(shrunk, 0.0, 1.0).astype(np.float32)
        out[f"stable_fallback_{name}"] = np.clip(fallback_values, 0.0, 1.0).astype(np.float32)

    return out


def _variant_features(
    variant: str,
    *,
    stable_traits: list[str],
    negative_control_trait: str,
) -> list[str]:
    if variant == "prior_success_only":
        return ["stable_raw_success"]
    if variant == "stable4_raw":
        return [f"stable_raw_{name}" for name in stable_traits]
    if variant == "stable4_reliable":
        return [
            *[f"stable_shrunk_{name}" for name in stable_traits],
            "stable_prior_log_n",
            "stable_reliability",
            "stable_has_prior",
        ]
    if variant == "stable4_reliable_plus_middle":
        return [
            *[f"stable_shrunk_{name}" for name in stable_traits],
            f"stable_shrunk_{negative_control_trait}",
            "stable_prior_log_n",
            "stable_reliability",
            "stable_has_prior",
        ]
    raise ValueError(f"unknown EX5 variant: {variant}")


def _prepare_x(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = frame[features].copy()
    for name in features:
        x[name] = pd.to_numeric(x[name], errors="coerce").astype(np.float32)
        x[name] = x[name].replace([np.inf, -np.inf], np.nan)
    return x


def _model_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(cfg)
    params.update(
        {
            "loss_function": "Logloss",
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "logging_level": "Silent",
        }
    )
    return params


def _experience_label(n: np.ndarray, bins: list[int]) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    bins = sorted(int(x) for x in bins)
    if len(bins) != 3:
        raise ValueError("EX5 currently expects exactly three experience bin boundaries")
    b0, b1, b2 = bins
    return np.where(
        n <= 0,
        "no_prior",
        np.where(n < b0, f"lt{b0}", np.where(n < b1, f"{b0}_{b1-1}", np.where(n < b2, f"{b1}_{b2-1}", f"ge{b2}"))),
    )


def _group_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    prior_success: np.ndarray,
    prior_n: np.ndarray,
    bins: list[int],
) -> dict[str, Any]:
    labels = _experience_label(prior_n, bins)
    result: dict[str, Any] = {}
    for name in ["no_prior", f"lt{bins[0]}", f"{bins[0]}_{bins[1]-1}", f"{bins[1]}_{bins[2]-1}", f"ge{bins[2]}"]:
        mask = labels == name
        if not mask.any():
            continue
        result[name] = {
            "rows": int(mask.sum()),
            "target_rate": float(np.mean(y[mask])),
            "brier_model": _brier(y[mask], pred[mask]),
            "brier_prior_success": _brier(y[mask], prior_success[mask]),
            "auc_model": _auc(y[mask], pred[mask]),
            "prob_std": float(np.std(pred[mask])),
        }
    return result


def _independence_audit(model, frame: pd.DataFrame, features: list[str], rows: int) -> float:
    from catboost import Pool

    n = min(int(rows), len(frame))
    if n <= 0:
        return 0.0
    sample = frame.iloc[:n].copy()
    xb = _prepare_x(sample, features)
    batch = np.asarray(model.predict_proba(Pool(xb))[:, 1], dtype=np.float64)
    singles = []
    for i in range(n):
        xo = _prepare_x(sample.iloc[[i]].copy(), features)
        singles.append(float(model.predict_proba(Pool(xo))[0, 1]))
    return float(np.max(np.abs(batch - np.asarray(singles, dtype=np.float64))))


def run(config_path: str | Path) -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool

    exp = _load_experiment_config(config_path)
    base_cfg = load_config(_path(exp, exp["baseline_config"]))
    configure_cuda(base_cfg)
    configure_warnings(base_cfg)

    out_dir = _path(exp, exp["output_dir"])
    model_dir = _path(exp, exp["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with stage("EX5 minimal canonical frame + auxiliary outcomes"):
        frame, aux = _prepare_minimal_frame(base_cfg)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))
    stable_traits = [str(x) for x in exp.get("stable_traits", ["ball", "reverse", "success", "strike"])]
    negative_control_trait = str(exp.get("negative_control_trait", "middle"))
    reliability_k = float(exp.get("reliability_k", 200.0))

    with stage("EX5 frozen previous-season stable traits"):
        profiles = _season_profiles(
            frame,
            aux,
            season_col=season_col,
            pitcher_col=pitcher_col,
            target_col=target_col,
        )
        frame = _attach_frozen_previous_season_traits(
            frame,
            profiles,
            season_col=season_col,
            pitcher_col=pitcher_col,
            stable_traits=stable_traits,
            negative_control_trait=negative_control_trait,
            reliability_k=reliability_k,
        )

    # 2019 has no 2018 profile. We deliberately start from 2020 rather than inventing
    # an unavailable historical state.
    frame = frame[pd.to_numeric(frame[season_col], errors="raise").astype(int) >= 2020].reset_index(drop=True)

    variants = [str(x) for x in exp.get("variants", ["prior_success_only", "stable4_reliable"])]
    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    experience_bins = [int(x) for x in exp.get("experience_bins", [50, 200, 500])]
    audit_rows = int(exp.get("independence_audit_rows", 8))

    results: dict[str, Any] = {
        "experiment": "EX5 frozen bidirectional-stable trait pitch-level probe",
        "definition": (
            "For a pitch in season s, use only the pitcher's frozen season s-1 profile. "
            "The EX4 direction-invariant subset is probed without SAFE or current-season target aggregates."
        ),
        "stable_traits": stable_traits,
        "negative_control_trait": negative_control_trait,
        "reliability_k": reliability_k,
        "rows": int(len(frame)),
        "variants": {},
    }

    for variant in variants:
        features = _variant_features(
            variant,
            stable_traits=stable_traits,
            negative_control_trait=negative_control_trait,
        )
        variant_result: dict[str, Any] = {"features": features, "folds": {}}

        for test_season in fold_seasons:
            train = frame[pd.to_numeric(frame[season_col], errors="raise").astype(int) < test_season].reset_index(drop=True)
            test = frame[pd.to_numeric(frame[season_col], errors="raise").astype(int) == test_season].reset_index(drop=True)
            if len(train) == 0 or len(test) == 0:
                variant_result["folds"][str(test_season)] = {
                    "skipped": True,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                }
                continue

            x_train = _prepare_x(train, features)
            x_test = _prepare_x(test, features)
            y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8)
            y_test = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.int8)

            with stage(f"EX5 pitch probe [{variant}] test={test_season}"):
                model = CatBoostClassifier(**_model_params(exp["catboost"]))
                model.fit(Pool(x_train, y_train, feature_names=features))
                pred = np.asarray(model.predict_proba(Pool(x_test, feature_names=features))[:, 1], dtype=np.float64)

            pred = np.clip(pred, 0.0, 1.0)
            prior = float(np.mean(y_train))
            prior_pred = np.full(len(y_test), prior, dtype=np.float64)
            previous_success = pd.to_numeric(test["stable_raw_success"], errors="coerce").fillna(prior).to_numpy(np.float64)
            prior_n = pd.to_numeric(test["stable_prior_n"], errors="coerce").fillna(0).to_numpy(np.float64)
            has_prior = prior_n > 0

            metrics = {
                "rows": int(len(test)),
                "target_rate": float(np.mean(y_test)),
                "auc": _auc(y_test, pred),
                "brier": _brier(y_test, pred),
                "prior_brier": _brier(y_test, prior_pred),
                "previous_success_brier": _brier(y_test, previous_success),
                "delta_brier_vs_prior": _brier(y_test, pred) - _brier(y_test, prior_pred),
                "delta_brier_vs_previous_success": _brier(y_test, pred) - _brier(y_test, previous_success),
                "prob_mean": float(np.mean(pred)),
                "prob_std": float(np.std(pred)),
                "previous_profile_coverage": float(np.mean(has_prior)),
            }
            groups = _group_metrics(y_test, pred, previous_success, prior_n, experience_bins)
            independence = _independence_audit(model, test, features, audit_rows)

            fold_result = {
                "skipped": False,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_seasons": sorted(int(x) for x in pd.to_numeric(train[season_col], errors="raise").astype(int).unique().tolist()),
                "metrics": metrics,
                "experience_groups": groups,
                "row_independence_max_abs_diff": independence,
            }
            variant_result["folds"][str(test_season)] = fold_result

            np.savez_compressed(
                out_dir / f"pitch_probe_{variant}_{test_season}.npz",
                y=y_test,
                pred=pred,
                row_id=test["row_id"].astype(str).to_numpy(dtype="U"),
                prior_n=prior_n.astype(np.float32),
                previous_success=previous_success.astype(np.float32),
            )

            if test_season == max(fold_seasons):
                model_path = model_dir / f"stable_trait_pitch_probe_{variant}.cbm"
                model.save_model(str(model_path))
                fold_result["model_path"] = str(model_path.relative_to(Path(exp["_repo_root"])))

            log(
                f"[EX5:{variant}:{test_season}] auc={metrics['auc'] if metrics['auc'] is not None else float('nan'):.5f} "
                f"brier={metrics['brier']:.8f} prior={metrics['prior_brier']:.8f} "
                f"prev_success={metrics['previous_success_brier']:.8f} "
                f"coverage={metrics['previous_profile_coverage']:.3f} row_diff={independence:.3e}"
            )

        valid_folds = [f for f in variant_result["folds"].values() if not f.get("skipped")]
        if valid_folds:
            variant_result["summary"] = {
                "mean_auc": float(np.mean([f["metrics"]["auc"] for f in valid_folds if f["metrics"]["auc"] is not None])),
                "mean_brier": float(np.mean([f["metrics"]["brier"] for f in valid_folds])),
                "mean_prior_brier": float(np.mean([f["metrics"]["prior_brier"] for f in valid_folds])),
                "mean_delta_brier_vs_prior": float(np.mean([f["metrics"]["delta_brier_vs_prior"] for f in valid_folds])),
            }
        results["variants"][variant] = variant_result

    metrics_path = out_dir / "metrics_stable_trait_pitch_probe.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log(f"[EX5] complete -> {metrics_path}")
    log("[EX5 summary]")
    for variant, payload in results["variants"].items():
        summary = payload.get("summary", {})
        if not summary:
            continue
        log(
            f"  {variant:30s} mean_auc={summary['mean_auc']:.5f} "
            f"mean_brier={summary['mean_brier']:.8f} prior={summary['mean_prior_brier']:.8f} "
            f"delta={summary['mean_delta_brier_vs_prior']:+.8f}"
        )
    return results
