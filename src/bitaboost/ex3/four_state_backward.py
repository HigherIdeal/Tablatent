from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.ex1.pitcher_season_backward import STATE_NAMES, _season_profiles
from bitaboost.ex2.hypothesis_backward import (
    HISTORY_PREFIXES,
    ID_FEATURES,
    _attach_previous_state,
    _auc,
    _base_features,
    _brier,
    _logloss,
    _prepare_minimal_frame,
    _prepare_x,
)
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage

CF_FEATURES = [
    "cf_post_success_rate",
    "cf_post_middle_rate",
    "cf_post_log_n",
    "cf_delta_success_rate",
    "cf_delta_middle_rate",
]
CLASS_NAMES = ("F_hard", "F_soft", "S_soft", "S_hard")


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


def _model_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(cfg)
    params.update(
        {
            "loss_function": "MultiRMSE",
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "logging_level": "Silent",
        }
    )
    return params


def _context_features() -> list[str]:
    base = _base_features()
    return [
        feature
        for feature in base
        if feature not in ID_FEATURES
        and not any(feature.startswith(prefix) for prefix in HISTORY_PREFIXES)
    ]


def _features_for_variant(variant: str) -> list[str]:
    if variant == "state_only":
        return list(CF_FEATURES)
    if variant == "context_state":
        features = _context_features() + list(CF_FEATURES)
        if len(features) != len(set(features)):
            raise RuntimeError("duplicate EX3 features")
        return features
    raise ValueError(f"unknown EX3 variant: {variant}")


def _numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    if np.isnan(values).any():
        values = np.where(np.isnan(values), default, values)
    return values


def _assimilated_state(
    frame: pd.DataFrame,
    *,
    event_success: np.ndarray | float,
    event_middle: np.ndarray | float,
    evidence_weight: float,
) -> pd.DataFrame:
    """Create a row-local post-pitch state under a hypothetical event.

    The official pre-pitch asof state is not overwritten. Instead, EX3 derives a
    counterfactual post-state using an effective event weight. Hard states use a
    full pitch (w=1); soft states use a fractional pitch (0<w<1). This makes the
    hypothesis alter the state itself rather than appear as an ignorable label column.
    """
    out = frame.copy()
    n = np.maximum(_numeric(out["asof_pitcher_n"]), 0.0)
    success_rate = np.clip(_numeric(out["asof_pitcher_success_rate"]), 0.0, 1.0)
    middle_rate = np.clip(_numeric(out["asof_pitcher_middle_rate"]), 0.0, 1.0)

    success_event = np.broadcast_to(np.asarray(event_success, dtype=np.float64), n.shape)
    middle_event = np.broadcast_to(np.asarray(event_middle, dtype=np.float64), n.shape)
    w = float(evidence_weight)
    if not (0.0 < w <= 1.0):
        raise ValueError(f"evidence_weight must be in (0, 1], got {w}")

    denom = np.maximum(n + w, 1e-12)
    post_success = np.clip((n * success_rate + w * success_event) / denom, 0.0, 1.0)
    post_middle = np.clip((n * middle_rate + w * middle_event) / denom, 0.0, 1.0)

    out["cf_post_success_rate"] = post_success.astype(np.float32)
    out["cf_post_middle_rate"] = post_middle.astype(np.float32)
    out["cf_post_log_n"] = np.log1p(n + w).astype(np.float32)
    out["cf_delta_success_rate"] = (post_success - success_rate).astype(np.float32)
    out["cf_delta_middle_rate"] = (post_middle - middle_rate).astype(np.float32)
    return out


def _actual_training_state(frame: pd.DataFrame, aux: pd.DataFrame, target_col: str) -> pd.DataFrame:
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(np.float64)
    middle = pd.to_numeric(aux.loc[frame.index, "middle"], errors="coerce").fillna(0).to_numpy(np.float64)
    return _assimilated_state(
        frame,
        event_success=y,
        event_middle=middle,
        evidence_weight=1.0,
    )


def _predict(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    from catboost import Pool

    x, cats = _prepare_x(frame, features)
    pred = np.asarray(model.predict(Pool(x, cat_features=cats, feature_names=features)), dtype=np.float64)
    return np.clip(pred, 0.0, 1.0)


def _state_energy(target: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.mean(((target - pred) / sigma[None, :]) ** 2, axis=1)


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vmax = np.max(values, axis=axis, keepdims=True)
    result = vmax + np.log(np.sum(np.exp(values - vmax), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _mix_failure_energy(e_middle0: np.ndarray, e_middle1: np.ndarray, p_middle: float, temperature: float) -> np.ndarray:
    p = float(np.clip(p_middle, 1e-8, 1.0 - 1e-8))
    t = float(temperature)
    scores = np.stack(
        [np.log1p(-p) - e_middle0 / t, np.log(p) - e_middle1 / t],
        axis=1,
    )
    return -t * _logsumexp(scores, axis=1)


def _class_energies(
    model,
    frame: pd.DataFrame,
    target_state: np.ndarray,
    sigma: np.ndarray,
    features: list[str],
    *,
    soft_weight: float,
    failure_middle_prior: float,
    substate_temperature: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Failure classes marginalize an unobserved middle/non-middle failure mechanism.
    candidate_frames = {
        "F_hard_m0": _assimilated_state(frame, event_success=0.0, event_middle=0.0, evidence_weight=1.0),
        "F_hard_m1": _assimilated_state(frame, event_success=0.0, event_middle=1.0, evidence_weight=1.0),
        "F_soft_m0": _assimilated_state(frame, event_success=0.0, event_middle=0.0, evidence_weight=soft_weight),
        "F_soft_m1": _assimilated_state(frame, event_success=0.0, event_middle=1.0, evidence_weight=soft_weight),
        "S_soft": _assimilated_state(frame, event_success=1.0, event_middle=0.0, evidence_weight=soft_weight),
        "S_hard": _assimilated_state(frame, event_success=1.0, event_middle=0.0, evidence_weight=1.0),
    }
    predictions = {name: _predict(model, block, features) for name, block in candidate_frames.items()}
    energy = {name: _state_energy(target_state, pred, sigma) for name, pred in predictions.items()}

    f_hard = _mix_failure_energy(
        energy["F_hard_m0"], energy["F_hard_m1"], failure_middle_prior, substate_temperature
    )
    f_soft = _mix_failure_energy(
        energy["F_soft_m0"], energy["F_soft_m1"], failure_middle_prior, substate_temperature
    )
    energies = np.stack([f_hard, f_soft, energy["S_soft"], energy["S_hard"]], axis=1)
    return energies, predictions


def _probabilities(energies: np.ndarray, temperature: float) -> np.ndarray:
    logits = -np.asarray(energies, dtype=np.float64) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _binary_margin(energies: np.ndarray) -> np.ndarray:
    failure_score = _logsumexp(-energies[:, :2], axis=1)
    success_score = _logsumexp(-energies[:, 2:], axis=1)
    return success_score - failure_score


def _select_temperature(
    history_energies: list[np.ndarray],
    history_labels: list[np.ndarray],
    grid: list[float],
) -> float:
    if not history_energies:
        return 1.0
    energies = np.concatenate(history_energies, axis=0)
    labels = np.concatenate(history_labels, axis=0).astype(np.float64)
    best_t = float(grid[0])
    best_brier = float("inf")
    for t in grid:
        probs = _probabilities(energies, float(t))
        success = probs[:, 2] + probs[:, 3]
        score = _brier(labels, success)
        if score < best_brier:
            best_brier = score
            best_t = float(t)
    return best_t


def _independence_audit(
    model,
    frame: pd.DataFrame,
    target_state: np.ndarray,
    sigma: np.ndarray,
    features: list[str],
    *,
    rows: int,
    soft_weight: float,
    failure_middle_prior: float,
    substate_temperature: float,
) -> float:
    n = min(int(rows), len(frame))
    if n <= 0:
        return 0.0
    batch_e, _ = _class_energies(
        model,
        frame.iloc[:n].copy(),
        target_state[:n],
        sigma,
        features,
        soft_weight=soft_weight,
        failure_middle_prior=failure_middle_prior,
        substate_temperature=substate_temperature,
    )
    singles = []
    for i in range(n):
        e, _ = _class_energies(
            model,
            frame.iloc[[i]].copy(),
            target_state[i : i + 1],
            sigma,
            features,
            soft_weight=soft_weight,
            failure_middle_prior=failure_middle_prior,
            substate_temperature=substate_temperature,
        )
        singles.append(e[0])
    return float(np.max(np.abs(batch_e - np.asarray(singles))))


def _class_diagnostics(y: np.ndarray, probs: np.ndarray, energies: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    winners = np.argmin(energies, axis=1)
    counts = {name: int(np.sum(winners == i)) for i, name in enumerate(CLASS_NAMES)}
    means = {name: float(np.mean(probs[:, i])) for i, name in enumerate(CLASS_NAMES)}

    fail = y == 0
    success = y == 1
    out: dict[str, Any] = {
        "winner_counts": counts,
        "mean_probability": means,
        "ambiguity_mean": float(np.mean(probs[:, 1] + probs[:, 2])),
        "confidence_mean": float(np.mean(probs[:, 0] + probs[:, 3])),
    }
    if fail.any():
        denom = np.maximum(probs[fail, 0] + probs[fail, 1], 1e-12)
        out["failure_true_branch"] = {
            "hard_share": float(np.mean(probs[fail, 0] / denom)),
            "soft_share": float(np.mean(probs[fail, 1] / denom)),
        }
    if success.any():
        denom = np.maximum(probs[success, 2] + probs[success, 3], 1e-12)
        out["success_true_branch"] = {
            "soft_share": float(np.mean(probs[success, 2] / denom)),
            "hard_share": float(np.mean(probs[success, 3] / denom)),
        }
    return out


def run(config_path: str | Path) -> dict[str, Any]:
    from catboost import CatBoostRegressor, Pool

    exp = _load_experiment_config(config_path)
    base_cfg = load_config(_path(exp, exp["baseline_config"]))
    configure_cuda(base_cfg)
    configure_warnings(base_cfg)

    out_dir = _path(exp, exp["output_dir"])
    model_dir = _path(exp, exp["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with stage("EX3 minimal canonical frame + auxiliary outcomes"):
        frame, aux = _prepare_minimal_frame(base_cfg)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))

    # Keep auxiliary rows aligned after later filtering by carrying them on the frame.
    frame = frame.reset_index(drop=True)
    aux = aux.reset_index(drop=True)
    frame["_ex3_middle_event"] = pd.to_numeric(aux["middle"], errors="coerce").fillna(0).astype(np.float32)

    with stage("EX3 previous-season pitcher states"):
        profiles = _season_profiles(
            frame,
            aux,
            season_col=season_col,
            pitcher_col=pitcher_col,
            target_col=target_col,
        )
        frame = _attach_previous_state(
            frame,
            profiles,
            season_col=season_col,
            pitcher_col=pitcher_col,
        )

    target_cols = [f"past_{name}" for name in STATE_NAMES]
    min_past = int(exp.get("min_past_pitch_count", 200))
    complete = frame[target_cols].notna().all(axis=1).to_numpy()
    complete &= pd.to_numeric(frame["past_pitch_count"], errors="coerce").fillna(0).to_numpy() >= min_past
    frame = frame.loc[complete].reset_index(drop=True)

    variants = [str(v) for v in exp.get("variants", ["state_only", "context_state"])]
    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    soft_weight = float(exp.get("soft_event_weight", 0.35))
    temp_grid = [float(x) for x in exp.get("temperature_grid", [0.25, 0.5, 1.0, 2.0, 4.0])]
    sigma_floor = float(exp.get("energy_sigma_floor", 0.01))
    substate_temperature = float(exp.get("substate_temperature", 1.0))
    prior_floor = float(exp.get("failure_middle_prior_floor", 0.02))
    prior_ceiling = float(exp.get("failure_middle_prior_ceiling", 0.98))
    audit_rows = int(exp.get("independence_audit_rows", 8))

    results: dict[str, Any] = {
        "experiment": "EX3 four-state counterfactual backward energy",
        "definition": (
            "F_hard/F_soft/S_soft/S_hard are counterfactual post-pitch states. "
            "Each is mapped backward to the known previous-season pitcher profile; "
            "lower reconstruction energy receives higher latent-state probability."
        ),
        "safe_contract": "no forward success classifier; no interaction across evaluation rows",
        "class_names": list(CLASS_NAMES),
        "soft_event_weight": soft_weight,
        "min_past_pitch_count": min_past,
        "eligible_rows": int(len(frame)),
        "variants": {},
    }

    for variant in variants:
        features = _features_for_variant(variant)
        past_energies: list[np.ndarray] = []
        past_labels: list[np.ndarray] = []
        variant_result: dict[str, Any] = {"feature_count": len(features), "folds": {}}

        for test_season in fold_seasons:
            train = frame[frame[season_col] < test_season].reset_index(drop=True)
            test = frame[frame[season_col] == test_season].reset_index(drop=True)
            if len(train) == 0 or len(test) == 0:
                variant_result["folds"][str(test_season)] = {
                    "skipped": True,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                }
                continue

            train_state = _assimilated_state(
                train,
                event_success=pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float64),
                event_middle=pd.to_numeric(train["_ex3_middle_event"], errors="coerce").fillna(0).to_numpy(np.float64),
                evidence_weight=1.0,
            )
            x_train, cats = _prepare_x(train_state, features)
            z_train = train[target_cols].to_numpy(np.float32)

            group_sizes = train.groupby([season_col, pitcher_col], dropna=False)[pitcher_col].transform("size")
            weights = (1.0 / group_sizes.to_numpy(np.float64)).astype(np.float32)
            weights *= float(len(weights) / weights.sum())

            with stage(f"EX3 fit realized-state backward [{variant}] test={test_season}"):
                pool = Pool(
                    x_train,
                    z_train,
                    weight=weights,
                    cat_features=cats,
                    feature_names=features,
                )
                model = CatBoostRegressor(**_model_params(exp["catboost"])).fit(pool)

            train_pred = np.clip(np.asarray(model.predict(pool), dtype=np.float64), 0.0, 1.0)
            sigma = np.sqrt(np.mean((train_pred - z_train.astype(np.float64)) ** 2, axis=0))
            sigma = np.maximum(sigma, sigma_floor)

            failure_train = train[pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8) == 0]
            if len(failure_train):
                middle_prior = float(pd.to_numeric(failure_train["_ex3_middle_event"], errors="coerce").fillna(0).mean())
            else:
                middle_prior = 0.5
            middle_prior = float(np.clip(middle_prior, prior_floor, prior_ceiling))

            z_test = test[target_cols].to_numpy(np.float64)
            y_test = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.int8)
            with stage(f"EX3 evaluate four counterfactual states [{variant}] test={test_season}"):
                energies, _ = _class_energies(
                    model,
                    test,
                    z_test,
                    sigma,
                    features,
                    soft_weight=soft_weight,
                    failure_middle_prior=middle_prior,
                    substate_temperature=substate_temperature,
                )

            temperature = _select_temperature(past_energies, past_labels, temp_grid)
            probs = _probabilities(energies, temperature)
            success_prob = probs[:, 2] + probs[:, 3]
            margin = _binary_margin(energies)
            prior = float(pd.to_numeric(train[target_col], errors="raise").mean())

            metrics = {
                "rows": int(len(test)),
                "target_rate": float(y_test.mean()),
                "auc_margin": _auc(y_test, margin),
                "brier": _brier(y_test, success_prob),
                "logloss": _logloss(y_test, success_prob),
                "prior_brier": _brier(y_test, np.full(len(y_test), prior)),
                "temperature_from_previous_folds": temperature,
                "failure_middle_prior_from_train": middle_prior,
                "prob_mean": float(success_prob.mean()),
                "prob_std": float(success_prob.std()),
            }
            diagnostics = _class_diagnostics(y_test, probs, energies)
            independence = _independence_audit(
                model,
                test,
                z_test,
                sigma,
                features,
                rows=audit_rows,
                soft_weight=soft_weight,
                failure_middle_prior=middle_prior,
                substate_temperature=substate_temperature,
            )

            fold_result = {
                "skipped": False,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_seasons": sorted(int(x) for x in train[season_col].unique().tolist()),
                "metrics": metrics,
                "latent_state_diagnostics": diagnostics,
                "energy_sigma": {name: float(sigma[i]) for i, name in enumerate(STATE_NAMES)},
                "row_independence_max_abs_diff": independence,
            }
            variant_result["folds"][str(test_season)] = fold_result

            np.savez_compressed(
                out_dir / f"four_state_{variant}_{test_season}.npz",
                y=y_test,
                row_id=test["row_id"].astype(str).to_numpy(dtype="U"),
                energies=energies,
                probabilities=probs,
                success_probability=success_prob,
                margin=margin,
                past_state=z_test,
            )

            if test_season == max(fold_seasons):
                model_path = model_dir / f"four_state_backward_{variant}.cbm"
                model.save_model(str(model_path))
                fold_result["model_path"] = str(model_path.relative_to(Path(exp["_repo_root"])))

            log(
                f"[EX3:{variant}:{test_season}] "
                f"auc={metrics['auc_margin'] if metrics['auc_margin'] is not None else float('nan'):.5f} "
                f"brier={metrics['brier']:.8f} prior={metrics['prior_brier']:.8f} "
                f"T={temperature:.3f} middle_prior={middle_prior:.3f} "
                f"amb={diagnostics['ambiguity_mean']:.3f} conf={diagnostics['confidence_mean']:.3f} "
                f"row_diff={independence:.3e}"
            )

            past_energies.append(energies.copy())
            past_labels.append(y_test.copy())

        aucs = [
            fold["metrics"]["auc_margin"]
            for fold in variant_result["folds"].values()
            if not fold.get("skipped") and fold["metrics"]["auc_margin"] is not None
        ]
        briers = [
            fold["metrics"]["brier"]
            for fold in variant_result["folds"].values()
            if not fold.get("skipped")
        ]
        priors = [
            fold["metrics"]["prior_brier"]
            for fold in variant_result["folds"].values()
            if not fold.get("skipped")
        ]
        variant_result["summary"] = {
            "mean_auc_margin": float(np.mean(aucs)) if aucs else None,
            "mean_brier": float(np.mean(briers)) if briers else None,
            "mean_prior_brier": float(np.mean(priors)) if priors else None,
        }
        results["variants"][variant] = variant_result

    metrics_path = out_dir / "metrics_four_state_backward.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"[EX3] complete -> {metrics_path}")
    return results
