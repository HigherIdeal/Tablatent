from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitaboost.config import load_config
from bitaboost.ex1.pitcher_season_backward import STATE_NAMES, _season_profiles
from bitaboost.ex2.hypothesis_backward import (
    _attach_previous_state,
    _auc,
    _brier,
    _logloss,
    _prepare_minimal_frame,
    _prepare_x,
)
from bitaboost.ex3.four_state_backward import (
    CLASS_NAMES,
    _binary_margin,
    _class_diagnostics,
    _features_for_variant,
    _load_experiment_config,
    _mix_failure_energy,
    _model_params,
    _path,
    _probabilities,
    _predict,
    _select_temperature,
    _state_energy,
)
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage


def _numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    if np.isnan(values).any():
        values = np.where(np.isnan(values), default, values)
    return values


def _assimilated_state_strength(
    frame: pd.DataFrame,
    *,
    event_success: np.ndarray | float,
    event_middle: np.ndarray | float,
    evidence_weight: float,
) -> pd.DataFrame:
    """Create a row-local pseudo-post state with arbitrary positive strength.

    EX3 originally used w<=1, so a single hypothetical pitch changed a veteran's
    cumulative state by O(1/n). This diagnostic deliberately allows w>1. It must
    not be interpreted as a literal number of future pitches; w is a pseudo-count
    that controls how strongly the candidate latent state displaces the pre-pitch
    sufficient statistics.
    """
    out = frame.copy()
    n = np.maximum(_numeric(out["asof_pitcher_n"]), 0.0)
    success_rate = np.clip(_numeric(out["asof_pitcher_success_rate"]), 0.0, 1.0)
    middle_rate = np.clip(_numeric(out["asof_pitcher_middle_rate"]), 0.0, 1.0)

    success_event = np.broadcast_to(np.asarray(event_success, dtype=np.float64), n.shape)
    middle_event = np.broadcast_to(np.asarray(event_middle, dtype=np.float64), n.shape)
    w = float(evidence_weight)
    if not np.isfinite(w) or w <= 0.0:
        raise ValueError(f"evidence_weight must be finite and > 0, got {w}")

    denom = np.maximum(n + w, 1e-12)
    post_success = np.clip((n * success_rate + w * success_event) / denom, 0.0, 1.0)
    post_middle = np.clip((n * middle_rate + w * middle_event) / denom, 0.0, 1.0)

    out["cf_post_success_rate"] = post_success.astype(np.float32)
    out["cf_post_middle_rate"] = post_middle.astype(np.float32)
    out["cf_post_log_n"] = np.log1p(n + w).astype(np.float32)
    out["cf_delta_success_rate"] = (post_success - success_rate).astype(np.float32)
    out["cf_delta_middle_rate"] = (post_middle - middle_rate).astype(np.float32)
    return out


def _class_energies_strength(
    model,
    frame: pd.DataFrame,
    target_state: np.ndarray,
    sigma: np.ndarray,
    features: list[str],
    *,
    hard_strength: float,
    soft_ratio: float,
    failure_middle_prior: float,
    substate_temperature: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    hard = float(hard_strength)
    soft = hard * float(soft_ratio)
    if hard <= 0.0:
        raise ValueError("hard_strength must be > 0")
    if not (0.0 < soft_ratio < 1.0):
        raise ValueError("soft_ratio must be in (0, 1)")

    candidate_frames = {
        "F_hard_m0": _assimilated_state_strength(
            frame, event_success=0.0, event_middle=0.0, evidence_weight=hard
        ),
        "F_hard_m1": _assimilated_state_strength(
            frame, event_success=0.0, event_middle=1.0, evidence_weight=hard
        ),
        "F_soft_m0": _assimilated_state_strength(
            frame, event_success=0.0, event_middle=0.0, evidence_weight=soft
        ),
        "F_soft_m1": _assimilated_state_strength(
            frame, event_success=0.0, event_middle=1.0, evidence_weight=soft
        ),
        "S_soft": _assimilated_state_strength(
            frame, event_success=1.0, event_middle=0.0, evidence_weight=soft
        ),
        "S_hard": _assimilated_state_strength(
            frame, event_success=1.0, event_middle=0.0, evidence_weight=hard
        ),
    }
    predictions = {name: _predict(model, block, features) for name, block in candidate_frames.items()}
    energy = {name: _state_energy(target_state, pred, sigma) for name, pred in predictions.items()}

    f_hard = _mix_failure_energy(
        energy["F_hard_m0"],
        energy["F_hard_m1"],
        failure_middle_prior,
        substate_temperature,
    )
    f_soft = _mix_failure_energy(
        energy["F_soft_m0"],
        energy["F_soft_m1"],
        failure_middle_prior,
        substate_temperature,
    )
    energies = np.stack([f_hard, f_soft, energy["S_soft"], energy["S_hard"]], axis=1)

    # Quantify how large the pseudo-count perturbation actually is in the input state.
    fh = candidate_frames["F_hard_m0"]
    fs = candidate_frames["F_soft_m0"]
    sh = candidate_frames["S_hard"]
    ss = candidate_frames["S_soft"]
    displacement = {
        "hard_weight": hard,
        "soft_weight": soft,
        "hard_success_gap_mean_abs": float(
            np.mean(np.abs(_numeric(sh["cf_post_success_rate"]) - _numeric(fh["cf_post_success_rate"])))
        ),
        "soft_success_gap_mean_abs": float(
            np.mean(np.abs(_numeric(ss["cf_post_success_rate"]) - _numeric(fs["cf_post_success_rate"])))
        ),
        "hard_delta_success_mean_abs": float(np.mean(np.abs(_numeric(sh["cf_delta_success_rate"])))),
        "soft_delta_success_mean_abs": float(np.mean(np.abs(_numeric(ss["cf_delta_success_rate"])))),
    }
    return energies, predictions, displacement


def _independence_audit_strength(
    model,
    frame: pd.DataFrame,
    target_state: np.ndarray,
    sigma: np.ndarray,
    features: list[str],
    *,
    rows: int,
    hard_strength: float,
    soft_ratio: float,
    failure_middle_prior: float,
    substate_temperature: float,
) -> float:
    n = min(int(rows), len(frame))
    if n <= 0:
        return 0.0
    batch, _, _ = _class_energies_strength(
        model,
        frame.iloc[:n].copy(),
        target_state[:n],
        sigma,
        features,
        hard_strength=hard_strength,
        soft_ratio=soft_ratio,
        failure_middle_prior=failure_middle_prior,
        substate_temperature=substate_temperature,
    )
    singles: list[np.ndarray] = []
    for i in range(n):
        one, _, _ = _class_energies_strength(
            model,
            frame.iloc[[i]].copy(),
            target_state[i : i + 1],
            sigma,
            features,
            hard_strength=hard_strength,
            soft_ratio=soft_ratio,
            failure_middle_prior=failure_middle_prior,
            substate_temperature=substate_temperature,
        )
        singles.append(one[0])
    return float(np.max(np.abs(batch - np.asarray(singles))))


def _strength_key(value: float) -> str:
    return f"{float(value):g}"


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

    with stage("EX3-SWEEP minimal canonical frame + auxiliary outcomes"):
        frame, aux = _prepare_minimal_frame(base_cfg)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))

    frame = frame.reset_index(drop=True)
    aux = aux.reset_index(drop=True)
    frame["_ex3_middle_event"] = (
        pd.to_numeric(aux["middle"], errors="coerce").fillna(0).astype(np.float32)
    )

    with stage("EX3-SWEEP previous-season pitcher states"):
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
    complete &= (
        pd.to_numeric(frame["past_pitch_count"], errors="coerce").fillna(0).to_numpy()
        >= min_past
    )
    frame = frame.loc[complete].reset_index(drop=True)

    variants = [str(v) for v in exp.get("variants", ["state_only", "context_state"])]
    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    strengths = [float(x) for x in exp.get("hard_event_strengths", [1, 5, 20, 50, 100])]
    if not strengths or any((not np.isfinite(x) or x <= 0.0) for x in strengths):
        raise ValueError("hard_event_strengths must contain positive finite values")
    if len({_strength_key(x) for x in strengths}) != len(strengths):
        raise ValueError("hard_event_strengths contains duplicate values")
    soft_ratio = float(exp.get("soft_to_hard_ratio", 0.35))
    temp_grid = [float(x) for x in exp.get("temperature_grid", [0.25, 0.5, 1, 2, 4])]
    sigma_floor = float(exp.get("energy_sigma_floor", 0.01))
    substate_temperature = float(exp.get("substate_temperature", 1.0))
    prior_floor = float(exp.get("failure_middle_prior_floor", 0.02))
    prior_ceiling = float(exp.get("failure_middle_prior_ceiling", 0.98))
    audit_rows = int(exp.get("independence_audit_rows", 8))

    results: dict[str, Any] = {
        "experiment": "EX3 counterfactual pseudo-count strength sweep",
        "question": (
            "Did four-state EX3 collapse because one-pitch counterfactual displacement "
            "was too small relative to backward reconstruction noise?"
        ),
        "training_contract": (
            "One backward model per variant/fold is trained only on the realized one-pitch state (w=1). "
            "The same frozen fold model is reused for every lambda; only candidate-state pseudo-count strength changes."
        ),
        "safe_contract": "no forward success classifier; every evaluation row is transformed independently",
        "class_names": list(CLASS_NAMES),
        "hard_event_strengths": strengths,
        "soft_to_hard_ratio": soft_ratio,
        "min_past_pitch_count": min_past,
        "eligible_rows": int(len(frame)),
        "variants": {},
    }

    for variant in variants:
        features = _features_for_variant(variant)
        histories: dict[str, dict[str, list[np.ndarray]]] = {
            _strength_key(s): {"energies": [], "labels": []} for s in strengths
        }
        variant_result: dict[str, Any] = {
            "feature_count": len(features),
            "folds": {},
            "strength_summary": {},
        }

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

            # Training remains literal one-pitch assimilation. The sweep is evaluation-only.
            train_state = _assimilated_state_strength(
                train,
                event_success=pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float64),
                event_middle=(
                    pd.to_numeric(train["_ex3_middle_event"], errors="coerce")
                    .fillna(0)
                    .to_numpy(np.float64)
                ),
                evidence_weight=1.0,
            )
            x_train, cats = _prepare_x(train_state, features)
            z_train = train[target_cols].to_numpy(np.float32)

            group_sizes = train.groupby(
                [season_col, pitcher_col], dropna=False
            )[pitcher_col].transform("size")
            weights = (1.0 / group_sizes.to_numpy(np.float64)).astype(np.float32)
            weights *= float(len(weights) / weights.sum())

            with stage(f"EX3-SWEEP fit realized-state backward [{variant}] test={test_season}"):
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

            failure_mask = (
                pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8) == 0
            )
            failure_train = train.loc[failure_mask]
            if len(failure_train):
                middle_prior = float(
                    pd.to_numeric(failure_train["_ex3_middle_event"], errors="coerce")
                    .fillna(0)
                    .mean()
                )
            else:
                middle_prior = 0.5
            middle_prior = float(np.clip(middle_prior, prior_floor, prior_ceiling))

            z_test = test[target_cols].to_numpy(np.float64)
            y_test = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.int8)
            prior = float(pd.to_numeric(train[target_col], errors="raise").mean())
            fold_result: dict[str, Any] = {
                "skipped": False,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_seasons": sorted(int(x) for x in train[season_col].unique().tolist()),
                "failure_middle_prior_from_train": middle_prior,
                "energy_sigma": {name: float(sigma[i]) for i, name in enumerate(STATE_NAMES)},
                "strengths": {},
            }

            for strength in strengths:
                key = _strength_key(strength)
                with stage(
                    f"EX3-SWEEP evaluate lambda={key} [{variant}] test={test_season}"
                ):
                    energies, _, displacement = _class_energies_strength(
                        model,
                        test,
                        z_test,
                        sigma,
                        features,
                        hard_strength=strength,
                        soft_ratio=soft_ratio,
                        failure_middle_prior=middle_prior,
                        substate_temperature=substate_temperature,
                    )

                hist = histories[key]
                temperature = _select_temperature(
                    hist["energies"], hist["labels"], temp_grid
                )
                probs = _probabilities(energies, temperature)
                success_prob = probs[:, 2] + probs[:, 3]
                margin = _binary_margin(energies)
                auc = _auc(y_test, margin)
                diagnostics = _class_diagnostics(y_test, probs, energies)
                independence = _independence_audit_strength(
                    model,
                    test,
                    z_test,
                    sigma,
                    features,
                    rows=audit_rows,
                    hard_strength=strength,
                    soft_ratio=soft_ratio,
                    failure_middle_prior=middle_prior,
                    substate_temperature=substate_temperature,
                )

                metrics = {
                    "rows": int(len(test)),
                    "target_rate": float(y_test.mean()),
                    "auc_margin": auc,
                    "brier": _brier(y_test, success_prob),
                    "logloss": _logloss(y_test, success_prob),
                    "prior_brier": _brier(y_test, np.full(len(y_test), prior)),
                    "temperature_from_previous_folds": temperature,
                    "prob_mean": float(success_prob.mean()),
                    "prob_std": float(success_prob.std()),
                }
                fold_result["strengths"][key] = {
                    "metrics": metrics,
                    "displacement": displacement,
                    "latent_state_diagnostics": diagnostics,
                    "row_independence_max_abs_diff": independence,
                }

                log(
                    f"[EX3-SWEEP:{variant}:{test_season}:lambda={key}] "
                    f"auc={auc if auc is not None else float('nan'):.5f} "
                    f"brier={metrics['brier']:.8f} prior={metrics['prior_brier']:.8f} "
                    f"gap_h={displacement['hard_success_gap_mean_abs']:.5f} "
                    f"gap_s={displacement['soft_success_gap_mean_abs']:.5f} "
                    f"amb={diagnostics['ambiguity_mean']:.3f} "
                    f"T={temperature:.3f} row_diff={independence:.3e}"
                )

                hist["energies"].append(energies.copy())
                hist["labels"].append(y_test.copy())

            variant_result["folds"][str(test_season)] = fold_result

            if test_season == max(fold_seasons):
                model_path = model_dir / f"four_state_strength_sweep_{variant}.cbm"
                model.save_model(str(model_path))
                fold_result["model_path"] = str(
                    model_path.relative_to(Path(exp["_repo_root"]))
                )

        # Summarize each lambda across chronological folds without using it to train anything.
        for strength in strengths:
            key = _strength_key(strength)
            fold_entries = [
                fold["strengths"][key]
                for fold in variant_result["folds"].values()
                if not fold.get("skipped") and key in fold.get("strengths", {})
            ]
            aucs = [
                entry["metrics"]["auc_margin"]
                for entry in fold_entries
                if entry["metrics"]["auc_margin"] is not None
            ]
            briers = [entry["metrics"]["brier"] for entry in fold_entries]
            priors = [entry["metrics"]["prior_brier"] for entry in fold_entries]
            gaps = [entry["displacement"]["hard_success_gap_mean_abs"] for entry in fold_entries]
            variant_result["strength_summary"][key] = {
                "mean_auc_margin": float(np.mean(aucs)) if aucs else None,
                "min_auc_margin": float(np.min(aucs)) if aucs else None,
                "max_auc_margin": float(np.max(aucs)) if aucs else None,
                "mean_brier": float(np.mean(briers)) if briers else None,
                "mean_prior_brier": float(np.mean(priors)) if priors else None,
                "mean_hard_success_gap": float(np.mean(gaps)) if gaps else None,
                "fold_aucs": aucs,
            }

        valid_summaries = [
            (key, value)
            for key, value in variant_result["strength_summary"].items()
            if value["mean_auc_margin"] is not None
        ]
        if valid_summaries:
            best_key, best = max(valid_summaries, key=lambda item: item[1]["mean_auc_margin"])
            variant_result["best_by_mean_auc"] = {
                "strength": float(best_key),
                **best,
            }
        results["variants"][variant] = variant_result

    metrics_path = out_dir / "metrics_counterfactual_strength_sweep.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"[EX3-SWEEP] complete -> {metrics_path}")
    return results
