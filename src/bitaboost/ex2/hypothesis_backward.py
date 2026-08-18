from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.ex1.backward_suite import _prepare_minimal_frame
from bitaboost.ex1.pitcher_season_backward import STATE_NAMES, _season_profiles
from bitaboost.legacy import activate
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage

ID_FEATURES = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
HISTORY_PREFIXES = ("asof_", "eng_ps_")
HYPOTHESIS_COL = "hypothesis_y"


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


def _attach_previous_state(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
) -> pd.DataFrame:
    lookup = profiles.copy()
    lookup["current_season"] = lookup[season_col].astype(int) + 1
    lookup = lookup.rename(
        columns={
            "pitch_count": "past_pitch_count",
            **{name: f"past_{name}" for name in STATE_NAMES},
        }
    )
    lookup = lookup[
        ["current_season", pitcher_col, "past_pitch_count", *[f"past_{x}" for x in STATE_NAMES]]
    ]

    keys = frame[[season_col, pitcher_col]].reset_index(drop=False).rename(
        columns={"index": "_row", season_col: "current_season"}
    )
    attached = keys.merge(lookup, on=["current_season", pitcher_col], how="left", sort=False)
    attached = attached.sort_values("_row", kind="stable").set_index("_row")

    out = frame.copy()
    for name in ["past_pitch_count", *[f"past_{x}" for x in STATE_NAMES]]:
        out[name] = attached[name].reindex(range(len(frame))).to_numpy()
    return out


def _base_features() -> list[str]:
    activate()
    import build_recent_regime_submissions as recent_core

    features = list(recent_core.feature_set("recent_raw_game_type"))
    if "control_success" in features:
        raise RuntimeError("target leakage: control_success appeared in EX2 features")
    return features


def _variant_features(base: list[str], variant: str) -> list[str]:
    if variant == "context_only":
        features = [
            f
            for f in base
            if f not in ID_FEATURES and not any(f.startswith(prefix) for prefix in HISTORY_PREFIXES)
        ]
    elif variant == "history_no_id":
        features = [f for f in base if f not in ID_FEATURES]
    elif variant == "history_plus_id":
        features = list(base)
    else:
        raise ValueError(f"unknown EX2 feature variant: {variant}")
    features.append(HYPOTHESIS_COL)
    if len(features) != len(set(features)):
        raise RuntimeError(f"duplicate EX2 features in {variant}")
    return features


def _prepare_x(frame: pd.DataFrame, features: list[str]):
    activate()
    import run_context_interaction_screen as context_core

    return context_core.prepare_x(frame, features)


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


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


def _classification_metrics(y: np.ndarray, margin: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    pred = (np.asarray(margin) > 0.0).astype(np.int8)
    return {
        "rows": int(len(y)),
        "target_rate": float(y.mean()),
        "auc_margin": _auc(y, margin),
        "sign_accuracy": float(np.mean(pred == y)),
        "brier": _brier(y, probability),
        "logloss": _logloss(y, probability),
        "prob_mean": float(np.mean(probability)),
        "prob_std": float(np.std(probability)),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
    }


def _rmse_by_target(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, name in enumerate(STATE_NAMES):
        out[name] = float(np.sqrt(np.mean((pred[:, i] - y[:, i]) ** 2)))
    out["macro"] = float(np.mean(list(out.values())))
    return out


def _temperature_from_history(
    margins: list[np.ndarray],
    labels: list[np.ndarray],
    grid: list[float],
) -> float:
    if not margins:
        return 1.0
    m = np.concatenate(margins).astype(np.float64)
    y = np.concatenate(labels).astype(np.float64)
    best_t = float(grid[0])
    best = float("inf")
    for t in grid:
        p = _sigmoid(m / float(t))
        score = _brier(y, p)
        if score < best:
            best = score
            best_t = float(t)
    return best_t


def _candidate_predictions(model, frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from catboost import Pool

    predictions: list[np.ndarray] = []
    for candidate in (0.0, 1.0):
        block = frame.copy()
        block[HYPOTHESIS_COL] = np.float32(candidate)
        x, cats = _prepare_x(block, features)
        pool = Pool(x, cat_features=cats, feature_names=features)
        pred = np.clip(np.asarray(model.predict(pool), dtype=np.float64), 0.0, 1.0)
        predictions.append(pred)
    return predictions[0], predictions[1]


def _independence_audit(model, frame: pd.DataFrame, features: list[str], rows: int) -> float:
    from catboost import Pool

    n = min(int(rows), len(frame))
    if n <= 0:
        return 0.0
    sample = frame.iloc[:n].copy()
    max_diff = 0.0
    for candidate in (0.0, 1.0):
        batch_frame = sample.copy()
        batch_frame[HYPOTHESIS_COL] = np.float32(candidate)
        xb, cats = _prepare_x(batch_frame, features)
        batch = np.asarray(
            model.predict(Pool(xb, cat_features=cats, feature_names=features)), dtype=np.float64
        )
        single: list[np.ndarray] = []
        for i in range(n):
            row = sample.iloc[[i]].copy()
            row[HYPOTHESIS_COL] = np.float32(candidate)
            xo, cats_one = _prepare_x(row, features)
            value = np.asarray(
                model.predict(Pool(xo, cat_features=cats_one, feature_names=features)),
                dtype=np.float64,
            )[0]
            single.append(value)
        diff = float(np.max(np.abs(batch - np.asarray(single))))
        max_diff = max(max_diff, diff)
    return max_diff


def _tf_groups(
    y: np.ndarray,
    past_success: np.ndarray,
    margin: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    trait_threshold: float,
) -> dict[str, Any]:
    trait = np.asarray(past_success) >= float(trait_threshold)
    yb = np.asarray(y, dtype=np.int8)
    labels = np.where(
        trait & (yb == 1),
        "TT",
        np.where(trait & (yb == 0), "TF", np.where((~trait) & (yb == 1), "FT", "FF")),
    )
    out: dict[str, Any] = {"trait_threshold": float(trait_threshold), "groups": {}}
    for name in ("TT", "TF", "FT", "FF"):
        mask = labels == name
        if not mask.any():
            continue
        true_energy = np.where(yb[mask] == 1, e1[mask], e0[mask])
        false_energy = np.where(yb[mask] == 1, e0[mask], e1[mask])
        correct = (margin[mask] > 0).astype(np.int8) == yb[mask]
        out["groups"][name] = {
            "rows": int(mask.sum()),
            "margin_mean": float(np.mean(margin[mask])),
            "energy_advantage_false_minus_true": float(np.mean(false_energy - true_energy)),
            "sign_accuracy": float(np.mean(correct)),
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

    with stage("EX2 minimal canonical frame + auxiliary outcomes"):
        frame, aux = _prepare_minimal_frame(base_cfg)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))

    with stage("EX2 previous-season pitcher states"):
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

    base_features = _base_features()
    variants = [str(x) for x in exp.get("variants", ["history_no_id", "history_plus_id"])]
    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    min_past_pitch_count = int(exp.get("min_past_pitch_count", 200))
    temperature_grid = [float(x) for x in exp.get("temperature_grid", [0.25, 0.5, 1, 2, 4, 8])]
    audit_rows = int(exp.get("independence_audit_rows", 8))
    target_cols = [f"past_{x}" for x in STATE_NAMES]

    complete = frame[target_cols].notna().all(axis=1).to_numpy()
    complete &= pd.to_numeric(frame["past_pitch_count"], errors="coerce").fillna(0).to_numpy() >= min_past_pitch_count
    frame = frame.loc[complete].reset_index(drop=True)

    results: dict[str, Any] = {
        "experiment": "EX2 candidate-label-conditioned backward reconstruction",
        "definition": "for each row, assume y=0 and y=1 separately; prefer the hypothesis that reconstructs the known previous-season pitcher state with lower standardized energy",
        "safe_contract": "row-independent; previous-season state is built only from earlier official training seasons",
        "state_names": list(STATE_NAMES),
        "min_past_pitch_count": min_past_pitch_count,
        "eligible_rows": int(len(frame)),
        "variants": {},
    }

    for variant in variants:
        features = _variant_features(base_features, variant)
        calibration_margins: list[np.ndarray] = []
        calibration_labels: list[np.ndarray] = []
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

            train_fit = train.copy()
            train_fit[HYPOTHESIS_COL] = pd.to_numeric(
                train_fit[target_col], errors="raise"
            ).astype(np.float32)
            x_train, cats = _prepare_x(train_fit, features)
            z_train = train_fit[target_cols].to_numpy(np.float32)

            group_sizes = train_fit.groupby(
                [season_col, pitcher_col], dropna=False
            )[pitcher_col].transform("size")
            weights = (1.0 / group_sizes.to_numpy(np.float64)).astype(np.float32)
            weights *= float(len(weights) / weights.sum())

            with stage(f"EX2 fit hypothesis-backward [{variant}] test={test_season}"):
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
            sigma = np.maximum(sigma, float(exp.get("energy_sigma_floor", 0.01)))

            with stage(f"EX2 evaluate y=0/y=1 hypotheses [{variant}] test={test_season}"):
                pred0, pred1 = _candidate_predictions(model, test, features)

            z_test = test[target_cols].to_numpy(np.float64)
            y_test = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.int8)
            e0_components = ((z_test - pred0) / sigma[None, :]) ** 2
            e1_components = ((z_test - pred1) / sigma[None, :]) ** 2
            e0 = e0_components.mean(axis=1)
            e1 = e1_components.mean(axis=1)
            component_margin = e0_components - e1_components
            margin = e0 - e1

            temperature = _temperature_from_history(
                calibration_margins,
                calibration_labels,
                temperature_grid,
            )
            probability = _sigmoid(margin / temperature)
            metrics = _classification_metrics(y_test, margin, probability)
            prior = float(pd.to_numeric(train[target_col], errors="raise").mean())
            metrics["prior_brier"] = _brier(y_test, np.full(len(y_test), prior))
            metrics["temperature_from_previous_folds"] = temperature

            true_pred = np.where(y_test[:, None] == 1, pred1, pred0)
            false_pred = np.where(y_test[:, None] == 1, pred0, pred1)
            reconstruction = {
                "true_hypothesis_rmse": _rmse_by_target(z_test, true_pred),
                "false_hypothesis_rmse": _rmse_by_target(z_test, false_pred),
                "counterfactual_gap_mean_abs": {
                    name: float(np.mean(np.abs(pred1[:, i] - pred0[:, i])))
                    for i, name in enumerate(STATE_NAMES)
                },
                "energy_sigma": {name: float(sigma[i]) for i, name in enumerate(STATE_NAMES)},
            }

            target_contribution: dict[str, Any] = {}
            for i, name in enumerate(STATE_NAMES):
                cm = component_margin[:, i]
                target_contribution[name] = {
                    "auc_margin": _auc(y_test, cm),
                    "sign_accuracy": float(np.mean((cm > 0).astype(np.int8) == y_test)),
                    "margin_mean": float(np.mean(cm)),
                    "margin_std": float(np.std(cm)),
                }

            unique_trait = train.drop_duplicates([season_col, pitcher_col], keep="last")
            trait_threshold = float(pd.to_numeric(unique_trait["past_success"], errors="coerce").median())
            tf = _tf_groups(
                y_test,
                test["past_success"].to_numpy(np.float64),
                margin,
                e0,
                e1,
                trait_threshold,
            )
            independence = _independence_audit(model, test, features, audit_rows)

            fold_result = {
                "skipped": False,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_seasons": sorted(int(x) for x in train[season_col].unique().tolist()),
                "metrics": metrics,
                "reconstruction": reconstruction,
                "target_contribution": target_contribution,
                "tf_ft_groups": tf,
                "row_independence_max_abs_diff": independence,
            }
            variant_result["folds"][str(test_season)] = fold_result

            np.savez_compressed(
                out_dir / f"hypothesis_{variant}_{test_season}.npz",
                y=y_test,
                row_id=test["row_id"].astype(str).to_numpy(),
                margin=margin,
                probability=probability,
                energy0=e0,
                energy1=e1,
                pred0=pred0,
                pred1=pred1,
                past_state=z_test,
            )

            if test_season == max(fold_seasons):
                model_path = model_dir / f"hypothesis_backward_{variant}.cbm"
                model.save_model(str(model_path))
                fold_result["model_path"] = str(model_path.relative_to(Path(exp["_repo_root"])))

            log(
                f"[EX2:{variant}:{test_season}] "
                f"auc={metrics['auc_margin'] if metrics['auc_margin'] is not None else float('nan'):.5f} "
                f"acc={metrics['sign_accuracy']:.5f} brier={metrics['brier']:.8f} "
                f"prior={metrics['prior_brier']:.8f} temp={temperature:.3f} "
                f"row_diff={independence:.3e}"
            )

            calibration_margins.append(margin.copy())
            calibration_labels.append(y_test.copy())

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

    (out_dir / "metrics_hypothesis_backward.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"[EX2] complete -> {out_dir / 'metrics_hypothesis_backward.json'}")
    return results
