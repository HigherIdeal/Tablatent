from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.features import AUX_NAMES, prepare
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage

STATE_NAMES = ("success", *AUX_NAMES)


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


def _season_profiles(
    frame: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    target_col: str,
) -> pd.DataFrame:
    """One row per pitcher-season. No row-level repeated targets."""
    base = frame[[season_col, pitcher_col, target_col]].reset_index(drop=True).copy()
    aux = aux.reset_index(drop=True)
    base["success"] = pd.to_numeric(base[target_col], errors="coerce")
    for name in AUX_NAMES:
        base[name] = pd.to_numeric(aux[name], errors="coerce")

    grouped = base.groupby([season_col, pitcher_col], dropna=False, sort=True)
    out = grouped["success"].agg(["count", "mean"]).reset_index()
    out = out.rename(columns={"count": "pitch_count", "mean": "success"})
    for name in AUX_NAMES:
        rate = grouped[name].mean().rename(name).reset_index()
        out = out.merge(rate, on=[season_col, pitcher_col], how="left", sort=False)
    out[season_col] = pd.to_numeric(out[season_col], errors="raise").astype(int)
    return out


def _adjacent_pairs(
    profiles: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
) -> pd.DataFrame:
    """Build current-season -> previous-season pitcher pairs."""
    cur = profiles.copy()
    cur = cur.rename(
        columns={
            season_col: "current_season",
            "pitch_count": "current_pitch_count",
            **{name: f"current_{name}" for name in STATE_NAMES},
        }
    )
    past = profiles.copy()
    past["current_season"] = past[season_col].astype(int) + 1
    past = past.rename(
        columns={
            "pitch_count": "past_pitch_count",
            **{name: f"past_{name}" for name in STATE_NAMES},
        }
    )
    past = past[
        ["current_season", pitcher_col, "past_pitch_count", *[f"past_{x}" for x in STATE_NAMES]]
    ]
    pairs = cur.merge(past, on=["current_season", pitcher_col], how="inner", sort=True)
    pairs["current_log_pitch_count"] = np.log1p(
        pd.to_numeric(pairs["current_pitch_count"], errors="coerce").astype(float)
    )
    return pairs.reset_index(drop=True)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = pd.Series(np.asarray(a, dtype=np.float64)).rank(method="average")
    b = pd.Series(np.asarray(b, dtype=np.float64)).rank(method="average")
    value = a.corr(b)
    return None if pd.isna(value) else float(value)


def _target_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rmses: list[float] = []
    maes: list[float] = []
    for i, name in enumerate(STATE_NAMES):
        err = pred[:, i] - y[:, i]
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        rmses.append(rmse)
        maes.append(mae)
        result[name] = {
            "rmse": rmse,
            "mae": mae,
            "pearson": _corr(y[:, i], pred[:, i]),
            "spearman": _spearman(y[:, i], pred[:, i]),
        }
    result["macro"] = {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
    }
    return result


def _prepare_model_frame(
    table: pd.DataFrame,
    *,
    pitcher_col: str,
    variant: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = [*[f"current_{x}" for x in STATE_NAMES], "current_log_pitch_count"]
    categorical: list[str] = []
    if variant == "state_plus_id":
        features.append(pitcher_col)
        categorical.append(pitcher_col)
    elif variant != "state_only":
        raise ValueError(f"unknown backward variant: {variant}")

    x = table[features].copy()
    for name in features:
        if name in categorical:
            x[name] = x[name].astype("string").fillna("__NA__").astype(str)
        else:
            x[name] = pd.to_numeric(x[name], errors="coerce").astype(np.float32)
    return x, features, categorical


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    pitcher_col: str,
    variant: str,
    model_cfg: dict[str, Any],
):
    from catboost import CatBoostRegressor, Pool

    x_train, features, cats = _prepare_model_frame(
        train, pitcher_col=pitcher_col, variant=variant
    )
    x_test, _, _ = _prepare_model_frame(
        test, pitcher_col=pitcher_col, variant=variant
    )
    target_cols = [f"past_{x}" for x in STATE_NAMES]
    y_train = train[target_cols].to_numpy(np.float32)
    y_test = test[target_cols].to_numpy(np.float64)

    params = dict(model_cfg)
    params.update(
        {
            "loss_function": "MultiRMSE",
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "logging_level": "Silent",
        }
    )
    model = CatBoostRegressor(**params)
    model.fit(Pool(x_train, y_train, cat_features=cats, feature_names=features))
    pred = np.clip(
        np.asarray(
            model.predict(Pool(x_test, cat_features=cats, feature_names=features)),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    identity = test[[f"current_{x}" for x in STATE_NAMES]].to_numpy(np.float64)
    mean_pred = np.repeat(
        y_train.mean(axis=0, keepdims=True).astype(np.float64), len(test), axis=0
    )

    metrics = {
        "model": _target_metrics(y_test, pred),
        "identity_future_equals_past": _target_metrics(y_test, identity),
        "train_mean": _target_metrics(y_test, mean_pred),
    }
    metrics["delta_macro_rmse"] = {
        "model_minus_identity": (
            metrics["model"]["macro"]["rmse"]
            - metrics["identity_future_equals_past"]["macro"]["rmse"]
        ),
        "model_minus_train_mean": (
            metrics["model"]["macro"]["rmse"] - metrics["train_mean"]["macro"]["rmse"]
        ),
    }
    return model, pred, y_test, metrics


def run(config_path: str | Path) -> dict[str, Any]:
    exp = _load_experiment_config(config_path)
    base_cfg = load_config(_path(exp, exp["baseline_config"]))
    configure_cuda(base_cfg)
    configure_warnings(base_cfg)

    out_dir = _path(exp, exp["output_dir"])
    model_dir = _path(exp, exp["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with stage("EX1-B prepare canonical training frame"):
        data = prepare(base_cfg)
        frame = data.frame.reset_index(drop=True)
        aux = data.aux.reset_index(drop=True)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))

    with stage("EX1-B build pitcher-season states and adjacent backward pairs"):
        profiles = _season_profiles(
            frame,
            aux,
            season_col=season_col,
            pitcher_col=pitcher_col,
            target_col=target_col,
        )
        pairs = _adjacent_pairs(
            profiles,
            season_col=season_col,
            pitcher_col=pitcher_col,
        )

    target_cols = [f"past_{x}" for x in STATE_NAMES]
    input_cols = [f"current_{x}" for x in STATE_NAMES]
    complete = pairs[target_cols + input_cols].notna().all(axis=1)
    pairs = pairs.loc[complete].reset_index(drop=True)

    thresholds = [int(x) for x in exp.get("min_pitch_thresholds", [50])]
    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    variants = [str(x) for x in exp.get("variants", ["state_only"])]
    model_cfg = dict(exp["catboost"])

    results: dict[str, Any] = {
        "experiment": "EX1-B pure pitcher-season backward reconstruction",
        "definition": "future pitcher-season state -> same pitcher's immediately previous-season state",
        "state_names": list(STATE_NAMES),
        "rows": {
            "pitcher_seasons": int(len(profiles)),
            "adjacent_pairs_complete": int(len(pairs)),
        },
        "thresholds": {},
    }

    for threshold in thresholds:
        eligible = pairs[
            (pairs["current_pitch_count"] >= threshold)
            & (pairs["past_pitch_count"] >= threshold)
        ].reset_index(drop=True)
        threshold_key = str(threshold)
        threshold_result: dict[str, Any] = {
            "eligible_pairs": int(len(eligible)),
            "unique_pitchers": int(eligible[pitcher_col].nunique()),
            "variants": {},
        }

        for variant in variants:
            variant_result: dict[str, Any] = {"folds": {}}
            for test_season in fold_seasons:
                train = eligible[eligible["current_season"] < test_season].reset_index(drop=True)
                test = eligible[eligible["current_season"] == test_season].reset_index(drop=True)
                if len(train) == 0 or len(test) == 0:
                    variant_result["folds"][str(test_season)] = {
                        "skipped": True,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                    }
                    continue

                with stage(
                    f"EX1-B backward [{variant}] min{threshold} test={test_season}"
                ):
                    model, pred, y_test, metrics = _fit_fold(
                        train,
                        test,
                        pitcher_col=pitcher_col,
                        variant=variant,
                        model_cfg=model_cfg,
                    )

                fold_result = {
                    "skipped": False,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_seasons": sorted(
                        int(x) for x in train["current_season"].unique().tolist()
                    ),
                    "metrics": metrics,
                }
                variant_result["folds"][str(test_season)] = fold_result

                np.savez_compressed(
                    out_dir / f"backward_{variant}_min{threshold}_{test_season}.npz",
                    y=y_test,
                    pred=pred,
                    pitcher_id=test[pitcher_col].astype(str).to_numpy(),
                    current_season=test["current_season"].to_numpy(np.int16),
                )

                if test_season == max(fold_seasons):
                    model_path = (
                        model_dir / f"pitcher_season_backward_{variant}_min{threshold}.cbm"
                    )
                    model.save_model(str(model_path))
                    fold_result["model_path"] = str(
                        model_path.relative_to(Path(exp["_repo_root"]))
                    )

                log(
                    f"[EX1-B:{variant}:min{threshold}:{test_season}] "
                    f"model_rmse={metrics['model']['macro']['rmse']:.6f} "
                    f"identity_rmse={metrics['identity_future_equals_past']['macro']['rmse']:.6f} "
                    f"mean_rmse={metrics['train_mean']['macro']['rmse']:.6f}"
                )

            variant_result["summary"] = _summarize_variant(variant_result["folds"])
            threshold_result["variants"][variant] = variant_result

        results["thresholds"][threshold_key] = threshold_result

    (out_dir / "metrics_backward.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pairs.to_csv(out_dir / "pitcher_season_pairs.csv.gz", index=False, compression="gzip")
    return results


def _summarize_variant(folds: dict[str, Any]) -> dict[str, Any]:
    model: list[float] = []
    identity: list[float] = []
    mean: list[float] = []
    for fold in folds.values():
        if fold.get("skipped"):
            continue
        m = fold["metrics"]
        model.append(float(m["model"]["macro"]["rmse"]))
        identity.append(float(m["identity_future_equals_past"]["macro"]["rmse"]))
        mean.append(float(m["train_mean"]["macro"]["rmse"]))
    if not model:
        return {}
    return {
        "folds": len(model),
        "mean_macro_rmse": {
            "model": float(np.mean(model)),
            "identity_future_equals_past": float(np.mean(identity)),
            "train_mean": float(np.mean(mean)),
        },
        "model_minus_identity": float(np.mean(model) - np.mean(identity)),
        "model_minus_train_mean": float(np.mean(model) - np.mean(mean)),
    }
