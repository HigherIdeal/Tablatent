from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.features import _legacy_config, auxiliary_targets
from bitaboost.legacy import activate
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage
from bitaboost.ex1.pitcher_season_backward import (
    STATE_NAMES,
    _adjacent_pairs,
    _season_profiles,
    _target_metrics,
)


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


def _prepare_minimal_frame(base_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the canonical row frame and auxiliary outcomes needed by EX1.

    Unlike the SAFE baseline prepare() path, this deliberately skips frozen anchors,
    matchup profiles, regime interactions, and every other forward-model feature.
    """
    activate()
    import build_recent_regime_submissions as recent_core

    frame, _ = recent_core.prepare_frame(_legacy_config(base_cfg))
    frame = frame.reset_index(drop=True)
    season_col = base_cfg["data"]["season_col"]
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    aux = auxiliary_targets(frame).reset_index(drop=True)
    return frame, aux


def _add_career_state(
    pairs: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
) -> pd.DataFrame:
    history = profiles[[season_col, pitcher_col, "pitch_count"]].copy()
    history = history.sort_values([pitcher_col, season_col], kind="stable").reset_index(drop=True)
    history["career_before_current"] = (
        history.groupby(pitcher_col, dropna=False)["pitch_count"].cumsum()
        - history["pitch_count"]
    )
    history = history.rename(columns={season_col: "current_season"})
    history = history[["current_season", pitcher_col, "career_before_current"]]
    out = pairs.merge(history, on=["current_season", pitcher_col], how="left", sort=False)
    out["career_before_current"] = out["career_before_current"].fillna(0).astype(np.int64)
    out["past_log_pitch_count"] = np.log1p(
        pd.to_numeric(out["past_pitch_count"], errors="coerce").astype(float)
    )
    return out


def _model_params(model_cfg: dict[str, Any]) -> dict[str, Any]:
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
    return params


def _prepare_direction_frame(
    table: pd.DataFrame,
    *,
    pitcher_col: str,
    variant: str,
    direction: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if direction == "backward":
        prefix = "current"
        count_feature = "current_log_pitch_count"
    elif direction == "forward":
        prefix = "past"
        count_feature = "past_log_pitch_count"
    else:
        raise ValueError(f"unknown direction: {direction}")

    features = [*[f"{prefix}_{x}" for x in STATE_NAMES], count_feature]
    cats: list[str] = []
    if variant == "state_plus_id":
        features.append(pitcher_col)
        cats.append(pitcher_col)
    elif variant != "state_only":
        raise ValueError(f"unknown variant: {variant}")

    x = table[features].copy()
    for name in features:
        if name in cats:
            x[name] = x[name].astype("string").fillna("__NA__").astype(str)
        else:
            x[name] = pd.to_numeric(x[name], errors="coerce").astype(np.float32)
    return x, features, cats


def _fit_direct(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    pitcher_col: str,
    variant: str,
    direction: str,
    model_cfg: dict[str, Any],
):
    from catboost import CatBoostRegressor, Pool

    x_train, features, cats = _prepare_direction_frame(
        train, pitcher_col=pitcher_col, variant=variant, direction=direction
    )
    x_test, _, _ = _prepare_direction_frame(
        test, pitcher_col=pitcher_col, variant=variant, direction=direction
    )

    if direction == "backward":
        target_prefix = "past"
        identity_prefix = "current"
    else:
        target_prefix = "current"
        identity_prefix = "past"

    target_cols = [f"{target_prefix}_{x}" for x in STATE_NAMES]
    identity_cols = [f"{identity_prefix}_{x}" for x in STATE_NAMES]
    y_train = train[target_cols].to_numpy(np.float32)
    y_test = test[target_cols].to_numpy(np.float64)

    model = CatBoostRegressor(**_model_params(model_cfg))
    model.fit(Pool(x_train, y_train, cat_features=cats, feature_names=features))
    pred = np.clip(
        np.asarray(
            model.predict(Pool(x_test, cat_features=cats, feature_names=features)),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    identity = test[identity_cols].to_numpy(np.float64)
    mean_pred = np.repeat(
        y_train.mean(axis=0, keepdims=True).astype(np.float64), len(test), axis=0
    )
    return model, y_test, pred, identity, mean_pred


def _fit_delta_backward(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    pitcher_col: str,
    variant: str,
    model_cfg: dict[str, Any],
):
    from catboost import CatBoostRegressor, Pool

    x_train, features, cats = _prepare_direction_frame(
        train, pitcher_col=pitcher_col, variant=variant, direction="backward"
    )
    x_test, _, _ = _prepare_direction_frame(
        test, pitcher_col=pitcher_col, variant=variant, direction="backward"
    )

    past_cols = [f"past_{x}" for x in STATE_NAMES]
    current_cols = [f"current_{x}" for x in STATE_NAMES]
    past_train = train[past_cols].to_numpy(np.float32)
    current_train = train[current_cols].to_numpy(np.float32)
    delta_train = past_train - current_train

    y_test = test[past_cols].to_numpy(np.float64)
    current_test = test[current_cols].to_numpy(np.float64)

    model = CatBoostRegressor(**_model_params(model_cfg))
    model.fit(Pool(x_train, delta_train, cat_features=cats, feature_names=features))
    delta_pred = np.asarray(
        model.predict(Pool(x_test, cat_features=cats, feature_names=features)),
        dtype=np.float64,
    )
    pred = np.clip(current_test + delta_pred, 0.0, 1.0)
    return model, y_test, pred, current_test, delta_pred


def _metric_bundle(y: np.ndarray, pred: np.ndarray, identity: np.ndarray, mean_pred: np.ndarray) -> dict[str, Any]:
    model = _target_metrics(y, pred)
    identity_metrics = _target_metrics(y, identity)
    mean_metrics = _target_metrics(y, mean_pred)
    return {
        "model": model,
        "identity": identity_metrics,
        "train_mean": mean_metrics,
        "delta_macro_rmse": {
            "model_minus_identity": float(model["macro"]["rmse"] - identity_metrics["macro"]["rmse"]),
            "model_minus_train_mean": float(model["macro"]["rmse"] - mean_metrics["macro"]["rmse"]),
        },
    }


def _eligible(pairs: pd.DataFrame, threshold: int) -> pd.DataFrame:
    mask = (pairs["current_pitch_count"] >= threshold) & (pairs["past_pitch_count"] >= threshold)
    return pairs.loc[mask].reset_index(drop=True)


def _fold_split(table: pd.DataFrame, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = table[table["current_season"] < test_season].reset_index(drop=True)
    test = table[table["current_season"] == test_season].reset_index(drop=True)
    return train, test


def _stage1_target_decomposition(
    pairs: pd.DataFrame,
    *,
    pitcher_col: str,
    thresholds: list[int],
    folds: list[int],
    variants: list[str],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"description": "direct backward reconstruction by target", "thresholds": {}}
    for threshold in thresholds:
        table = _eligible(pairs, threshold)
        tresult: dict[str, Any] = {"eligible_pairs": int(len(table)), "variants": {}}
        for variant in variants:
            vresult: dict[str, Any] = {"folds": {}}
            for test_season in folds:
                train, test = _fold_split(table, test_season)
                if len(train) == 0 or len(test) == 0:
                    vresult["folds"][str(test_season)] = {"skipped": True, "train_rows": len(train), "test_rows": len(test)}
                    continue
                with stage(f"EX1-C1 target decomposition [{variant}] min{threshold} test={test_season}"):
                    _, y, pred, identity, mean_pred = _fit_direct(
                        train,
                        test,
                        pitcher_col=pitcher_col,
                        variant=variant,
                        direction="backward",
                        model_cfg=model_cfg,
                    )
                metrics = _metric_bundle(y, pred, identity, mean_pred)
                vresult["folds"][str(test_season)] = {
                    "skipped": False,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "metrics": metrics,
                }
                log(
                    f"[C1:{variant}:min{threshold}:{test_season}] "
                    f"model={metrics['model']['macro']['rmse']:.6f} "
                    f"identity={metrics['identity']['macro']['rmse']:.6f} "
                    f"mean={metrics['train_mean']['macro']['rmse']:.6f}"
                )
            tresult["variants"][variant] = vresult
        result["thresholds"][str(threshold)] = tresult
    return result


def _stage2_delta_backward(
    pairs: pd.DataFrame,
    *,
    pitcher_col: str,
    thresholds: list[int],
    folds: list[int],
    variants: list[str],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"description": "predict past-current delta, then reconstruct past state", "thresholds": {}}
    for threshold in thresholds:
        table = _eligible(pairs, threshold)
        tresult: dict[str, Any] = {"eligible_pairs": int(len(table)), "variants": {}}
        for variant in variants:
            vresult: dict[str, Any] = {"folds": {}}
            for test_season in folds:
                train, test = _fold_split(table, test_season)
                if len(train) == 0 or len(test) == 0:
                    vresult["folds"][str(test_season)] = {"skipped": True, "train_rows": len(train), "test_rows": len(test)}
                    continue
                with stage(f"EX1-C2 delta backward [{variant}] min{threshold} test={test_season}"):
                    _, y, pred, identity, delta_pred = _fit_delta_backward(
                        train,
                        test,
                        pitcher_col=pitcher_col,
                        variant=variant,
                        model_cfg=model_cfg,
                    )
                past_cols = [f"past_{x}" for x in STATE_NAMES]
                mean_target = train[past_cols].to_numpy(np.float64).mean(axis=0, keepdims=True)
                mean_pred = np.repeat(mean_target, len(test), axis=0)
                metrics = _metric_bundle(y, pred, identity, mean_pred)
                true_delta = y - identity
                delta_metrics = _target_metrics(true_delta, delta_pred)
                vresult["folds"][str(test_season)] = {
                    "skipped": False,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "reconstruction_metrics": metrics,
                    "delta_metrics": delta_metrics,
                }
                log(
                    f"[C2:{variant}:min{threshold}:{test_season}] "
                    f"recon={metrics['model']['macro']['rmse']:.6f} "
                    f"identity={metrics['identity']['macro']['rmse']:.6f} "
                    f"delta_rmse={delta_metrics['macro']['rmse']:.6f}"
                )
            tresult["variants"][variant] = vresult
        result["thresholds"][str(threshold)] = tresult
    return result


def _career_mask(values: pd.Series, group: dict[str, Any]) -> np.ndarray:
    lo = int(group.get("min", 0))
    hi = group.get("max")
    mask = values.to_numpy(np.int64) >= lo
    if hi is not None:
        mask &= values.to_numpy(np.int64) <= int(hi)
    return mask


def _stage3_career_groups(
    pairs: pd.DataFrame,
    *,
    pitcher_col: str,
    threshold: int,
    folds: list[int],
    variant: str,
    career_groups: list[dict[str, Any]],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    table = _eligible(pairs, threshold)
    result: dict[str, Any] = {
        "description": "slice one backward model by cumulative career entering current season",
        "min_season_pitches": threshold,
        "variant": variant,
        "folds": {},
    }
    for test_season in folds:
        train, test = _fold_split(table, test_season)
        if len(train) == 0 or len(test) == 0:
            result["folds"][str(test_season)] = {"skipped": True, "train_rows": len(train), "test_rows": len(test)}
            continue
        with stage(f"EX1-C3 career groups [{variant}] min{threshold} test={test_season}"):
            _, y, pred, identity, mean_pred = _fit_direct(
                train,
                test,
                pitcher_col=pitcher_col,
                variant=variant,
                direction="backward",
                model_cfg=model_cfg,
            )
        fold_result: dict[str, Any] = {
            "skipped": False,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "overall": _metric_bundle(y, pred, identity, mean_pred),
            "groups": {},
        }
        for group in career_groups:
            name = str(group["name"])
            mask = _career_mask(test["career_before_current"], group)
            if not mask.any():
                fold_result["groups"][name] = {"rows": 0, "skipped": True}
                continue
            gm = _metric_bundle(y[mask], pred[mask], identity[mask], mean_pred[mask])
            fold_result["groups"][name] = {
                "rows": int(mask.sum()),
                "skipped": False,
                "career_min": int(group.get("min", 0)),
                "career_max": None if group.get("max") is None else int(group["max"]),
                "metrics": gm,
            }
            log(
                f"[C3:{test_season}:{name}] rows={int(mask.sum())} "
                f"model={gm['model']['macro']['rmse']:.6f} "
                f"identity={gm['identity']['macro']['rmse']:.6f}"
            )
        result["folds"][str(test_season)] = fold_result
    return result


def _stage4_symmetry(
    pairs: pd.DataFrame,
    *,
    pitcher_col: str,
    threshold: int,
    folds: list[int],
    variants: list[str],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    table = _eligible(pairs, threshold)
    result: dict[str, Any] = {
        "description": "same samples/model capacity: backward current->past versus forward past->current",
        "min_season_pitches": threshold,
        "variants": {},
    }
    for variant in variants:
        vresult: dict[str, Any] = {"folds": {}}
        for test_season in folds:
            train, test = _fold_split(table, test_season)
            if len(train) == 0 or len(test) == 0:
                vresult["folds"][str(test_season)] = {"skipped": True, "train_rows": len(train), "test_rows": len(test)}
                continue
            with stage(f"EX1-C4 symmetry backward [{variant}] min{threshold} test={test_season}"):
                _, y_b, p_b, id_b, mean_b = _fit_direct(
                    train,
                    test,
                    pitcher_col=pitcher_col,
                    variant=variant,
                    direction="backward",
                    model_cfg=model_cfg,
                )
            with stage(f"EX1-C4 symmetry forward [{variant}] min{threshold} test={test_season}"):
                _, y_f, p_f, id_f, mean_f = _fit_direct(
                    train,
                    test,
                    pitcher_col=pitcher_col,
                    variant=variant,
                    direction="forward",
                    model_cfg=model_cfg,
                )
            mb = _metric_bundle(y_b, p_b, id_b, mean_b)
            mf = _metric_bundle(y_f, p_f, id_f, mean_f)
            backward_rmse = float(mb["model"]["macro"]["rmse"])
            forward_rmse = float(mf["model"]["macro"]["rmse"])
            vresult["folds"][str(test_season)] = {
                "skipped": False,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "backward": mb,
                "forward": mf,
                "backward_minus_forward_macro_rmse": backward_rmse - forward_rmse,
            }
            log(
                f"[C4:{variant}:{test_season}] backward={backward_rmse:.6f} "
                f"forward={forward_rmse:.6f} diff={backward_rmse-forward_rmse:+.6f}"
            )
        result["variants"][variant] = vresult
    return result


def run(config_path: str | Path) -> dict[str, Any]:
    exp = _load_experiment_config(config_path)
    base_cfg = load_config(_path(exp, exp["baseline_config"]))
    configure_cuda(base_cfg)
    configure_warnings(base_cfg)

    out_dir = _path(exp, exp["output_dir"])
    model_dir = _path(exp, exp["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with stage("EX1 suite minimal canonical frame + auxiliary outcomes"):
        frame, aux = _prepare_minimal_frame(base_cfg)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))

    with stage("EX1 suite pitcher-season states + adjacent pairs"):
        profiles = _season_profiles(
            frame,
            aux,
            season_col=season_col,
            pitcher_col=pitcher_col,
            target_col=target_col,
        )
        pairs = _adjacent_pairs(profiles, season_col=season_col, pitcher_col=pitcher_col)
        pairs = _add_career_state(
            pairs,
            profiles,
            season_col=season_col,
            pitcher_col=pitcher_col,
        )

    target_cols = [f"past_{x}" for x in STATE_NAMES]
    current_cols = [f"current_{x}" for x in STATE_NAMES]
    complete = pairs[target_cols + current_cols].notna().all(axis=1)
    pairs = pairs.loc[complete].reset_index(drop=True)

    thresholds = [int(x) for x in exp.get("min_pitch_thresholds", [50, 200])]
    folds = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    variants = [str(x) for x in exp.get("variants", ["state_only"])]
    primary_variant = str(exp.get("primary_variant", "state_only"))
    model_cfg = dict(exp["catboost"])

    results: dict[str, Any] = {
        "experiment": "EX1 four-stage pure backward research suite",
        "safe_forward_loaded": False,
        "definition": "historical pitcher-season state research only; no competition forward predictor",
        "state_names": list(STATE_NAMES),
        "rows": {
            "raw_pitch_rows": int(len(frame)),
            "pitcher_seasons": int(len(profiles)),
            "adjacent_pairs_complete": int(len(pairs)),
        },
    }

    results["stage1_target_decomposition"] = _stage1_target_decomposition(
        pairs,
        pitcher_col=pitcher_col,
        thresholds=thresholds,
        folds=folds,
        variants=variants,
        model_cfg=model_cfg,
    )

    results["stage2_delta_backward"] = _stage2_delta_backward(
        pairs,
        pitcher_col=pitcher_col,
        thresholds=thresholds,
        folds=folds,
        variants=variants,
        model_cfg=model_cfg,
    )

    results["stage3_career_groups"] = _stage3_career_groups(
        pairs,
        pitcher_col=pitcher_col,
        threshold=int(exp.get("career_min_season_pitches", 50)),
        folds=folds,
        variant=primary_variant,
        career_groups=list(exp["career_groups"]),
        model_cfg=model_cfg,
    )

    results["stage4_forward_backward_symmetry"] = _stage4_symmetry(
        pairs,
        pitcher_col=pitcher_col,
        threshold=int(exp.get("symmetry_min_season_pitches", 200)),
        folds=folds,
        variants=variants,
        model_cfg=model_cfg,
    )

    (out_dir / "metrics_backward_suite.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pairs.to_csv(out_dir / "pitcher_season_pairs.csv.gz", index=False, compression="gzip")
    log(f"[EX1 suite] complete -> {out_dir / 'metrics_backward_suite.json'}")
    return results
