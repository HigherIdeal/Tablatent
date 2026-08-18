from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.ex1.backward_suite import _prepare_minimal_frame
from bitaboost.ex1.pitcher_season_backward import (
    STATE_NAMES,
    _adjacent_pairs,
    _corr,
    _season_profiles,
    _spearman,
    _target_metrics,
)
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


def _prepare_pairs(
    frame: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    pairs["past_log_pitch_count"] = np.log1p(
        pd.to_numeric(pairs["past_pitch_count"], errors="coerce").astype(float)
    )
    target_cols = [f"past_{x}" for x in STATE_NAMES] + [f"current_{x}" for x in STATE_NAMES]
    complete = pairs[target_cols].notna().all(axis=1)
    return profiles, pairs.loc[complete].reset_index(drop=True)


def _prepare_direction_frame(
    table: pd.DataFrame,
    *,
    pitcher_col: str,
    variant: str,
    direction: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if direction == "forward":
        prefix = "past"
        count_feature = "past_log_pitch_count"
    elif direction == "backward":
        prefix = "current"
        count_feature = "current_log_pitch_count"
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


def _direction_arrays(
    table: pd.DataFrame,
    *,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "forward":
        target_prefix = "current"
        identity_prefix = "past"
    elif direction == "backward":
        target_prefix = "past"
        identity_prefix = "current"
    else:
        raise ValueError(direction)
    y = table[[f"{target_prefix}_{x}" for x in STATE_NAMES]].to_numpy(np.float64)
    identity = table[[f"{identity_prefix}_{x}" for x in STATE_NAMES]].to_numpy(np.float64)
    return y, identity


def _fit_direction(
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
        train,
        pitcher_col=pitcher_col,
        variant=variant,
        direction=direction,
    )
    x_test, _, _ = _prepare_direction_frame(
        test,
        pitcher_col=pitcher_col,
        variant=variant,
        direction=direction,
    )
    y_train, _ = _direction_arrays(train, direction=direction)
    y_test, identity = _direction_arrays(test, direction=direction)

    model = CatBoostRegressor(**_model_params(model_cfg))
    model.fit(Pool(x_train, y_train.astype(np.float32), cat_features=cats, feature_names=features))
    pred = np.clip(
        np.asarray(
            model.predict(Pool(x_test, cat_features=cats, feature_names=features)),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(test), axis=0)
    metrics = {
        "model": _target_metrics(y_test, pred),
        "identity": _target_metrics(y_test, identity),
        "train_mean": _target_metrics(y_test, mean_pred),
    }
    return model, pred, y_test, metrics


def _predict_from_state_array(
    model,
    base: pd.DataFrame,
    state: np.ndarray,
    *,
    pitcher_col: str,
    variant: str,
    direction: str,
) -> np.ndarray:
    from catboost import Pool

    table = base.copy()
    prefix = "past" if direction == "forward" else "current"
    for i, name in enumerate(STATE_NAMES):
        table[f"{prefix}_{name}"] = np.asarray(state[:, i], dtype=np.float32)
    x, features, cats = _prepare_direction_frame(
        table,
        pitcher_col=pitcher_col,
        variant=variant,
        direction=direction,
    )
    return np.clip(
        np.asarray(model.predict(Pool(x, cat_features=cats, feature_names=features)), dtype=np.float64),
        0.0,
        1.0,
    )


def _raw_pair_metrics(test: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in STATE_NAMES:
        past = pd.to_numeric(test[f"past_{name}"], errors="coerce").to_numpy(np.float64)
        current = pd.to_numeric(test[f"current_{name}"], errors="coerce").to_numpy(np.float64)
        diff = current - past
        out[name] = {
            "pearson": _corr(past, current),
            "spearman": _spearman(past, current),
            "rmse_identity": float(np.sqrt(np.mean(diff * diff))),
            "mae_change": float(np.mean(np.abs(diff))),
            "mean_change_current_minus_past": float(np.mean(diff)),
            "std_change": float(np.std(diff)),
        }
    return out


def _cycle_metrics(
    forward_model,
    backward_model,
    test: pd.DataFrame,
    forward_pred: np.ndarray,
    backward_pred: np.ndarray,
    *,
    pitcher_col: str,
    variant: str,
) -> dict[str, Any]:
    # Use the observed season pitch counts only as reliability/context inputs. The state
    # values themselves are replaced by the opposite-direction predictions.
    past_true = test[[f"past_{x}" for x in STATE_NAMES]].to_numpy(np.float64)
    current_true = test[[f"current_{x}" for x in STATE_NAMES]].to_numpy(np.float64)

    past_cycle = _predict_from_state_array(
        backward_model,
        test,
        forward_pred,
        pitcher_col=pitcher_col,
        variant=variant,
        direction="backward",
    )
    current_cycle = _predict_from_state_array(
        forward_model,
        test,
        backward_pred,
        pitcher_col=pitcher_col,
        variant=variant,
        direction="forward",
    )
    return {
        "past_to_future_to_past": _target_metrics(past_true, past_cycle),
        "future_to_past_to_future": _target_metrics(current_true, current_cycle),
    }


def _rmse_skill(model_rmse: float, baseline_rmse: float) -> float:
    denom = max(float(baseline_rmse) ** 2, 1e-12)
    return float(1.0 - (float(model_rmse) ** 2) / denom)


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(clean)) if clean else None


def _target_aggregate(
    folds: dict[str, Any],
    target: str,
    *,
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    valid = [f for f in folds.values() if not f.get("skipped")]
    if not valid:
        return {"classification": "no_data"}

    direction_summary: dict[str, Any] = {}
    for direction in ("forward", "backward"):
        pearson: list[float | None] = []
        spearman: list[float | None] = []
        skill_mean: list[float] = []
        skill_identity: list[float] = []
        beats_mean: list[bool] = []
        beats_identity: list[bool] = []
        rmses: list[float] = []
        for fold in valid:
            m = fold["directions"][direction]["metrics"]
            mm = m["model"][target]
            tm = m["train_mean"][target]
            im = m["identity"][target]
            rmses.append(float(mm["rmse"]))
            pearson.append(mm.get("pearson"))
            spearman.append(mm.get("spearman"))
            skill_mean.append(_rmse_skill(mm["rmse"], tm["rmse"]))
            skill_identity.append(_rmse_skill(mm["rmse"], im["rmse"]))
            beats_mean.append(float(mm["rmse"]) <= float(tm["rmse"]))
            beats_identity.append(float(mm["rmse"]) <= float(im["rmse"]))

        direction_summary[direction] = {
            "mean_rmse": float(np.mean(rmses)),
            "mean_pearson": _mean_or_none(pearson),
            "mean_spearman": _mean_or_none(spearman),
            "mean_skill_vs_train_mean": float(np.mean(skill_mean)),
            "mean_skill_vs_identity": float(np.mean(skill_identity)),
            "better_than_mean_fraction": float(np.mean(beats_mean)),
            "better_than_identity_fraction": float(np.mean(beats_identity)),
            "rmse_std_across_folds": float(np.std(rmses)),
        }

    raw_corr = _mean_or_none([fold["raw_pair_metrics"][target]["pearson"] for fold in valid])
    raw_spearman = _mean_or_none([fold["raw_pair_metrics"][target]["spearman"] for fold in valid])
    raw_change = float(np.mean([fold["raw_pair_metrics"][target]["mae_change"] for fold in valid]))

    cycle_past = float(
        np.mean(
            [
                fold["cycle_metrics"]["past_to_future_to_past"][target]["rmse"]
                for fold in valid
            ]
        )
    )
    cycle_current = float(
        np.mean(
            [
                fold["cycle_metrics"]["future_to_past_to_future"][target]["rmse"]
                for fold in valid
            ]
        )
    )

    fwd = direction_summary["forward"]
    bwd = direction_summary["backward"]
    min_corr = float(selection_cfg.get("min_mean_pearson", 0.20))
    min_spear = float(selection_cfg.get("min_mean_spearman", 0.20))
    min_fraction = float(selection_cfg.get("min_better_than_mean_fraction", 2.0 / 3.0))

    fwd_ok = (
        fwd["mean_pearson"] is not None
        and fwd["mean_spearman"] is not None
        and fwd["mean_pearson"] >= min_corr
        and fwd["mean_spearman"] >= min_spear
        and fwd["better_than_mean_fraction"] >= min_fraction
    )
    bwd_ok = (
        bwd["mean_pearson"] is not None
        and bwd["mean_spearman"] is not None
        and bwd["mean_pearson"] >= min_corr
        and bwd["mean_spearman"] >= min_spear
        and bwd["better_than_mean_fraction"] >= min_fraction
    )

    if fwd_ok and bwd_ok:
        classification = "bidirectional_stable"
    elif fwd_ok:
        classification = "forward_only"
    elif bwd_ok:
        classification = "backward_only"
    else:
        classification = "weak_or_regime_sensitive"

    corr_floor = min(
        fwd["mean_pearson"] if fwd["mean_pearson"] is not None else -1.0,
        bwd["mean_pearson"] if bwd["mean_pearson"] is not None else -1.0,
    )
    skill_floor = min(fwd["mean_skill_vs_train_mean"], bwd["mean_skill_vs_train_mean"])
    fold_floor = min(fwd["better_than_mean_fraction"], bwd["better_than_mean_fraction"])
    # Transparent ranking score: only the weakest direction matters; negative components
    # are clipped because they are evidence against a stable trait.
    stability_score = float(
        max(corr_floor, 0.0)
        * max(skill_floor, 0.0)
        * max(fold_floor, 0.0)
    )

    return {
        "classification": classification,
        "stability_score": stability_score,
        "raw_pair": {
            "mean_pearson": raw_corr,
            "mean_spearman": raw_spearman,
            "mean_absolute_season_change": raw_change,
        },
        "forward": fwd,
        "backward": bwd,
        "cycle": {
            "past_to_future_to_past_mean_rmse": cycle_past,
            "future_to_past_to_future_mean_rmse": cycle_current,
        },
    }


def _aggregate_traits(
    folds: dict[str, Any],
    *,
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    targets = {
        name: _target_aggregate(folds, name, selection_cfg=selection_cfg)
        for name in STATE_NAMES
    }
    ranking = sorted(
        STATE_NAMES,
        key=lambda name: float(targets[name].get("stability_score", 0.0)),
        reverse=True,
    )
    stable = [
        name
        for name in ranking
        if targets[name].get("classification") == "bidirectional_stable"
    ]
    return {
        "targets": targets,
        "ranking": ranking,
        "selected_bidirectional_stable_subset": stable,
    }


def _save_fold_npz(
    path: Path,
    test: pd.DataFrame,
    *,
    pitcher_col: str,
    forward_pred: np.ndarray,
    backward_pred: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        pitcher_id=test[pitcher_col].astype(str).to_numpy(dtype="U"),
        current_season=test["current_season"].to_numpy(np.int16),
        past_state=test[[f"past_{x}" for x in STATE_NAMES]].to_numpy(np.float32),
        current_state=test[[f"current_{x}" for x in STATE_NAMES]].to_numpy(np.float32),
        forward_pred=forward_pred.astype(np.float32),
        backward_pred=backward_pred.astype(np.float32),
    )


def run(config_path: str | Path) -> dict[str, Any]:
    exp = _load_experiment_config(config_path)
    base_cfg = load_config(_path(exp, exp["baseline_config"]))
    configure_cuda(base_cfg)
    configure_warnings(base_cfg)

    out_dir = _path(exp, exp["output_dir"])
    model_dir = _path(exp, exp["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with stage("EX4 minimal canonical frame + auxiliary outcomes"):
        frame, aux = _prepare_minimal_frame(base_cfg)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))

    with stage("EX4 pitcher-season states + adjacent pairs"):
        profiles, pairs = _prepare_pairs(
            frame,
            aux,
            season_col=season_col,
            pitcher_col=pitcher_col,
            target_col=target_col,
        )

    thresholds = [int(x) for x in exp.get("min_pitch_thresholds", [50, 200, 500])]
    variants = [str(x) for x in exp.get("variants", ["state_only"])]
    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    model_cfg = dict(exp["catboost"])
    selection_cfg = dict(exp.get("selection", {}))
    primary_threshold = int(exp.get("primary_threshold", 200))
    primary_variant = str(exp.get("primary_variant", "state_only"))

    results: dict[str, Any] = {
        "experiment": "EX4 bidirectional stable-trait discovery",
        "definition": (
            "discover pitcher-season state components that remain predictable when time is "
            "modeled in both directions; backward is a stability probe, not a success predictor"
        ),
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

                raw_metrics = _raw_pair_metrics(test)

                with stage(f"EX4 forward [{variant}] min{threshold} test={test_season}"):
                    forward_model, forward_pred, _, forward_metrics = _fit_direction(
                        train,
                        test,
                        pitcher_col=pitcher_col,
                        variant=variant,
                        direction="forward",
                        model_cfg=model_cfg,
                    )

                with stage(f"EX4 backward [{variant}] min{threshold} test={test_season}"):
                    backward_model, backward_pred, _, backward_metrics = _fit_direction(
                        train,
                        test,
                        pitcher_col=pitcher_col,
                        variant=variant,
                        direction="backward",
                        model_cfg=model_cfg,
                    )

                with stage(f"EX4 cycle consistency [{variant}] min{threshold} test={test_season}"):
                    cycle_metrics = _cycle_metrics(
                        forward_model,
                        backward_model,
                        test,
                        forward_pred,
                        backward_pred,
                        pitcher_col=pitcher_col,
                        variant=variant,
                    )

                fold_result = {
                    "skipped": False,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_seasons": sorted(int(x) for x in train["current_season"].unique()),
                    "raw_pair_metrics": raw_metrics,
                    "directions": {
                        "forward": {"metrics": forward_metrics},
                        "backward": {"metrics": backward_metrics},
                    },
                    "cycle_metrics": cycle_metrics,
                }
                variant_result["folds"][str(test_season)] = fold_result

                _save_fold_npz(
                    out_dir / f"bidirectional_{variant}_min{threshold}_{test_season}.npz",
                    test,
                    pitcher_col=pitcher_col,
                    forward_pred=forward_pred,
                    backward_pred=backward_pred,
                )

                if test_season == max(fold_seasons):
                    fwd_path = model_dir / f"forward_{variant}_min{threshold}.cbm"
                    bwd_path = model_dir / f"backward_{variant}_min{threshold}.cbm"
                    forward_model.save_model(str(fwd_path))
                    backward_model.save_model(str(bwd_path))
                    fold_result["model_paths"] = {
                        "forward": str(fwd_path.relative_to(Path(exp["_repo_root"]))),
                        "backward": str(bwd_path.relative_to(Path(exp["_repo_root"]))),
                    }

                log(
                    f"[EX4:{variant}:min{threshold}:{test_season}] "
                    f"forward={forward_metrics['model']['macro']['rmse']:.6f} "
                    f"backward={backward_metrics['model']['macro']['rmse']:.6f} "
                    f"cycle_past={cycle_metrics['past_to_future_to_past']['macro']['rmse']:.6f} "
                    f"cycle_current={cycle_metrics['future_to_past_to_future']['macro']['rmse']:.6f}"
                )
                for name in STATE_NAMES:
                    log(
                        f"  {name:<8} raw_r={raw_metrics[name]['pearson'] if raw_metrics[name]['pearson'] is not None else float('nan'):+.3f} "
                        f"fwd_r={forward_metrics['model'][name]['pearson'] if forward_metrics['model'][name]['pearson'] is not None else float('nan'):+.3f} "
                        f"bwd_r={backward_metrics['model'][name]['pearson'] if backward_metrics['model'][name]['pearson'] is not None else float('nan'):+.3f}"
                    )

            variant_result["stable_trait_summary"] = _aggregate_traits(
                variant_result["folds"],
                selection_cfg=selection_cfg,
            )
            threshold_result["variants"][variant] = variant_result

        results["thresholds"][str(threshold)] = threshold_result

    primary = results["thresholds"].get(str(primary_threshold), {}).get("variants", {}).get(primary_variant)
    if primary is not None:
        results["primary"] = {
            "threshold": primary_threshold,
            "variant": primary_variant,
            "stable_trait_summary": primary["stable_trait_summary"],
        }

    metrics_path = out_dir / "metrics_bidirectional_stable_traits.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pairs.to_csv(out_dir / "pitcher_season_pairs.csv.gz", index=False, compression="gzip")

    log(f"[EX4] complete -> {metrics_path}")
    if primary is not None:
        summary = primary["stable_trait_summary"]
        log("[EX4 primary stable-trait ranking]")
        for name in summary["ranking"]:
            item = summary["targets"][name]
            log(
                f"  {name:<8} class={item['classification']:<24} "
                f"score={item['stability_score']:.6f} "
                f"raw_r={item['raw_pair']['mean_pearson'] if item['raw_pair']['mean_pearson'] is not None else float('nan'):+.3f} "
                f"fwd_r={item['forward']['mean_pearson'] if item['forward']['mean_pearson'] is not None else float('nan'):+.3f} "
                f"bwd_r={item['backward']['mean_pearson'] if item['backward']['mean_pearson'] is not None else float('nan'):+.3f}"
            )
        log(f"  selected={summary['selected_bidirectional_stable_subset']}")

    return results
