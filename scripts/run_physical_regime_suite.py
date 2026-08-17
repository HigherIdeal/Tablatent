#!/usr/bin/env python3
"""Compare core and all-feature physical mixtures with the reference system.

Experts per temporal fold:
  old_full, old_recent, physical_full, physical_recent,
  physical_all_v1_full, physical_all_v1_recent, r_fast

The old and physical full/recent mixtures share alpha_recent. Their predictions
are blended by lambda_physical, after which the unchanged R-fast specialist is
applied only to game_type=R with beta_r. The augmented dataset already enforces
Trackman season < feature season, so no current/future physical rows are used.

This is a heavy CatBoost suite. The script writes prediction caches and compact
metrics but must be executed explicitly by the user.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_physical_augmented_datasets import feature_columns  # noqa: E402

DEFAULT_DATA = (
    ROOT
    / "data"
    / "processed"
    / "physical_features"
    / "train_with_pitcher_physical.parquet"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "physical_regime_suite"
TARGET = "control_success"

PITCHER_TEAM_WIN_EXPECTANCY = "pitcher_team_win_expectancy"

CANONICAL_FEATURES = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_total_before",
    "score_diff_home",
    "base_state",
    PITCHER_TEAM_WIN_EXPECTANCY,
    "li",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

SUCCESS_STATE = [
    "eng_ps_prev1_minus_long",
    "eng_ps_prev3_minus_long",
    "eng_ps_prev5_minus_long",
    "eng_ps_prev1_minus_prev3",
    "eng_ps_prev3_minus_prev5",
    "eng_ps_prev1_minus_prev5",
    "eng_ps_recent_mean_135",
    "eng_ps_recent_mean_minus_long",
    "eng_ps_recent_range_135",
]

CATEGORICAL = {
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "tm_profile_source",
}

PHYSICAL_CORE = [
    "tm_pitch_count",
    "tm_seasons_observed",
    "tm_rel_speed_mean",
    "tm_rel_speed_std",
    "tm_spin_rate_mean",
    "tm_spin_rate_std",
    "tm_induced_vert_break_mean",
    "tm_induced_vert_break_std",
    "tm_horz_break_mean",
    "tm_horz_break_std",
    "tm_extension_mean",
    "tm_extension_std",
    "tm_rel_height_mean",
    "tm_rel_height_std",
    "tm_rel_side_mean",
    "tm_rel_side_std",
    "tm_zone_speed_mean",
    "tm_zone_speed_std",
    "tm_player_profile_available",
    "tm_profile_source",
]

PHYSICAL_PITCH_MIX = [
    "tm_fastball_count",
    "tm_fastball_rate",
    "tm_breaking_count",
    "tm_breaking_rate",
    "tm_offspeed_count",
    "tm_offspeed_rate",
    "tm_other_count",
    "tm_other_rate",
]
for _group in ("fastball", "breaking", "offspeed"):
    for _measure in (
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
    ):
        PHYSICAL_PITCH_MIX.append(f"tm_{_group}_{_measure}_mean")

BASE_FEATURES = CANONICAL_FEATURES + SUCCESS_STATE
PHYSICAL_FEATURES = BASE_FEATURES + PHYSICAL_CORE + PHYSICAL_PITCH_MIX
PHYSICAL_ALL_COLUMNS = feature_columns() + [
    "tm_player_profile_available",
    "tm_profile_source",
]
PHYSICAL_ALL_FEATURES = BASE_FEATURES + PHYSICAL_ALL_COLUMNS

R_FAST_FEATURES = [
    "game_month",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


@dataclass(frozen=True)
class FoldSpec:
    name: str
    weight: float
    kind: str
    cutoff_month: int | None = None
    valid_start: int | None = None
    valid_end: int | None = None


FOLDS = (
    FoldSpec("season_forward_2024", 0.50, "season_forward"),
    FoldSpec("mid_2024", 0.20, "within", 5, 6, 7),
    FoldSpec("late_2024", 0.30, "within", 7, 8, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--alpha-values", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--lambda-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--beta-r", type=float, default=0.10)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def parse_float_values(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError("blend grids must contain values in [0, 1]")
    return values


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float32)


def add_derived_features(frame: pd.DataFrame) -> None:
    top_bottom = frame["top_bottom"].astype(str)
    unknown = sorted(set(top_bottom.unique()) - {"T", "B"})
    if unknown:
        raise ValueError(f"Unexpected top_bottom values: {unknown}")
    home = pd.to_numeric(frame["home_win_expectancy"], errors="coerce")
    away = pd.to_numeric(frame["away_win_expectancy"], errors="coerce")
    frame[PITCHER_TEAM_WIN_EXPECTANCY] = np.where(top_bottom.eq("T"), home, away)

    long_rate = numeric(frame, "asof_pitcher_success_rate")
    prev1 = numeric(frame, "asof_pitcher_prev1_game_success_rate")
    prev3 = numeric(frame, "asof_pitcher_prev3_game_success_rate")
    prev5 = numeric(frame, "asof_pitcher_prev5_game_success_rate")
    frame["eng_ps_prev1_minus_long"] = prev1 - long_rate
    frame["eng_ps_prev3_minus_long"] = prev3 - long_rate
    frame["eng_ps_prev5_minus_long"] = prev5 - long_rate
    frame["eng_ps_prev1_minus_prev3"] = prev1 - prev3
    frame["eng_ps_prev3_minus_prev5"] = prev3 - prev5
    frame["eng_ps_prev1_minus_prev5"] = prev1 - prev5
    recent = pd.concat([prev1, prev3, prev5], axis=1)
    frame["eng_ps_recent_mean_135"] = recent.mean(axis=1, skipna=False)
    frame["eng_ps_recent_mean_minus_long"] = (
        frame["eng_ps_recent_mean_135"] - long_rate
    )
    frame["eng_ps_recent_range_135"] = (
        recent.max(axis=1, skipna=False) - recent.min(axis=1, skipna=False)
    )


def required_columns() -> list[str]:
    derived = {PITCHER_TEAM_WIN_EXPECTANCY, *SUCCESS_STATE}
    required = (set(BASE_FEATURES) - derived) | set(PHYSICAL_ALL_COLUMNS)
    required |= {
        TARGET,
        "row_id",
        "home_win_expectancy",
        "away_win_expectancy",
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    }
    return sorted(required)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=required_columns())
    missing = sorted(set(required_columns()) - set(frame.columns))
    if missing:
        raise ValueError(f"Augmented dataset missing columns: {missing}")
    add_derived_features(frame)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_month"] = pd.to_numeric(
        frame["game_month"], errors="raise"
    ).astype(int)
    frame = frame.sort_values(
        ["season", "game_month", "row_id"], kind="stable"
    ).reset_index(drop=True)
    return frame


def fold_masks(frame: pd.DataFrame, spec: FoldSpec) -> tuple[pd.Series, pd.Series, pd.Series]:
    season = frame["season"]
    month = frame["game_month"]
    if spec.kind == "season_forward":
        full = season.le(2023)
        recent = season.eq(2023)
        valid = season.eq(2024)
    else:
        if spec.cutoff_month is None or spec.valid_start is None:
            raise ValueError(f"Invalid fold spec: {spec}")
        observed = season.eq(2024) & month.le(spec.cutoff_month)
        full = season.lt(2024) | observed
        recent = season.eq(2023) | observed
        valid = season.eq(2024) & month.ge(spec.valid_start)
        if spec.valid_end is not None:
            valid &= month.le(spec.valid_end)
    if bool((full & valid).any()) or bool((recent & valid).any()):
        raise RuntimeError(f"Temporal leakage in {spec.name}")
    return full, recent, valid


def prepare_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = frame.loc[:, features].copy()
    categorical = [feature for feature in features if feature in CATEGORICAL]
    categorical_set = set(categorical)
    for column in features:
        if column in categorical_set:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x, categorical


def catboost_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "iterations": args.iterations,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
        "random_seed": args.seed,
        "loss_function": "Logloss",
        "has_time": True,
        "one_hot_max_size": 10,
        "allow_writing_files": False,
        "task_type": args.task_type,
        "thread_count": args.thread_count,
        "verbose": args.verbose,
    }
    if args.task_type == "GPU":
        params["devices"] = args.devices
    return params


def cache_signature(args: argparse.Namespace) -> str:
    payload = {
        "data": str(args.data.resolve()),
        "iterations": args.iterations,
        "seed": args.seed,
        "task_type": args.task_type,
        "features": {
            "base": BASE_FEATURES,
            "physical": PHYSICAL_FEATURES,
            "r_fast": R_FAST_FEATURES,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = prepare_x(train, features)
    x_valid, valid_categorical = prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("Categorical feature mismatch")
    y_train = pd.to_numeric(train[TARGET], errors="raise").to_numpy(np.float32)
    train_pool = Pool(
        x_train,
        label=y_train,
        cat_features=categorical,
        feature_names=features,
    )
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**catboost_params(args))
    model.fit(train_pool, verbose=args.verbose)
    prediction = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)
    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return prediction


def load_cache(path: Path, expected_rows: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        value = np.load(path)["prediction"]
        if len(value) != expected_rows:
            return None
        return np.asarray(value, dtype=np.float64)
    except Exception:
        return None


def brier(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        y = y[mask]
        prediction = prediction[mask]
    return float(np.mean((y - prediction) ** 2)) if len(y) else math.nan


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    value = brier(y, prediction)
    reference = float(y.mean() * (1.0 - y.mean()))
    return float(100000.0 * (1.0 - value / reference)) if reference > 0 else math.nan


def weighted_summary(rows: pd.DataFrame) -> pd.DataFrame:
    weights = {fold.name: fold.weight for fold in FOLDS}
    output = []
    for key, group in rows.groupby(
        ["physical_family", "alpha_recent", "lambda_physical", "beta_r"],
        sort=False,
    ):
        fold_weights = np.asarray([weights[name] for name in group["fold"]], dtype=float)
        fold_weights /= fold_weights.sum()
        values = group["brier"].to_numpy(float)
        output.append(
            {
                "physical_family": key[0],
                "alpha_recent": key[1],
                "lambda_physical": key[2],
                "beta_r": key[3],
                "weighted_brier": float(np.dot(fold_weights, values)),
                "weighted_raw_score": float(
                    np.dot(fold_weights, group["raw_score"].to_numpy(float))
                ),
                "weighted_r_brier": float(
                    np.dot(fold_weights, group["r_brier"].to_numpy(float))
                ),
                "weighted_f_brier": float(
                    np.dot(fold_weights, group["f_brier"].to_numpy(float))
                ),
                "worst_brier": float(values.max()),
                "improved_folds_vs_reference": int(
                    group["delta_vs_reference"].lt(0.0).sum()
                ),
            }
        )
    return pd.DataFrame(output).sort_values(
        ["weighted_brier", "worst_brier"]
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.iterations <= 0 or args.thread_count <= 0:
        raise ValueError("iterations and thread count must be positive")
    if not 0.0 <= args.beta_r <= 1.0:
        raise ValueError("--beta-r must be in [0, 1]")
    alpha_values = parse_float_values(args.alpha_values)
    lambda_values = parse_float_values(args.lambda_values)

    try:
        import catboost
    except ImportError as error:
        raise RuntimeError("catboost is required in the active environment") from error

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    signature = cache_signature(args)

    print("stage 1/4: loading compact columns from augmented Parquet")
    frame = load_frame(args.data)
    print(f"rows: {len(frame):,}")
    print(f"base_features: {len(BASE_FEATURES):,}")
    print(f"physical_features: {len(PHYSICAL_FEATURES):,}")
    print(f"physical_all_features: {len(PHYSICAL_ALL_FEATURES):,}")
    print(f"catboost: {catboost.__version__}")

    expert_specs = {
        "old_full": BASE_FEATURES,
        "old_recent": BASE_FEATURES,
        "physical_full": PHYSICAL_FEATURES,
        "physical_recent": PHYSICAL_FEATURES,
        # Versioned names preserve the existing five-expert cache signature.
        "physical_all_v1_full": PHYSICAL_ALL_FEATURES,
        "physical_all_v1_recent": PHYSICAL_ALL_FEATURES,
        "r_fast": R_FAST_FEATURES,
    }
    expert_rows: list[dict[str, Any]] = []
    fold_predictions: dict[str, dict[str, np.ndarray]] = {}
    fold_targets: dict[str, np.ndarray] = {}
    fold_types: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []

    print("stage 2/4: fitting or loading five experts per temporal fold")
    with tqdm(
        total=len(FOLDS) * len(expert_specs),
        desc="CatBoost experts",
        unit="model",
        dynamic_ncols=True,
    ) as expert_progress:
        for fold in FOLDS:
            full_mask, recent_mask, valid_mask = fold_masks(frame, fold)
            r_mask = recent_mask & frame["game_type"].astype(str).eq("R")
            train_masks = {
                "old_full": full_mask,
                "old_recent": recent_mask,
                "physical_full": full_mask,
                "physical_recent": recent_mask,
                "physical_all_v1_full": full_mask,
                "physical_all_v1_recent": recent_mask,
                "r_fast": r_mask,
            }
            valid = frame.loc[valid_mask].copy()
            y = pd.to_numeric(valid[TARGET], errors="raise").to_numpy(np.float64)
            game_type = valid["game_type"].astype("string").fillna("<MISSING>")
            is_r = game_type.eq("R").to_numpy()
            is_f = game_type.eq("F").to_numpy()
            fold_predictions[fold.name] = {}
            fold_targets[fold.name] = y
            fold_types[fold.name] = (is_r, is_f)
            tqdm.write(
                f"[{fold.name}] valid={len(valid):,} full={int(full_mask.sum()):,} "
                f"recent={int(recent_mask.sum()):,} recent_R={int(r_mask.sum()):,}"
            )

            for expert, features in expert_specs.items():
                cache = cache_dir / f"{signature}__{fold.name}__{expert}.npz"
                prediction = None if args.no_resume else load_cache(cache, len(valid))
                expert_progress.set_description_str(f"{fold.name} / {expert}")
                if prediction is None:
                    train = frame.loc[train_masks[expert]].copy()
                    expert_progress.set_postfix_str(
                        f"fit rows={len(train):,} features={len(features)} "
                        f"trees={args.iterations}"
                    )
                    tqdm.write(
                        f"  fit {expert:<16s} train={len(train):,} "
                        f"features={len(features)} trees={args.iterations}"
                    )
                    prediction = fit_predict(train, valid, features, args)
                    np.savez_compressed(cache, prediction=prediction.astype(np.float32))
                    del train
                    gc.collect()
                else:
                    expert_progress.set_postfix_str("cache hit")
                    tqdm.write(f"  cache {expert:<14s} {cache.name}")
                fold_predictions[fold.name][expert] = prediction
                expert_rows.append(
                    {
                        "fold": fold.name,
                        "expert": expert,
                        "brier": brier(y, prediction),
                        "raw_score": raw_score(y, prediction),
                        "r_brier": brier(y, prediction, is_r),
                        "f_brier": brier(y, prediction, is_f),
                    }
                )
                expert_progress.update(1)

            source_counts = valid["tm_profile_source"].astype(str).value_counts()
            diagnostics.append(
                {
                    "fold": fold.name,
                    "weight": fold.weight,
                    "valid_rows": len(valid),
                    "target_rate": float(y.mean()),
                    "r_rows": int(is_r.sum()),
                    "f_rows": int(is_f.sum()),
                    "profile_sources": {
                        str(key): int(value) for key, value in source_counts.items()
                    },
                }
            )
            del valid
            gc.collect()

    print("stage 3/4: evaluating physical and regime blend grid")
    blend_rows: list[dict[str, Any]] = []
    reference_by_fold: dict[str, float] = {}
    physical_variants = {
        "core": ("physical_full", "physical_recent"),
        "all_v1": ("physical_all_v1_full", "physical_all_v1_recent"),
    }
    with tqdm(
        total=(
            len(FOLDS)
            * len(physical_variants)
            * len(alpha_values)
            * len(lambda_values)
        ),
        desc="Blend grid",
        unit="config",
        dynamic_ncols=True,
    ) as blend_progress:
        for fold in FOLDS:
            y = fold_targets[fold.name]
            is_r, is_f = fold_types[fold.name]
            predictions = fold_predictions[fold.name]
            for alpha in alpha_values:
                old_base = (
                    (1.0 - alpha) * predictions["old_full"]
                    + alpha * predictions["old_recent"]
                )
                for family, (full_name, recent_name) in physical_variants.items():
                    physical_base = (
                        (1.0 - alpha) * predictions[full_name]
                        + alpha * predictions[recent_name]
                    )
                    for physical_weight in lambda_values:
                        blend_progress.set_postfix_str(
                            f"fold={fold.name} family={family} "
                            f"alpha={alpha:.2f} lambda={physical_weight:.2f}"
                        )
                        base = (
                            (1.0 - physical_weight) * old_base
                            + physical_weight * physical_base
                        )
                        final = base.copy()
                        final[is_r] = (
                            (1.0 - args.beta_r) * base[is_r]
                            + args.beta_r * predictions["r_fast"][is_r]
                        )
                        row = {
                            "fold": fold.name,
                            "physical_family": family,
                            "alpha_recent": alpha,
                            "lambda_physical": physical_weight,
                            "beta_r": args.beta_r,
                            "brier": brier(y, final),
                            "raw_score": raw_score(y, final),
                            "r_brier": brier(y, final, is_r),
                            "f_brier": brier(y, final, is_f),
                        }
                        blend_rows.append(row)
                        if (
                            family == "core"
                            and alpha == 0.2
                            and physical_weight == 0.0
                        ):
                            reference_by_fold[fold.name] = row["brier"]
                        blend_progress.update(1)

    if len(reference_by_fold) != len(FOLDS):
        raise ValueError("alpha grid must include 0.2 and lambda grid must include 0")
    for row in blend_rows:
        row["delta_vs_reference"] = (
            row["brier"] - reference_by_fold[row["fold"]]
        )

    print("stage 4/4: writing weighted summaries")
    expert_frame = pd.DataFrame(expert_rows)
    blend_frame = pd.DataFrame(blend_rows)
    summary = weighted_summary(blend_frame)
    expert_frame.to_csv(args.output_dir / "expert_results.csv", index=False)
    blend_frame.to_csv(args.output_dir / "blend_results.csv", index=False)
    summary.to_csv(args.output_dir / "blend_summary.csv", index=False)
    (
        summary.sort_values("weighted_brier")
        .groupby(
            ["physical_family", "lambda_physical"],
            as_index=False,
            sort=False,
        )
        .first()
        .sort_values("weighted_brier")
        .to_csv(args.output_dir / "best_by_physical_weight.csv", index=False)
    )

    run_config = {
        "data": str(args.data),
        "rows": len(frame),
        "iterations": args.iterations,
        "seed": args.seed,
        "task_type": args.task_type,
        "devices": args.devices if args.task_type == "GPU" else None,
        "alpha_values": alpha_values,
        "lambda_physical_values": lambda_values,
        "beta_r": args.beta_r,
        "reference_weighted_brier_from_prior_run": 0.24789305,
        "features": {
            "base": BASE_FEATURES,
            "physical_core": PHYSICAL_CORE,
            "physical_pitch_mix": PHYSICAL_PITCH_MIX,
            "physical_all": PHYSICAL_ALL_COLUMNS,
            "r_fast": R_FAST_FEATURES,
        },
        "folds": [fold.__dict__ for fold in FOLDS],
        "fold_diagnostics": diagnostics,
        "semantics": {
            "old_base": "(1-alpha)*old_full + alpha*old_recent",
            "physical_base": "(1-alpha)*physical_full + alpha*physical_recent",
            "base": "(1-lambda)*old_base + lambda*physical_base",
            "r_gate": "R=(1-beta)*base + beta*r_fast; non-R=base",
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.head(12).to_string(index=False))
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
