from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_features import (
    APPROX_REDUNDANT_OFFICIAL,
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    EXACT_REDUNDANT_ENGINEERED,
    EXACT_REDUNDANT_OFFICIAL,
    NON_EXACT_OVERLAPS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.utils import load_config, save_json, seed_everything


CATEGORICAL = set(CANONICAL_CATEGORICAL) | {"pitcher_id", "batter_id"}

GROUPS = {
    "pitcher_team_id": ["pitcher_team_id"],
    "batter_team_id": ["batter_team_id"],
    "team_ids": ["pitcher_team_id", "batter_team_id"],
    "season": ["season"],
    "calendar": ["game_month", "game_dayofweek"],
    "game_phase": ["inning", "top_bottom", "game_type"],
    "count": ["balls_before", "strikes_before", "outs_before"],
    "score": ["run_total_before", "score_diff_home"],
    "base_state": ["base_state"],
    "leverage": [PITCHER_TEAM_WIN_EXPECTANCY, "li"],
    "win_expectancy": [PITCHER_TEAM_WIN_EXPECTANCY],
    "handedness": ["pitcher_hand", "batter_hand"],
    "pitcher_profile": [
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    ],
    "pitcher_recent": [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ],
    "batter_profile": [
        "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    ],
    "pitchmix": [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],
}

DEFAULT_VARIANTS = [
    "reference_canonical",
    "add_pitcher_id",
    "add_batter_id",
    "drop_pitcher_team_id",
    "drop_batter_team_id",
    "drop_team_ids",
    "drop_season",
    "drop_calendar",
    "drop_game_phase",
    "drop_count",
    "drop_score",
    "drop_base_state",
    "drop_win_expectancy",
    "drop_leverage",
    "drop_handedness",
    "drop_pitcher_profile",
    "drop_pitcher_recent",
    "drop_batter_profile",
    "drop_pitchmix",
]


def variant_features(name: str) -> list[str]:
    if name == "reference_canonical":
        return list(CANONICAL_FEATURES)
    if name == "add_pitcher_id":
        return [*CANONICAL_FEATURES, "pitcher_id"]
    if name == "add_batter_id":
        return [*CANONICAL_FEATURES, "batter_id"]
    if not name.startswith("drop_"):
        raise ValueError(f"Unknown variant: {name}")
    group = name[5:]
    if group not in GROUPS:
        raise ValueError(f"Unknown ablation group: {group}")
    drop = set(GROUPS[group])
    return [feature for feature in CANONICAL_FEATURES if feature not in drop]


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


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    brier = float(np.mean((p - y) ** 2))
    reference = float(y.mean() * (1.0 - y.mean()))
    return {
        "brier": brier,
        "brier_skill": float(1.0 - brier / reference),
        "competition_score": float(max(0.0, 100000.0 * (1.0 - brier / reference))),
        "auc": float(roc_auc_score(y, p)),
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "target_mean": float(y.mean()),
        "reference_brier": reference,
    }


def run_ablation(
    config: dict,
    folds: list[int],
    variants: list[str],
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> dict:
    try:
        import catboost
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError("catboost가 없습니다.") from exc

    seed_everything(int(config["seed"]))
    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [
        target, season, row_id, "pitcher_id", "batter_id"
    ])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "catboost_ablation_canonical"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = {variant: variant_features(variant) for variant in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[Canonical Ablation] folds={folds}, variants={len(variants)}, "
        f"iterations={iterations}, task_type={task_type}, catboost={catboost.__version__}"
    )
    print(
        f"[Canonical Ablation] reference features={len(CANONICAL_FEATURES)}; "
        "deterministic/rounding redundancy normalized"
    )
    print(
        "[Canonical Ablation] home/away win expectancy -> "
        "pitcher_team_win_expectancy"
    )
    print("[Canonical Ablation] invariants: OK")
    print("[Canonical Ablation] pitcher_id/batter_id excluded in reference; add-back variants test them")

    rows: list[dict] = []
    for val_year in folds:
        use_mask = frame[season] <= val_year
        fold = frame.loc[use_mask].copy()
        train_mask = fold[season] < val_year
        val_mask = fold[season] == val_year
        if not train_mask.any() or not val_mask.any():
            raise ValueError(f"Fold {val_year}: empty train or validation split")

        train = fold.loc[train_mask]
        valid = fold.loc[val_mask]
        y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float32)
        prior = float(y_train.mean())
        print(
            f"\n[Fold {val_year}] train={len(train):,}, val={len(valid):,}, "
            f"prior={prior:.6f}"
        )

        for index, variant in enumerate(variants, start=1):
            features = feature_sets[variant]
            x_train, categorical = prepare_x(train, features)
            x_valid, _ = prepare_x(valid, features)
            train_pool = Pool(
                x_train,
                label=y_train,
                cat_features=categorical,
                feature_names=features,
            )
            valid_pool = Pool(
                x_valid,
                label=y_valid,
                cat_features=categorical,
                feature_names=features,
            )

            params = {
                "iterations": int(iterations),
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 10.0,
                "random_strength": 0.5,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 0.5,
                "border_count": 128,
                "random_seed": int(config["seed"]),
                "loss_function": "Logloss",
                "has_time": True,
                "one_hot_max_size": 10,
                "allow_writing_files": False,
                "task_type": task_type,
                "verbose": verbose,
            }
            if task_type == "GPU":
                params["devices"] = devices

            print(
                f"  [{index:02d}/{len(variants):02d}] {variant:<24s} "
                f"features={len(features):2d}",
                flush=True,
            )
            model = CatBoostClassifier(**params)
            model.fit(train_pool, verbose=verbose)
            prediction = model.predict_proba(valid_pool)[:, 1]
            metric = metrics(y_valid, prediction)
            rows.append(
                {
                    "variant": variant,
                    "validation_year": int(val_year),
                    "train_start_year": int(train[season].min()),
                    "train_end_year": int(train[season].max()),
                    "train_rows": int(len(train)),
                    "val_rows": int(len(valid)),
                    "feature_count": int(len(features)),
                    "categorical_count": int(len(categorical)),
                    "prior": prior,
                    **metric,
                }
            )
            print(
                f"       brier={metric['brier']:.8f} "
                f"skill={metric['brier_skill']:+.3e} "
                f"auc={metric['auc']:.5f} "
                f"p_std={metric['prediction_std']:.5f}"
            )

            del model, train_pool, valid_pool, x_train, x_valid, prediction
            gc.collect()

        del fold, train, valid, y_train, y_valid
        gc.collect()

    fold_results = pd.DataFrame(rows)
    reference = (
        fold_results.loc[
            fold_results["variant"] == "reference_canonical",
            ["validation_year", "brier"],
        ]
        .rename(columns={"brier": "reference_variant_brier"})
    )
    fold_results = fold_results.merge(reference, on="validation_year", how="left")
    fold_results["delta_brier_vs_reference"] = (
        fold_results["brier"] - fold_results["reference_variant_brier"]
    )
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        fold_results.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_brier=("delta_brier_vs_reference", "mean"),
            worst_delta_brier=("delta_brier_vs_reference", "max"),
            mean_skill=("brier_skill", "mean"),
            mean_auc=("auc", "mean"),
            mean_prediction_std=("prediction_std", "mean"),
        )
        .sort_values(["mean_brier", "worst_brier"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    run_config = {
        "reference": "canonical de-duplicated feature set",
        "canonical_features": CANONICAL_FEATURES,
        "removed_exact_official": EXACT_REDUNDANT_OFFICIAL,
        "normalized_approx_official": APPROX_REDUNDANT_OFFICIAL,
        "removed_exact_engineered": EXACT_REDUNDANT_ENGINEERED,
        "retained_non_exact_overlaps": NON_EXACT_OVERLAPS,
        "invariant_check": invariant_check,
        "folds": folds,
        "variants": variants,
        "iterations": int(iterations),
        "output_dir": str(output_dir),
    }
    save_json(run_config, output_dir / "run_config.json")

    print("\n[Canonical Ablation summary] lower mean_delta_brier is better")
    display_columns = [
        "variant", "feature_count", "mean_brier", "mean_delta_brier",
        "worst_delta_brier", "mean_auc",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved: {output_dir / 'summary.csv'}")
    return run_config


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate the canonical de-duplicated CatBoost feature set."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--iterations", type=int, default=520)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    folds = parse_ints(args.folds)
    variants = (
        list(DEFAULT_VARIANTS)
        if args.variants == "all"
        else parse_strings(args.variants)
    )
    unknown = [variant for variant in variants if variant not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    run_ablation(
        config=config,
        folds=folds,
        variants=variants,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
