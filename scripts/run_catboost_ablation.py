
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import load_frame
from src.utils import load_config, save_json, seed_everything
TARGET = "control_success"

BASE_790 = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before",
    "run_bot_before", "run_total_before", "score_diff_home",
    "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state", "home_win_expectancy", "away_win_expectancy",
    "li", "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id",
    "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

ENGINEERED_790 = [
    "pitcher_success_eb100", "pitcher_success_eb500",
    "pitcher_reliability_500",
    "batter_success_eb100", "batter_success_eb500",
    "batter_reliability_500",
    "pitcher_n_log", "batter_n_log", "pitchmix_n_log",
]

REFERENCE_790 = BASE_790 + ENGINEERED_790

CATEGORICAL = {
    "game_month", "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
}

GROUPS = {
    "batter_id": ["batter_id"],
    "pitcher_team_id": ["pitcher_team_id"],
    "batter_team_id": ["batter_team_id"],
    "team_ids": ["pitcher_team_id", "batter_team_id"],
    "retained_ids": ["batter_id", "pitcher_team_id", "batter_team_id"],
    "season": ["season"],
    "game_context": [
        "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
        "balls_before", "strikes_before", "outs_before", "run_top_before",
        "run_bot_before", "run_total_before", "score_diff_home",
        "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
        "runner_on_3b", "num_runners_on", "base_state",
    ],
    "leverage": ["home_win_expectancy", "away_win_expectancy", "li"],
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
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ],
    "empirical_bayes": [
        "pitcher_success_eb100", "pitcher_success_eb500",
        "batter_success_eb100", "batter_success_eb500",
    ],
    "reliability_logs": [
        "pitcher_reliability_500", "batter_reliability_500",
        "pitcher_n_log", "batter_n_log", "pitchmix_n_log",
    ],
    "all_engineered": ENGINEERED_790,
}

DEFAULT_VARIANTS = [
    "reference_790",
    "add_pitcher_id",
    "drop_batter_id",
    "drop_pitcher_team_id",
    "drop_batter_team_id",
    "drop_team_ids",
    "drop_retained_ids",
    "drop_season",
    "drop_game_context",
    "drop_leverage",
    "drop_handedness",
    "drop_pitcher_profile",
    "drop_pitcher_recent",
    "drop_batter_profile",
    "drop_pitchmix",
    "drop_empirical_bayes",
    "drop_reliability_logs",
    "drop_all_engineered",
]


def _add_790_features(frame: pd.DataFrame, prior: float) -> None:
    pitcher_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
    pitcher_rate = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(prior).clip(0, 1)
    batter_n = pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)
    batter_rate = pd.to_numeric(
        frame["asof_batter_success_rate"], errors="coerce"
    ).fillna(prior).clip(0, 1)

    for alpha in (100, 500):
        frame[f"pitcher_success_eb{alpha}"] = (
            pitcher_rate * pitcher_n + alpha * prior
        ) / (pitcher_n + alpha)
        frame[f"batter_success_eb{alpha}"] = (
            batter_rate * batter_n + alpha * prior
        ) / (batter_n + alpha)

    frame["pitcher_reliability_500"] = pitcher_n / (pitcher_n + 500)
    frame["batter_reliability_500"] = batter_n / (batter_n + 500)
    frame["pitcher_n_log"] = np.log1p(pitcher_n)
    frame["batter_n_log"] = np.log1p(batter_n)
    frame["pitchmix_n_log"] = np.log1p(
        pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )


def _variant_features(name: str) -> list[str]:
    if name == "reference_790":
        return list(REFERENCE_790)
    if name == "add_pitcher_id":
        return list(REFERENCE_790) + ["pitcher_id"]
    prefix = "drop_"
    if not name.startswith(prefix):
        raise ValueError(f"Unknown variant: {name}")
    group_name = name[len(prefix):]
    if group_name not in GROUPS:
        raise ValueError(f"Unknown ablation group: {group_name}")
    drop = set(GROUPS[group_name])
    return [f for f in REFERENCE_790 if f not in drop]


def _prepare_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = frame.loc[:, features].copy()
    categorical = [f for f in features if f in CATEGORICAL]
    categorical_set = set(categorical)
    for col in features:
        if col in categorical_set:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(np.float32)
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    return x, categorical


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
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
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    required = set(REFERENCE_790 + [target_col, season_col, row_id_col, "pitcher_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sort_cols = [c for c in [season_col, "game_month", row_id_col] if c in frame.columns]
    frame = frame.sort_values(sort_cols).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "catboost_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = {name: _variant_features(name) for name in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict] = []
    print(
        f"[Ablation] folds={folds}, variants={len(variants)}, iterations={iterations}, "
        f"task_type={task_type}, catboost={catboost.__version__}"
    )
    print("[Ablation] fixed 790-point reference hyperparameters; no early stopping")

    for val_year in folds:
        train_mask = frame[season_col] < val_year
        val_mask = frame[season_col] == val_year
        if not train_mask.any() or not val_mask.any():
            raise ValueError(f"Fold {val_year}: empty train or validation split")

        fold = frame.loc[train_mask | val_mask].copy()
        fold_train_mask = fold[season_col] < val_year
        fold_val_mask = fold[season_col] == val_year
        prior = float(pd.to_numeric(
            fold.loc[fold_train_mask, target_col], errors="raise"
        ).mean())
        _add_790_features(fold, prior)

        train_frame = fold.loc[fold_train_mask]
        val_frame = fold.loc[fold_val_mask]
        train_y = pd.to_numeric(train_frame[target_col], errors="raise").to_numpy(np.float32)
        val_y = pd.to_numeric(val_frame[target_col], errors="raise").to_numpy(np.float32)

        print(
            f"\n[Fold {val_year}] train={len(train_frame):,} "
            f"({int(train_frame[season_col].min())}-{val_year-1}), "
            f"val={len(val_frame):,}, prior={prior:.6f}"
        )

        for idx, variant in enumerate(variants, start=1):
            features = feature_sets[variant]
            train_x, categorical = _prepare_x(train_frame, features)
            val_x, _ = _prepare_x(val_frame, features)
            train_pool = Pool(
                train_x,
                label=train_y,
                cat_features=categorical,
                feature_names=features,
            )
            val_pool = Pool(
                val_x,
                label=val_y,
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

            model = CatBoostClassifier(**params)
            print(
                f"  [{idx:02d}/{len(variants):02d}] {variant:<24s} "
                f"features={len(features):2d}",
                flush=True,
            )
            model.fit(train_pool, verbose=verbose)
            pred = model.predict_proba(val_pool)[:, 1]
            metric = _metrics(val_y, pred)
            row = {
                "variant": variant,
                "validation_year": int(val_year),
                "train_start_year": int(train_frame[season_col].min()),
                "train_end_year": int(train_frame[season_col].max()),
                "train_rows": int(len(train_frame)),
                "val_rows": int(len(val_frame)),
                "feature_count": int(len(features)),
                "categorical_count": int(len(categorical)),
                "prior": prior,
                **metric,
            }
            rows.append(row)
            print(
                f"       brier={metric['brier']:.8f} "
                f"skill={metric['brier_skill']:+.3e} "
                f"auc={metric['auc']:.5f} "
                f"p_std={metric['prediction_std']:.5f}"
            )

            del model, train_pool, val_pool, train_x, val_x, pred
            gc.collect()

        del fold, train_frame, val_frame, train_y, val_y
        gc.collect()

    fold_results = pd.DataFrame(rows)
    ref = (
        fold_results.loc[fold_results["variant"] == "reference_790",
                         ["validation_year", "brier"]]
        .rename(columns={"brier": "reference_variant_brier"})
    )
    fold_results = fold_results.merge(ref, on="validation_year", how="left")
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

    result = {
        "reference": "uploaded 790-point H2/J0 feature set",
        "folds": folds,
        "variants": variants,
        "iterations": int(iterations),
        "fixed_params": {
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 10.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "border_count": 128,
            "has_time": True,
            "one_hot_max_size": 10,
        },
        "output_dir": str(output_dir),
    }
    save_json(result, output_dir / "run_config.json")

    print("\n[Ablation summary] lower mean_delta_brier is better")
    display_cols = [
        "variant", "feature_count", "mean_brier", "mean_delta_brier",
        "worst_delta_brier", "mean_auc",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved: {output_dir / 'summary.csv'}")
    return result


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate the uploaded 790-point CatBoost feature set on temporal folds."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--folds",
        default="2023,2024",
        help="Comma-separated temporal validation seasons; training uses all earlier seasons.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument("--iterations", type=int, default=520)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    folds = _parse_csv_ints(args.folds)
    variants = (
        list(DEFAULT_VARIANTS)
        if args.variants == "all"
        else _parse_csv_strings(args.variants)
    )
    unknown = [v for v in variants if v not in DEFAULT_VARIANTS]
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
