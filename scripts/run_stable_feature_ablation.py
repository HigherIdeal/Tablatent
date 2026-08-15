from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything

RECENT_VARIANT = "recent_raw_game_type"
STABLE_VARIANT = "recent_drop_game_type"

RECENT_ITERATIONS = 400
STABLE_ITERATIONS_GRID = [250, 300, 400]
ALPHAS = [0.35, 0.375, 0.40, 0.425, 0.45]
LOCKED_STABLE_ITERATIONS = 300
LOCKED_ALPHA = 0.425

SUCCESS_STATE_FEATURES = [
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

# First-pass ablations are deliberately small and interpretable. The audit is a
# hypothesis generator; these features are not automatically discarded.
STABLE_DROP_VARIANTS: dict[str, list[str]] = {
    "baseline_drop_game_type": [],
    "drop_season": ["season"],
    "drop_game_month": ["game_month"],
    "drop_inning": ["inning"],
    "drop_batter_n": ["asof_batter_n"],
    "drop_batter_success_rate": ["asof_batter_success_rate"],
    "drop_batter_core": ["asof_batter_n", "asof_batter_success_rate"],
    "drop_pitcher_n": ["asof_pitcher_n"],
    "drop_pitcher_success_rate": ["asof_pitcher_success_rate"],
    "drop_pitcher_control_rates": ["asof_pitcher_ball_rate", "asof_pitcher_strike_rate"],
    "drop_pitcher_recent_success_family": [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        *SUCCESS_STATE_FEATURES,
    ],
    "drop_pitchmix": [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],
    "drop_team_ids": ["pitcher_team_id", "batter_team_id"],
    "drop_calendar": ["game_month", "game_dayofweek"],
}


def build_stable_feature_sets() -> dict[str, list[str]]:
    base = recent_core.feature_set(STABLE_VARIANT)
    base_set = set(base)
    if "game_type" in base_set:
        raise RuntimeError("stable baseline unexpectedly contains game_type")

    result: dict[str, list[str]] = {}
    for name, drops in STABLE_DROP_VARIANTS.items():
        missing = sorted(set(drops) - base_set)
        if missing:
            raise RuntimeError(f"variant {name} references missing stable features: {missing}")
        drop_set = set(drops)
        features = [feature for feature in base if feature not in drop_set]
        if not features:
            raise RuntimeError(f"variant {name} removed every feature")
        if len(features) != len(set(features)):
            raise RuntimeError(f"variant {name} contains duplicate features")
        result[name] = features
    return result


def _aggregate_locked(rows: pd.DataFrame, fold_weights: dict[str, float]) -> pd.DataFrame:
    summary_rows: list[dict] = []
    baseline = rows.loc[rows["variant"].eq("baseline_drop_game_type")].set_index("fold")
    if len(baseline) != len(fold_weights):
        raise RuntimeError("locked baseline is missing proxy folds")

    for variant, group in rows.groupby("variant", sort=False):
        group = group.set_index("fold").loc[list(fold_weights)]
        weights = np.asarray([fold_weights[name] for name in group.index], dtype=np.float64)
        weights /= weights.sum()
        raw = group["raw_score"].to_numpy(np.float64)
        brier = group["brier"].to_numpy(np.float64)
        baseline_raw = baseline.loc[group.index, "raw_score"].to_numpy(np.float64)
        delta = raw - baseline_raw
        summary_rows.append(
            {
                "variant": variant,
                "weighted_raw_score": float(np.dot(weights, raw)),
                "weighted_brier": float(np.dot(weights, brier)),
                "weighted_delta_raw_vs_baseline": float(np.dot(weights, delta)),
                "worst_delta_raw_vs_baseline": float(delta.min()),
                "best_delta_raw_vs_baseline": float(delta.max()),
                "improved_folds_vs_baseline": int(np.count_nonzero(delta > 0.0)),
                "fold_count": int(len(group)),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["weighted_delta_raw_vs_baseline", "worst_delta_raw_vs_baseline"],
        ascending=[False, False],
    )


def _aggregate_tuned(rows: pd.DataFrame, fold_weights: dict[str, float]) -> pd.DataFrame:
    summary_rows: list[dict] = []
    group_cols = ["variant", "stable_iterations", "alpha_recent"]
    for key, group in rows.groupby(group_cols, sort=False):
        group = group.set_index("fold").loc[list(fold_weights)]
        weights = np.asarray([fold_weights[name] for name in group.index], dtype=np.float64)
        weights /= weights.sum()
        raw = group["raw_score"].to_numpy(np.float64)
        brier = group["brier"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": key[0],
                "stable_iterations": int(key[1]),
                "alpha_recent": float(key[2]),
                "weighted_raw_score": float(np.dot(weights, raw)),
                "weighted_brier": float(np.dot(weights, brier)),
                "worst_raw_score": float(raw.min()),
                "raw_score_std": float(np.sqrt(np.dot(weights, (raw - np.dot(weights, raw)) ** 2))),
            }
        )
    full = pd.DataFrame(summary_rows)
    baseline_best = full.loc[full["variant"].eq("baseline_drop_game_type")].sort_values(
        ["weighted_raw_score", "worst_raw_score"], ascending=[False, False]
    ).iloc[0]
    best_per_variant = (
        full.sort_values(["variant", "weighted_raw_score", "worst_raw_score"], ascending=[True, False, False])
        .groupby("variant", as_index=False, sort=False)
        .first()
    )
    best_per_variant["delta_weighted_raw_vs_tuned_baseline"] = (
        best_per_variant["weighted_raw_score"] - float(baseline_best["weighted_raw_score"])
    )
    return best_per_variant.sort_values(
        ["delta_weighted_raw_vs_tuned_baseline", "worst_raw_score"], ascending=[False, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate temporal-instability candidates only from the long-history stable expert. "
            "The recent expert stays fixed at 400 trees. Primary ranking uses the locked "
            "A400/B300/alpha=.425 configuration; a small B-tree/alpha grid is secondary."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--variants", default="all", help="Comma-separated names, or all")
    parser.add_argument("--output-dir", default="outputs/stable_feature_ablation")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    month_col = "game_month"
    row_id_col = config["data"].get("row_id_col", "row_id")
    seed_everything(seed)

    frame, invariant_check = recent_core.prepare_frame(config)
    sort_columns = [season_col, month_col]
    if row_id_col in frame.columns:
        sort_columns.append(row_id_col)
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    recent_features = recent_core.feature_set(RECENT_VARIANT)
    all_stable_feature_sets = build_stable_feature_sets()
    if args.variants == "all":
        variant_names = list(all_stable_feature_sets)
    else:
        variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
        unknown = sorted(set(variant_names) - set(all_stable_feature_sets))
        if unknown:
            raise ValueError(f"unknown variants: {unknown}")
        if "baseline_drop_game_type" not in variant_names:
            variant_names.insert(0, "baseline_drop_game_type")
    stable_feature_sets = {name: all_stable_feature_sets[name] for name in variant_names}

    fold_weights = {spec.name: spec.weight for spec in proxy.DEFAULT_FOLDS}
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_rows: list[dict] = []
    tuned_rows: list[dict] = []
    expert_rows: list[dict] = []

    print("[Stable Expert Feature Ablation]")
    print(f"  recent expert : fixed {RECENT_ITERATIONS} trees, raw game_type")
    print(f"  stable trees  : {STABLE_ITERATIONS_GRID}")
    print(f"  alpha grid    : {ALPHAS}")
    print(f"  locked config : A={RECENT_ITERATIONS}, B={LOCKED_STABLE_ITERATIONS}, alpha={LOCKED_ALPHA}")
    print(f"  variants      : {len(stable_feature_sets)}")

    for spec in proxy.DEFAULT_FOLDS:
        recent_mask, stable_mask, valid_mask = proxy.fold_masks(frame, spec, season_col, month_col)
        print(
            f"\n[{spec.name}] recent_train={int(recent_mask.sum()):,} "
            f"stable_train={int(stable_mask.sum()):,} valid={int(valid_mask.sum()):,}"
        )

        print("  fitting shared recent expert...")
        seed_everything(seed)
        recent_preds, y_valid, _ = proxy._fit_prefix_predictions(
            frame=frame,
            train_mask=recent_mask,
            valid_mask=valid_mask,
            features=recent_features,
            target_col=target_col,
            config=config,
            iterations_grid=[RECENT_ITERATIONS],
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
        )
        p_recent = recent_preds[RECENT_ITERATIONS]

        for idx, (variant, stable_features) in enumerate(stable_feature_sets.items(), start=1):
            print(
                f"  [{idx:02d}/{len(stable_feature_sets):02d}] {variant:<38} "
                f"features={len(stable_features)}"
            )
            seed_everything(seed)
            stable_preds, y_stable, _ = proxy._fit_prefix_predictions(
                frame=frame,
                train_mask=stable_mask,
                valid_mask=valid_mask,
                features=stable_features,
                target_col=target_col,
                config=config,
                iterations_grid=STABLE_ITERATIONS_GRID,
                task_type=args.task_type,
                devices=args.devices,
                verbose=args.verbose,
            )
            if not np.array_equal(y_valid, y_stable):
                raise RuntimeError(f"validation target mismatch for {variant} in {spec.name}")

            for stable_iterations, p_stable in stable_preds.items():
                sm = probability_metrics(y_valid, p_stable)
                expert_rows.append(
                    {
                        "fold": spec.name,
                        "variant": variant,
                        "stable_iterations": stable_iterations,
                        **sm,
                    }
                )
                for alpha in ALPHAS:
                    pred = alpha * p_recent + (1.0 - alpha) * p_stable
                    metrics = probability_metrics(y_valid, pred)
                    row = {
                        "fold": spec.name,
                        "fold_weight": spec.weight,
                        "variant": variant,
                        "stable_iterations": stable_iterations,
                        "alpha_recent": alpha,
                        **metrics,
                    }
                    tuned_rows.append(row)
                    if stable_iterations == LOCKED_STABLE_ITERATIONS and np.isclose(alpha, LOCKED_ALPHA):
                        locked_rows.append(row.copy())

            del stable_preds, y_stable
            gc.collect()

        del recent_preds, p_recent, y_valid
        gc.collect()

    locked_df = pd.DataFrame(locked_rows)
    tuned_df = pd.DataFrame(tuned_rows)
    expert_df = pd.DataFrame(expert_rows)
    locked_summary = _aggregate_locked(locked_df, fold_weights)
    tuned_summary = _aggregate_tuned(tuned_df, fold_weights)

    locked_df.to_csv(output_dir / "locked_results_by_fold.csv", index=False)
    locked_summary.to_csv(output_dir / "locked_variant_summary.csv", index=False)
    tuned_df.to_csv(output_dir / "tuned_results_by_fold.csv", index=False)
    tuned_summary.to_csv(output_dir / "tuned_variant_summary.csv", index=False)
    expert_df.to_csv(output_dir / "stable_expert_results.csv", index=False)

    best_locked = locked_summary.iloc[0].to_dict()
    recommendation = {
        "primary_selection": "locked A400/B300/alpha=.425 weighted temporal proxy",
        "best_variant": best_locked["variant"],
        "weighted_raw_score": float(best_locked["weighted_raw_score"]),
        "weighted_delta_raw_vs_baseline": float(best_locked["weighted_delta_raw_vs_baseline"]),
        "worst_delta_raw_vs_baseline": float(best_locked["worst_delta_raw_vs_baseline"]),
        "improved_folds_vs_baseline": int(best_locked["improved_folds_vs_baseline"]),
        "dropped_features": STABLE_DROP_VARIANTS[best_locked["variant"]],
        "note": "Adopt a drop only after checking fold consistency; the temporal audit itself is not causal evidence.",
    }
    save_json(recommendation, output_dir / "recommended_feature_policy.json")
    save_json(
        {
            "seed": seed,
            "fold_weights": fold_weights,
            "recent_iterations": RECENT_ITERATIONS,
            "stable_iterations_grid": STABLE_ITERATIONS_GRID,
            "alphas": ALPHAS,
            "locked_stable_iterations": LOCKED_STABLE_ITERATIONS,
            "locked_alpha": LOCKED_ALPHA,
            "variants": {name: STABLE_DROP_VARIANTS[name] for name in variant_names},
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    print("\n[Locked-config feature ranking]")
    print(
        locked_summary.head(20)[
            [
                "variant",
                "weighted_raw_score",
                "weighted_delta_raw_vs_baseline",
                "worst_delta_raw_vs_baseline",
                "improved_folds_vs_baseline",
            ]
        ].to_string(index=False)
    )
    print("\n[Secondary tuned ranking]")
    print(
        tuned_summary.head(20)[
            [
                "variant",
                "stable_iterations",
                "alpha_recent",
                "weighted_raw_score",
                "delta_weighted_raw_vs_tuned_baseline",
                "worst_raw_score",
            ]
        ].to_string(index=False)
    )
    print("\n[Recommended feature policy]")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
