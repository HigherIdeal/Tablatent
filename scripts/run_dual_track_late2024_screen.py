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
import run_context_interaction_screen as context_core
import run_dual_track_blend_screen as blend_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything

RECENT_VARIANT = "recent_raw_game_type"
STABLE_VARIANT = "recent_drop_game_type"


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    if any(v <= 0 for v in values):
        raise ValueError("all integers must be positive")
    return values


def make_masks(
    frame: pd.DataFrame,
    season_col: str,
    month_col: str,
    valid_year: int,
    cutoff_month: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    month = pd.to_numeric(frame[month_col], errors="raise").astype(int)
    early_valid_year = season.eq(valid_year) & month.le(cutoff_month)
    late_valid_year = season.eq(valid_year) & month.gt(cutoff_month)

    # Recent expert sees the immediately preceding new-regime season plus the
    # early part of the validation season. This is the closest available proxy
    # for 2025, where 2023 and 2024 are both already observed.
    recent_train = season.eq(valid_year - 1) | early_valid_year

    # Stable expert retains all historical rows available before the holdout,
    # but it never receives raw game_type.
    stable_train = season.lt(valid_year) | early_valid_year
    return recent_train, stable_train, late_valid_year


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    features: list[str],
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = context_core.prepare_x(train, features)
    x_valid, valid_categorical = context_core.prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("train/valid categorical mismatch")
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)

    params = context_core.catboost_params(config, iterations, task_type, devices, verbose)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)
    pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)

    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Late-2024 dual-track proxy. Recent expert trains on all 2023 plus early 2024; "
            "stable expert trains on 2019-2023 plus early 2024 with game_type removed; "
            "both are evaluated on late 2024."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--valid-year", type=int, default=2024)
    parser.add_argument("--cutoff-month", type=int, default=7)
    parser.add_argument("--iterations-grid", default="100,150,200,250,300,400")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/dual_track_late2024_screen")
    parser.add_argument("--locked-recent-iterations", type=int, default=400)
    parser.add_argument("--locked-stable-iterations", type=int, default=250)
    parser.add_argument("--locked-alpha-recent", type=float, default=0.30)
    args = parser.parse_args()

    if not (0.0 <= args.locked_alpha_recent <= 1.0):
        raise ValueError("locked alpha must be in [0,1]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    month_col = "game_month"
    iterations_grid = parse_ints(args.iterations_grid)
    alphas = blend_core.alpha_grid(args.alpha_step)

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[month_col] = pd.to_numeric(frame[month_col], errors="raise").astype(int)
    sort_columns = [season_col, month_col]
    if row_id_col in frame.columns:
        sort_columns.append(row_id_col)
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    recent_mask, stable_mask, valid_mask = make_masks(
        frame, season_col, month_col, args.valid_year, args.cutoff_month
    )
    recent_train = frame.loc[recent_mask].copy()
    stable_train = frame.loc[stable_mask].copy()
    valid = frame.loc[valid_mask].copy()
    if recent_train.empty or stable_train.empty or valid.empty:
        raise RuntimeError(
            f"empty split: recent={len(recent_train)}, stable={len(stable_train)}, valid={len(valid)}"
        )

    valid_months = sorted(valid[month_col].unique().tolist())
    train_months_2024 = sorted(
        frame.loc[frame[season_col].eq(args.valid_year) & frame[month_col].le(args.cutoff_month), month_col]
        .unique()
        .tolist()
    )
    if not valid_months:
        raise RuntimeError("no late-season validation months")

    recent_features = recent_core.feature_set(RECENT_VARIANT)
    stable_features = recent_core.feature_set(STABLE_VARIANT)
    if set(recent_features) - {"game_type"} != set(stable_features):
        raise RuntimeError("expert feature sets must differ only by game_type")

    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Late-2024 Dual-Track] valid_year={args.valid_year} cutoff_month={args.cutoff_month} "
        f"early_months={train_months_2024} late_months={valid_months}"
    )
    print(
        f"[Late-2024 Dual-Track] recent_rows={len(recent_train):,} stable_rows={len(stable_train):,} "
        f"valid_rows={len(valid):,} target={y_valid.mean():.6f}"
    )
    print(
        "[Late-2024 Dual-Track] recent = previous season + early current season + raw game_type; "
        "stable = all available history + early current season - game_type"
    )

    recent_predictions: dict[int, np.ndarray] = {}
    stable_predictions: dict[int, np.ndarray] = {}
    recent_metrics: dict[int, dict[str, float]] = {}
    stable_metrics: dict[int, dict[str, float]] = {}
    expert_rows: list[dict] = []

    print("\n[Recent expert]")
    for iterations in iterations_grid:
        seed_everything(seed)
        pred = fit_predict(
            recent_train, valid, target_col, recent_features, config,
            iterations, args.task_type, args.devices, args.verbose,
        )
        metrics = probability_metrics(y_valid, pred)
        recent_predictions[iterations] = pred
        recent_metrics[iterations] = metrics
        expert_rows.append({"expert": "recent", "iterations": iterations, **metrics})
        print(
            f"  trees={iterations:>3d} brier={metrics['brier']:.8f} "
            f"raw_score={metrics['raw_score']:+.2f} auc={metrics['auc']:.5f} "
            f"p_std={metrics['prediction_std']:.5f}"
        )

    print("\n[Stable expert]")
    for iterations in iterations_grid:
        seed_everything(seed)
        pred = fit_predict(
            stable_train, valid, target_col, stable_features, config,
            iterations, args.task_type, args.devices, args.verbose,
        )
        metrics = probability_metrics(y_valid, pred)
        stable_predictions[iterations] = pred
        stable_metrics[iterations] = metrics
        expert_rows.append({"expert": "stable", "iterations": iterations, **metrics})
        print(
            f"  trees={iterations:>3d} brier={metrics['brier']:.8f} "
            f"raw_score={metrics['raw_score']:+.2f} auc={metrics['auc']:.5f} "
            f"p_std={metrics['prediction_std']:.5f}"
        )

    blend_rows: list[dict] = []
    for ri, p_recent in recent_predictions.items():
        for si, p_stable in stable_predictions.items():
            analytic_alpha = blend_core.clipped_analytic_alpha(y_valid, p_recent, p_stable)
            for selection, alpha in (("analytic", analytic_alpha),):
                pred = alpha * p_recent + (1.0 - alpha) * p_stable
                metrics = probability_metrics(y_valid, pred)
                blend_rows.append(
                    {
                        "selection": selection,
                        "recent_iterations": ri,
                        "stable_iterations": si,
                        "alpha_recent": alpha,
                        "delta_vs_recent": metrics["brier"] - recent_metrics[ri]["brier"],
                        "delta_vs_stable": metrics["brier"] - stable_metrics[si]["brier"],
                        **metrics,
                    }
                )

            grid_alpha, grid_metrics = blend_core._best_grid_blend(
                y_valid, p_recent, p_stable, alphas
            )
            blend_rows.append(
                {
                    "selection": "grid",
                    "recent_iterations": ri,
                    "stable_iterations": si,
                    "alpha_recent": grid_alpha,
                    "delta_vs_recent": grid_metrics["brier"] - recent_metrics[ri]["brier"],
                    "delta_vs_stable": grid_metrics["brier"] - stable_metrics[si]["brier"],
                    **grid_metrics,
                }
            )

    expert_df = pd.DataFrame(expert_rows).sort_values(["expert", "brier", "iterations"])
    blend_df = pd.DataFrame(blend_rows).sort_values(["brier", "selection"])
    expert_df.to_csv(output_dir / "expert_results.csv", index=False)
    blend_df.to_csv(output_dir / "blend_results.csv", index=False)

    locked = None
    if (
        args.locked_recent_iterations in recent_predictions
        and args.locked_stable_iterations in stable_predictions
    ):
        p_locked = (
            args.locked_alpha_recent * recent_predictions[args.locked_recent_iterations]
            + (1.0 - args.locked_alpha_recent) * stable_predictions[args.locked_stable_iterations]
        )
        locked = {
            "recent_iterations": int(args.locked_recent_iterations),
            "stable_iterations": int(args.locked_stable_iterations),
            "alpha_recent": float(args.locked_alpha_recent),
            **probability_metrics(y_valid, p_locked),
        }

    best_grid = blend_df.loc[blend_df["selection"].eq("grid")].iloc[0].to_dict()
    best_analytic = blend_df.loc[blend_df["selection"].eq("analytic")].iloc[0].to_dict()
    best_recent = expert_df.loc[expert_df["expert"].eq("recent")].iloc[0].to_dict()
    best_stable = expert_df.loc[expert_df["expert"].eq("stable")].iloc[0].to_dict()

    recommendation = {
        "proxy": "late_2024",
        "valid_year": int(args.valid_year),
        "cutoff_month": int(args.cutoff_month),
        "early_valid_year_months": train_months_2024,
        "late_valid_year_months": valid_months,
        "recent_iterations": int(best_grid["recent_iterations"]),
        "stable_iterations": int(best_grid["stable_iterations"]),
        "alpha_recent": float(best_grid["alpha_recent"]),
        "validation_brier": float(best_grid["brier"]),
        "validation_raw_score": float(best_grid["raw_score"]),
        "best_recent_only": {
            "iterations": int(best_recent["iterations"]),
            "brier": float(best_recent["brier"]),
            "raw_score": float(best_recent["raw_score"]),
        },
        "best_stable_only": {
            "iterations": int(best_stable["iterations"]),
            "brier": float(best_stable["brier"]),
            "raw_score": float(best_stable["raw_score"]),
        },
        "best_analytic": {
            "recent_iterations": int(best_analytic["recent_iterations"]),
            "stable_iterations": int(best_analytic["stable_iterations"]),
            "alpha_recent": float(best_analytic["alpha_recent"]),
            "brier": float(best_analytic["brier"]),
            "raw_score": float(best_analytic["raw_score"]),
        },
        "locked_fullseason_2024_candidate": locked,
    }
    save_json(recommendation, output_dir / "recommended_config.json")
    save_json(
        {
            "seed": seed,
            "iterations_grid": iterations_grid,
            "alpha_step": args.alpha_step,
            "task_type": args.task_type,
            "recent_features": recent_features,
            "stable_features": stable_features,
            "canonical_invariants": invariant_check,
            "training_order": sort_columns,
        },
        output_dir / "run_config.json",
    )

    payload: dict[str, np.ndarray] = {
        "y": y_valid.astype(np.float32),
    }
    if row_id_col in valid.columns:
        payload["row_id"] = valid[row_id_col].to_numpy()
    for k, v in recent_predictions.items():
        payload[f"recent_{k}"] = v.astype(np.float32)
    for k, v in stable_predictions.items():
        payload[f"stable_{k}"] = v.astype(np.float32)
    np.savez_compressed(output_dir / "validation_predictions.npz", **payload)

    print("\n[Best late-2024 blends]")
    cols = [
        "selection", "recent_iterations", "stable_iterations", "alpha_recent",
        "brier", "raw_score", "auc", "delta_vs_recent", "delta_vs_stable",
    ]
    print(blend_df.head(12)[cols].to_string(index=False))
    if locked is not None:
        print("\n[Locked candidate from full-2024 proxy: A400/B250/alpha=.30]")
        print(json.dumps(locked, ensure_ascii=False, indent=2))
    print("\n[Late-2024 recommendation]")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
