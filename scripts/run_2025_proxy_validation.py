from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


RECENT_VARIANT = "recent_raw_game_type"
STABLE_VARIANT = "recent_drop_game_type"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    weight: float
    kind: str
    cutoff_month: int | None = None
    valid_month_start: int | None = None
    valid_month_end: int | None = None


DEFAULT_FOLDS = (
    FoldSpec("season_forward_2024", 0.50, "season_forward"),
    FoldSpec("mid_2024", 0.20, "within_2024", cutoff_month=5, valid_month_start=6, valid_month_end=7),
    FoldSpec("late_2024", 0.30, "within_2024", cutoff_month=7, valid_month_start=8, valid_month_end=None),
)


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    if any(v <= 0 for v in values):
        raise ValueError("all integers must be positive")
    if values != sorted(set(values)):
        raise ValueError(f"iterations must be unique and sorted: {values}")
    return values


def alpha_grid(step: float) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("alpha-step must be in (0, 1]")
    values = np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)
    return np.unique(np.clip(np.append(values, 1.0), 0.0, 1.0))


def fold_masks(
    frame: pd.DataFrame,
    spec: FoldSpec,
    season_col: str,
    month_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return recent-train, stable-train, validation masks for one 2025 proxy fold.

    The recent expert is intentionally restricted to the post-change regime.
    The stable expert can use all older seasons but never sees game_type.
    Validation is always strictly later than every 2024 row admitted to training.
    """
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    month = pd.to_numeric(frame[month_col], errors="raise").astype(int)

    if spec.kind == "season_forward":
        recent = season.eq(2023)
        stable = season.le(2023)
        valid = season.eq(2024)
    elif spec.kind == "within_2024":
        if spec.cutoff_month is None or spec.valid_month_start is None:
            raise ValueError(f"invalid within-2024 fold: {spec}")
        observed_2024 = season.eq(2024) & month.le(spec.cutoff_month)
        recent = season.eq(2023) | observed_2024
        stable = season.lt(2024) | observed_2024
        valid = season.eq(2024) & month.ge(spec.valid_month_start)
        if spec.valid_month_end is not None:
            valid &= month.le(spec.valid_month_end)
    else:
        raise ValueError(f"unknown fold kind: {spec.kind}")

    if bool((recent & valid).any()) or bool((stable & valid).any()):
        raise RuntimeError(f"validation leakage in fold {spec.name}")
    return recent, stable, valid


def _prepare_xy(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"empty split: train={len(train):,}, valid={len(valid):,}")

    x_train, categorical = context_core.prepare_x(train, features)
    x_valid, valid_categorical = context_core.prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("train/valid categorical feature mismatch")
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    return x_train, y_train, x_valid, y_valid, categorical


def _fit_prefix_predictions(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    features: list[str],
    target_col: str,
    config: dict,
    iterations_grid: list[int],
    task_type: str,
    devices: str,
    verbose: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """Train once at max(iterations_grid), then evaluate exact tree prefixes.

    With fixed CatBoost parameters and no eval-set early stopping, the first N trees
    are the same prefix that would be used by an N-tree model. This makes the
    multi-fold screen far cheaper than retraining every tree count separately.
    """
    from catboost import CatBoostClassifier, Pool

    x_train, y_train, x_valid, y_valid, categorical = _prepare_xy(
        frame, train_mask, valid_mask, features, target_col
    )
    max_iterations = max(iterations_grid)
    params = context_core.catboost_params(
        config=config,
        iterations=max_iterations,
        task_type=task_type,
        devices=devices,
        verbose=verbose,
    )
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)

    if int(model.tree_count_) < max_iterations:
        raise RuntimeError(
            f"model produced only {model.tree_count_} trees, expected at least {max_iterations}"
        )

    predictions: dict[int, np.ndarray] = {}
    for iterations in iterations_grid:
        pred = model.predict_proba(valid_pool, ntree_start=0, ntree_end=iterations)[:, 1]
        predictions[iterations] = np.asarray(pred, dtype=np.float64)

    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return predictions, y_valid, categorical


def _weighted_summary(candidate_df: pd.DataFrame, fold_weights: dict[str, float]) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["recent_iterations", "stable_iterations", "alpha_recent"]
    for key, group in candidate_df.groupby(group_cols, sort=False):
        weights = np.asarray([fold_weights[name] for name in group["fold"]], dtype=np.float64)
        weights /= weights.sum()
        raw = group["raw_score"].to_numpy(np.float64)
        brier = group["brier"].to_numpy(np.float64)
        delta = group["delta_raw_vs_best_recent"].to_numpy(np.float64)
        weighted_raw = float(np.dot(weights, raw))
        weighted_brier = float(np.dot(weights, brier))
        weighted_delta = float(np.dot(weights, delta))
        weighted_var = float(np.dot(weights, (raw - weighted_raw) ** 2))
        rows.append(
            {
                "recent_iterations": int(key[0]),
                "stable_iterations": int(key[1]),
                "alpha_recent": float(key[2]),
                "weighted_raw_score": weighted_raw,
                "weighted_brier": weighted_brier,
                "weighted_delta_raw_vs_best_recent": weighted_delta,
                "raw_score_std": float(np.sqrt(weighted_var)),
                "worst_raw_score": float(raw.min()),
                "worst_delta_raw_vs_best_recent": float(delta.min()),
                "improved_folds_vs_best_recent": int(np.count_nonzero(delta > 0.0)),
                "fold_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["weighted_raw_score", "worst_raw_score"], ascending=[False, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robust 2025 proxy validation. Combines an inter-season 2023->2024 fold "
            "with two expanding-window 2024 temporal folds; no random split is used."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations-grid", default="100,150,200,250,300,400")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/proxy_2025_validation")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    month_col = "game_month"
    iterations_grid = parse_ints(args.iterations_grid)
    alphas = alpha_grid(args.alpha_step)

    frame, invariant_check = recent_core.prepare_frame(config)
    sort_columns = [season_col, month_col]
    if row_id_col in frame.columns:
        sort_columns.append(row_id_col)
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    recent_features = recent_core.feature_set(RECENT_VARIANT)
    stable_features = recent_core.feature_set(STABLE_VARIANT)
    if set(recent_features) - {"game_type"} != set(stable_features):
        raise RuntimeError("expert feature sets must differ only by game_type")

    fold_weights = {spec.name: spec.weight for spec in DEFAULT_FOLDS}
    if not np.isclose(sum(fold_weights.values()), 1.0):
        raise RuntimeError(f"fold weights must sum to 1: {fold_weights}")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_rows: list[dict] = []
    candidate_rows: list[dict] = []
    fold_diagnostics: list[dict] = []

    print("[2025 Proxy Validation]")
    print("  recent expert: post-2023 regime only + raw game_type")
    print("  stable expert: all available history - game_type")
    print(f"  tree prefixes: {iterations_grid} (one max-tree fit per expert/fold)")
    print(f"  alpha grid: step={args.alpha_step:.3f}")
    print("  selection weights: season-forward=0.50, mid-2024=0.20, late-2024=0.30")

    for spec in DEFAULT_FOLDS:
        recent_mask, stable_mask, valid_mask = fold_masks(frame, spec, season_col, month_col)
        valid_rows = frame.loc[valid_mask]
        if valid_rows.empty:
            raise RuntimeError(f"fold {spec.name} has no validation rows")

        diagnostic = {
            "fold": spec.name,
            "weight": spec.weight,
            "recent_train_rows": int(recent_mask.sum()),
            "stable_train_rows": int(stable_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "valid_target_rate": float(pd.to_numeric(valid_rows[target_col], errors="raise").mean()),
            "valid_month_min": int(pd.to_numeric(valid_rows[month_col], errors="raise").min()),
            "valid_month_max": int(pd.to_numeric(valid_rows[month_col], errors="raise").max()),
        }
        fold_diagnostics.append(diagnostic)
        print(
            f"\n[{spec.name}] weight={spec.weight:.2f} "
            f"recent_train={diagnostic['recent_train_rows']:,} "
            f"stable_train={diagnostic['stable_train_rows']:,} "
            f"valid={diagnostic['valid_rows']:,} months={diagnostic['valid_month_min']}-{diagnostic['valid_month_max']}"
        )

        print("  fitting recent expert...")
        seed_everything(seed)
        recent_preds, y_valid_recent, _ = _fit_prefix_predictions(
            frame=frame,
            train_mask=recent_mask,
            valid_mask=valid_mask,
            features=recent_features,
            target_col=target_col,
            config=config,
            iterations_grid=iterations_grid,
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
        )

        print("  fitting stable expert...")
        seed_everything(seed)
        stable_preds, y_valid_stable, _ = _fit_prefix_predictions(
            frame=frame,
            train_mask=stable_mask,
            valid_mask=valid_mask,
            features=stable_features,
            target_col=target_col,
            config=config,
            iterations_grid=iterations_grid,
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
        )
        if not np.array_equal(y_valid_recent, y_valid_stable):
            raise RuntimeError(f"validation target mismatch in fold {spec.name}")
        y_valid = y_valid_recent

        recent_metrics: dict[int, dict[str, float]] = {}
        stable_metrics: dict[int, dict[str, float]] = {}
        for iterations in iterations_grid:
            rm = probability_metrics(y_valid, recent_preds[iterations])
            sm = probability_metrics(y_valid, stable_preds[iterations])
            recent_metrics[iterations] = rm
            stable_metrics[iterations] = sm
            expert_rows.append({"fold": spec.name, "expert": "recent", "iterations": iterations, **rm})
            expert_rows.append({"fold": spec.name, "expert": "stable", "iterations": iterations, **sm})

        best_recent_iter = min(iterations_grid, key=lambda n: recent_metrics[n]["brier"])
        best_stable_iter = min(iterations_grid, key=lambda n: stable_metrics[n]["brier"])
        best_recent = recent_metrics[best_recent_iter]
        best_stable = stable_metrics[best_stable_iter]
        print(
            f"  recent best: trees={best_recent_iter} brier={best_recent['brier']:.8f} "
            f"raw_score={best_recent['raw_score']:+.2f}"
        )
        print(
            f"  stable best: trees={best_stable_iter} brier={best_stable['brier']:.8f} "
            f"raw_score={best_stable['raw_score']:+.2f}"
        )

        for recent_iterations, p_recent in recent_preds.items():
            for stable_iterations, p_stable in stable_preds.items():
                for alpha in alphas:
                    pred = alpha * p_recent + (1.0 - alpha) * p_stable
                    metrics = probability_metrics(y_valid, pred)
                    candidate_rows.append(
                        {
                            "fold": spec.name,
                            "fold_weight": spec.weight,
                            "recent_iterations": recent_iterations,
                            "stable_iterations": stable_iterations,
                            "alpha_recent": float(alpha),
                            "delta_raw_vs_best_recent": metrics["raw_score"] - best_recent["raw_score"],
                            "delta_raw_vs_best_stable": metrics["raw_score"] - best_stable["raw_score"],
                            "delta_brier_vs_best_recent": metrics["brier"] - best_recent["brier"],
                            "delta_brier_vs_best_stable": metrics["brier"] - best_stable["brier"],
                            **metrics,
                        }
                    )

        del recent_preds, stable_preds, y_valid_recent, y_valid_stable, y_valid
        gc.collect()

    expert_df = pd.DataFrame(expert_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df = _weighted_summary(candidate_df, fold_weights)

    expert_df.to_csv(output_dir / "expert_results.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_results_by_fold.csv", index=False)
    summary_df.to_csv(output_dir / "candidate_summary.csv", index=False)

    best = summary_df.iloc[0].to_dict()
    recommendation = {
        "selection_method": "weighted mean competition raw score across temporal proxy folds",
        "fold_weights": fold_weights,
        "recent_variant": RECENT_VARIANT,
        "stable_variant": STABLE_VARIANT,
        "final_recent_train_seasons": [2023, 2024],
        "final_stable_train_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "recent_iterations": int(best["recent_iterations"]),
        "stable_iterations": int(best["stable_iterations"]),
        "alpha_recent": float(best["alpha_recent"]),
        "weighted_raw_score": float(best["weighted_raw_score"]),
        "weighted_brier": float(best["weighted_brier"]),
        "weighted_delta_raw_vs_best_recent": float(best["weighted_delta_raw_vs_best_recent"]),
        "worst_delta_raw_vs_best_recent": float(best["worst_delta_raw_vs_best_recent"]),
        "improved_folds_vs_best_recent": int(best["improved_folds_vs_best_recent"]),
        "note": "This is a model-selection proxy, not an estimate of the hidden 2025 leaderboard score.",
    }
    save_json(recommendation, output_dir / "recommended_config.json")
    save_json(
        {
            "seed": seed,
            "iterations_grid": iterations_grid,
            "alpha_step": args.alpha_step,
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "folds": [spec.__dict__ for spec in DEFAULT_FOLDS],
            "fold_diagnostics": fold_diagnostics,
            "recent_features": recent_features,
            "stable_features": stable_features,
            "canonical_invariants": invariant_check,
            "training_order": sort_columns,
        },
        output_dir / "run_config.json",
    )

    print("\n[Top robust candidates]")
    print(
        summary_df.head(15)[
            [
                "recent_iterations",
                "stable_iterations",
                "alpha_recent",
                "weighted_raw_score",
                "weighted_delta_raw_vs_best_recent",
                "worst_delta_raw_vs_best_recent",
                "improved_folds_vs_best_recent",
                "raw_score_std",
            ]
        ].to_string(index=False)
    )
    print("\n[Recommended configuration]")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
