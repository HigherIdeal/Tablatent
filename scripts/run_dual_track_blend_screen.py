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


def parse_seasons(value: str) -> list[int]:
    seasons = parse_ints(value)
    if len(seasons) != len(set(seasons)):
        raise ValueError(f"duplicate seasons: {seasons}")
    return sorted(seasons)


def clipped_analytic_alpha(y: np.ndarray, p_recent: np.ndarray, p_stable: np.ndarray) -> float:
    """Brier-optimal alpha for p=alpha*recent + (1-alpha)*stable, clipped to [0, 1]."""
    y = np.asarray(y, dtype=np.float64)
    p_recent = np.asarray(p_recent, dtype=np.float64)
    p_stable = np.asarray(p_stable, dtype=np.float64)
    if y.shape != p_recent.shape or y.shape != p_stable.shape:
        raise ValueError("shape mismatch in blend alpha calculation")
    direction = p_recent - p_stable
    denom = float(np.dot(direction, direction))
    if denom <= 0.0:
        return 0.5
    alpha = float(np.dot(y - p_stable, direction) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def alpha_grid(step: float) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("alpha-step must be in (0, 1]")
    values = np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)
    values = np.unique(np.clip(np.append(values, 1.0), 0.0, 1.0))
    return values


def _prepare_expert_data(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    train_seasons: list[int],
    valid_year: int,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    train = frame.loc[frame[season_col].isin(train_seasons)].copy()
    valid = frame.loc[frame[season_col].eq(valid_year)].copy()
    if train.empty or valid.empty:
        raise RuntimeError(
            f"empty split: train_seasons={train_seasons}, valid_year={valid_year}, "
            f"train_rows={len(train)}, valid_rows={len(valid)}"
        )
    x_train, categorical = context_core.prepare_x(train, features)
    x_valid, valid_categorical = context_core.prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("train/valid categorical feature mismatch")
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    return x_train, y_train, x_valid, y_valid, categorical


def _fit_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    params = context_core.catboost_params(config, iterations, task_type, devices, verbose)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)
    pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)
    del model, train_pool, valid_pool
    gc.collect()
    return pred


def _best_grid_blend(
    y: np.ndarray,
    p_recent: np.ndarray,
    p_stable: np.ndarray,
    alphas: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_alpha = None
    best_metrics = None
    for alpha in alphas:
        pred = alpha * p_recent + (1.0 - alpha) * p_stable
        metrics = probability_metrics(y, pred)
        if best_metrics is None or metrics["brier"] < best_metrics["brier"]:
            best_alpha = float(alpha)
            best_metrics = metrics
    assert best_alpha is not None and best_metrics is not None
    return best_alpha, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dual-track 2024 proxy validation: recent expert learns 2023 with raw game_type; "
            "stable expert learns 2019-2023 with game_type removed; then screen tree counts and blends."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--valid-year", type=int, default=2024)
    parser.add_argument("--recent-train-seasons", default="2023")
    parser.add_argument("--stable-train-seasons", default="2019,2020,2021,2022,2023")
    parser.add_argument("--iterations-grid", default="100,150,200,250,300,400")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/dual_track_blend_screen")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    recent_train_seasons = parse_seasons(args.recent_train_seasons)
    stable_train_seasons = parse_seasons(args.stable_train_seasons)
    iterations_grid = parse_ints(args.iterations_grid)
    alphas = alpha_grid(args.alpha_step)
    valid_year = int(args.valid_year)

    for name, seasons in (
        ("recent", recent_train_seasons),
        ("stable", stable_train_seasons),
    ):
        if any(year >= valid_year for year in seasons):
            raise ValueError(f"{name} train seasons must all be < valid_year={valid_year}: {seasons}")

    frame, invariant_check = recent_core.prepare_frame(config)
    sort_columns = [season_col]
    if "game_month" in frame.columns:
        sort_columns.append("game_month")
    if row_id_col in frame.columns:
        sort_columns.append(row_id_col)
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    recent_features = recent_core.feature_set(RECENT_VARIANT)
    stable_features = recent_core.feature_set(STABLE_VARIANT)
    if set(recent_features) - {"game_type"} != set(stable_features):
        raise RuntimeError("expert feature sets must differ only by game_type")

    recent_data = _prepare_expert_data(
        frame, season_col, target_col, recent_train_seasons, valid_year, recent_features
    )
    stable_data = _prepare_expert_data(
        frame, season_col, target_col, stable_train_seasons, valid_year, stable_features
    )
    xr_train, yr_train, xr_valid, y_valid, recent_categorical = recent_data
    xs_train, ys_train, xs_valid, y_valid_stable, stable_categorical = stable_data
    if not np.array_equal(y_valid, y_valid_stable):
        raise RuntimeError("recent/stable validation target ordering mismatch")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Dual-Track Blend Screen] valid={valid_year:,}; "
        f"recent_train={recent_train_seasons} rows={len(xr_train):,}; "
        f"stable_train={stable_train_seasons} rows={len(xs_train):,}"
    )
    print(
        f"[Dual-Track Blend Screen] iterations={iterations_grid}, alpha_step={args.alpha_step:.3f}, "
        f"task_type={args.task_type}"
    )
    print(
        "[Dual-Track Blend Screen] recent expert keeps raw game_type; "
        "stable expert removes game_type but retains full canonical + success_state otherwise."
    )

    expert_rows: list[dict] = []
    recent_predictions: dict[int, np.ndarray] = {}
    stable_predictions: dict[int, np.ndarray] = {}
    recent_metrics: dict[int, dict[str, float]] = {}
    stable_metrics: dict[int, dict[str, float]] = {}

    print("\n[Recent expert: recent regime + raw game_type]")
    for iterations in iterations_grid:
        seed_everything(seed)
        pred = _fit_predict(
            xr_train, yr_train, xr_valid, recent_features, recent_categorical,
            config, iterations, args.task_type, args.devices, args.verbose,
        )
        metrics = probability_metrics(y_valid, pred)
        recent_predictions[iterations] = pred
        recent_metrics[iterations] = metrics
        expert_rows.append(
            {
                "expert": "recent_raw_game_type",
                "iterations": iterations,
                "train_seasons": ",".join(map(str, recent_train_seasons)),
                "feature_count": len(recent_features),
                **metrics,
            }
        )
        print(
            f"  trees={iterations:>3d} brier={metrics['brier']:.8f} "
            f"raw_score={metrics['raw_score']:+.2f} auc={metrics['auc']:.5f} "
            f"p_std={metrics['prediction_std']:.5f}"
        )

    print("\n[Stable expert: full history - game_type]")
    for iterations in iterations_grid:
        seed_everything(seed)
        pred = _fit_predict(
            xs_train, ys_train, xs_valid, stable_features, stable_categorical,
            config, iterations, args.task_type, args.devices, args.verbose,
        )
        metrics = probability_metrics(y_valid, pred)
        stable_predictions[iterations] = pred
        stable_metrics[iterations] = metrics
        expert_rows.append(
            {
                "expert": "stable_full_drop_game_type",
                "iterations": iterations,
                "train_seasons": ",".join(map(str, stable_train_seasons)),
                "feature_count": len(stable_features),
                **metrics,
            }
        )
        print(
            f"  trees={iterations:>3d} brier={metrics['brier']:.8f} "
            f"raw_score={metrics['raw_score']:+.2f} auc={metrics['auc']:.5f} "
            f"p_std={metrics['prediction_std']:.5f}"
        )

    blend_rows: list[dict] = []
    for recent_iter, p_recent in recent_predictions.items():
        for stable_iter, p_stable in stable_predictions.items():
            alpha = clipped_analytic_alpha(y_valid, p_recent, p_stable)
            pred = alpha * p_recent + (1.0 - alpha) * p_stable
            metrics = probability_metrics(y_valid, pred)
            blend_rows.append(
                {
                    "selection": "analytic",
                    "recent_iterations": recent_iter,
                    "stable_iterations": stable_iter,
                    "alpha_recent": alpha,
                    "delta_vs_recent": metrics["brier"] - recent_metrics[recent_iter]["brier"],
                    "delta_vs_stable": metrics["brier"] - stable_metrics[stable_iter]["brier"],
                    **metrics,
                }
            )

            grid_alpha, grid_metrics = _best_grid_blend(y_valid, p_recent, p_stable, alphas)
            blend_rows.append(
                {
                    "selection": "grid",
                    "recent_iterations": recent_iter,
                    "stable_iterations": stable_iter,
                    "alpha_recent": grid_alpha,
                    "delta_vs_recent": grid_metrics["brier"] - recent_metrics[recent_iter]["brier"],
                    "delta_vs_stable": grid_metrics["brier"] - stable_metrics[stable_iter]["brier"],
                    **grid_metrics,
                }
            )

    expert_df = pd.DataFrame(expert_rows).sort_values(["expert", "brier", "iterations"])
    blend_df = pd.DataFrame(blend_rows).sort_values(["selection", "brier"])
    expert_df.to_csv(output_dir / "expert_results.csv", index=False)
    blend_df.to_csv(output_dir / "blend_results.csv", index=False)

    prediction_payload: dict[str, np.ndarray] = {
        "y": y_valid.astype(np.float32),
    }
    valid_rows = frame.loc[frame[season_col].eq(valid_year)]
    if row_id_col in valid_rows.columns:
        prediction_payload["row_id"] = valid_rows[row_id_col].to_numpy()
    for iterations, pred in recent_predictions.items():
        prediction_payload[f"recent_{iterations}"] = pred.astype(np.float32)
    for iterations, pred in stable_predictions.items():
        prediction_payload[f"stable_{iterations}"] = pred.astype(np.float32)
    np.savez_compressed(output_dir / "validation_predictions.npz", **prediction_payload)

    best_grid = blend_df.loc[blend_df["selection"].eq("grid")].iloc[0].to_dict()
    best_analytic = blend_df.loc[blend_df["selection"].eq("analytic")].iloc[0].to_dict()
    recommendation = {
        "validation_year": valid_year,
        "recent_train_seasons": recent_train_seasons,
        "stable_train_seasons": stable_train_seasons,
        "final_recent_train_seasons": [2023, 2024],
        "final_stable_train_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "recent_variant": RECENT_VARIANT,
        "stable_variant": STABLE_VARIANT,
        "alpha_semantics": "p = alpha_recent * recent + (1 - alpha_recent) * stable",
        "recommended_source": "grid",
        "recent_iterations": int(best_grid["recent_iterations"]),
        "stable_iterations": int(best_grid["stable_iterations"]),
        "alpha_recent": float(best_grid["alpha_recent"]),
        "validation_brier": float(best_grid["brier"]),
        "validation_raw_score": float(best_grid["raw_score"]),
        "best_analytic": {
            "recent_iterations": int(best_analytic["recent_iterations"]),
            "stable_iterations": int(best_analytic["stable_iterations"]),
            "alpha_recent": float(best_analytic["alpha_recent"]),
            "validation_brier": float(best_analytic["brier"]),
            "validation_raw_score": float(best_analytic["raw_score"]),
        },
    }
    save_json(recommendation, output_dir / "recommended_config.json")
    save_json(
        {
            "seed": seed,
            "valid_year": valid_year,
            "recent_train_seasons": recent_train_seasons,
            "stable_train_seasons": stable_train_seasons,
            "iterations_grid": iterations_grid,
            "alpha_step": args.alpha_step,
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "recent_features": recent_features,
            "stable_features": stable_features,
            "canonical_invariants": invariant_check,
            "training_order": sort_columns,
        },
        output_dir / "run_config.json",
    )

    print("\n[Best dual-track blends on validation]")
    display_cols = [
        "selection", "recent_iterations", "stable_iterations", "alpha_recent",
        "brier", "raw_score", "auc", "delta_vs_recent", "delta_vs_stable",
    ]
    print(blend_df.sort_values("brier").head(12)[display_cols].to_string(index=False))
    print("\n[Recommended final configuration: coarse grid]")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
