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
import run_2025_proxy_validation as proxy_core
import run_context_interaction_screen as context_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    policy: str  # full_raw | stable | recent | specialist
    features: tuple[str, ...]
    description: str


def parse_ints(value: str) -> list[int]:
    values = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not values or any(v <= 0 for v in values):
        raise ValueError("iterations-grid must contain positive integers")
    return values


def parse_devices(value: str) -> list[str]:
    values = [x.strip() for x in value.replace(":", ",").split(",") if x.strip()]
    return values or ["0"]


def float_grid(step: float, maximum: float = 1.0) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("grid step must be in (0, 1]")
    if not (0.0 <= maximum <= 1.0):
        raise ValueError("grid maximum must be in [0, 1]")
    values = np.arange(0.0, maximum + step * 0.5, step, dtype=np.float64)
    return np.unique(np.clip(np.append(values, maximum), 0.0, maximum))


def _feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    base_no_gt = [f for f in base_features if f != "game_type"]

    fast = [
        "game_month", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
        "base_state", "li", "pitcher_hand", "batter_hand",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ]
    recent_range = [
        "game_month", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
        "base_state", "li", "pitcher_hand", "batter_hand", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
        "eng_ps_recent_mean_135", "eng_ps_recent_mean_minus_long", "eng_ps_recent_range_135",
        "eng_ps_prev1_minus_long", "eng_ps_prev3_minus_long", "eng_ps_prev5_minus_long",
        "eng_ps_prev1_minus_prev3", "eng_ps_prev3_minus_prev5", "eng_ps_prev1_minus_prev5",
    ]
    both = list(dict.fromkeys(fast + recent_range))
    for name, features in {"r_fast": fast, "r_range": recent_range, "r_both": both}.items():
        missing = sorted(set(features) - set(base_no_gt))
        if missing:
            raise ValueError(f"{name} asks for features not in prepared base feature set: {missing}")
    return {
        "full_raw": list(base_features),
        "stable_drop_gt": base_no_gt,
        "recent_raw": list(base_features),
        "r_full": base_no_gt,
        "r_fast": fast,
        "r_range": recent_range,
        "r_both": both,
    }


def _token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _prepare_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    return context_core.prepare_x(frame, features)


def _fit_prefixes(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_col: str,
    config: dict,
    iterations_grid: list[int],
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = _prepare_x(train, features)
    x_valid, valid_categorical = _prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)

    params = context_core.catboost_params(
        config=config,
        iterations=max(iterations_grid),
        task_type=task_type,
        devices=device,
        verbose=verbose,
    )
    params["thread_count"] = int(thread_count)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)

    predictions: dict[int, np.ndarray] = {}
    for iterations in iterations_grid:
        predictions[int(iterations)] = np.asarray(
            model.predict_proba(valid_pool, ntree_start=0, ntree_end=int(iterations))[:, 1],
            dtype=np.float64,
        )
    stats = {
        "train_rows": int(len(train)),
        "feature_count": int(len(features)),
        "categorical_count": int(len(categorical)),
        "target_rate": float(y_train.mean()),
    }
    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return predictions, stats


def _cache_path(cache_dir: Path, fold: str, expert: str) -> Path:
    return cache_dir / f"{fold}__{expert}.npz"


def save_prediction_cache(path: Path, predictions: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"p_{k}": v.astype(np.float32) for k, v in predictions.items()})


def load_prediction_cache(path: Path, iterations_grid: list[int], expected_rows: int) -> dict[int, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            out = {}
            for iterations in iterations_grid:
                key = f"p_{iterations}"
                if key not in data or len(data[key]) != expected_rows:
                    return None
                out[int(iterations)] = np.asarray(data[key], dtype=np.float64)
        return out
    except Exception:
        return None


def gated_prediction(
    p_old: np.ndarray,
    p_recent: np.ndarray,
    p_specialist: np.ndarray,
    is_r: np.ndarray,
    alpha_recent: float,
    beta_r: float,
) -> np.ndarray:
    base = (1.0 - alpha_recent) * p_old + alpha_recent * p_recent
    out = base.copy()
    out[is_r] = (1.0 - beta_r) * base[is_r] + beta_r * p_specialist[is_r]
    return out


def _moments(y: np.ndarray, predictions: list[np.ndarray], mask: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, np.nan, np.full(len(predictions), np.nan), np.full((len(predictions), len(predictions)), np.nan)
    yy = y[mask].astype(np.float64)
    pp = np.column_stack([p[mask] for p in predictions]).astype(np.float64)
    n = int(len(yy))
    y2 = float(np.mean(yy * yy))
    yp = np.mean(yy[:, None] * pp, axis=0)
    gram = (pp.T @ pp) / float(n)
    return n, y2, yp, gram


def quadratic_brier(moment: tuple[int, float, np.ndarray, np.ndarray], coeff: np.ndarray) -> float:
    n, y2, yp, gram = moment
    if n <= 0:
        return float("nan")
    coeff = np.asarray(coeff, dtype=np.float64)
    return float(y2 - 2.0 * np.dot(coeff, yp) + coeff @ gram @ coeff)


def _score_from_brier(brier: float, target_mean: float) -> float:
    ref = float(target_mean * (1.0 - target_mean))
    return float(100000.0 * (1.0 - brier / ref)) if ref > 0 else float("nan")


def evaluate_gated_grid(
    *,
    y: np.ndarray,
    is_r: np.ndarray,
    is_f: np.ndarray,
    p_old: np.ndarray,
    p_recent: np.ndarray,
    p_specialist: np.ndarray,
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
) -> list[dict[str, float]]:
    non_r = ~is_r
    r_moment = _moments(y, [p_old, p_recent, p_specialist], is_r)
    nr_moment = _moments(y, [p_old, p_recent, p_specialist], non_r)
    f_moment = _moments(y, [p_old, p_recent, p_specialist], is_f)
    n_total = len(y)
    target_mean = float(np.mean(y))
    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        coeff_nr = np.asarray([1.0 - alpha, alpha, 0.0], dtype=np.float64)
        brier_nr = quadratic_brier(nr_moment, coeff_nr)
        f_brier = quadratic_brier(f_moment, coeff_nr)
        for beta in beta_values:
            coeff_r = np.asarray(
                [(1.0 - beta) * (1.0 - alpha), (1.0 - beta) * alpha, beta],
                dtype=np.float64,
            )
            r_brier = quadratic_brier(r_moment, coeff_r)
            overall = (r_moment[0] * r_brier + nr_moment[0] * brier_nr) / float(n_total)
            rows.append(
                {
                    "alpha_recent": float(alpha),
                    "beta_r": float(beta),
                    "brier": float(overall),
                    "raw_score": _score_from_brier(float(overall), target_mean),
                    "r_brier": float(r_brier),
                    "f_brier": float(f_brier),
                }
            )
    return rows


def _weighted_summary(rows: pd.DataFrame, fold_weights: dict[str, float]) -> pd.DataFrame:
    key = [
        "base_mode", "specialist", "old_iterations", "recent_iterations",
        "specialist_iterations", "alpha_recent", "beta_r",
    ]
    summary_rows: list[dict] = []
    for config, group in rows.groupby(key, sort=False):
        weights = np.asarray([fold_weights[str(x)] for x in group["fold"]], dtype=np.float64)
        weights /= weights.sum()
        brier = group["brier"].to_numpy(np.float64)
        raw = group["raw_score"].to_numpy(np.float64)
        r_brier = group["r_brier"].to_numpy(np.float64)
        f_brier = group["f_brier"].to_numpy(np.float64)
        summary_rows.append(
            {
                **dict(zip(key, config)),
                "folds": int(group["fold"].nunique()),
                "weighted_brier": float(np.dot(weights, brier)),
                "weighted_raw_score": float(np.dot(weights, raw)),
                "weighted_r_brier": float(np.dot(weights, r_brier)),
                "weighted_f_brier": float(np.dot(weights, f_brier)),
                "worst_brier": float(brier.max()),
                "best_brier": float(brier.min()),
            }
        )
    out = pd.DataFrame(summary_rows)
    if out.empty:
        return out
    return out.sort_values(["weighted_brier", "worst_brier"]).reset_index(drop=True)


def _expert_weighted_summary(expert_rows: pd.DataFrame, fold_weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (expert, iterations), group in expert_rows.groupby(["expert", "iterations"], sort=False):
        weights = np.asarray([fold_weights[str(x)] for x in group["fold"]], dtype=np.float64)
        weights /= weights.sum()
        rows.append(
            {
                "expert": expert,
                "iterations": int(iterations),
                "weighted_brier": float(np.dot(weights, group["brier"].to_numpy(np.float64))),
                "weighted_raw_score": float(np.dot(weights, group["raw_score"].to_numpy(np.float64))),
                "weighted_r_brier": float(np.dot(weights, group["r_brier"].to_numpy(np.float64))),
                "weighted_f_brier": float(np.dot(weights, group["f_brier"].to_numpy(np.float64))),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_brier").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train general/recent/R-specialist CatBoost experts separately, then apply the specialist only on "
            "game_type=R through a gated mixture. Tree prefixes are reused; ensemble grids use quadratic Brier "
            "moments so thousands of weight combinations are cheap."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations-grid", default="200,300,400,500")
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--beta-step", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=0.8)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=6)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/gated_r_specialist_suite")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    iterations_grid = parse_ints(args.iterations_grid)
    alpha_values = float_grid(args.alpha_step, 1.0)
    beta_values = float_grid(args.beta_step, args.beta_max)
    devices = parse_devices(args.devices)

    frame, invariant_check = recent_core.prepare_frame(config)
    sort_cols = [season_col, "game_month"] + ([row_id_col] if row_id_col in frame.columns else [])
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    base_features = recent_core.feature_set("recent_raw_game_type")
    feature_sets = _feature_sets(base_features)

    output_dir = (ROOT / args.output_dir).resolve()
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[Gated R-Specialist Suite]")
    print(f"  catboost          : {catboost.__version__}")
    print(f"  task_type         : {args.task_type}")
    print(f"  devices           : {devices}")
    print(f"  tree prefixes     : {iterations_grid} (one max-tree fit per expert/fold)")
    print(f"  alpha recent grid : {alpha_values.tolist()}")
    print(f"  beta R gate grid  : {beta_values.tolist()}")
    print("  base modes        : stable_drop_gt + recent_raw ; full_raw + recent_raw")
    print("  R specialists     : r_full, r_fast, r_range, r_both (trained on recent R only)")
    print("  gate semantics    : F/non-R stays base; R=(1-beta)*base + beta*specialist")
    print("  checkpoint/resume : prediction cache per expert/fold")

    fold_weights = {spec.name: spec.weight for spec in proxy_core.DEFAULT_FOLDS}
    all_expert_rows: list[dict] = []
    all_ensemble_rows: list[dict] = []
    diagnostics: list[dict] = []

    for fold_spec in proxy_core.DEFAULT_FOLDS:
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(
            frame, fold_spec, season_col, "game_month"
        )
        valid = frame.loc[valid_mask].copy()
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt = _token(valid["game_type"]).to_numpy()
        is_r = gt == "R"
        is_f = gt == "F"
        recent_r_mask = recent_mask & _token(frame["game_type"]).eq("R")

        train_masks = {
            "full_raw": full_mask,
            "stable_drop_gt": full_mask,
            "recent_raw": recent_mask,
            "r_full": recent_r_mask,
            "r_fast": recent_r_mask,
            "r_range": recent_r_mask,
            "r_both": recent_r_mask,
        }
        print(
            f"\n[Fold {fold_spec.name}] valid={len(valid):,} rate={y.mean():.6f} "
            f"R={int(is_r.sum()):,} F={int(is_f.sum()):,} recentR_train={int(recent_r_mask.sum()):,}"
        )

        predictions_by_expert: dict[str, dict[int, np.ndarray]] = {}
        expert_stats: dict[str, dict[str, object]] = {}
        for job_index, expert in enumerate(train_masks):
            cache = _cache_path(cache_dir, fold_spec.name, expert)
            cached = None if args.no_resume else load_prediction_cache(cache, iterations_grid, len(valid))
            if cached is not None:
                predictions_by_expert[expert] = cached
                print(f"  cache {expert:<15s} -> {cache.name}")
                continue

            train = frame.loc[train_masks[expert]].copy()
            device = devices[job_index % len(devices)] if args.task_type == "GPU" else "CPU"
            print(
                f"  fit   {expert:<15s} device={device:<3s} train={len(train):,} features={len(feature_sets[expert])}",
                flush=True,
            )
            seed_everything(seed)
            predictions, stats = _fit_prefixes(
                train=train,
                valid=valid,
                features=feature_sets[expert],
                target_col=target_col,
                config=config,
                iterations_grid=iterations_grid,
                task_type=args.task_type,
                device=device,
                verbose=args.verbose,
                thread_count=args.thread_count,
            )
            predictions_by_expert[expert] = predictions
            expert_stats[expert] = stats
            save_prediction_cache(cache, predictions)
            del train
            gc.collect()

        # Individual expert diagnostics. Specialists are evaluated on all rows for bookkeeping,
        # but their meaningful score is R-only because they are only ever gated onto R.
        for expert, pred_map in predictions_by_expert.items():
            for iterations, pred in pred_map.items():
                metric = probability_metrics(y, pred)
                all_expert_rows.append(
                    {
                        "fold": fold_spec.name,
                        "expert": expert,
                        "iterations": int(iterations),
                        "brier": metric["brier"],
                        "raw_score": metric["raw_score"],
                        "r_brier": float(np.mean((y[is_r] - pred[is_r]) ** 2)),
                        "f_brier": float(np.mean((y[is_f] - pred[is_f]) ** 2)) if is_f.any() else np.nan,
                    }
                )

        # Ensemble grid. The expensive work is only a 3x3 moment matrix per tree triplet;
        # alpha/beta sweeps do not revisit every validation row.
        for base_mode, old_expert in [("stable_recent", "stable_drop_gt"), ("full_recent", "full_raw")]:
            for specialist in ["r_full", "r_fast", "r_range", "r_both"]:
                for old_iter in iterations_grid:
                    p_old = predictions_by_expert[old_expert][old_iter]
                    for recent_iter in iterations_grid:
                        p_recent = predictions_by_expert["recent_raw"][recent_iter]
                        for specialist_iter in iterations_grid:
                            p_spec = predictions_by_expert[specialist][specialist_iter]
                            grid_rows = evaluate_gated_grid(
                                y=y,
                                is_r=is_r,
                                is_f=is_f,
                                p_old=p_old,
                                p_recent=p_recent,
                                p_specialist=p_spec,
                                alpha_values=alpha_values,
                                beta_values=beta_values,
                            )
                            for row in grid_rows:
                                all_ensemble_rows.append(
                                    {
                                        "fold": fold_spec.name,
                                        "base_mode": base_mode,
                                        "specialist": specialist,
                                        "old_iterations": int(old_iter),
                                        "recent_iterations": int(recent_iter),
                                        "specialist_iterations": int(specialist_iter),
                                        **row,
                                    }
                                )

        fold_ensemble = pd.DataFrame([x for x in all_ensemble_rows if x["fold"] == fold_spec.name])
        fold_best = fold_ensemble.nsmallest(5, "brier")
        print("  [fold best gated]")
        print(
            fold_best[
                ["base_mode", "specialist", "old_iterations", "recent_iterations", "specialist_iterations",
                 "alpha_recent", "beta_r", "brier", "raw_score", "r_brier", "f_brier"]
            ].to_string(index=False, formatters={"brier": "{:.8f}".format, "raw_score": "{:+.2f}".format})
        )
        diagnostics.append(
            {
                "fold": fold_spec.name,
                "weight": float(fold_spec.weight),
                "valid_rows": int(len(valid)),
                "target_rate": float(y.mean()),
                "r_rows": int(is_r.sum()),
                "f_rows": int(is_f.sum()),
                "recent_r_train_rows": int(recent_r_mask.sum()),
            }
        )
        pd.DataFrame(all_expert_rows).to_csv(output_dir / "expert_results_checkpoint.csv", index=False)
        pd.DataFrame(all_ensemble_rows).to_csv(output_dir / "ensemble_results_checkpoint.csv", index=False)

    expert_df = pd.DataFrame(all_expert_rows)
    ensemble_df = pd.DataFrame(all_ensemble_rows)
    expert_df.to_csv(output_dir / "expert_results.csv", index=False)
    ensemble_df.to_csv(output_dir / "ensemble_results.csv", index=False)

    expert_summary = _expert_weighted_summary(expert_df, fold_weights)
    ensemble_summary = _weighted_summary(ensemble_df, fold_weights)

    # Compare every gated configuration with the best fixed ungated configuration in the same base family.
    ungated = ensemble_summary.loc[ensemble_summary["beta_r"].eq(0.0)]
    ungated_best = ungated.groupby("base_mode", as_index=False)["weighted_brier"].min().rename(
        columns={"weighted_brier": "best_ungated_brier"}
    )
    ensemble_summary = ensemble_summary.merge(ungated_best, on="base_mode", how="left")
    ensemble_summary["delta_vs_best_ungated"] = (
        ensemble_summary["weighted_brier"] - ensemble_summary["best_ungated_brier"]
    )
    recent_best = float(
        expert_summary.loc[expert_summary["expert"].eq("recent_raw"), "weighted_brier"].min()
    )
    ensemble_summary["delta_vs_best_recent"] = ensemble_summary["weighted_brier"] - recent_best
    ensemble_summary = ensemble_summary.sort_values(["weighted_brier", "worst_brier"]).reset_index(drop=True)

    expert_summary.to_csv(output_dir / "expert_summary.csv", index=False)
    ensemble_summary.to_csv(output_dir / "ensemble_summary.csv", index=False)
    best_per_specialist = (
        ensemble_summary.sort_values("weighted_brier")
        .groupby(["base_mode", "specialist"], as_index=False, sort=False)
        .first()
        .sort_values("weighted_brier")
    )
    best_per_specialist.to_csv(output_dir / "best_per_specialist.csv", index=False)

    print("\n[Expert Summary: fixed tree count across folds]")
    print(
        expert_summary.groupby("expert", as_index=False, sort=False).first()[
            ["expert", "iterations", "weighted_brier", "weighted_raw_score", "weighted_r_brier", "weighted_f_brier"]
        ].sort_values("weighted_brier").to_string(
            index=False,
            formatters={
                "weighted_brier": "{:.8f}".format,
                "weighted_raw_score": "{:+.2f}".format,
                "weighted_r_brier": "{:.8f}".format,
                "weighted_f_brier": "{:.8f}".format,
            },
        )
    )

    print("\n[Best Gated Configuration per Specialist]")
    print(
        best_per_specialist[
            ["base_mode", "specialist", "old_iterations", "recent_iterations", "specialist_iterations",
             "alpha_recent", "beta_r", "weighted_brier", "weighted_raw_score", "weighted_r_brier",
             "weighted_f_brier", "delta_vs_best_ungated", "delta_vs_best_recent"]
        ].to_string(
            index=False,
            formatters={
                "weighted_brier": "{:.8f}".format,
                "weighted_raw_score": "{:+.2f}".format,
                "weighted_r_brier": "{:.8f}".format,
                "weighted_f_brier": "{:.8f}".format,
                "delta_vs_best_ungated": "{:+.8f}".format,
                "delta_vs_best_recent": "{:+.8f}".format,
            },
        )
    )

    print("\n[Top 15 Fixed Gated Configurations]")
    print(
        ensemble_summary.head(15)[
            ["base_mode", "specialist", "old_iterations", "recent_iterations", "specialist_iterations",
             "alpha_recent", "beta_r", "weighted_brier", "weighted_raw_score", "weighted_r_brier",
             "weighted_f_brier", "delta_vs_best_ungated", "delta_vs_best_recent"]
        ].to_string(
            index=False,
            formatters={
                "weighted_brier": "{:.8f}".format,
                "weighted_raw_score": "{:+.2f}".format,
                "weighted_r_brier": "{:.8f}".format,
                "weighted_f_brier": "{:.8f}".format,
                "delta_vs_best_ungated": "{:+.8f}".format,
                "delta_vs_best_recent": "{:+.8f}".format,
            },
        )
    )

    save_json(
        {
            "seed": seed,
            "iterations_grid": iterations_grid,
            "alpha_values": alpha_values.tolist(),
            "beta_values": beta_values.tolist(),
            "task_type": args.task_type,
            "devices": devices if args.task_type == "GPU" else None,
            "thread_count": int(args.thread_count),
            "fold_weights": fold_weights,
            "feature_sets": feature_sets,
            "fold_diagnostics": diagnostics,
            "canonical_invariants": invariant_check,
            "semantics": {
                "stable_recent": "base=(1-alpha)*stable_drop_game_type + alpha*recent_raw_game_type",
                "full_recent": "base=(1-alpha)*full_raw_game_type + alpha*recent_raw_game_type",
                "r_gate": "non-R=base; R=(1-beta)*base + beta*R-specialist",
                "specialist_training": "recent temporal mask AND game_type=R only",
            },
        },
        output_dir / "run_config.json",
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
