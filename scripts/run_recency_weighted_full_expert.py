from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy_core
import run_context_interaction_screen as context_core
import run_gated_r_specialist_suite as gated_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


def parse_floats(value: str) -> list[float]:
    out = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not out or any(x < 0 for x in out):
        raise ValueError("half-lives must be non-negative; 0 means unweighted")
    return out


def _token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _row_time(frame: pd.DataFrame) -> np.ndarray:
    season = pd.to_numeric(frame["season"], errors="raise").to_numpy(np.int64)
    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(np.int64)
    return season * 12 + (month - 1)


def recency_weights(train: pd.DataFrame, half_life_months: float) -> np.ndarray | None:
    if half_life_months <= 0:
        return None
    t = _row_time(train)
    cutoff = int(t.max())
    age = np.maximum(cutoff - t, 0).astype(np.float64)
    w = np.power(0.5, age / float(half_life_months))
    mean = float(w.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise RuntimeError("invalid recency weights")
    return (w / mean).astype(np.float64)


def fit_predict(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_col: str,
    config: dict,
    iterations: int,
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = context_core.prepare_x(train, features)
    x_valid, valid_cat = context_core.prepare_x(valid, features)
    if categorical != valid_cat:
        raise RuntimeError("categorical mismatch")
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    if sample_weight is not None and len(sample_weight) != len(train):
        raise ValueError("sample_weight row mismatch")
    params = context_core.catboost_params(
        config=config,
        iterations=iterations,
        task_type=task_type,
        devices=device,
        verbose=verbose,
    )
    params["thread_count"] = int(thread_count)
    train_pool = Pool(
        x_train,
        label=y_train,
        weight=sample_weight,
        cat_features=categorical,
        feature_names=features,
    )
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)
    pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)
    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return pred


def _blend(
    p_full: np.ndarray,
    p_recent: np.ndarray,
    p_r: np.ndarray,
    is_r: np.ndarray,
    alpha_recent: float,
    beta_r: float,
) -> np.ndarray:
    base = (1.0 - alpha_recent) * p_full + alpha_recent * p_recent
    out = base.copy()
    out[is_r] = (1.0 - beta_r) * base[is_r] + beta_r * p_r[is_r]
    return np.clip(out, 0.0, 1.0)


def _subset_brier(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean((np.asarray(y)[mask] - np.asarray(p)[mask]) ** 2))


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Test smooth recency weighting only on the full-history raw-game_type CatBoost expert while holding "
            "the recent and R-fast experts plus ensemble weights fixed. All weighting is train-time only."
        )
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--half-life-months", default="0,12,24,36,60")
    p.add_argument("--alpha-recent", type=float, default=0.20)
    p.add_argument("--beta-r", type=float, default=0.10)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--thread-count", type=int, default=6)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/recency_weighted_full_expert")
    args = p.parse_args()

    if not (0.0 <= args.alpha_recent <= 1.0 and 0.0 <= args.beta_r <= 1.0):
        raise ValueError("ensemble weights must be in [0,1]")
    half_lives = parse_floats(args.half_life_months)
    if 0.0 not in half_lives:
        half_lives = [0.0] + half_lives

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target_col = config["data"]["target_col"]
    frame, invariant_check = recent_core.prepare_frame(config)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_month"] = pd.to_numeric(frame["game_month"], errors="raise").astype(int)

    base_features = recent_core.feature_set("recent_raw_game_type")
    r_fast_features = gated_core._feature_sets(base_features)["r_fast"]
    devices = gated_core.parse_devices(args.devices)
    device = devices[0] if args.task_type == "GPU" else "CPU"
    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Recency-Weighted Full Expert]")
    print(f"  half_life_months={half_lives} iterations={args.iterations}")
    print(f"  alpha_recent={args.alpha_recent:.3f} beta_r={args.beta_r:.3f}")
    print("  only full-history expert changes; recent and R-fast experts remain fixed")

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    fold_weights = {spec.name: float(spec.weight) for spec in proxy_core.DEFAULT_FOLDS}

    for spec in proxy_core.DEFAULT_FOLDS:
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(frame, spec, "season", "game_month")
        full_train = frame.loc[full_mask].copy()
        recent_train = frame.loc[recent_mask].copy()
        valid = frame.loc[valid_mask].copy()
        recent_r_mask = recent_mask & _token(frame["game_type"]).eq("R")
        r_train = frame.loc[recent_r_mask].copy()
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        is_r = _token(valid["game_type"]).eq("R").to_numpy()
        is_f = _token(valid["game_type"]).eq("F").to_numpy()

        # These two experts are deliberately trained once per fold and then held fixed.
        p_recent = fit_predict(
            train=recent_train, valid=valid, features=base_features, target_col=target_col,
            config=config, iterations=args.iterations, task_type=args.task_type, device=device,
            verbose=args.verbose, thread_count=args.thread_count,
        )
        p_r = fit_predict(
            train=r_train, valid=valid, features=r_fast_features, target_col=target_col,
            config=config, iterations=args.iterations, task_type=args.task_type, device=device,
            verbose=args.verbose, thread_count=args.thread_count,
        )

        print(f"\n  [{spec.name}] full={len(full_train):,} recent={len(recent_train):,} valid={len(valid):,}")
        baseline_brier = None
        for half_life in half_lives:
            weights = recency_weights(full_train, half_life)
            p_full = fit_predict(
                train=full_train, valid=valid, features=base_features, target_col=target_col,
                config=config, iterations=args.iterations, task_type=args.task_type, device=device,
                verbose=args.verbose, thread_count=args.thread_count, sample_weight=weights,
            )
            pred = _blend(p_full, p_recent, p_r, is_r, args.alpha_recent, args.beta_r)
            metric = probability_metrics(y, pred)
            if half_life == 0.0:
                baseline_brier = float(metric["brier"])
            if baseline_brier is None:
                raise RuntimeError("unweighted half-life=0 must be evaluated first")
            row = {
                "fold": spec.name,
                "weight": float(spec.weight),
                "half_life_months": float(half_life),
                "brier": float(metric["brier"]),
                "raw_score": float(metric["raw_score"]),
                "delta_brier_vs_unweighted": float(metric["brier"] - baseline_brier),
                "r_brier": _subset_brier(y, pred, is_r),
                "f_brier": _subset_brier(y, pred, is_f),
                "full_prediction_mean": float(p_full.mean()),
            }
            rows.append(row)
            if weights is None:
                w_min = w_p10 = w_p50 = w_p90 = w_max = 1.0
            else:
                w_min = float(weights.min())
                w_p10 = float(np.quantile(weights, 0.10))
                w_p50 = float(np.quantile(weights, 0.50))
                w_p90 = float(np.quantile(weights, 0.90))
                w_max = float(weights.max())
            weight_rows.append(
                {
                    "fold": spec.name,
                    "half_life_months": float(half_life),
                    "weight_min": w_min,
                    "weight_p10": w_p10,
                    "weight_p50": w_p50,
                    "weight_p90": w_p90,
                    "weight_max": w_max,
                }
            )
            print(
                f"    half_life={half_life:>5g} brier={row['brier']:.8f} "
                f"raw={row['raw_score']:+.2f} dBrier={row['delta_brier_vs_unweighted']:+.8f}"
            )
            del p_full, pred, weights
            gc.collect()

        del full_train, recent_train, r_train, valid, p_recent, p_r
        gc.collect()

    folds = pd.DataFrame(rows)
    folds.to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(outdir / "weight_diagnostics.csv", index=False)

    summary_rows = []
    for half_life, group in folds.groupby("half_life_months", sort=True):
        weights = np.asarray([fold_weights[str(f)] for f in group["fold"]], dtype=np.float64)
        weights /= weights.sum()
        brier = group["brier"].to_numpy(np.float64)
        delta = group["delta_brier_vs_unweighted"].to_numpy(np.float64)
        raw = group["raw_score"].to_numpy(np.float64)
        summary_rows.append(
            {
                "half_life_months": float(half_life),
                "weighted_brier": float(np.dot(weights, brier)),
                "weighted_raw_score": float(np.dot(weights, raw)),
                "weighted_delta_brier_vs_unweighted": float(np.dot(weights, delta)),
                "worst_delta_brier_vs_unweighted": float(delta.max()),
                "best_delta_brier_vs_unweighted": float(delta.min()),
                "improved_folds": int(np.count_nonzero(delta < 0.0)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("weighted_brier").reset_index(drop=True)
    summary.to_csv(outdir / "summary.csv", index=False)

    metadata = {
        "experiment": "recency_weighted_full_expert",
        "row_independent_inference": True,
        "weight_formula": "w_i = 0.5 ** (age_months / half_life_months), normalized to mean 1; half_life=0 is unweighted",
        "weighted_branch": "full_raw only",
        "recent_branch": "unchanged",
        "r_fast_branch": "unchanged",
        "iterations": int(args.iterations),
        "alpha_recent": float(args.alpha_recent),
        "beta_r": float(args.beta_r),
        "half_life_months": half_lives,
        "canonical_invariant_check": invariant_check,
    }
    save_json(metadata, outdir / "metadata.json")

    print("\n[Summary]")
    print(summary.to_string(index=False, formatters={
        "weighted_brier": "{:.8f}".format,
        "weighted_raw_score": "{:+.2f}".format,
        "weighted_delta_brier_vs_unweighted": "{:+.8f}".format,
        "worst_delta_brier_vs_unweighted": "{:+.8f}".format,
        "best_delta_brier_vs_unweighted": "{:+.8f}".format,
    }))
    print("\nPromotion rule: require weighted improvement with no material proxy-fold regression; otherwise keep the unweighted full expert.")
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
