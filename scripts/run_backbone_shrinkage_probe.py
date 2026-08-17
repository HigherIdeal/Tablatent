from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# Project server policy: physical GPU 2 -> process-local device 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy_core
import run_gated_r_specialist_suite as gated_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, seed_everything


def parse_floats(value: str) -> list[float]:
    out = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("at least one alpha is required")
    if any(x < 0.0 or x > 1.5 for x in out):
        raise ValueError("shrink alphas must be in [0, 1.5]")
    if 1.0 not in out:
        out.append(1.0)
        out.sort()
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def oracle_alpha(y: np.ndarray, p: np.ndarray, center: float) -> float:
    """Validation-label optimum for diagnosis only; never deploy this per-fold value."""
    d = np.asarray(p, dtype=np.float64) - float(center)
    denom = float(np.dot(d, d))
    if denom <= 0.0:
        return 0.0
    a = float(np.dot(d, np.asarray(y, dtype=np.float64) - float(center)) / denom)
    return float(np.clip(a, 0.0, 1.5))


def shrink(p: np.ndarray, center: float, alpha: float) -> np.ndarray:
    return np.clip(float(center) + float(alpha) * (np.asarray(p, dtype=np.float64) - float(center)), 0.0, 1.0)


def load_or_fit(
    *,
    cache_dir: Path,
    fold: str,
    expert: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_col: str,
    config: dict,
    iterations: int,
    task_type: str,
    device: str,
    thread_count: int,
    verbose: int,
    no_resume: bool,
) -> np.ndarray:
    cache = gated_core._cache_path(cache_dir, fold, expert)
    cached = None if no_resume else gated_core.load_prediction_cache(cache, [iterations], len(valid))
    if cached is not None:
        print(f"  cache {expert:<12s} -> {cache.name}")
        return np.asarray(cached[iterations], dtype=np.float64)

    print(f"  fit   {expert:<12s} train={len(train):,} features={len(features)}", flush=True)
    pred_map, _ = gated_core._fit_prefixes(
        train=train,
        valid=valid,
        features=features,
        target_col=target_col,
        config=config,
        iterations_grid=[iterations],
        task_type=task_type,
        device=device,
        verbose=verbose,
        thread_count=thread_count,
    )
    gated_core.save_prediction_cache(cache, pred_map)
    return np.asarray(pred_map[iterations], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration/shrinkage probe for the current deployable CatBoost backbone: "
            "full_raw + recent_raw + R-fast with fixed alpha_recent=0.20, beta_r=0.10."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--alpha-recent", type=float, default=0.20)
    parser.add_argument("--beta-r", type=float, default=0.10)
    parser.add_argument(
        "--shrink-alphas",
        default="0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
    )
    parser.add_argument("--center", choices=["train_prior", "half", "valid_mean_diagnostic"], default="train_prior")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=6)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/backbone_shrinkage_probe")
    parser.add_argument(
        "--cache-dir",
        default="outputs/gated_r_specialist_suite/cache",
        help="Reuse existing exact expert predictions when available.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.alpha_recent <= 1.0:
        raise ValueError("alpha-recent must be in [0,1]")
    if not 0.0 <= args.beta_r <= 1.0:
        raise ValueError("beta-r must be in [0,1]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    alphas = parse_floats(args.shrink_alphas)

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    base_features = recent_core.feature_set("recent_raw_game_type")
    feature_sets = gated_core._feature_sets(base_features)
    r_fast_features = feature_sets["r_fast"]

    cache_dir = (ROOT / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Backbone shrinkage probe]")
    print(
        f"  backbone: full_raw + recent_raw + r_fast | trees={args.iterations} "
        f"alpha_recent={args.alpha_recent:.2f} beta_r={args.beta_r:.2f}"
    )
    print(f"  shrink center={args.center} alphas={alphas}")
    print("  train_prior/half centers are row-independent and deployable; valid_mean_diagnostic is NOT deployable.")

    fold_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    fold_weights = {spec.name: float(spec.weight) for spec in proxy_core.DEFAULT_FOLDS}

    for spec in proxy_core.DEFAULT_FOLDS:
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(frame, spec, season_col, "game_month")
        valid = frame.loc[valid_mask].copy()
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        is_r = valid["game_type"].astype("string").fillna("<MISSING>").astype(str).eq("R").to_numpy()
        recent_r_mask = recent_mask & frame["game_type"].astype("string").fillna("<MISSING>").astype(str).eq("R")

        full_train = frame.loc[full_mask].copy()
        recent_train = frame.loc[recent_mask].copy()
        r_train = frame.loc[recent_r_mask].copy()

        print(f"\n[Fold {spec.name}] valid={len(valid):,} rate={y.mean():.6f}")
        p_full = load_or_fit(
            cache_dir=cache_dir, fold=spec.name, expert="full_raw", train=full_train, valid=valid,
            features=base_features, target_col=target_col, config=config, iterations=args.iterations,
            task_type=args.task_type, device=args.devices, thread_count=args.thread_count,
            verbose=args.verbose, no_resume=args.no_resume,
        )
        p_recent = load_or_fit(
            cache_dir=cache_dir, fold=spec.name, expert="recent_raw", train=recent_train, valid=valid,
            features=base_features, target_col=target_col, config=config, iterations=args.iterations,
            task_type=args.task_type, device=args.devices, thread_count=args.thread_count,
            verbose=args.verbose, no_resume=args.no_resume,
        )
        p_r = load_or_fit(
            cache_dir=cache_dir, fold=spec.name, expert="r_fast", train=r_train, valid=valid,
            features=r_fast_features, target_col=target_col, config=config, iterations=args.iterations,
            task_type=args.task_type, device=args.devices, thread_count=args.thread_count,
            verbose=args.verbose, no_resume=args.no_resume,
        )

        p_fixed = gated_core.gated_prediction(
            p_full, p_recent, p_r, is_r, args.alpha_recent, args.beta_r
        )
        raw_metrics = probability_metrics(y, p_fixed)

        if args.center == "train_prior":
            center = float(pd.to_numeric(full_train[target_col], errors="raise").mean())
        elif args.center == "half":
            center = 0.5
        else:
            center = float(p_fixed.mean())

        a_oracle = oracle_alpha(y, p_fixed, center)
        p_oracle = shrink(p_fixed, center, a_oracle)
        print(
            f"  fixed: brier={raw_metrics['brier']:.8f} score={raw_metrics['raw_score']:+.2f} "
            f"mean={p_fixed.mean():.5f} p_std={p_fixed.std():.5f} center={center:.5f}"
        )
        print(
            f"  oracle diagnostic: alpha={a_oracle:.4f} brier={brier(y, p_oracle):.8f} "
            f"delta={brier(y, p_oracle) - raw_metrics['brier']:+.8f}"
        )

        for alpha in alphas:
            pred = shrink(p_fixed, center, alpha)
            metrics = probability_metrics(y, pred)
            fold_rows.append(
                {
                    "fold": spec.name,
                    "weight": float(spec.weight),
                    "alpha": float(alpha),
                    "center_mode": args.center,
                    "center": center,
                    "oracle_alpha": a_oracle,
                    "brier": float(metrics["brier"]),
                    "raw_score": float(metrics["raw_score"]),
                    "auc": float(metrics["auc"]),
                    "prediction_mean": float(metrics["prediction_mean"]),
                    "prediction_std": float(metrics["prediction_std"]),
                    "delta_brier_vs_fixed": float(metrics["brier"] - raw_metrics["brier"]),
                    "valid_rows": int(len(valid)),
                }
            )
            print(
                f"    alpha={alpha:>4.2f} brier={metrics['brier']:.8f} "
                f"delta={metrics['brier'] - raw_metrics['brier']:+.8f} "
                f"p_std={metrics['prediction_std']:.5f}"
            )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "fold": spec.name,
                    "target": y,
                    "p_full": p_full,
                    "p_recent": p_recent,
                    "p_r_fast": p_r,
                    "is_r": is_r.astype(np.int8),
                    "p_fixed": p_fixed,
                }
            )
        )

        del valid, full_train, recent_train, r_train, p_full, p_recent, p_r, p_fixed, p_oracle
        gc.collect()

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(outdir / "fold_alpha_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        outdir / "backbone_predictions.csv.gz", index=False, compression="gzip"
    )

    summary_rows = []
    for alpha, group in folds.groupby("alpha", sort=True):
        weights = np.asarray([fold_weights[str(f)] for f in group["fold"]], dtype=np.float64)
        weights /= weights.sum()
        delta = group["delta_brier_vs_fixed"].to_numpy(np.float64)
        summary_rows.append(
            {
                "alpha": float(alpha),
                "weighted_brier": float(np.dot(weights, group["brier"].to_numpy(np.float64))),
                "weighted_raw_score": float(np.dot(weights, group["raw_score"].to_numpy(np.float64))),
                "weighted_delta_brier_vs_fixed": float(np.dot(weights, delta)),
                "improved_folds": int(np.count_nonzero(delta < 0.0)),
                "worst_delta": float(delta.max()),
                "mean_prediction_std": float(group["prediction_std"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["weighted_brier", "worst_delta"]).reset_index(drop=True)
    summary.to_csv(outdir / "alpha_summary.csv", index=False)

    print("\n[Alpha summary: lower is better]")
    print(summary.to_string(index=False, formatters={
        "weighted_brier": "{:.8f}".format,
        "weighted_raw_score": "{:+.2f}".format,
        "weighted_delta_brier_vs_fixed": "{:+.8f}".format,
        "worst_delta": "{:+.8f}".format,
    }))
    print(f"saved: {outdir / 'fold_alpha_metrics.csv'}")
    print(f"saved: {outdir / 'alpha_summary.csv'}")
    print(f"saved: {outdir / 'backbone_predictions.csv.gz'}")
    print(f"canonical invariant check: {invariant_check}")


if __name__ == "__main__":
    main()
