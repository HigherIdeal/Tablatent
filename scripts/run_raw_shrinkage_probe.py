from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# Project server policy: expose physical GPU 2 unless caller overrides it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gbdt_guided_piecewise as core
from src.canonical_features import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config


def parse_floats(value: str) -> list[float]:
    out = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("at least one alpha is required")
    if any(x < 0.0 or x > 1.5 for x in out):
        raise ValueError("alphas must be in [0, 1.5]")
    return out


def stratified_sample(
    frame: pd.DataFrame,
    fraction: float,
    strata: list[str],
    seed: int,
) -> pd.DataFrame:
    """Deterministic order-preserving per-stratum sample."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction >= 1.0:
        return frame.copy()

    chosen: list[np.ndarray] = []
    grouped = frame.groupby(strata, observed=True, sort=True, dropna=False)
    for group_id, (_, idx) in enumerate(grouped.groups.items()):
        ids = np.asarray(list(idx))
        n = min(len(ids), max(1, int(round(len(ids) * fraction))))
        rng = np.random.default_rng(seed + 104729 * (group_id + 1))
        chosen.append(rng.choice(ids, size=n, replace=False))

    if not chosen:
        raise RuntimeError("sampling produced no rows")
    return frame.loc[np.sort(np.concatenate(chosen))].copy()


def oracle_alpha(y: np.ndarray, p: np.ndarray, center: float) -> float:
    """Closed-form Brier-optimal alpha for p'=center+alpha*(p-center).

    This uses validation labels and is diagnostic only; never use the per-fold
    oracle alpha as a deployable calibration rule.
    """
    d = p - center
    denom = float(np.dot(d, d))
    if denom <= 0.0:
        return 0.0
    alpha = float(np.dot(d, y - center) / denom)
    return float(np.clip(alpha, 0.0, 1.5))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast diagnostic: train only the raw neural baseline, then shrink "
            "validation probability dispersion while preserving prediction mean."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--valid-fraction", type=float, default=1.0)
    parser.add_argument(
        "--alphas",
        default="0.40,0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        help="p' = mean(p) + alpha * (p - mean(p)); alpha=1 is raw",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--clip-z", type=float, default=12.0)
    parser.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    core.set_seed(seed)
    folds = core.parse_ints(args.folds)
    alphas = parse_floats(args.alphas)
    if 1.0 not in alphas:
        alphas.append(1.0)
        alphas.sort()

    target = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    device = core.torch_device(args.torch_device)

    frame = load_frame(config).copy()
    validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame = frame.sort_values([season_col, "game_month", row_id], kind="stable").reset_index(drop=True)

    numerical = core.numeric_features()
    categorical = [f for f in CANONICAL_FEATURES if f in set(CANONICAL_CATEGORICAL)]
    output_dir = Path(config["paths"]["output_dir"]) / "raw_shrinkage_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Raw shrinkage probe] folds={folds} train_fraction={args.train_fraction:.3f} "
        f"valid_fraction={args.valid_fraction:.3f} alphas={alphas} device={device}"
    )
    print("[Raw shrinkage probe] center=mean(raw prediction), so shrinkage changes dispersion only.")
    print("[Raw shrinkage probe] oracle alpha uses validation labels and is diagnostic ONLY.")

    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for val_year in folds:
        full_train = frame.loc[frame[season_col] < val_year].copy()
        full_valid = frame.loc[frame[season_col] == val_year].copy()
        train = stratified_sample(
            full_train,
            args.train_fraction,
            [season_col, target],
            seed + val_year,
        )
        valid_strata = [target]
        if "game_type" in full_valid.columns:
            valid_strata.append("game_type")
        valid = stratified_sample(
            full_valid,
            args.valid_fraction,
            valid_strata,
            seed + 10000 + val_year,
        )

        ytr = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
        yva = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        scaler = core.fit_scaler(train, numerical)
        xtr_num = core.transform_numeric(train, numerical, scaler, args.clip_z)
        xva_num = core.transform_numeric(valid, numerical, scaler, args.clip_z)
        cat_maps = core.fit_category_maps(train, categorical)
        xtr_cat = core.transform_categories(train, categorical, cat_maps)
        xva_cat = core.transform_categories(valid, categorical, cat_maps)
        cat_sizes = [len(cat_maps[f]) + 1 for f in categorical]

        print(
            f"\n[Fold {val_year}] train={len(train):,}/{len(full_train):,} "
            f"valid={len(valid):,}/{len(full_valid):,} "
            f"train_rate={ytr.mean():.6f} valid_rate={yva.mean():.6f}"
        )

        p, history, best_epoch = core.train_model(
            xtr_num,
            xtr_cat,
            ytr,
            xva_num,
            xva_cat,
            yva,
            cat_sizes,
            [],
            args,
            device,
            seed,
        )
        p = p.astype(np.float64, copy=False)
        center = float(p.mean())
        raw_metrics = probability_metrics(yva, p)
        a_oracle = oracle_alpha(yva, p, center)
        p_oracle = np.clip(center + a_oracle * (p - center), 0.0, 1.0)
        oracle_metrics = probability_metrics(yva, p_oracle)

        print(
            f"  raw: best_epoch={best_epoch} brier={raw_metrics['brier']:.8f} "
            f"auc={raw_metrics['auc']:.5f} mean={center:.5f} "
            f"p_std={raw_metrics['prediction_std']:.5f}"
        )
        print(
            f"  oracle diagnostic: alpha={a_oracle:.4f} "
            f"brier={oracle_metrics['brier']:.8f} "
            f"delta={oracle_metrics['brier'] - raw_metrics['brier']:+.8f}"
        )

        for alpha in alphas:
            shrunk = np.clip(center + alpha * (p - center), 0.0, 1.0)
            metrics = probability_metrics(yva, shrunk)
            rows.append(
                {
                    "fold": val_year,
                    "alpha": alpha,
                    "train_fraction": args.train_fraction,
                    "valid_fraction": args.valid_fraction,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "best_epoch": best_epoch,
                    "center": center,
                    "oracle_alpha": a_oracle,
                    "delta_brier_vs_raw": metrics["brier"] - raw_metrics["brier"],
                    **metrics,
                }
            )
            print(
                f"    alpha={alpha:>4.2f} brier={metrics['brier']:.8f} "
                f"delta={metrics['brier'] - raw_metrics['brier']:+.8f} "
                f"p_std={metrics['prediction_std']:.5f}"
            )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": val_year,
                    "row_id": valid[row_id].astype(str).to_numpy(),
                    "target": yva,
                    "raw_prediction": p,
                }
            )
        )
        pd.DataFrame(history).to_csv(output_dir / f"history_fold{val_year}.csv", index=False)

        del xtr_num, xva_num, xtr_cat, xva_cat, p, p_oracle
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "fold_alpha_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "raw_predictions.csv.gz", index=False, compression="gzip"
    )

    summary_rows = []
    for alpha, group in result.groupby("alpha", sort=True):
        weights = group["valid_rows"].to_numpy(np.float64)
        summary_rows.append(
            {
                "alpha": alpha,
                "folds": len(group),
                "weighted_brier": float(np.average(group["brier"], weights=weights)),
                "mean_delta_brier_vs_raw": float(group["delta_brier_vs_raw"].mean()),
                "improved_folds": int((group["delta_brier_vs_raw"] < 0).sum()),
                "mean_prediction_std": float(group["prediction_std"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("weighted_brier")
    summary.to_csv(output_dir / "alpha_summary.csv", index=False)

    print("\n[Alpha summary: lower is better]")
    print(summary.to_string(index=False))
    print(f"saved: {output_dir / 'fold_alpha_metrics.csv'}")
    print(f"saved: {output_dir / 'alpha_summary.csv'}")
    print(f"saved: {output_dir / 'raw_predictions.csv.gz'}")


if __name__ == "__main__":
    main()
