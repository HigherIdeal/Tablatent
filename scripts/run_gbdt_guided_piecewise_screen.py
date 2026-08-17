from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Server policy: use physical GPU 2 unless the caller already set visibility.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gbdt_guided_piecewise as core
from src.canonical_features import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config


def stratified_sample(
    frame: pd.DataFrame,
    fraction: float,
    strata: list[str],
    seed: int,
) -> pd.DataFrame:
    """Deterministic per-stratum sampling without replacement.

    Every non-empty stratum keeps at least one row. The returned rows preserve
    original order after sampling so CatBoost has_time semantics are not
    accidentally replaced by random row order.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample fraction must be in (0, 1]")
    if fraction >= 1.0:
        return frame.copy()

    missing = sorted(set(strata) - set(frame.columns))
    if missing:
        raise ValueError(f"missing sampling strata columns: {missing}")

    chosen: list[np.ndarray] = []
    grouped = frame.groupby(strata, observed=True, sort=True, dropna=False)
    for group_id, (_, idx) in enumerate(grouped.groups.items()):
        ids = np.asarray(list(idx))
        n = max(1, int(round(len(ids) * fraction)))
        n = min(n, len(ids))
        rng = np.random.default_rng(seed + 104729 * (group_id + 1))
        chosen.append(rng.choice(ids, size=n, replace=False))

    if not chosen:
        raise RuntimeError("sampling produced no rows")
    keep = np.concatenate(chosen)
    return frame.loc[np.sort(keep)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast screening for the GBDT-guided PWL ablation. Train and validation "
            "are deterministic stratified subsets; use the full script for promotion."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--valid-fraction", type=float, default=0.25)
    parser.add_argument("--border-iterations", type=int, default=200)
    parser.add_argument("--max-borders", type=int, default=24)
    parser.add_argument("--recent-seasons", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--clip-z", type=float, default=12.0)
    parser.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("train-fraction must be in (0, 1]")
    if not 0.0 < args.valid_fraction <= 1.0:
        raise ValueError("valid-fraction must be in (0, 1]")
    if args.max_borders < 1 or args.recent_seasons < 1:
        raise ValueError("max-borders and recent-seasons must be >= 1")

    try:
        import catboost
        import torch
    except ImportError as exc:
        raise RuntimeError("catboost and torch are required") from exc

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    core.set_seed(seed)
    target = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    folds = core.parse_ints(args.folds)
    variants = core.parse_variants(args.variants)
    device = core.torch_device(args.torch_device)

    frame = load_frame(config).copy()
    validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(int)
    frame = frame.sort_values([season_col, "game_month", row_id], kind="stable").reset_index(drop=True)

    features = list(CANONICAL_FEATURES)
    numerical = core.numeric_features()
    categorical = [f for f in features if f in set(CANONICAL_CATEGORICAL)]
    if PITCHER_TEAM_WIN_EXPECTANCY not in numerical:
        raise RuntimeError("canonical win expectancy must be numerical")

    output_dir = Path(config["paths"]["output_dir"]) / "gbdt_guided_piecewise_screen"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[GBDT-PWL SCREEN] folds={folds} variants={variants} "
        f"train_fraction={args.train_fraction:.3f} valid_fraction={args.valid_fraction:.3f} "
        f"batch={args.batch_size} epochs={args.epochs} physical_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"torch={torch.__version__} device={device} catboost={catboost.__version__}"
    )
    print("[GBDT-PWL SCREEN] train strata = season x target")
    print("[GBDT-PWL SCREEN] valid strata = target x game_type; absolute Brier is screening-only")
    print("[GBDT-PWL SCREEN] CatBoost borders and neural fitting see sampled TRAIN rows only")

    rows: list[dict] = []
    metadata: dict[str, dict] = {}

    for val_year in folds:
        train_full = frame.loc[frame[season_col] < val_year].copy()
        valid_full = frame.loc[frame[season_col] == val_year].copy()
        if train_full.empty or valid_full.empty:
            raise ValueError(f"Fold {val_year}: empty train or validation")

        train = stratified_sample(
            train_full,
            args.train_fraction,
            [season_col, target],
            seed + val_year * 10,
        )
        valid = stratified_sample(
            valid_full,
            args.valid_fraction,
            [target, "game_type"],
            seed + val_year * 10 + 1,
        )
        recent = train.loc[train[season_col] >= val_year - args.recent_seasons].copy()
        if recent.empty:
            raise ValueError(f"Fold {val_year}: sampled recent split is empty")

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
            f"\n[Fold {val_year}] train={len(train):,}/{len(train_full):,} "
            f"recent={len(recent):,} valid={len(valid):,}/{len(valid_full):,} "
            f"train_rate={ytr.mean():.6f} valid_rate={yva.mean():.6f}"
        )

        q_raw = core.quantile_knots(train, numerical, args.max_borders)
        q_norm = core.normalize_knots(q_raw, numerical, scaler, args.clip_z)
        global_raw = recent_raw = None

        if any(v in variants for v in ("gbdt_global_pl", "gbdt_dual_pl")):
            print("  discovering global CatBoost borders on sampled train...")
            global_raw = core.fit_catboost_borders(
                train,
                features,
                numerical,
                target,
                config,
                args.border_iterations,
                args.task_type,
                args.devices,
                args.verbose,
                args.max_borders,
            )
        if "gbdt_dual_pl" in variants:
            print("  discovering recent CatBoost borders on sampled recent...")
            recent_raw = core.fit_catboost_borders(
                recent,
                features,
                numerical,
                target,
                config,
                args.border_iterations,
                args.task_type,
                args.devices,
                args.verbose,
                args.max_borders,
            )

        global_norm = (
            core.normalize_knots(global_raw, numerical, scaler, args.clip_z)
            if global_raw is not None
            else None
        )
        recent_norm = (
            core.normalize_knots(recent_raw, numerical, scaler, args.clip_z)
            if recent_raw is not None
            else None
        )

        metadata[str(val_year)] = {
            "train_rows_full": int(len(train_full)),
            "train_rows_sampled": int(len(train)),
            "valid_rows_full": int(len(valid_full)),
            "valid_rows_sampled": int(len(valid)),
            "recent_seasons": sorted(int(x) for x in recent[season_col].unique()),
            "train_rate_full": float(train_full[target].mean()),
            "train_rate_sampled": float(train[target].mean()),
            "valid_rate_full": float(valid_full[target].mean()),
            "valid_rate_sampled": float(valid[target].mean()),
        }

        fold_rows: list[dict] = []
        for i, variant in enumerate(variants, start=1):
            if variant == "raw":
                knot_sets = []
            elif variant == "quantile_pl":
                knot_sets = [core.pad_knots(q_norm, numerical, args.max_borders)]
            elif variant == "gbdt_global_pl":
                if global_norm is None:
                    raise RuntimeError("global borders missing")
                knot_sets = [core.pad_knots(global_norm, numerical, args.max_borders)]
            elif variant == "gbdt_dual_pl":
                if global_norm is None or recent_norm is None:
                    raise RuntimeError("dual borders missing")
                knot_sets = [
                    core.pad_knots(global_norm, numerical, args.max_borders),
                    core.pad_knots(recent_norm, numerical, args.max_borders),
                ]
            else:
                raise RuntimeError(f"unhandled variant: {variant}")

            print(f"  [{i:02d}/{len(variants):02d}] {variant}")
            p, history, best_epoch = core.train_model(
                xtr_num,
                xtr_cat,
                ytr,
                xva_num,
                xva_cat,
                yva,
                cat_sizes,
                knot_sets,
                args,
                device,
                seed + val_year * 100 + i,
            )
            metrics = probability_metrics(yva, p)
            row = {
                "fold": int(val_year),
                "variant": variant,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "best_epoch": int(best_epoch),
                **metrics,
            }
            fold_rows.append(row)
            print(
                f"       best={best_epoch} brier={metrics['brier']:.8f} "
                f"score={metrics['raw_score']:.2f} auc={metrics['auc']:.5f} "
                f"p_std={metrics['prediction_std']:.5f}"
            )

        raw_brier = next((r["brier"] for r in fold_rows if r["variant"] == "raw"), None)
        for row in fold_rows:
            row["delta_brier_vs_raw"] = (
                float(row["brier"] - raw_brier) if raw_brier is not None else float("nan")
            )
            rows.append(row)

        del train_full, valid_full, train, valid, recent
        del xtr_num, xva_num, xtr_cat, xva_cat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "screen_results.csv", index=False)
    (output_dir / "screen_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n[Screen summary: negative delta is better than raw]")
    for _, row in result.iterrows():
        print(
            f"  fold={int(row['fold'])} {row['variant']:<18} "
            f"brier={row['brier']:.8f} delta={row['delta_brier_vs_raw']:+.8f}"
        )
    print(f"[saved] {output_dir / 'screen_results.csv'}")
    print("[promotion rule] Only variants with consistent negative delta should be rerun on full data.")


if __name__ == "__main__":
    main()
