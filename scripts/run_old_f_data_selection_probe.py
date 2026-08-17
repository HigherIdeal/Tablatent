from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_game_type_temporal_regime_ablation as regime_core
from src.utils import load_config, save_json, seed_everything


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_floats(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one float is required")
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"weights must be in [0,1], got {v}")
    return values


def fit_predict_weighted(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    features: list[str],
    weights: np.ndarray,
    params: dict,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    weights = np.asarray(weights, dtype=np.float32)
    if len(weights) != len(train):
        raise ValueError("weight length mismatch")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")

    # For exact hard-drop, physically remove zero-weight rows rather than relying
    # on CatBoost's treatment of zero weights.
    keep = weights > 0.0
    if not np.any(keep):
        raise ValueError("all training rows were removed")

    train_used = train.loc[keep].copy()
    weight_used = weights[keep]

    x_train, categorical = regime_core.prepare_x(
        train_used,
        features,
        extra_categorical=set(),
    )
    x_valid, valid_categorical = regime_core.prepare_x(
        valid,
        features,
        extra_categorical=set(),
    )
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")

    y_train = pd.to_numeric(train_used[target_col], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float32)

    train_pool = Pool(
        x_train,
        label=y_train,
        weight=weight_used,
        cat_features=categorical,
        feature_names=features,
    )
    valid_pool = Pool(
        x_valid,
        label=y_valid,
        cat_features=categorical,
        feature_names=features,
    )

    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=False)
    pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)

    del model, train_pool, valid_pool, x_train, x_valid, y_train, y_valid, train_used
    gc.collect()
    return pred


def evaluate_groups(
    y: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> list[dict]:
    masks = {
        "ALL": np.ones(len(y), dtype=bool),
        "R": gt == "R",
        "F": gt == "F",
    }
    rows: list[dict] = []
    for group, mask in masks.items():
        metric = regime_core.binary_metrics(y[mask], pred[mask])
        base = regime_core.binary_metrics(y[mask], baseline_pred[mask])
        rows.append(
            {
                "group": group,
                "rows": int(mask.sum()),
                **metric,
                "delta_brier_vs_full_weight": float(metric["brier"] - base["brier"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select/downweight temporally old F-domain rows while keeping all R rows. "
            "The rule is forward-valid: in fold y, the latest N prior F seasons keep weight 1; "
            "older F rows receive a swept weight. No target is used to decide which rows are old."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--recent-f-seasons", default="1,2")
    parser.add_argument("--old-f-weights", default="1,0.75,0.5,0.25,0.1,0")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/old_f_data_selection_probe")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    folds = sorted(parse_ints(args.folds))
    recent_windows = sorted(set(parse_ints(args.recent_f_seasons)))
    if any(v <= 0 for v in recent_windows):
        raise ValueError("recent-f-seasons values must be positive")
    old_f_weights = parse_floats(args.old_f_weights)

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)

    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()

    unexpected = sorted(set(frame["game_type"].dropna().unique()) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    features = [*recent_core.feature_set("recent_raw_game_type"), "eng_recent_f"]
    if len(features) != len(set(features)):
        raise RuntimeError("duplicate features")

    params = regime_core.build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"Old-F data selection | folds={folds} | recent_F_windows={recent_windows} | "
        f"old_F_weights={old_f_weights} | rows={len(frame):,} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations}"
    )
    tqdm.write(
        "RULE: all R rows keep weight=1. For fold y and recent_F_seasons=N, "
        "F rows from seasons >= y-N keep weight=1; earlier F rows get swept weight."
    )

    results: list[dict] = []
    diagnostics: list[dict] = []
    total_models = len(folds) * (1 + len(recent_windows) * max(0, len(old_f_weights) - 1))
    progress = tqdm(total=total_models, desc="old-F selection models", unit="model", dynamic_ncols=True)

    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col] == val_year].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid")

        y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt_valid = valid["game_type"].astype(str).to_numpy()

        # Fit the full-weight baseline once per fold. Weight=1 is identical across windows.
        baseline_weights = np.ones(len(train), dtype=np.float32)
        seed_everything(seed)
        baseline_pred = fit_predict_weighted(
            train=train,
            valid=valid,
            target_col=target_col,
            features=features,
            weights=baseline_weights,
            params=params,
        )
        progress.update(1)

        for row in evaluate_groups(y_valid, gt_valid, baseline_pred, baseline_pred):
            results.append(
                {
                    "fold": int(val_year),
                    "recent_f_seasons": 0,
                    "old_f_weight": 1.0,
                    "variant": "BASE_FULL_WEIGHT",
                    **row,
                }
            )

        train_season = train[season_col].to_numpy(np.int32)
        train_gt = train["game_type"].astype(str).to_numpy()

        for recent_n in recent_windows:
            recent_cutoff = val_year - int(recent_n)
            old_f_mask = (train_gt == "F") & (train_season < recent_cutoff)
            recent_f_mask = (train_gt == "F") & ~old_f_mask
            r_mask = train_gt == "R"

            diagnostics.append(
                {
                    "fold": int(val_year),
                    "recent_f_seasons": int(recent_n),
                    "recent_cutoff": int(recent_cutoff),
                    "train_rows": int(len(train)),
                    "r_rows": int(r_mask.sum()),
                    "recent_f_rows": int(recent_f_mask.sum()),
                    "old_f_rows": int(old_f_mask.sum()),
                    "old_f_share": float(old_f_mask.mean()),
                }
            )

            for old_weight in old_f_weights:
                if abs(old_weight - 1.0) < 1e-12:
                    # Already fitted as the baseline.
                    continue

                weights = np.ones(len(train), dtype=np.float32)
                weights[old_f_mask] = np.float32(old_weight)

                seed_everything(seed)
                pred = fit_predict_weighted(
                    train=train,
                    valid=valid,
                    target_col=target_col,
                    features=features,
                    weights=weights,
                    params=params,
                )
                progress.update(1)

                for row in evaluate_groups(y_valid, gt_valid, pred, baseline_pred):
                    results.append(
                        {
                            "fold": int(val_year),
                            "recent_f_seasons": int(recent_n),
                            "old_f_weight": float(old_weight),
                            "variant": f"RECENT{recent_n}_OLD_F_W{old_weight:g}",
                            **row,
                        }
                    )

    progress.close()

    result_df = pd.DataFrame(results)
    diag_df = pd.DataFrame(diagnostics)
    result_df.to_csv(output_dir / "results.csv", index=False)
    diag_df.to_csv(output_dir / "selection_diagnostics.csv", index=False)

    tqdm.write("\n[Fold results | ALL | lower dB is better]")
    all_rows = result_df.loc[result_df["group"].eq("ALL")].copy()
    for fold in folds:
        part = all_rows.loc[all_rows["fold"].eq(fold)].sort_values("brier")
        tqdm.write(f"fold={fold}")
        for _, row in part.iterrows():
            tqdm.write(
                f"  {row['variant']:<24s} brier={float(row['brier']):.8f} "
                f"score={float(row['score']):+9.2f} dB={float(row['delta_brier_vs_full_weight']):+.8f}"
            )

    candidates = all_rows.loc[~all_rows["variant"].eq("BASE_FULL_WEIGHT")].copy()
    if not candidates.empty:
        summary = (
            candidates.groupby(["recent_f_seasons", "old_f_weight"], as_index=False)
            .agg(
                mean_dB=("delta_brier_vs_full_weight", "mean"),
                worst_dB=("delta_brier_vs_full_weight", "max"),
                best_dB=("delta_brier_vs_full_weight", "min"),
                wins=("delta_brier_vs_full_weight", lambda x: int((x < 0).sum())),
            )
            .sort_values(["mean_dB", "worst_dB"])
        )
        summary.to_csv(output_dir / "cross_fold_summary.csv", index=False)

        tqdm.write("\n[Cross-fold summary | ALL]")
        for _, row in summary.iterrows():
            tqdm.write(
                f"recentF={int(row['recent_f_seasons'])} oldF_w={float(row['old_f_weight']):.2f} "
                f"mean_dB={float(row['mean_dB']):+.8f} worst_dB={float(row['worst_dB']):+.8f} "
                f"best_dB={float(row['best_dB']):+.8f} wins={int(row['wins'])}/{len(folds)}"
            )

    latest = max(folds)
    latest_rows = result_df.loc[result_df["fold"].eq(latest)].copy()
    tqdm.write(f"\n[{latest} diagnostics | best by group]")
    for group in ("ALL", "R", "F"):
        part = latest_rows.loc[latest_rows["group"].eq(group)].sort_values("brier")
        row = part.iloc[0]
        tqdm.write(
            f"{group:<3s} {row['variant']:<24s} brier={float(row['brier']):.8f} "
            f"score={float(row['score']):+9.2f} dB={float(row['delta_brier_vs_full_weight']):+.8f}"
        )

    save_json(
        output_dir / "run_info.json",
        {
            "folds": folds,
            "recent_f_seasons": recent_windows,
            "old_f_weights": old_f_weights,
            "iterations": int(args.iterations),
            "regime_start_year": int(args.regime_start_year),
            "invariant_check": invariant_check,
        },
    )
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
