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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_recent_regime_submissions as recent_core
import run_game_type_temporal_regime_ablation as regime_core
from src.utils import load_config, save_json, seed_everything


PITCHER_RATES = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
BATTER_RATES = {
    "success": "asof_batter_success_rate",
    "middle": "asof_batter_middle_rate",
}
VARIANTS = ("A0_REGIME", "A1_ROW_COUNTS", "A2_ROW_COUNT_STATE")


def parse_ints(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("at least one fold is required")
    return values


def add_rowlocal_count_features(frame: pd.DataFrame) -> dict[str, float | int]:
    """Expose cumulative as-of sufficient statistics using this row only."""
    specs = (("pitcher", "asof_pitcher_n", PITCHER_RATES), ("batter", "asof_batter_n", BATTER_RATES))
    diagnostics: dict[str, float | int] = {"rows": int(len(frame))}
    for prefix, n_col, rates in specs:
        n = pd.to_numeric(frame[n_col], errors="raise").to_numpy(np.float64)
        if np.any(n < 0):
            raise ValueError(f"negative counts in {n_col}")
        for short, rate_col in rates.items():
            rate = pd.to_numeric(frame[rate_col], errors="coerce").to_numpy(np.float64)
            raw_count = n * rate
            rounded = np.rint(raw_count)
            exact = np.isfinite(raw_count) & (np.abs(raw_count - rounded) <= 0.05)
            count = np.where(exact, rounded, raw_count)
            frame[f"eng_{prefix}_{short}_count"] = count.astype(np.float32)
            frame[f"eng_{prefix}_{short}_uncertainty"] = np.sqrt(
                np.clip(rate * (1.0 - rate), 0.0, None) / (n + 1.0)
            ).astype(np.float32)
            diagnostics[f"{prefix}_{short}_integer_share"] = float(exact.mean())

    pitcher_success = pd.to_numeric(frame["eng_pitcher_success_count"], errors="coerce")
    batter_success = pd.to_numeric(frame["eng_batter_success_count"], errors="coerce")
    frame["eng_pitcher_non_success_count"] = (
        pd.to_numeric(frame["asof_pitcher_n"], errors="raise") - pitcher_success
    ).astype(np.float32)
    frame["eng_batter_non_success_count"] = (
        pd.to_numeric(frame["asof_batter_n"], errors="raise") - batter_success
    ).astype(np.float32)
    return diagnostics


def feature_sets(base: list[str]) -> dict[str, list[str]]:
    counts = [
        *[f"eng_pitcher_{name}_count" for name in PITCHER_RATES],
        *[f"eng_batter_{name}_count" for name in BATTER_RATES],
        "eng_pitcher_non_success_count",
        "eng_batter_non_success_count",
    ]
    uncertainty = [
        *[f"eng_pitcher_{name}_uncertainty" for name in PITCHER_RATES],
        *[f"eng_batter_{name}_uncertainty" for name in BATTER_RATES],
    ]
    return {
        "A0_REGIME": [*base, "eng_recent_f"],
        "A1_ROW_COUNTS": [*base, "eng_recent_f", *counts],
        "A2_ROW_COUNT_STATE": [*base, "eng_recent_f", *counts, *uncertainty],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-safe row-local cumulative as-of count probe.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--folds", default="2022,2023,2024")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--target-score", type=float, default=1200.0)
    ap.add_argument("--regime-start-year", type=int, default=2023)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--gpu-ram-part", type=float, default=0.95)
    ap.add_argument("--pinned-memory-size", default="4GB")
    ap.add_argument("--output-dir", default="outputs/rowlocal_asof_count_probe")
    args = ap.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    folds = parse_ints(args.folds)
    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame, invariants = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    diagnostics = add_rowlocal_count_features(frame)
    regime_core.add_regime_features(frame, season_col=season, regime_start_year=args.regime_start_year)
    base = recent_core.feature_set("recent_raw_game_type")
    variants = feature_sets(base)
    params = regime_core.build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    progress = tqdm(total=len(folds) * len(VARIANTS), desc="row-local asof", unit="model", dynamic_ncols=True)

    for val_year in folds:
        train = frame.loc[frame[season] < val_year].copy()
        valid = frame.loc[frame[season].eq(val_year)].copy()
        if train.empty or valid.empty:
            raise ValueError(f"fold {val_year}: empty train/valid")
        y = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        gt = valid["game_type"].astype(str).to_numpy()
        fold_pred: dict[str, np.ndarray] = {}

        for name in VARIANTS:
            seed_everything(seed)
            pred = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target,
                features=variants[name],
                extra_categorical=set(),
                params=params,
            )
            fold_pred[name] = pred
            metric = regime_core.binary_metrics(y, pred)
            progress.set_postfix_str(f"{val_year} {name} score={metric['score']:.1f} brier={metric['brier']:.6f}")
            progress.update(1)

        base_brier = regime_core.binary_metrics(y, fold_pred["A0_REGIME"])["brier"]
        for name, pred in fold_pred.items():
            for group, mask in {"ALL": np.ones(len(valid), bool), "R": gt == "R", "F": gt == "F"}.items():
                metric = regime_core.binary_metrics(y[mask], pred[mask])
                rows.append({
                    "validation_year": val_year,
                    "variant": name,
                    "group": group,
                    "rows": int(mask.sum()),
                    "feature_count": len(variants[name]),
                    **metric,
                    "delta_brier_vs_A0_all": metric["brier"] - base_brier if group == "ALL" else np.nan,
                })
        if val_year == 2024:
            predictions = fold_pred
            pred_frame = pd.DataFrame({row_id: valid[row_id].to_numpy(), "target": y, "game_type": gt})
            for name, pred in fold_pred.items():
                pred_frame[f"{name}_probability"] = pred
            pred_frame.to_csv(out / "validation_2024_predictions.csv", index=False)
        del train, valid, y, fold_pred
        gc.collect()

    progress.close()
    results = pd.DataFrame(rows)
    results.to_csv(out / "metrics.csv", index=False)
    overall = results.loc[results["group"].eq("ALL")]
    summary = overall.groupby("variant", as_index=False).agg(
        folds=("validation_year", "count"),
        mean_brier=("brier", "mean"),
        mean_delta=("delta_brier_vs_A0_all", "mean"),
        worst_delta=("delta_brier_vs_A0_all", "max"),
        wins=("delta_brier_vs_A0_all", lambda x: int((x < 0).sum())),
    ).sort_values("mean_delta")
    summary.to_csv(out / "summary.csv", index=False)
    save_json({
        "experiment": "row-local cumulative as-of sufficient-statistic representation",
        "folds": folds,
        "variants": variants,
        "diagnostics": diagnostics,
        "iterations": args.iterations,
        "target_score": args.target_score,
        "task_type": args.task_type,
        "devices": args.devices if args.task_type == "GPU" else None,
        "canonical_invariants": invariants,
        "leakage_guard": "every engineered value is a pure function of columns in the same row; no ID grouping, sorting, target, other validation row, or state update",
    }, out / "run_config.json")

    print("\n2024 results")
    latest = results.loc[(results.validation_year == 2024) & (results.group == "ALL")].copy()
    reference = float(frame.loc[frame[season].eq(2024), target].mean())
    reference_brier = reference * (1.0 - reference)
    target_brier = reference_brier * (1.0 - args.target_score / 100000.0)
    latest["gap_to_target_brier"] = latest["brier"] - target_brier
    print(latest[["variant", "score", "brier", "delta_brier_vs_A0_all", "gap_to_target_brier"]].sort_values("brier").to_string(index=False))
    best = latest.sort_values("brier").iloc[0]
    print(
        f"target_score={args.target_score:.1f} target_brier={target_brier:.9f} "
        f"status={'PASS' if float(best['score']) >= args.target_score else 'MISS'}"
    )
    print(f"output_dir: {out}")


if __name__ == "__main__":
    main()
