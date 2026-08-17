#!/usr/bin/env python3
"""Cheap regime-specific OOF blend scan using existing predictions."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / "data" / "processed" / "physical_features"
    / "train_with_pitcher_physical.parquet"
)
DEFAULT_CACHE = ROOT / "outputs" / "physical_regime_suite" / "cache"
DEFAULT_MOE = (
    ROOT / "outputs" / "pitch_arsenal_moe_v2"
    / "validation_predictions.parquet"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "catboost_moe_blend_scan"


@dataclass(frozen=True)
class Fold:
    name: str
    weight: float
    kind: str
    cutoff_month: int | None = None
    valid_start: int | None = None
    valid_end: int | None = None


FOLDS = (
    Fold("season_forward_2024", 0.50, "season_forward"),
    Fold("mid_2024", 0.20, "within", 5, 6, 7),
    Fold("late_2024", 0.30, "within", 7, 8, None),
)
EXPERTS = ("old_full", "old_recent", "physical_full", "physical_recent", "r_fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--moe-predictions", type=Path, default=DEFAULT_MOE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-moe-weight", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def valid_mask(frame: pd.DataFrame, fold: Fold) -> np.ndarray:
    season = frame["season"].to_numpy(np.int64)
    month = frame["game_month"].to_numpy(np.int64)
    if fold.kind == "season_forward":
        return season == 2024
    mask = (season == 2024) & (month >= int(fold.valid_start))
    if fold.valid_end is not None:
        mask &= month <= fold.valid_end
    return mask


def load_cache(cache_dir: Path, fold: str, expert: str) -> np.ndarray:
    matches = list(cache_dir.glob(f"*__{fold}__{expert}.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cache for {fold}/{expert}, found {len(matches)}"
        )
    with np.load(matches[0]) as payload:
        return payload["prediction"].astype(np.float64)


def brier(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        target, prediction = target[mask], prediction[mask]
    return float(np.mean(np.square(prediction - target)))


def score(target: np.ndarray, prediction: np.ndarray) -> float:
    reference = float(target.mean() * (1.0 - target.mean()))
    return max(0.0, 100_000.0 * (1.0 - brier(target, prediction) / reference))


def main() -> None:
    args = parse_args()
    for name in ("data", "cache_dir", "moe_predictions", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (args.data, args.cache_dir, args.moe_predictions):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.step <= 0 or not 0 <= args.max_moe_weight <= 1:
        raise ValueError("Require --step > 0 and --max-moe-weight in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "regime_blend_results.csv",
        args.output_dir / "regime_blend_summary.csv",
    ]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("Output exists; use --force")

    order = pd.read_parquet(
        args.data, columns=["row_id", "season", "game_month"]
    ).sort_values(["season", "game_month", "row_id"], kind="stable")
    moe = pd.read_parquet(args.moe_predictions)
    if moe.duplicated(["fold", "row_id"]).any():
        raise ValueError("Duplicate fold/row_id in MoE predictions")

    fold_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold in FOLDS:
        row_ids = order.loc[valid_mask(order, fold), "row_id"].astype("string")
        neural = moe.loc[moe["fold"].eq(fold.name)].copy()
        neural["row_id"] = neural["row_id"].astype("string")
        neural = neural.set_index("row_id").reindex(row_ids)
        if neural[["target", "probability", "game_type"]].isna().any().any():
            raise ValueError(f"MoE row alignment failed for {fold.name}")
        target = neural["target"].to_numpy(np.float64)
        moe_prediction = neural["probability"].to_numpy(np.float64)
        is_r = neural["game_type"].astype(str).eq("R").to_numpy()
        experts = {
            expert: load_cache(args.cache_dir, fold.name, expert)
            for expert in EXPERTS
        }
        if any(len(values) != len(target) for values in experts.values()):
            raise ValueError(f"CatBoost cache length mismatch for {fold.name}")

        old = 0.8 * experts["old_full"] + 0.2 * experts["old_recent"]
        physical = 0.8 * experts["physical_full"] + 0.2 * experts["physical_recent"]
        catboost = 0.25 * old + 0.75 * physical
        catboost[is_r] = 0.9 * catboost[is_r] + 0.1 * experts["r_fast"][is_r]
        fold_data[fold.name] = (target, catboost, moe_prediction, is_r)

    weights = np.arange(0.0, args.max_moe_weight + args.step / 2, args.step)
    rows: list[dict[str, float | str]] = []
    for fold in FOLDS:
        target, catboost, neural, is_r = fold_data[fold.name]
        for r_weight in weights:
            for f_weight in weights:
                prediction = catboost.copy()
                prediction[is_r] = (
                    (1.0 - r_weight) * catboost[is_r] + r_weight * neural[is_r]
                )
                prediction[~is_r] = (
                    (1.0 - f_weight) * catboost[~is_r] + f_weight * neural[~is_r]
                )
                rows.append(
                    {
                        "fold": fold.name,
                        "fold_weight": fold.weight,
                        "r_moe_weight": float(r_weight),
                        "f_moe_weight": float(f_weight),
                        "brier": brier(target, prediction),
                        "score": score(target, prediction),
                        "r_brier": brier(target, prediction, is_r),
                        "f_brier": brier(target, prediction, ~is_r),
                    }
                )
    results = pd.DataFrame(rows)
    baseline = results.loc[
        results["r_moe_weight"].eq(0) & results["f_moe_weight"].eq(0)
    ].set_index("fold")["brier"]
    results["improved_vs_catboost"] = results.apply(
        lambda row: row["brier"] < baseline[row["fold"]], axis=1
    )
    summaries: list[dict[str, float | int]] = []
    for (r_weight, f_weight), group in results.groupby(
        ["r_moe_weight", "f_moe_weight"], sort=True
    ):
        fold_weights = group["fold_weight"].to_numpy(np.float64)
        summaries.append(
            {
                "r_moe_weight": float(r_weight),
                "f_moe_weight": float(f_weight),
                "weighted_brier": float(np.average(group["brier"], weights=fold_weights)),
                "weighted_score": float(np.average(group["score"], weights=fold_weights)),
                "weighted_r_brier": float(np.average(group["r_brier"], weights=fold_weights)),
                "weighted_f_brier": float(np.average(group["f_brier"], weights=fold_weights)),
                "worst_brier": float(group["brier"].max()),
                "improved_folds": int(group["improved_vs_catboost"].sum()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("weighted_brier")
    results.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    print(summary.head(12).to_string(index=False))
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
