from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_regime_atlas as atlas
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config, save_json


RESIDUAL_COL = "__gt_controlled_residual__"


def _year_profile_residual(
    groups: pd.Series,
    residual: pd.Series,
    season: pd.Series,
    year: int,
) -> pd.DataFrame:
    mask = season.eq(year)
    temp = pd.DataFrame(
        {
            "group": groups.loc[mask].to_numpy(),
            "y": pd.to_numeric(residual.loc[mask], errors="raise").to_numpy(np.float64),
        }
    )
    prof = temp.groupby("group", dropna=False)["y"].agg(["count", "mean"]).reset_index()
    # Residuals are centered within season x game_type. Re-center once more at season
    # level only to remove tiny floating-point/sample-weight differences.
    prior = float(temp["y"].mean())
    prof["effect"] = prof["mean"] - prior
    prof["season"] = int(year)
    return prof[["season", "group", "count", "mean", "effect"]]


def _metrics_for_signal(
    frame: pd.DataFrame,
    *,
    signal: str,
    components: list[str],
    season_col: str,
    residual_col: str,
    categorical_features: set[str],
    numeric_bins: int,
    max_auto_categories: int,
    min_era_count: int,
    min_effect_for_flip: float,
    same_player_mask: pd.Series,
) -> dict:
    groups, grouping, group_count = atlas._make_signal_groups(
        frame,
        components,
        categorical_features,
        numeric_bins,
        max_auto_categories,
    )
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    residual = pd.to_numeric(frame[residual_col], errors="raise").astype(float)
    profiles = {
        year: _year_profile_residual(groups, residual, season, year)
        for year in atlas.YEARS
    }

    fixed = atlas._era_shift(
        profiles,
        atlas.OLD_YEARS,
        atlas.RECENT_YEARS,
        min_era_count=min_era_count,
        min_effect_for_flip=min_effect_for_flip,
    )

    cp_rows = []
    for change_year in atlas.CHANGE_YEARS:
        left = [y for y in atlas.YEARS if y < change_year]
        right = [y for y in atlas.YEARS if y >= change_year]
        metrics = atlas._era_shift(
            profiles,
            left,
            right,
            min_era_count=min_era_count,
            min_effect_for_flip=min_effect_for_flip,
        )
        cp_rows.append((change_year, metrics))
    valid = [
        (year, m)
        for year, m in cp_rows
        if np.isfinite(float(m["changepoint_ratio"]))
    ]
    if valid:
        best_year, best_metrics = max(
            valid,
            key=lambda item: (
                float(item[1]["changepoint_ratio"]),
                float(item[1]["shift_rmse"]),
            ),
        )
        best_ratio = float(best_metrics["changepoint_ratio"])
    else:
        best_year, best_ratio = np.nan, np.nan

    same_groups = groups.loc[same_player_mask]
    same_season = season.loc[same_player_mask]
    same_residual = residual.loc[same_player_mask]
    same_profiles = {
        year: _year_profile_residual(same_groups, same_residual, same_season, year)
        for year in atlas.YEARS
    }
    same = atlas._era_shift(
        same_profiles,
        atlas.OLD_YEARS,
        atlas.RECENT_YEARS,
        min_era_count=max(100, min_era_count // 3),
        min_effect_for_flip=min_effect_for_flip,
    )

    shift = float(fixed["shift_rmse"])
    same_shift = float(same["shift_rmse"])
    same_preservation = (
        same_shift / shift
        if np.isfinite(shift) and shift > 0 and np.isfinite(same_shift)
        else np.nan
    )

    return {
        "signal": signal,
        "components": "+".join(components),
        "signal_type": "feature" if len(components) == 1 else "interaction",
        "grouping": grouping,
        "groups": int(group_count),
        "controlled_shift_2023_rmse": shift,
        "controlled_changepoint_ratio_2023": float(fixed["changepoint_ratio"]),
        "controlled_sign_flip_rate_2023": float(fixed["sign_flip_rate"]),
        "controlled_best_change_year": best_year,
        "controlled_best_changepoint_ratio": best_ratio,
        "controlled_same_player_preservation": same_preservation,
    }


def _verdict(row: pd.Series) -> str:
    shift = float(row["controlled_shift_2023_rmse"])
    ratio = float(row["controlled_changepoint_ratio_2023"])
    flip = float(row["controlled_sign_flip_rate_2023"])
    year = row["controlled_best_change_year"]
    same = float(row["controlled_same_player_preservation"])
    retention = float(row["shift_retention_after_game_type_control"])

    if (
        np.isfinite(shift)
        and shift >= 0.005
        and np.isfinite(ratio)
        and ratio >= 1.5
        and year == 2023
        and np.isfinite(same)
        and same >= 0.6
        and np.isfinite(retention)
        and retention >= 0.45
    ):
        if flip >= 0.35:
            return "INDEPENDENT_REGIME_FLIP"
        return "INDEPENDENT_REGIME_SHIFT"
    if np.isfinite(retention) and retention < 0.30:
        return "MOSTLY_EXPLAINED_BY_GAME_TYPE"
    return "WEAK_OR_MIXED_AFTER_CONTROL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast CPU-only follow-up to Regime Atlas. It removes the season-specific "
            "game_type mean effect from control_success, then re-tests the strongest "
            "non-game_type regime candidates."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--atlas", default="outputs/regime_atlas/regime_atlas.csv")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--numeric-bins", type=int, default=6)
    parser.add_argument("--max-auto-categories", type=int, default=20)
    parser.add_argument("--min-era-count", type=int, default=500)
    parser.add_argument("--min-effect-for-flip", type=float, default=0.002)
    parser.add_argument("--same-player-min-old-seasons", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/game_type_controlled_regime")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, _ = atlas.recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)

    if "game_type" not in frame.columns:
        raise ValueError("game_type is required")

    # Remove only the season-specific marginal game_type effect. This deliberately
    # does not use any candidate signal, so persistence afterwards is evidence that
    # the candidate carries temporal structure beyond game_type itself.
    y = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    gt_mean = y.groupby([frame[season_col], frame["game_type"].astype(str)]).transform("mean")
    frame[RESIDUAL_COL] = y - gt_mean

    atlas_path = (ROOT / args.atlas).resolve()
    if not atlas_path.is_file():
        raise FileNotFoundError(
            f"Regime Atlas output not found: {atlas_path}. Run scripts/run_regime_atlas.py first."
        )
    original = pd.read_csv(atlas_path)
    original = original.loc[
        ~original["components"].astype(str).str.contains("game_type", regex=False)
        & original["classification"].isin(
            ["post_2023_regime", "sign_flip_or_reversal", "drifting_or_changepoint"]
        )
    ].sort_values(["regime_score", "shift_2023_rmse"], ascending=False)
    selected = original.head(args.top).copy()
    if selected.empty:
        raise RuntimeError("No non-game_type regime candidates found in atlas output")

    same_player_mask, same_stats = atlas._same_player_mask(
        frame,
        pitcher_col="pitcher_id",
        season_col=season_col,
        min_old_seasons=args.same_player_min_old_seasons,
    )
    categorical_features = set(CANONICAL_CATEGORICAL)

    print("[Game-Type Controlled Regime Validation]")
    print("  training              : NONE (CPU statistical screen only)")
    print("  control               : residual = y - E[y | season, game_type]")
    print(f"  candidates            : top {len(selected)} non-game_type atlas signals")
    print(
        f"  same-player cohort    : {same_stats['pitchers']:,} pitchers / "
        f"{same_stats['rows']:,} rows ({same_stats['row_fraction']:.1%})"
    )

    results = []
    interaction_lookup = dict(atlas.DEFAULT_INTERACTIONS)
    for idx, row in enumerate(selected.itertuples(index=False), start=1):
        signal = str(row.signal)
        if signal in interaction_lookup:
            components = interaction_lookup[signal]
        else:
            components = str(row.components).split("+")
        metrics = _metrics_for_signal(
            frame,
            signal=signal,
            components=components,
            season_col=season_col,
            residual_col=RESIDUAL_COL,
            categorical_features=categorical_features,
            numeric_bins=args.numeric_bins,
            max_auto_categories=args.max_auto_categories,
            min_era_count=args.min_era_count,
            min_effect_for_flip=args.min_effect_for_flip,
            same_player_mask=same_player_mask,
        )
        metrics["original_shift_2023_rmse"] = float(row.shift_2023_rmse)
        metrics["original_sign_flip_rate_2023"] = float(row.sign_flip_rate_2023)
        metrics["original_regime_score"] = float(row.regime_score)
        orig_shift = metrics["original_shift_2023_rmse"]
        ctrl_shift = metrics["controlled_shift_2023_rmse"]
        metrics["shift_retention_after_game_type_control"] = (
            ctrl_shift / orig_shift if np.isfinite(orig_shift) and orig_shift > 0 else np.nan
        )
        results.append(metrics)
        print(
            f"  [{idx:02d}/{len(selected):02d}] {signal:<38} "
            f"orig={orig_shift:.5f} ctrl={ctrl_shift:.5f} "
            f"retain={metrics['shift_retention_after_game_type_control']:.2f} "
            f"flip={metrics['controlled_sign_flip_rate_2023']:.2f} "
            f"cp={metrics['controlled_best_change_year']} "
            f"same={metrics['controlled_same_player_preservation']:.2f}"
        )

    result_df = pd.DataFrame(results)
    result_df["verdict"] = result_df.apply(_verdict, axis=1)
    verdict_rank = {
        "INDEPENDENT_REGIME_FLIP": 0,
        "INDEPENDENT_REGIME_SHIFT": 1,
        "WEAK_OR_MIXED_AFTER_CONTROL": 2,
        "MOSTLY_EXPLAINED_BY_GAME_TYPE": 3,
    }
    result_df["_rank"] = result_df["verdict"].map(verdict_rank).fillna(9)
    result_df = result_df.sort_values(
        ["_rank", "controlled_changepoint_ratio_2023", "controlled_shift_2023_rmse"],
        ascending=[True, False, False],
    ).drop(columns="_rank")

    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out / "game_type_controlled_regime.csv", index=False)
    save_json(
        {
            "control": "y - E[y | season, game_type]",
            "same_player_cohort": same_stats,
            "candidate_count": int(len(result_df)),
            "independent_flip_signals": result_df.loc[
                result_df["verdict"].eq("INDEPENDENT_REGIME_FLIP"), "signal"
            ].tolist(),
            "independent_shift_signals": result_df.loc[
                result_df["verdict"].eq("INDEPENDENT_REGIME_SHIFT"), "signal"
            ].tolist(),
        },
        out / "summary.json",
    )

    cols = [
        "signal",
        "verdict",
        "original_shift_2023_rmse",
        "controlled_shift_2023_rmse",
        "shift_retention_after_game_type_control",
        "controlled_changepoint_ratio_2023",
        "controlled_sign_flip_rate_2023",
        "controlled_best_change_year",
        "controlled_same_player_preservation",
    ]
    print("\n[Controlled Regime Candidates]")
    print(result_df[cols].to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
