from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config, save_json


YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
OLD_YEARS = [2019, 2020, 2021, 2022]
RECENT_YEARS = [2023, 2024]
CHANGE_YEARS = [2021, 2022, 2023]
VARIANT = "recent_raw_game_type"

# Hand-picked interactions that are both interpretable and plausible sources of
# temporal non-stationarity.  They intentionally avoid player IDs as model-like
# predictors; pitcher_id is used only later to build a same-player control cohort.
DEFAULT_INTERACTIONS: dict[str, list[str]] = {
    "game_type_x_count": ["game_type", "balls_before", "strikes_before"],
    "game_type_x_hand_matchup": ["game_type", "pitcher_hand", "batter_hand"],
    "game_type_x_pitcher_n": ["game_type", "asof_pitcher_n"],
    "game_type_x_pitcher_success": ["game_type", "asof_pitcher_success_rate"],
    "game_type_x_pitcher_ball_rate": ["game_type", "asof_pitcher_ball_rate"],
    "game_type_x_pitcher_strike_rate": ["game_type", "asof_pitcher_strike_rate"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "hand_matchup_x_count": ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"],
    "pitcher_ball_rate_x_experience": ["asof_pitcher_ball_rate", "asof_pitcher_n"],
    "pitcher_strike_rate_x_experience": ["asof_pitcher_strike_rate", "asof_pitcher_n"],
    "pitcher_success_x_experience": ["asof_pitcher_success_rate", "asof_pitcher_n"],
    "batter_success_x_experience": ["asof_batter_success_rate", "asof_batter_n"],
    "fastball_rate_x_batter_hand": ["asof_pitcher_fastball_rate", "batter_hand"],
    "breaking_rate_x_batter_hand": ["asof_pitcher_breaking_rate", "batter_hand"],
    "offspeed_rate_x_batter_hand": ["asof_pitcher_offspeed_rate", "batter_hand"],
    "recent_success_x_experience": ["eng_ps_recent_mean_135", "asof_pitcher_n"],
    "recent_delta_x_experience": ["eng_ps_recent_mean_minus_long", "asof_pitcher_n"],
    "inning_x_li": ["inning", "li"],
    "base_state_x_count": ["base_state", "balls_before", "strikes_before"],
}


def _safe_float(value: float | int | np.floating) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.average(values[mask] ** 2, weights=weights[mask])))


def _make_component_groups(
    series: pd.Series,
    *,
    categorical: bool,
    numeric_bins: int,
    max_auto_categories: int,
) -> tuple[pd.Series, str, int]:
    if categorical or int(series.nunique(dropna=False)) <= max_auto_categories:
        groups = series.astype("string").fillna("<MISSING>").astype(str)
        return groups, "categorical", int(groups.nunique(dropna=False))

    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or len(np.unique(finite)) < 3:
        groups = numeric.astype("string").fillna("<MISSING>").astype(str)
        return groups, "categorical", int(groups.nunique(dropna=False))

    edges = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, numeric_bins + 1)))
    if len(edges) < 3:
        groups = numeric.astype("string").fillna("<MISSING>").astype(str)
        return groups, "categorical", int(groups.nunique(dropna=False))

    labels = np.full(len(values), -1, dtype=np.int16)
    valid = np.isfinite(values)
    labels[valid] = np.digitize(values[valid], edges[1:-1], right=True).astype(np.int16)
    groups = pd.Series(labels, index=series.index).astype(str)
    groups = groups.where(groups.ne("-1"), "<MISSING>")
    return groups, "quantile_bins", int(pd.Series(labels[valid]).nunique()) + int((~valid).any())


def _make_signal_groups(
    frame: pd.DataFrame,
    components: list[str],
    categorical_features: set[str],
    numeric_bins: int,
    max_auto_categories: int,
) -> tuple[pd.Series, str, int]:
    grouped_parts: list[pd.Series] = []
    modes: list[str] = []
    for component in components:
        groups, mode, _ = _make_component_groups(
            frame[component],
            categorical=component in categorical_features,
            numeric_bins=numeric_bins,
            max_auto_categories=max_auto_categories,
        )
        grouped_parts.append(groups.rename(component))
        modes.append(mode)

    if len(grouped_parts) == 1:
        groups = grouped_parts[0]
    else:
        joined = pd.concat(grouped_parts, axis=1).astype(str)
        groups = joined.agg("|".join, axis=1)
    grouping = "+".join(modes)
    return groups, grouping, int(groups.nunique(dropna=False))


def _year_profile(groups: pd.Series, target: pd.Series, season: pd.Series, year: int) -> pd.DataFrame:
    mask = season.eq(year)
    y = target.loc[mask].astype(float)
    g = groups.loc[mask]
    prior = float(y.mean())
    temp = pd.DataFrame({"group": g.to_numpy(), "y": y.to_numpy(np.float64)})
    prof = temp.groupby("group", dropna=False)["y"].agg(["count", "mean"]).reset_index()
    prof["effect"] = prof["mean"] - prior
    prof["season"] = int(year)
    prof["season_prior"] = prior
    return prof[["season", "group", "count", "mean", "effect", "season_prior"]]


def _aggregate_era(profiles: dict[int, pd.DataFrame], years: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in years:
        part = profiles[year][["group", "count", "effect"]].copy()
        part["weighted_effect"] = part["count"] * part["effect"]
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)
    agg = merged.groupby("group", dropna=False).agg(
        count=("count", "sum"), weighted_effect=("weighted_effect", "sum")
    )
    agg["effect"] = agg["weighted_effect"] / agg["count"].clip(lower=1)
    return agg[["count", "effect"]]


def _internal_rmse(profiles: dict[int, pd.DataFrame], years: list[int], era: pd.DataFrame) -> float:
    era_effect = era["effect"].to_dict()
    total_sq = 0.0
    total_weight = 0.0
    for year in years:
        for row in profiles[year].itertuples(index=False):
            if row.group not in era_effect:
                continue
            weight = float(row.count)
            diff = float(row.effect) - float(era_effect[row.group])
            total_sq += weight * diff * diff
            total_weight += weight
    return float(math.sqrt(total_sq / total_weight)) if total_weight > 0 else float("nan")


def _era_shift(
    profiles: dict[int, pd.DataFrame],
    left_years: list[int],
    right_years: list[int],
    min_era_count: int,
    min_effect_for_flip: float,
) -> dict[str, float | int]:
    left = _aggregate_era(profiles, left_years).rename(
        columns={"count": "left_count", "effect": "left_effect"}
    )
    right = _aggregate_era(profiles, right_years).rename(
        columns={"count": "right_count", "effect": "right_effect"}
    )
    joined = left.join(right, how="inner")
    supported = joined.loc[
        joined["left_count"].ge(min_era_count) & joined["right_count"].ge(min_era_count)
    ].copy()
    if supported.empty:
        return {
            "supported_groups": 0,
            "shift_rmse": float("nan"),
            "left_internal_rmse": float("nan"),
            "right_internal_rmse": float("nan"),
            "changepoint_ratio": float("nan"),
            "sign_flip_rate": float("nan"),
            "left_effect_rms": float("nan"),
            "right_effect_rms": float("nan"),
        }

    weights = np.minimum(
        supported["left_count"].to_numpy(np.float64),
        supported["right_count"].to_numpy(np.float64),
    )
    left_effect = supported["left_effect"].to_numpy(np.float64)
    right_effect = supported["right_effect"].to_numpy(np.float64)
    shift = _weighted_rms(right_effect - left_effect, weights)
    left_internal = _internal_rmse(profiles, left_years, left.rename(columns={"left_effect": "effect"}))
    right_internal = _internal_rmse(profiles, right_years, right.rename(columns={"right_effect": "effect"}))
    floor = max(
        left_internal if np.isfinite(left_internal) else 0.0,
        right_internal if np.isfinite(right_internal) else 0.0,
        0.002,
    )
    strong = (np.abs(left_effect) >= min_effect_for_flip) & (
        np.abs(right_effect) >= min_effect_for_flip
    )
    flips = strong & (left_effect * right_effect < 0.0)
    strong_weight = float(weights[strong].sum())
    flip_rate = float(weights[flips].sum() / strong_weight) if strong_weight > 0 else 0.0
    return {
        "supported_groups": int(len(supported)),
        "shift_rmse": shift,
        "left_internal_rmse": left_internal,
        "right_internal_rmse": right_internal,
        "changepoint_ratio": float(shift / floor) if np.isfinite(shift) else float("nan"),
        "sign_flip_rate": flip_rate,
        "left_effect_rms": _weighted_rms(left_effect, weights),
        "right_effect_rms": _weighted_rms(right_effect, weights),
    }


def _distribution_jsd(
    groups: pd.Series,
    season: pd.Series,
    left_years: list[int],
    right_years: list[int],
) -> float:
    left = groups.loc[season.isin(left_years)].value_counts(normalize=True, dropna=False)
    right = groups.loc[season.isin(right_years)].value_counts(normalize=True, dropna=False)
    support = left.index.union(right.index)
    p = left.reindex(support, fill_value=0.0).to_numpy(np.float64)
    q = right.reindex(support, fill_value=0.0).to_numpy(np.float64)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _same_player_mask(
    frame: pd.DataFrame,
    *,
    pitcher_col: str,
    season_col: str,
    min_old_seasons: int,
) -> tuple[pd.Series, dict[str, int | float]]:
    pitcher_seasons = (
        frame[[pitcher_col, season_col]]
        .dropna()
        .drop_duplicates()
        .groupby(pitcher_col)[season_col]
        .agg(lambda s: set(map(int, s)))
    )
    keep = []
    for pitcher, seasons in pitcher_seasons.items():
        old_count = len(set(OLD_YEARS) & seasons)
        if old_count >= min_old_seasons and set(RECENT_YEARS).issubset(seasons):
            keep.append(pitcher)
    mask = frame[pitcher_col].isin(keep)
    stats = {
        "pitchers": int(len(keep)),
        "rows": int(mask.sum()),
        "row_fraction": float(mask.mean()),
        "min_old_seasons": int(min_old_seasons),
    }
    return mask, stats


def _signal_summary(
    frame: pd.DataFrame,
    *,
    signal: str,
    components: list[str],
    target_col: str,
    season_col: str,
    categorical_features: set[str],
    numeric_bins: int,
    max_auto_categories: int,
    min_era_count: int,
    min_effect_for_flip: float,
    same_player_mask: pd.Series,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    groups, grouping, group_count = _make_signal_groups(
        frame,
        components,
        categorical_features,
        numeric_bins,
        max_auto_categories,
    )
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    target = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    profiles = {year: _year_profile(groups, target, season, year) for year in YEARS}

    profile_rows: list[pd.DataFrame] = []
    strength_by_year: dict[int, float] = {}
    for year in YEARS:
        prof = profiles[year].copy()
        prof.insert(0, "signal", signal)
        prof.insert(1, "components", "+".join(components))
        prof.insert(2, "grouping", grouping)
        profile_rows.append(prof)
        strength_by_year[year] = _weighted_rms(
            prof["effect"].to_numpy(np.float64), prof["count"].to_numpy(np.float64)
        )

    cp_rows: list[dict] = []
    for change_year in CHANGE_YEARS:
        left_years = [year for year in YEARS if year < change_year]
        right_years = [year for year in YEARS if year >= change_year]
        metrics = _era_shift(
            profiles,
            left_years,
            right_years,
            min_era_count=min_era_count,
            min_effect_for_flip=min_effect_for_flip,
        )
        cp_rows.append(
            {
                "signal": signal,
                "change_year": int(change_year),
                "left_years": ",".join(map(str, left_years)),
                "right_years": ",".join(map(str, right_years)),
                **metrics,
            }
        )
    cp_df = pd.DataFrame(cp_rows)
    valid_cp = cp_df.loc[np.isfinite(cp_df["changepoint_ratio"].to_numpy(np.float64))]
    if valid_cp.empty:
        best_change_year = np.nan
        best_cp_ratio = np.nan
    else:
        best = valid_cp.sort_values(["changepoint_ratio", "shift_rmse"], ascending=False).iloc[0]
        best_change_year = int(best["change_year"])
        best_cp_ratio = float(best["changepoint_ratio"])

    fixed = _era_shift(
        profiles,
        OLD_YEARS,
        RECENT_YEARS,
        min_era_count=min_era_count,
        min_effect_for_flip=min_effect_for_flip,
    )
    distribution_jsd = _distribution_jsd(groups, season, OLD_YEARS, RECENT_YEARS)

    same = frame.loc[same_player_mask]
    same_groups = groups.loc[same_player_mask]
    same_season = season.loc[same_player_mask]
    same_target = target.loc[same_player_mask]
    same_profiles = {
        year: _year_profile(same_groups, same_target, same_season, year) for year in YEARS
    }
    same_metrics = _era_shift(
        same_profiles,
        OLD_YEARS,
        RECENT_YEARS,
        min_era_count=max(100, min_era_count // 3),
        min_effect_for_flip=min_effect_for_flip,
    )
    full_shift = float(fixed["shift_rmse"])
    same_shift = float(same_metrics["shift_rmse"])
    preservation = (
        float(same_shift / full_shift)
        if np.isfinite(full_shift) and full_shift > 0 and np.isfinite(same_shift)
        else float("nan")
    )

    fixed_ratio = float(fixed["changepoint_ratio"])
    flip = float(fixed["sign_flip_rate"])
    persistence_bonus = np.clip(preservation, 0.5, 1.5) if np.isfinite(preservation) else 0.75
    regime_score = (
        float(fixed_ratio * (1.0 + flip) * persistence_bonus)
        if np.isfinite(fixed_ratio)
        else float("nan")
    )

    summary = {
        "signal": signal,
        "components": "+".join(components),
        "signal_type": "feature" if len(components) == 1 else "interaction",
        "grouping": grouping,
        "groups": int(group_count),
        "supported_groups_2023": int(fixed["supported_groups"]),
        "shift_2023_rmse": _safe_float(fixed["shift_rmse"]),
        "old_internal_rmse": _safe_float(fixed["left_internal_rmse"]),
        "recent_internal_rmse": _safe_float(fixed["right_internal_rmse"]),
        "changepoint_ratio_2023": _safe_float(fixed["changepoint_ratio"]),
        "sign_flip_rate_2023": _safe_float(fixed["sign_flip_rate"]),
        "old_effect_rms": _safe_float(fixed["left_effect_rms"]),
        "recent_effect_rms": _safe_float(fixed["right_effect_rms"]),
        "distribution_jsd_2023": distribution_jsd,
        "best_change_year": best_change_year,
        "best_changepoint_ratio": best_cp_ratio,
        "same_player_shift_2023_rmse": _safe_float(same_metrics["shift_rmse"]),
        "same_player_changepoint_ratio_2023": _safe_float(same_metrics["changepoint_ratio"]),
        "same_player_sign_flip_rate_2023": _safe_float(same_metrics["sign_flip_rate"]),
        "same_player_preservation": preservation,
        "composition_gap_rmse": (
            abs(full_shift - same_shift)
            if np.isfinite(full_shift) and np.isfinite(same_shift)
            else float("nan")
        ),
        "regime_score": regime_score,
    }
    for year in YEARS:
        summary[f"effect_rms_{year}"] = strength_by_year[year]
    return summary, pd.concat(profile_rows, ignore_index=True), cp_df


def _classification(row: pd.Series) -> str:
    ratio = row.get("changepoint_ratio_2023", np.nan)
    shift = row.get("shift_2023_rmse", np.nan)
    recent_internal = row.get("recent_internal_rmse", np.nan)
    flip = row.get("sign_flip_rate_2023", np.nan)
    preservation = row.get("same_player_preservation", np.nan)
    best_year = row.get("best_change_year", np.nan)

    if not np.isfinite(shift):
        return "insufficient_support"
    if (
        shift >= 0.005
        and np.isfinite(ratio)
        and ratio >= 1.5
        and best_year == 2023
        and (not np.isfinite(recent_internal) or recent_internal <= 0.8 * shift)
        and (not np.isfinite(preservation) or preservation >= 0.6)
    ):
        return "post_2023_regime"
    if np.isfinite(flip) and flip >= 0.35 and shift >= 0.004:
        return "sign_flip_or_reversal"
    if shift >= 0.0035 and np.isfinite(ratio) and ratio >= 1.2:
        return "drifting_or_changepoint"
    if shift <= 0.0025 and (not np.isfinite(flip) or flip <= 0.10):
        return "stable"
    return "mixed_or_composition_sensitive"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CPU-only Regime Atlas over 2019-2024. The script screens single features and a curated set "
            "of interactions for post-2023 conditional-effect changes, searches the strongest changepoint, and "
            "repeats the old-vs-recent comparison on a bridge cohort of the same pitchers to separate composition "
            "shift from persistent regime shift."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--numeric-bins", type=int, default=6)
    parser.add_argument("--max-auto-categories", type=int, default=20)
    parser.add_argument("--min-era-count", type=int, default=500)
    parser.add_argument("--min-effect-for-flip", type=float, default=0.002)
    parser.add_argument("--same-player-min-old-seasons", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs/regime_atlas")
    args = parser.parse_args()

    if args.numeric_bins < 3:
        raise ValueError("numeric-bins must be >= 3")
    if args.same_player_min_old_seasons < 1 or args.same_player_min_old_seasons > 4:
        raise ValueError("same-player-min-old-seasons must be in [1, 4]")

    config = load_config(ROOT / args.config)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    missing_years = sorted(set(YEARS) - set(frame[season_col].unique().tolist()))
    if missing_years:
        raise RuntimeError(f"missing required seasons: {missing_years}")
    if "pitcher_id" not in frame.columns:
        raise ValueError("pitcher_id is required for same-player control")

    features = [f for f in recent_core.feature_set(VARIANT) if f != season_col]
    categorical_features = set(CANONICAL_CATEGORICAL)
    same_player_mask, same_player_stats = _same_player_mask(
        frame,
        pitcher_col="pitcher_id",
        season_col=season_col,
        min_old_seasons=args.same_player_min_old_seasons,
    )
    if int(same_player_mask.sum()) == 0:
        raise RuntimeError("same-player control cohort is empty")

    interactions = {
        name: components
        for name, components in DEFAULT_INTERACTIONS.items()
        if all(component in frame.columns for component in components)
    }
    signals = [(feature, [feature]) for feature in features] + list(interactions.items())

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Regime Atlas]")
    print(f"  years                  : {YEARS}")
    print(f"  fixed regime split     : {OLD_YEARS} -> {RECENT_YEARS}")
    print(f"  changepoint candidates : {CHANGE_YEARS}")
    print(f"  signals                : {len(features)} features + {len(interactions)} interactions")
    print(
        "  same-player cohort     : "
        f"{same_player_stats['pitchers']:,} pitchers / {same_player_stats['rows']:,} rows "
        f"({same_player_stats['row_fraction']:.1%})"
    )
    print("  training               : NONE (CPU statistical screen only)")

    summaries: list[dict] = []
    profiles: list[pd.DataFrame] = []
    changepoints: list[pd.DataFrame] = []
    for idx, (signal, components) in enumerate(signals, start=1):
        summary, profile_df, cp_df = _signal_summary(
            frame,
            signal=signal,
            components=components,
            target_col=target_col,
            season_col=season_col,
            categorical_features=categorical_features,
            numeric_bins=args.numeric_bins,
            max_auto_categories=args.max_auto_categories,
            min_era_count=args.min_era_count,
            min_effect_for_flip=args.min_effect_for_flip,
            same_player_mask=same_player_mask,
        )
        summaries.append(summary)
        profiles.append(profile_df)
        changepoints.append(cp_df)
        print(
            f"  [{idx:02d}/{len(signals):02d}] {signal:<38} "
            f"shift={summary['shift_2023_rmse']:.5f} "
            f"cp={summary['best_change_year']} "
            f"same={summary['same_player_preservation']:.2f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df["classification"] = summary_df.apply(_classification, axis=1)
    class_rank = {
        "post_2023_regime": 0,
        "sign_flip_or_reversal": 1,
        "drifting_or_changepoint": 2,
        "mixed_or_composition_sensitive": 3,
        "stable": 4,
        "insufficient_support": 5,
    }
    summary_df["class_rank"] = summary_df["classification"].map(class_rank).fillna(99)
    summary_df = summary_df.sort_values(
        ["class_rank", "regime_score", "shift_2023_rmse"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["class_rank"])

    summary_df.to_csv(output_dir / "regime_atlas.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(output_dir / "season_group_effects.csv", index=False)
    pd.concat(changepoints, ignore_index=True).to_csv(output_dir / "changepoint_scan.csv", index=False)

    top = summary_df.head(args.top).copy()
    top.to_csv(output_dir / "top_regime_candidates.csv", index=False)
    candidates = top.loc[
        top["classification"].isin(
            ["post_2023_regime", "sign_flip_or_reversal", "drifting_or_changepoint"]
        )
    ]
    save_json(
        output_dir / "regime_candidates.json",
        {
            "same_player_cohort": same_player_stats,
            "candidate_signals": candidates[
                [
                    "signal",
                    "components",
                    "signal_type",
                    "classification",
                    "shift_2023_rmse",
                    "changepoint_ratio_2023",
                    "sign_flip_rate_2023",
                    "best_change_year",
                    "same_player_preservation",
                    "distribution_jsd_2023",
                    "regime_score",
                ]
            ].to_dict(orient="records"),
        },
    )
    save_json(
        output_dir / "run_config.json",
        {
            "years": YEARS,
            "old_years": OLD_YEARS,
            "recent_years": RECENT_YEARS,
            "change_year_candidates": CHANGE_YEARS,
            "feature_variant": VARIANT,
            "numeric_bins": args.numeric_bins,
            "min_era_count": args.min_era_count,
            "same_player_min_old_seasons": args.same_player_min_old_seasons,
            "same_player_cohort": same_player_stats,
            "interactions": interactions,
            "canonical_invariant_check": invariant_check,
            "notes": [
                "This is a model-free discovery screen, not causal evidence.",
                "pitcher_id is used only to restrict the same-player control cohort; it is never a signal component.",
                "distribution_jsd_2023 measures composition change separately from target-effect change.",
                "Candidates should be validated next with frozen-baseline residual analysis before creating a new expert.",
            ],
        },
    )

    display_cols = [
        "signal",
        "signal_type",
        "classification",
        "shift_2023_rmse",
        "changepoint_ratio_2023",
        "sign_flip_rate_2023",
        "best_change_year",
        "same_player_preservation",
        "distribution_jsd_2023",
        "regime_score",
    ]
    print("\n[Top Regime Candidates]")
    print(top[display_cols].to_string(index=False))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
