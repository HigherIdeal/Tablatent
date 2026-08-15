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


OLD_YEARS = [2019, 2020, 2021, 2022]
RECENT_YEARS = [2023, 2024]
ALL_YEARS = OLD_YEARS + RECENT_YEARS
VARIANT = "recent_raw_game_type"


def _weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.average(values[mask] ** 2, weights=weights[mask])))


def _make_groups(
    series: pd.Series,
    *,
    categorical: bool,
    numeric_bins: int,
    max_auto_categories: int,
) -> tuple[pd.Series, str, int]:
    """Create target-independent groups shared across all seasons."""
    if categorical or int(series.nunique(dropna=False)) <= max_auto_categories:
        groups = series.astype("string").fillna("<MISSING>").astype(str)
        return groups, "categorical", int(groups.nunique(dropna=False))

    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    if finite.nunique(dropna=True) < 3:
        groups = numeric.astype("string").fillna("<MISSING>").astype(str)
        return groups, "categorical", int(groups.nunique(dropna=False))

    quantiles = np.linspace(0.0, 1.0, numeric_bins + 1)
    edges = np.unique(np.nanquantile(finite.to_numpy(np.float64), quantiles))
    if len(edges) < 3:
        groups = numeric.astype("string").fillna("<MISSING>").astype(str)
        return groups, "categorical", int(groups.nunique(dropna=False))

    values = numeric.to_numpy(np.float64)
    labels = np.full(len(values), -1, dtype=np.int16)
    valid = np.isfinite(values)
    labels[valid] = np.digitize(values[valid], edges[1:-1], right=True).astype(np.int16)
    groups = pd.Series(labels, index=series.index).astype(str)
    groups = groups.where(groups.ne("-1"), "<MISSING>")
    return groups, "quantile_bins", int(pd.Series(labels[valid]).nunique()) + int((~valid).any())


def _year_profile(
    groups: pd.Series,
    target: pd.Series,
    season: pd.Series,
    year: int,
) -> pd.DataFrame:
    mask = season.eq(year)
    y = target.loc[mask]
    g = groups.loc[mask]
    prior = float(y.mean())
    temp = pd.DataFrame({"group": g.to_numpy(), "y": y.to_numpy(np.float64)})
    prof = temp.groupby("group", dropna=False)["y"].agg(["count", "mean"]).reset_index()
    prof["effect"] = prof["mean"] - prior
    prof["season"] = int(year)
    prof["season_prior"] = prior
    return prof[["season", "group", "count", "mean", "effect", "season_prior"]]


def _aggregate_era(profiles: dict[int, pd.DataFrame], years: list[int]) -> pd.DataFrame:
    parts = []
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


def _internal_rmse(
    profiles: dict[int, pd.DataFrame],
    years: list[int],
    era_profile: pd.DataFrame,
) -> float:
    total_sq = 0.0
    total_weight = 0.0
    era_effect = era_profile["effect"].to_dict()
    for year in years:
        prof = profiles[year]
        for row in prof.itertuples(index=False):
            if row.group not in era_effect:
                continue
            diff = float(row.effect) - float(era_effect[row.group])
            weight = float(row.count)
            total_sq += weight * diff * diff
            total_weight += weight
    if total_weight <= 0:
        return float("nan")
    return float(math.sqrt(total_sq / total_weight))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0.0 or np.std(y[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _classify(
    shift_rmse: float,
    old_internal: float,
    recent_internal: float,
    sign_flip_rate: float,
) -> str:
    if not np.isfinite(shift_rmse):
        return "insufficient_support"
    recent_scale = recent_internal if np.isfinite(recent_internal) else 0.0
    old_scale = old_internal if np.isfinite(old_internal) else 0.0
    internal = max(old_scale, recent_scale, 0.002)

    if shift_rmse >= 0.005 and shift_rmse >= 1.5 * internal and recent_scale <= 0.75 * shift_rmse:
        return "regime_sensitive"
    if shift_rmse <= 0.0025 and sign_flip_rate <= 0.10:
        return "stable"
    if shift_rmse >= 0.0035 and recent_scale < shift_rmse:
        return "drifting"
    return "unstable_or_weak"


def audit_feature(
    frame: pd.DataFrame,
    feature: str,
    target_col: str,
    season_col: str,
    categorical_features: set[str],
    numeric_bins: int,
    max_auto_categories: int,
    min_era_count: int,
    min_effect_for_flip: float,
) -> tuple[dict, pd.DataFrame]:
    if feature == season_col:
        return (
            {
                "feature": feature,
                "grouping": "time_index",
                "groups": int(frame[feature].nunique(dropna=False)),
                "supported_groups": 0,
                "old_recent_effect_rmse": float("nan"),
                "old_internal_rmse": float("nan"),
                "recent_internal_rmse": float("nan"),
                "changepoint_ratio": float("nan"),
                "sign_flip_rate": float("nan"),
                "era_effect_corr": float("nan"),
                "old_effect_rms": float("nan"),
                "recent_effect_rms": float("nan"),
                "priority_score": float("nan"),
                "classification": "time_index",
            },
            pd.DataFrame(),
        )

    groups, grouping, group_count = _make_groups(
        frame[feature],
        categorical=feature in categorical_features,
        numeric_bins=numeric_bins,
        max_auto_categories=max_auto_categories,
    )
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    target = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    profiles = {year: _year_profile(groups, target, season, year) for year in ALL_YEARS}

    long_rows = []
    for year, prof in profiles.items():
        temp = prof.copy()
        temp.insert(0, "feature", feature)
        temp.insert(1, "grouping", grouping)
        long_rows.append(temp)
    long_df = pd.concat(long_rows, ignore_index=True)

    old = _aggregate_era(profiles, OLD_YEARS).rename(
        columns={"count": "old_count", "effect": "old_effect"}
    )
    recent = _aggregate_era(profiles, RECENT_YEARS).rename(
        columns={"count": "recent_count", "effect": "recent_effect"}
    )
    joined = old.join(recent, how="inner")
    supported = joined.loc[
        joined["old_count"].ge(min_era_count) & joined["recent_count"].ge(min_era_count)
    ].copy()

    if supported.empty:
        shift_rmse = float("nan")
        sign_flip_rate = float("nan")
        corr = float("nan")
        old_effect_rms = float("nan")
        recent_effect_rms = float("nan")
    else:
        weights = np.minimum(
            supported["old_count"].to_numpy(np.float64),
            supported["recent_count"].to_numpy(np.float64),
        )
        old_effect = supported["old_effect"].to_numpy(np.float64)
        recent_effect = supported["recent_effect"].to_numpy(np.float64)
        shift_rmse = _weighted_rmse(recent_effect - old_effect, weights)
        old_effect_rms = _weighted_rmse(old_effect, weights)
        recent_effect_rms = _weighted_rmse(recent_effect, weights)
        strong = (np.abs(old_effect) >= min_effect_for_flip) & (
            np.abs(recent_effect) >= min_effect_for_flip
        )
        flips = strong & (old_effect * recent_effect < 0.0)
        strong_weight = float(weights[strong].sum())
        sign_flip_rate = (
            float(weights[flips].sum() / strong_weight) if strong_weight > 0 else 0.0
        )
        corr = _safe_corr(old_effect, recent_effect)

    old_internal = _internal_rmse(profiles, OLD_YEARS, old)
    recent_internal = _internal_rmse(profiles, RECENT_YEARS, recent)
    internal_floor = max(
        old_internal if np.isfinite(old_internal) else 0.0,
        recent_internal if np.isfinite(recent_internal) else 0.0,
        0.002,
    )
    changepoint_ratio = (
        float(shift_rmse / internal_floor) if np.isfinite(shift_rmse) else float("nan")
    )
    if np.isfinite(shift_rmse):
        stability_bonus = 1.0 / max(recent_internal if np.isfinite(recent_internal) else 0.0, 0.002)
        priority_score = float(shift_rmse * (1.0 + 1.5 * sign_flip_rate) * stability_bonus)
    else:
        priority_score = float("nan")

    classification = _classify(
        shift_rmse=shift_rmse,
        old_internal=old_internal,
        recent_internal=recent_internal,
        sign_flip_rate=sign_flip_rate if np.isfinite(sign_flip_rate) else 0.0,
    )
    summary = {
        "feature": feature,
        "grouping": grouping,
        "groups": group_count,
        "supported_groups": int(len(supported)),
        "old_recent_effect_rmse": shift_rmse,
        "old_internal_rmse": old_internal,
        "recent_internal_rmse": recent_internal,
        "changepoint_ratio": changepoint_ratio,
        "sign_flip_rate": sign_flip_rate,
        "era_effect_corr": corr,
        "old_effect_rms": old_effect_rms,
        "recent_effect_rms": recent_effect_rms,
        "priority_score": priority_score,
        "classification": classification,
    }
    return summary, long_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit season-to-season target relationships for the current canonical + success_state feature set. "
            "The audit is target-aware but model-free: it ranks features whose conditional effects change between "
            "2019-2022 and the post-2023 regime while remaining relatively consistent in 2023-2024."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--numeric-bins", type=int, default=10)
    parser.add_argument("--max-auto-categories", type=int, default=20)
    parser.add_argument("--min-era-count", type=int, default=500)
    parser.add_argument("--min-effect-for-flip", type=float, default=0.002)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs/temporal_feature_stability_audit")
    args = parser.parse_args()

    if args.numeric_bins < 3:
        raise ValueError("numeric-bins must be >= 3")
    if args.min_era_count <= 0:
        raise ValueError("min-era-count must be positive")

    config = load_config(ROOT / args.config)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    observed_years = sorted(frame[season_col].unique().tolist())
    missing_years = sorted(set(ALL_YEARS) - set(observed_years))
    if missing_years:
        raise RuntimeError(f"missing required seasons: {missing_years}")

    features = recent_core.feature_set(VARIANT)
    categorical_features = set(CANONICAL_CATEGORICAL)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    long_parts = []
    print("[Temporal Feature Stability Audit]")
    print(f"  old era   : {OLD_YEARS}")
    print(f"  recent era: {RECENT_YEARS}")
    print(f"  features  : {len(features)}")
    print(
        f"  grouping  : categorical/raw or {args.numeric_bins} global quantile bins; "
        f"min era count={args.min_era_count}"
    )

    for idx, feature in enumerate(features, start=1):
        summary, long_df = audit_feature(
            frame=frame,
            feature=feature,
            target_col=target_col,
            season_col=season_col,
            categorical_features=categorical_features,
            numeric_bins=args.numeric_bins,
            max_auto_categories=args.max_auto_categories,
            min_era_count=args.min_era_count,
            min_effect_for_flip=args.min_effect_for_flip,
        )
        summaries.append(summary)
        if not long_df.empty:
            long_parts.append(long_df)
        shift = summary["old_recent_effect_rmse"]
        shift_text = f"{shift:.5f}" if np.isfinite(shift) else "nan"
        print(
            f"  [{idx:02d}/{len(features):02d}] {feature:<42} "
            f"class={summary['classification']:<18} shift={shift_text}"
        )

    summary_df = pd.DataFrame(summaries)
    class_rank = {
        "regime_sensitive": 0,
        "drifting": 1,
        "unstable_or_weak": 2,
        "stable": 3,
        "insufficient_support": 4,
        "time_index": 5,
    }
    summary_df["class_rank"] = summary_df["classification"].map(class_rank).fillna(99)
    summary_df = summary_df.sort_values(
        ["class_rank", "priority_score", "old_recent_effect_rmse"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["class_rank"])
    summary_df.to_csv(output_dir / "feature_stability.csv", index=False)

    if long_parts:
        pd.concat(long_parts, ignore_index=True).to_csv(
            output_dir / "per_season_group_effects.csv", index=False
        )

    eligible = summary_df.loc[
        summary_df["classification"].isin(["regime_sensitive", "drifting"])
        & summary_df["feature"].ne(season_col)
    ].head(args.top_candidates)
    candidate_payload = {
        "source": "2019-2022 vs 2023-2024 conditional-effect stability audit",
        "warning": "Candidates are hypotheses for stable-expert ablation, not automatic drops.",
        "features": eligible[
            [
                "feature",
                "classification",
                "priority_score",
                "old_recent_effect_rmse",
                "old_internal_rmse",
                "recent_internal_rmse",
                "changepoint_ratio",
                "sign_flip_rate",
            ]
        ].to_dict(orient="records"),
    }
    (output_dir / "regime_sensitive_candidates.json").write_text(
        json.dumps(candidate_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_json(
        {
            "old_years": OLD_YEARS,
            "recent_years": RECENT_YEARS,
            "variant": VARIANT,
            "numeric_bins": args.numeric_bins,
            "max_auto_categories": args.max_auto_categories,
            "min_era_count": args.min_era_count,
            "min_effect_for_flip": args.min_effect_for_flip,
            "top_candidates": args.top_candidates,
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    print("\n[Top regime-sensitive / drifting candidates]")
    display = summary_df.loc[
        summary_df["classification"].isin(["regime_sensitive", "drifting"])
    ].head(20)
    if display.empty:
        print("  none under current thresholds")
    else:
        print(
            display[
                [
                    "feature",
                    "classification",
                    "old_recent_effect_rmse",
                    "old_internal_rmse",
                    "recent_internal_rmse",
                    "changepoint_ratio",
                    "sign_flip_rate",
                    "priority_score",
                ]
            ].to_string(index=False)
        )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
