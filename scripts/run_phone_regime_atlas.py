from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD_YEARS = [2019, 2020, 2021, 2022]
RECENT_YEARS = [2023, 2024]
ALL_YEARS = OLD_YEARS + RECENT_YEARS

SCALAR_SIGNALS = [
    "game_type",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "li",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

CATEGORICAL_HINTS = {
    "game_type",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
}


def _safe_token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _find_data_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"data file not found: {path}")
        return path

    # CSV is preferred on a phone because usecols avoids loading every column.
    candidates = [
        ROOT / "data" / "raw" / "extracted" / "train.csv",
        ROOT / "data" / "raw" / "train.csv",
        ROOT / "data" / "processed" / "train.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No train data found. Expected one of data/raw/extracted/train.csv, "
        "data/raw/train.csv, data/processed/train.pkl. Pass --data PATH if it is elsewhere."
    )


def _needed_columns() -> list[str]:
    columns = {
        "season",
        "control_success",
        "pitcher_id",
        "batter_id",
        *SCALAR_SIGNALS,
    }
    return sorted(columns)


def _load_analysis_frame(path: Path, max_rows: int | None) -> pd.DataFrame:
    wanted = _needed_columns()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in wanted if c in header]
        frame = pd.read_csv(path, usecols=usecols, nrows=max_rows, low_memory=False)
    elif suffix in {".pkl", ".pickle"}:
        print("  note: pickle input loads the full frame before selecting columns; CSV is lower-RAM on Termux")
        frame = pd.read_pickle(path)
        keep = [c for c in wanted if c in frame.columns]
        frame = frame.loc[:, keep]
        if max_rows is not None:
            frame = frame.iloc[:max_rows].copy()
    else:
        raise ValueError(f"unsupported data format: {path.suffix}")

    required = {"season", "control_success"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"required columns missing: {missing}")

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["control_success"] = pd.to_numeric(frame["control_success"], errors="coerce")
    frame = frame.dropna(subset=["season", "control_success"]).copy()
    frame["season"] = frame["season"].astype(np.int16)
    frame["control_success"] = frame["control_success"].astype(np.float32)
    frame = frame[frame["season"].isin(ALL_YEARS)].reset_index(drop=True)
    return frame


def _quantile_groups(series: pd.Series, bins: int) -> tuple[pd.Series, str, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(np.float64)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        groups = pd.Series("<MISSING>", index=series.index, dtype="string")
        return groups, "categorical", 1
    unique = np.unique(values[valid])
    if len(unique) <= max(20, bins * 2):
        groups = numeric.astype("string").fillna("<MISSING>")
        return groups, "categorical", int(groups.nunique(dropna=False))
    edges = np.unique(np.quantile(values[valid], np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        groups = numeric.astype("string").fillna("<MISSING>")
        return groups, "categorical", int(groups.nunique(dropna=False))
    labels = np.full(len(values), -1, dtype=np.int16)
    labels[valid] = np.digitize(values[valid], edges[1:-1], right=True).astype(np.int16)
    groups = pd.Series(labels, index=series.index).astype("string")
    groups = groups.where(groups.ne("-1"), "<MISSING>")
    return groups, "quantile_bins", int(pd.Series(labels[valid]).nunique()) + int((~valid).any())


def _make_groups(series: pd.Series, categorical: bool, bins: int) -> tuple[pd.Series, str, int]:
    if categorical:
        groups = _safe_token(series)
        return groups, "categorical", int(groups.nunique(dropna=False))
    return _quantile_groups(series, bins)


def _bin_token(series: pd.Series, bins: int) -> pd.Series:
    groups, _, _ = _quantile_groups(series, bins)
    return groups.astype(str)


def _interaction_signals(frame: pd.DataFrame, bins: int) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    def has(*cols: str) -> bool:
        return all(c in frame.columns for c in cols)

    if has("balls_before", "strikes_before"):
        count = _safe_token(frame["balls_before"]) + "|" + _safe_token(frame["strikes_before"])
        out["ix_count"] = count
        if "game_type" in frame.columns:
            out["ix_game_type_x_count"] = _safe_token(frame["game_type"]) + "|" + count
        if "base_state" in frame.columns:
            out["ix_base_x_count"] = _safe_token(frame["base_state"]) + "|" + count

    if has("pitcher_hand", "batter_hand"):
        hand = _safe_token(frame["pitcher_hand"]) + "|" + _safe_token(frame["batter_hand"])
        out["ix_hand_matchup"] = hand
        if "game_type" in frame.columns:
            out["ix_game_type_x_hand"] = _safe_token(frame["game_type"]) + "|" + hand

    if has("game_type", "asof_pitcher_n"):
        out["ix_game_type_x_pitcher_exp"] = (
            _safe_token(frame["game_type"]) + "|" + _bin_token(frame["asof_pitcher_n"], bins)
        )

    if has("asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_pitcher_n"):
        out["ix_control_x_pitcher_exp"] = (
            _bin_token(frame["asof_pitcher_ball_rate"], max(4, bins // 2))
            + "|"
            + _bin_token(frame["asof_pitcher_strike_rate"], max(4, bins // 2))
            + "|"
            + _bin_token(frame["asof_pitcher_n"], max(4, bins // 2))
        )

    if has("asof_batter_success_rate", "asof_batter_n"):
        out["ix_batter_success_x_exp"] = (
            _bin_token(frame["asof_batter_success_rate"], bins)
            + "|"
            + _bin_token(frame["asof_batter_n"], bins)
        )

    if has("asof_pitcher_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_n"):
        delta = pd.to_numeric(frame["asof_pitcher_prev3_game_success_rate"], errors="coerce") - pd.to_numeric(
            frame["asof_pitcher_success_rate"], errors="coerce"
        )
        out["ix_recent_form_x_pitcher_exp"] = _bin_token(delta, bins) + "|" + _bin_token(
            frame["asof_pitcher_n"], bins
        )

    mix_cols = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    if has(*mix_cols):
        mix = frame[mix_cols].apply(pd.to_numeric, errors="coerce")
        arr = mix.to_numpy(np.float64)
        valid = np.isfinite(arr).any(axis=1)
        labels = np.array(["<MISSING>"] * len(frame), dtype=object)
        safe = np.where(np.isfinite(arr), arr, -np.inf)
        names = np.asarray(["fastball", "breaking", "offspeed"], dtype=object)
        labels[valid] = names[np.argmax(safe[valid], axis=1)]
        dominant = pd.Series(labels, index=frame.index, dtype="string")
        out["ix_dominant_pitchmix"] = dominant
        if "batter_hand" in frame.columns:
            out["ix_pitchmix_x_batter_hand"] = dominant.astype(str) + "|" + _safe_token(frame["batter_hand"])
        if "game_type" in frame.columns:
            out["ix_game_type_x_pitchmix"] = _safe_token(frame["game_type"]) + "|" + dominant.astype(str)

    if has("inning", "li"):
        out["ix_inning_x_li"] = _safe_token(frame["inning"]) + "|" + _bin_token(frame["li"], bins)

    return out


def _profile(groups: pd.Series, target: pd.Series, season: pd.Series, year: int) -> pd.DataFrame:
    mask = season.eq(year)
    g = groups.loc[mask]
    y = target.loc[mask]
    prior = float(y.mean())
    temp = pd.DataFrame({"group": g.to_numpy(), "y": y.to_numpy(np.float64)})
    prof = temp.groupby("group", dropna=False, sort=False)["y"].agg(["count", "mean"]).reset_index()
    prof["effect"] = prof["mean"] - prior
    prof["season"] = int(year)
    prof["season_prior"] = prior
    return prof[["season", "group", "count", "mean", "effect", "season_prior"]]


def _era_profile(profiles: dict[int, pd.DataFrame], years: list[int]) -> pd.DataFrame:
    parts = []
    for year in years:
        p = profiles[year][["group", "count", "effect"]].copy()
        p["weighted_effect"] = p["count"] * p["effect"]
        parts.append(p)
    merged = pd.concat(parts, ignore_index=True)
    agg = merged.groupby("group", sort=False).agg(count=("count", "sum"), weighted=("weighted_effect", "sum"))
    agg["effect"] = agg["weighted"] / agg["count"].clip(lower=1)
    return agg[["count", "effect"]]


def _weighted_rmse(delta: np.ndarray, weight: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mask = np.isfinite(delta) & np.isfinite(weight) & (weight > 0)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.average(delta[mask] ** 2, weights=weight[mask])))


def _pair_rmse(a: pd.DataFrame, b: pd.DataFrame, min_count: int) -> float:
    aa = a.set_index("group")[["count", "effect"]].rename(columns={"count": "a_n", "effect": "a_e"})
    bb = b.set_index("group")[["count", "effect"]].rename(columns={"count": "b_n", "effect": "b_e"})
    joined = aa.join(bb, how="inner")
    joined = joined[(joined["a_n"] >= min_count) & (joined["b_n"] >= min_count)]
    if joined.empty:
        return float("nan")
    w = np.minimum(joined["a_n"].to_numpy(float), joined["b_n"].to_numpy(float))
    return _weighted_rmse(joined["b_e"].to_numpy(float) - joined["a_e"].to_numpy(float), w)


def _js_divergence(old: pd.DataFrame, recent: pd.DataFrame) -> float:
    a = old.set_index("group")["count"].astype(float)
    b = recent.set_index("group")["count"].astype(float)
    idx = a.index.union(b.index)
    p = a.reindex(idx, fill_value=0.0).to_numpy(float)
    q = b.reindex(idx, fill_value=0.0).to_numpy(float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _same_entity_mask(frame: pd.DataFrame, id_col: str, min_old: int, min_recent: int) -> tuple[pd.Series, dict]:
    if id_col not in frame.columns:
        return pd.Series(False, index=frame.index), {"available": False}
    ids = _safe_token(frame[id_col])
    old_counts = ids[frame["season"].isin(OLD_YEARS)].value_counts()
    recent_counts = ids[frame["season"].isin(RECENT_YEARS)].value_counts()
    eligible = old_counts[old_counts >= min_old].index.intersection(recent_counts[recent_counts >= min_recent].index)
    mask = ids.isin(eligible)
    return mask, {
        "available": True,
        "eligible_entities": int(len(eligible)),
        "rows": int(mask.sum()),
        "row_fraction": float(mask.mean()),
        "min_old_rows": int(min_old),
        "min_recent_rows": int(min_recent),
    }


def _audit_one(
    name: str,
    groups: pd.Series,
    frame: pd.DataFrame,
    min_era_count: int,
    min_year_count: int,
    cohort_mask: pd.Series | None,
) -> tuple[dict, pd.DataFrame]:
    season = frame["season"]
    target = frame["control_success"]
    profiles = {year: _profile(groups, target, season, year) for year in ALL_YEARS}
    old = _era_profile(profiles, OLD_YEARS).rename(columns={"count": "old_n", "effect": "old_e"})
    recent = _era_profile(profiles, RECENT_YEARS).rename(columns={"count": "recent_n", "effect": "recent_e"})
    joined = old.join(recent, how="inner")
    supported = joined[(joined["old_n"] >= min_era_count) & (joined["recent_n"] >= min_era_count)].copy()

    if supported.empty:
        shift = sign_flip = corr = float("nan")
    else:
        weight = np.minimum(supported["old_n"].to_numpy(float), supported["recent_n"].to_numpy(float))
        old_e = supported["old_e"].to_numpy(float)
        recent_e = supported["recent_e"].to_numpy(float)
        shift = _weighted_rmse(recent_e - old_e, weight)
        strong = (np.abs(old_e) >= 0.002) & (np.abs(recent_e) >= 0.002)
        strong_w = float(weight[strong].sum())
        flips = strong & (old_e * recent_e < 0)
        sign_flip = float(weight[flips].sum() / strong_w) if strong_w > 0 else 0.0
        corr = _safe_corr(old_e, recent_e)

    shock_22_23 = _pair_rmse(profiles[2022], profiles[2023], min_year_count)
    post_23_24 = _pair_rmse(profiles[2023], profiles[2024], min_year_count)
    pre_pairs = [
        _pair_rmse(profiles[2019], profiles[2020], min_year_count),
        _pair_rmse(profiles[2020], profiles[2021], min_year_count),
        _pair_rmse(profiles[2021], profiles[2022], min_year_count),
    ]
    pre_finite = [x for x in pre_pairs if np.isfinite(x)]
    pre_adjacent = float(np.mean(pre_finite)) if pre_finite else float("nan")
    background = max(
        pre_adjacent if np.isfinite(pre_adjacent) else 0.0,
        post_23_24 if np.isfinite(post_23_24) else 0.0,
        0.002,
    )
    changepoint_ratio = float(shock_22_23 / background) if np.isfinite(shock_22_23) else float("nan")
    composition_js = _js_divergence(old.reset_index(), recent.reset_index())

    cohort_shift = float("nan")
    cohort_ratio = float("nan")
    cohort_groups_supported = 0
    if cohort_mask is not None and bool(cohort_mask.any()):
        sub = frame.loc[cohort_mask].reset_index(drop=True)
        sub_groups = groups.loc[cohort_mask].reset_index(drop=True)
        sub_profiles = {year: _profile(sub_groups, sub["control_success"], sub["season"], year) for year in ALL_YEARS}
        sub_old = _era_profile(sub_profiles, OLD_YEARS).rename(columns={"count": "old_n", "effect": "old_e"})
        sub_recent = _era_profile(sub_profiles, RECENT_YEARS).rename(columns={"count": "recent_n", "effect": "recent_e"})
        sj = sub_old.join(sub_recent, how="inner")
        sj = sj[(sj["old_n"] >= max(100, min_era_count // 4)) & (sj["recent_n"] >= max(100, min_era_count // 4))]
        cohort_groups_supported = int(len(sj))
        if not sj.empty:
            w = np.minimum(sj["old_n"].to_numpy(float), sj["recent_n"].to_numpy(float))
            cohort_shift = _weighted_rmse(sj["recent_e"].to_numpy(float) - sj["old_e"].to_numpy(float), w)
            if np.isfinite(shift) and shift > 1e-12:
                cohort_ratio = float(cohort_shift / shift)

    post_stability = (
        float(shift / max(post_23_24, 0.002))
        if np.isfinite(shift) and np.isfinite(post_23_24)
        else 1.0
    )
    cohort_factor = float(np.clip(cohort_ratio, 0.5, 1.5)) if np.isfinite(cohort_ratio) else 1.0
    if np.isfinite(shift):
        regime_score = float(
            shift
            * np.clip(changepoint_ratio if np.isfinite(changepoint_ratio) else 1.0, 0.5, 8.0)
            * np.clip(post_stability, 0.5, 5.0)
            * (1.0 + (sign_flip if np.isfinite(sign_flip) else 0.0))
            * cohort_factor
        )
    else:
        regime_score = float("nan")

    if not np.isfinite(shift):
        classification = "insufficient_support"
    elif shift <= 0.003 and (not np.isfinite(shock_22_23) or shock_22_23 <= 0.005):
        classification = "stable"
    elif (
        shift >= 0.005
        and np.isfinite(changepoint_ratio)
        and changepoint_ratio >= 1.5
        and (not np.isfinite(post_23_24) or post_23_24 <= max(0.006, 0.8 * shift))
    ):
        classification = "regime_candidate"
    elif np.isfinite(composition_js) and composition_js >= 0.05 and np.isfinite(cohort_ratio) and cohort_ratio < 0.6:
        classification = "composition_sensitive"
    elif shift >= 0.004:
        classification = "drifting_or_mixed"
    else:
        classification = "weak"

    long = pd.concat(
        [profiles[y].assign(signal=name) for y in ALL_YEARS],
        ignore_index=True,
    )
    summary = {
        "signal": name,
        "groups": int(groups.nunique(dropna=False)),
        "supported_groups": int(len(supported)),
        "old_recent_effect_rmse": shift,
        "shock_2022_2023_rmse": shock_22_23,
        "post_2023_2024_rmse": post_23_24,
        "pre_adjacent_mean_rmse": pre_adjacent,
        "changepoint_2023_ratio": changepoint_ratio,
        "sign_flip_rate": sign_flip,
        "old_recent_effect_corr": corr,
        "composition_js": composition_js,
        "same_pitcher_effect_rmse": cohort_shift,
        "same_pitcher_vs_all_ratio": cohort_ratio,
        "same_pitcher_supported_groups": cohort_groups_supported,
        "regime_score": regime_score,
        "classification": classification,
    }
    return summary, long


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only regime atlas for Termux/phone use. No CatBoost is trained. "
            "It scans scalar features and selected interactions for 2023 changepoints, "
            "composition shift, post-2023 persistence, and same-pitcher cohort persistence."
        )
    )
    parser.add_argument("--data", default=None, help="train.csv or train.pkl; auto-detected if omitted")
    parser.add_argument("--mode", choices=["phone", "full"], default="phone")
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--min-era-count", type=int, default=None)
    parser.add_argument("--min-year-count", type=int, default=None)
    parser.add_argument("--same-pitcher-min-old", type=int, default=200)
    parser.add_argument("--same-pitcher-min-recent", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None, help="debug/smoke only; leave unset for real analysis")
    parser.add_argument("--output-dir", default="outputs/phone_regime_atlas")
    args = parser.parse_args()

    defaults = {
        "phone": {"bins": 6, "min_era": 1000, "min_year": 250},
        "full": {"bins": 10, "min_era": 500, "min_year": 100},
    }[args.mode]
    bins = args.bins or defaults["bins"]
    min_era_count = args.min_era_count or defaults["min_era"]
    min_year_count = args.min_year_count or defaults["min_year"]
    if bins < 3:
        raise ValueError("--bins must be >= 3")

    path = _find_data_path(args.data)
    print("[Phone Regime Atlas | CPU only]")
    print(f"  data       : {path}")
    print(f"  mode       : {args.mode}")
    print(f"  bins       : {bins}")
    print(f"  min support: era={min_era_count}, year={min_year_count}")
    print("  training   : NONE (pandas/numpy aggregation only)")
    frame = _load_analysis_frame(path, args.max_rows)
    print(f"  rows       : {len(frame):,}")
    print(f"  columns    : {len(frame.columns)} loaded")

    observed = sorted(frame["season"].unique().tolist())
    missing_years = sorted(set(ALL_YEARS) - set(observed))
    if missing_years:
        raise RuntimeError(f"missing seasons for atlas: {missing_years}")

    season_rates = (
        frame.groupby("season")["control_success"]
        .agg(["size", "mean"])
        .rename(columns={"size": "rows", "mean": "success_rate"})
        .reset_index()
    )
    print("\n[Season target rates]")
    print(season_rates.to_string(index=False))

    cohort_mask, cohort_summary = _same_entity_mask(
        frame,
        "pitcher_id",
        args.same_pitcher_min_old,
        args.same_pitcher_min_recent,
    )
    if cohort_summary.get("available"):
        print(
            "\n[Same-pitcher cohort] "
            f"pitchers={cohort_summary['eligible_entities']:,}, rows={cohort_summary['rows']:,} "
            f"({cohort_summary['row_fraction']:.1%})"
        )
    else:
        print("\n[Same-pitcher cohort] pitcher_id unavailable; cohort control skipped")
        cohort_mask = None

    signals: list[tuple[str, pd.Series, str]] = []
    for feature in SCALAR_SIGNALS:
        if feature not in frame.columns:
            continue
        groups, grouping, _ = _make_groups(frame[feature], feature in CATEGORICAL_HINTS, bins)
        signals.append((feature, groups, grouping))

    interactions = _interaction_signals(frame, bins)
    for name, groups in interactions.items():
        signals.append((name, groups.astype("string"), "interaction"))

    print(f"\n[Atlas] signals={len(signals)} (scalar={len(signals)-len(interactions)}, interactions={len(interactions)})")
    summaries: list[dict] = []
    long_parts: list[pd.DataFrame] = []
    for idx, (name, groups, grouping) in enumerate(signals, start=1):
        summary, long = _audit_one(
            name,
            groups,
            frame,
            min_era_count=min_era_count,
            min_year_count=min_year_count,
            cohort_mask=cohort_mask,
        )
        summary["grouping"] = grouping
        summaries.append(summary)
        long_parts.append(long)
        shift = summary["old_recent_effect_rmse"]
        cp = summary["changepoint_2023_ratio"]
        shift_s = f"{shift:.5f}" if np.isfinite(shift) else "nan"
        cp_s = f"{cp:.2f}" if np.isfinite(cp) else "nan"
        print(
            f"  [{idx:02d}/{len(signals):02d}] {name:<38} "
            f"{summary['classification']:<22} shift={shift_s} cp={cp_s}"
        )

    atlas = pd.DataFrame(summaries).sort_values(
        ["regime_score", "old_recent_effect_rmse"], ascending=[False, False], na_position="last"
    )
    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(output_dir / "signal_regime_atlas.csv", index=False)
    long_df.to_csv(output_dir / "per_season_group_effects.csv", index=False)
    season_rates.to_csv(output_dir / "season_target_rates.csv", index=False)

    top = atlas.loc[
        atlas["classification"].isin(["regime_candidate", "drifting_or_mixed", "composition_sensitive"])
    ].head(15)
    payload = {
        "purpose": "find evidence-driven expert tracks before training triple/quad ensembles",
        "warning": "This is a model-free diagnostic. High regime_score is a hypothesis, not proof of causal regime change.",
        "top_candidates": top.to_dict(orient="records"),
    }
    (output_dir / "top_regime_candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    (output_dir / "cohort_summary.json").write_text(
        json.dumps(cohort_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_config = {
        "data": str(path),
        "mode": args.mode,
        "bins": bins,
        "min_era_count": min_era_count,
        "min_year_count": min_year_count,
        "same_pitcher_min_old": args.same_pitcher_min_old,
        "same_pitcher_min_recent": args.same_pitcher_min_recent,
        "rows": int(len(frame)),
        "signals": [name for name, _, _ in signals],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[Top regime signals]")
    cols = [
        "signal",
        "classification",
        "old_recent_effect_rmse",
        "shock_2022_2023_rmse",
        "post_2023_2024_rmse",
        "changepoint_2023_ratio",
        "sign_flip_rate",
        "composition_js",
        "same_pitcher_vs_all_ratio",
        "regime_score",
    ]
    print(atlas.head(20)[cols].to_string(index=False))
    print("\nRead first:")
    print("  - high old_recent_effect_rmse: target relationship changed across eras")
    print("  - high changepoint_2023_ratio: 2022->2023 is unusually abrupt")
    print("  - low post_2023_2024_rmse: new relationship persists into 2024")
    print("  - same_pitcher_vs_all_ratio near 1: shift survives player-composition control")
    print("  - high composition_js + low cohort ratio: likely composition-driven rather than a clean regime")
    print(f"\nSaved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
