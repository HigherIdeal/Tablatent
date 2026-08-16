from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_regime_atlas as atlas
from src.utils import load_config, save_json


YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
OLD_YEARS = [2019, 2020, 2021, 2022]
RECENT_YEARS = [2023, 2024]
RESIDUAL_COL = "__gt_controlled_residual__"

# These are the unresolved non-game_type candidates after the first controlled
# screen.  Each candidate uses target-independent global quantile bins so that
# the same group has the same numerical meaning in every season.
CANDIDATES: dict[str, list[tuple[str, str, int | None]]] = {
    "pitcher_strike_rate_x_experience": [
        ("asof_pitcher_strike_rate", "numeric", 4),
        ("asof_pitcher_n", "numeric", 4),
    ],
    "fastball_rate_x_batter_hand": [
        ("asof_pitcher_fastball_rate", "numeric", 4),
        ("batter_hand", "categorical", None),
    ],
    "eng_ps_recent_range_135": [
        ("eng_ps_recent_range_135", "numeric", 5),
    ],
    "breaking_rate_x_batter_hand": [
        ("asof_pitcher_breaking_rate", "numeric", 4),
        ("batter_hand", "categorical", None),
    ],
    "pitcher_ball_rate_x_experience": [
        ("asof_pitcher_ball_rate", "numeric", 4),
        ("asof_pitcher_n", "numeric", 4),
    ],
}


def _fmt_edge(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _global_quantile_groups(series: pd.Series, bins: int) -> tuple[pd.Series, list[float]]:
    """Target-independent global quantile groups with readable interval labels."""
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite values for {series.name}")
    edges = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError(f"Insufficient unique quantile edges for {series.name}: {edges}")

    labels = np.full(len(values), "<MISSING>", dtype=object)
    valid = np.isfinite(values)
    index = np.digitize(values[valid], edges[1:-1], right=True)
    interval_labels: list[str] = []
    for i in range(len(edges) - 1):
        left = _fmt_edge(float(edges[i]))
        right = _fmt_edge(float(edges[i + 1]))
        close = "]" if i == len(edges) - 2 else ")"
        interval_labels.append(f"Q{i + 1}[{left},{right}{close}")
    labels[valid] = np.asarray(interval_labels, dtype=object)[index]
    return pd.Series(labels, index=series.index, name=series.name), [float(x) for x in edges]


def _candidate_groups(
    frame: pd.DataFrame,
    specification: list[tuple[str, str, int | None]],
) -> tuple[pd.Series, dict[str, dict]]:
    pieces: list[pd.Series] = []
    metadata: dict[str, dict] = {}
    for column, kind, bins in specification:
        if column not in frame.columns:
            raise ValueError(f"Missing candidate component: {column}")
        if kind == "numeric":
            assert bins is not None
            grouped, edges = _global_quantile_groups(frame[column], int(bins))
            pieces.append(grouped.astype(str).map(lambda x, c=column: f"{c}={x}"))
            metadata[column] = {"kind": kind, "bins": int(bins), "edges": edges}
        elif kind == "categorical":
            grouped = frame[column].astype("string").fillna("<MISSING>").astype(str)
            pieces.append(grouped.map(lambda x, c=column: f"{c}={x}"))
            metadata[column] = {
                "kind": kind,
                "levels": sorted(grouped.unique().tolist()),
            }
        else:
            raise ValueError(f"Unknown candidate component kind: {kind}")

    combined = pieces[0].copy()
    for piece in pieces[1:]:
        combined = combined.str.cat(piece, sep=" | ")
    return combined, metadata


def _add_game_type_controlled_residual(
    frame: pd.DataFrame,
    *,
    target_col: str,
    season_col: str,
) -> None:
    y = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    gt = frame["game_type"].astype("string").fillna("<MISSING>").astype(str)
    gt_mean = y.groupby([frame[season_col], gt]).transform("mean")
    frame[RESIDUAL_COL] = y - gt_mean


def _profile_one_cohort(
    frame: pd.DataFrame,
    groups: pd.Series,
    *,
    target_col: str,
    season_col: str,
    cohort: str,
) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "season": pd.to_numeric(frame[season_col], errors="raise").astype(int),
            "group": groups.astype(str),
            "target": pd.to_numeric(frame[target_col], errors="raise").astype(float),
            "controlled_residual": pd.to_numeric(frame[RESIDUAL_COL], errors="raise").astype(float),
        },
        index=frame.index,
    )
    seasonal_prior = temp.groupby("season")["target"].transform("mean")
    temp["season_centered_target"] = temp["target"] - seasonal_prior
    prof = (
        temp.groupby(["season", "group"], dropna=False)
        .agg(
            count=("target", "size"),
            success_rate=("target", "mean"),
            season_centered_effect=("season_centered_target", "mean"),
            game_type_controlled_effect=("controlled_residual", "mean"),
        )
        .reset_index()
    )
    prof.insert(0, "cohort", cohort)
    return prof


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _group_shift_table(
    profile: pd.DataFrame,
    *,
    candidate: str,
    min_era_count: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for group, part in profile.groupby("group", dropna=False):
        old = part.loc[part["season"].isin(OLD_YEARS)]
        recent = part.loc[part["season"].isin(RECENT_YEARS)]
        old_count = int(old["count"].sum())
        recent_count = int(recent["count"].sum())
        if old_count < min_era_count or recent_count < min_era_count:
            continue

        old_ctrl = _weighted_mean(old["game_type_controlled_effect"], old["count"])
        recent_ctrl = _weighted_mean(recent["game_type_controlled_effect"], recent["count"])
        old_raw = _weighted_mean(old["season_centered_effect"], old["count"])
        recent_raw = _weighted_mean(recent["season_centered_effect"], recent["count"])
        y23 = part.loc[part["season"].eq(2023), "game_type_controlled_effect"]
        y24 = part.loc[part["season"].eq(2024), "game_type_controlled_effect"]
        ctrl23 = float(y23.iloc[0]) if len(y23) else float("nan")
        ctrl24 = float(y24.iloc[0]) if len(y24) else float("nan")
        recent_same_direction = (
            bool(np.sign(ctrl23) == np.sign(ctrl24))
            if np.isfinite(ctrl23) and np.isfinite(ctrl24) and abs(ctrl23) >= 0.001 and abs(ctrl24) >= 0.001
            else False
        )
        rows.append(
            {
                "candidate": candidate,
                "cohort": str(part["cohort"].iloc[0]),
                "group": str(group),
                "old_count": old_count,
                "recent_count": recent_count,
                "old_season_centered_effect": old_raw,
                "recent_season_centered_effect": recent_raw,
                "raw_effect_delta": recent_raw - old_raw,
                "old_game_type_controlled_effect": old_ctrl,
                "recent_game_type_controlled_effect": recent_ctrl,
                "controlled_effect_delta": recent_ctrl - old_ctrl,
                "controlled_effect_2023": ctrl23,
                "controlled_effect_2024": ctrl24,
                "recent_same_direction": recent_same_direction,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["abs_controlled_effect_delta"] = result["controlled_effect_delta"].abs()
    return result.sort_values("abs_controlled_effect_delta", ascending=False)


def _wide_year_table(profile: pd.DataFrame, candidate: str) -> pd.DataFrame:
    rows: list[dict] = []
    for group, part in profile.groupby("group", dropna=False):
        row: dict[str, object] = {
            "candidate": candidate,
            "cohort": str(part["cohort"].iloc[0]),
            "group": str(group),
        }
        for year in YEARS:
            year_part = part.loc[part["season"].eq(year)]
            if year_part.empty:
                row[f"count_{year}"] = 0
                row[f"rate_{year}"] = np.nan
                row[f"ctrl_{year}"] = np.nan
            else:
                item = year_part.iloc[0]
                row[f"count_{year}"] = int(item["count"])
                row[f"rate_{year}"] = float(item["success_rate"])
                row[f"ctrl_{year}"] = float(item["game_type_controlled_effect"])
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_summary(full_shifts: pd.DataFrame, same_shifts: pd.DataFrame, candidate: str) -> dict:
    if full_shifts.empty:
        return {
            "candidate": candidate,
            "supported_groups": 0,
            "weighted_abs_controlled_delta": np.nan,
            "recent_direction_consistency": np.nan,
            "same_player_delta_preservation": np.nan,
        }
    weights = np.minimum(
        full_shifts["old_count"].to_numpy(np.float64),
        full_shifts["recent_count"].to_numpy(np.float64),
    )
    delta = full_shifts["controlled_effect_delta"].to_numpy(np.float64)
    weighted_abs = float(np.average(np.abs(delta), weights=weights))
    consistency = float(np.average(full_shifts["recent_same_direction"].astype(float), weights=weights))

    same_map = same_shifts.set_index("group")["controlled_effect_delta"] if not same_shifts.empty else pd.Series(dtype=float)
    paired = full_shifts.loc[full_shifts["group"].isin(same_map.index)].copy()
    if paired.empty:
        preservation = np.nan
        corr = np.nan
    else:
        full_delta = paired["controlled_effect_delta"].to_numpy(np.float64)
        same_delta = paired["group"].map(same_map).to_numpy(np.float64)
        denom = float(np.sqrt(np.mean(full_delta**2)))
        numer = float(np.sqrt(np.mean(same_delta**2)))
        preservation = numer / denom if denom > 0 else np.nan
        if len(paired) >= 3 and np.std(full_delta) > 0 and np.std(same_delta) > 0:
            corr = float(np.corrcoef(full_delta, same_delta)[0, 1])
        else:
            corr = np.nan

    return {
        "candidate": candidate,
        "supported_groups": int(len(full_shifts)),
        "weighted_abs_controlled_delta": weighted_abs,
        "max_abs_controlled_delta": float(full_shifts["abs_controlled_effect_delta"].max()),
        "recent_direction_consistency": consistency,
        "same_player_delta_preservation": preservation,
        "same_player_delta_correlation": corr,
    }


def _print_candidate(
    candidate: str,
    wide: pd.DataFrame,
    shifts: pd.DataFrame,
    *,
    top_groups: int,
) -> None:
    print(f"\n[{candidate}]")
    if shifts.empty:
        print("  no groups satisfy support thresholds")
        return
    selected = shifts.head(top_groups).copy()
    wide_map = wide.set_index("group")
    display_rows = []
    for row in selected.itertuples(index=False):
        w = wide_map.loc[row.group]
        display_rows.append(
            {
                "group": row.group,
                "old_ctrl": row.old_game_type_controlled_effect,
                "recent_ctrl": row.recent_game_type_controlled_effect,
                "delta": row.controlled_effect_delta,
                "ctrl23": w.get("ctrl_2023", np.nan),
                "ctrl24": w.get("ctrl_2024", np.nan),
                "rate19": w.get("rate_2019", np.nan),
                "rate20": w.get("rate_2020", np.nan),
                "rate21": w.get("rate_2021", np.nan),
                "rate22": w.get("rate_2022", np.nan),
                "rate23": w.get("rate_2023", np.nan),
                "rate24": w.get("rate_2024", np.nan),
                "old_n": row.old_count,
                "recent_n": row.recent_count,
            }
        )
    display = pd.DataFrame(display_rows)
    formatters = {
        col: (lambda x: f"{float(x):+.4f}" if pd.notna(x) else "nan")
        for col in ["old_ctrl", "recent_ctrl", "delta", "ctrl23", "ctrl24"]
    }
    formatters.update(
        {
            col: (lambda x: f"{float(x):.4f}" if pd.notna(x) else "nan")
            for col in ["rate19", "rate20", "rate21", "rate22", "rate23", "rate24"]
        }
    )
    print(display.to_string(index=False, formatters=formatters))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only drilldown of unresolved temporal candidates. Prints the actual 2019-2024 group success rates "
            "and game_type-controlled signed effects, then repeats the shift on the same-pitcher bridge cohort."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--min-era-count", type=int, default=500)
    parser.add_argument("--same-player-min-era-count", type=int, default=150)
    parser.add_argument("--same-player-min-old-seasons", type=int, default=2)
    parser.add_argument("--top-groups", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/regime_candidate_drilldown")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, _ = atlas.recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    _add_game_type_controlled_residual(frame, target_col=target_col, season_col=season_col)

    same_mask, same_stats = atlas._same_player_mask(
        frame,
        pitcher_col="pitcher_id",
        season_col=season_col,
        min_old_seasons=args.same_player_min_old_seasons,
    )

    print("[Regime Candidate Drilldown]")
    print("  training           : NONE (CPU statistical analysis only)")
    print("  control            : residual = y - E[y | season, game_type]")
    print(f"  candidates         : {list(CANDIDATES)}")
    print(
        f"  same-player cohort : {same_stats['pitchers']:,} pitchers / {same_stats['rows']:,} rows "
        f"({same_stats['row_fraction']:.1%})"
    )
    print("  interpretation     : ctrl values are probability-point effects after game_type control")

    all_profiles: list[pd.DataFrame] = []
    all_wide: list[pd.DataFrame] = []
    all_shifts: list[pd.DataFrame] = []
    summaries: list[dict] = []
    bin_metadata: dict[str, dict] = {}

    for candidate, spec in CANDIDATES.items():
        groups, metadata = _candidate_groups(frame, spec)
        bin_metadata[candidate] = metadata

        full_profile = _profile_one_cohort(
            frame,
            groups,
            target_col=target_col,
            season_col=season_col,
            cohort="all",
        )
        same_profile = _profile_one_cohort(
            frame.loc[same_mask],
            groups.loc[same_mask],
            target_col=target_col,
            season_col=season_col,
            cohort="same_player",
        )
        full_profile.insert(0, "candidate", candidate)
        same_profile.insert(0, "candidate", candidate)
        all_profiles.extend([full_profile, same_profile])

        full_wide = _wide_year_table(full_profile, candidate)
        same_wide = _wide_year_table(same_profile, candidate)
        all_wide.extend([full_wide, same_wide])

        full_shifts = _group_shift_table(
            full_profile,
            candidate=candidate,
            min_era_count=args.min_era_count,
        )
        same_shifts = _group_shift_table(
            same_profile,
            candidate=candidate,
            min_era_count=args.same_player_min_era_count,
        )
        all_shifts.extend([full_shifts, same_shifts])
        summaries.append(_candidate_summary(full_shifts, same_shifts, candidate))

        _print_candidate(candidate, full_wide, full_shifts, top_groups=args.top_groups)

    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profiles_df = pd.concat(all_profiles, ignore_index=True)
    wide_df = pd.concat(all_wide, ignore_index=True)
    shifts_df = pd.concat([x for x in all_shifts if not x.empty], ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["weighted_abs_controlled_delta", "max_abs_controlled_delta"],
        ascending=False,
    )

    profiles_df.to_csv(out / "candidate_group_profiles_long.csv", index=False)
    wide_df.to_csv(out / "candidate_group_profiles_wide.csv", index=False)
    shifts_df.to_csv(out / "candidate_group_shifts.csv", index=False)
    summary_df.to_csv(out / "candidate_summary.csv", index=False)
    save_json(
        {
            "same_player_cohort": same_stats,
            "candidate_bin_metadata": bin_metadata,
            "notes": [
                "All numeric bins are global target-independent quantiles shared across seasons.",
                "game_type_controlled_effect is y - E[y | season, game_type] averaged within each candidate group.",
                "controlled_effect_delta is recent(2023-2024) minus old(2019-2022).",
                "same-player profiles use the bridge cohort only as a diagnostic; pitcher_id is never a candidate signal.",
            ],
        },
        out / "run_metadata.json",
    )

    print("\n[Candidate Summary]")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
