from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_frame
from src.utils import load_config


def _normalize_game_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _state_from_counts(r_rows: int, f_rows: int) -> str:
    if r_rows > 0 and f_rows > 0:
        return "RF"
    if r_rows > 0:
        return "R"
    if f_rows > 0:
        return "F"
    return "NONE"


def build_pitcher_season_presence(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    pitcher_col: str,
    game_type_col: str,
    n_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (pitcher, season), group in frame.groupby([pitcher_col, season_col], sort=True):
        r = group.loc[group[game_type_col] == "R"]
        f = group.loc[group[game_type_col] == "F"]
        r_rows = int(len(r))
        f_rows = int(len(f))
        rows.append(
            {
                pitcher_col: pitcher,
                season_col: int(season),
                "state": _state_from_counts(r_rows, f_rows),
                "rows": int(len(group)),
                "r_rows": r_rows,
                "f_rows": f_rows,
                "r_share": r_rows / len(group),
                "f_share": f_rows / len(group),
                "overall_success_rate": float(group[target_col].mean()),
                "r_success_rate": float(r[target_col].mean()) if r_rows else np.nan,
                "f_success_rate": float(f[target_col].mean()) if f_rows else np.nan,
                "r_n_min": float(r[n_col].min()) if r_rows else np.nan,
                "r_n_max": float(r[n_col].max()) if r_rows else np.nan,
                "f_n_min": float(f[n_col].min()) if f_rows else np.nan,
                "f_n_max": float(f[n_col].max()) if f_rows else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values([season_col, pitcher_col]).reset_index(drop=True)


def build_adjacent_season_transitions(
    presence: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for pitcher, group in presence.groupby(pitcher_col, sort=True):
        group = group.sort_values(season_col)
        records = group.to_dict("records")
        for prev, curr in zip(records[:-1], records[1:]):
            prev_year = int(prev[season_col])
            curr_year = int(curr[season_col])
            rows.append(
                {
                    pitcher_col: pitcher,
                    "from_season": prev_year,
                    "to_season": curr_year,
                    "season_gap": curr_year - prev_year,
                    "adjacent_season": (curr_year - prev_year) == 1,
                    "from_state": prev["state"],
                    "to_state": curr["state"],
                    "transition": f"{prev['state']}->{curr['state']}",
                    "from_rows": int(prev["rows"]),
                    "to_rows": int(curr["rows"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["from_season", "to_season", pitcher_col]).reset_index(drop=True)


def build_pitch_level_switches(
    frame: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    game_type_col: str,
    n_col: str,
    month_col: str,
    row_id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    switch_rows: list[dict] = []
    sequence_rows: list[dict] = []

    sort_cols = [n_col, season_col]
    if month_col in frame.columns:
        sort_cols.append(month_col)
    if row_id_col and row_id_col in frame.columns:
        sort_cols.append(row_id_col)

    for pitcher, group in frame.groupby(pitcher_col, sort=True):
        ordered = group.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        types = ordered[game_type_col].astype(str).to_numpy()
        if len(types) == 0:
            continue

        run_starts = np.r_[True, types[1:] != types[:-1]]
        compressed = ordered.loc[run_starts].copy().reset_index(drop=True)
        sequence = "->".join(compressed[game_type_col].astype(str).tolist())
        sequence_rows.append(
            {
                pitcher_col: pitcher,
                "rows": int(len(ordered)),
                "segments": int(len(compressed)),
                "switches": int(max(0, len(compressed) - 1)),
                "first_type": str(compressed.iloc[0][game_type_col]),
                "last_type": str(compressed.iloc[-1][game_type_col]),
                "sequence": sequence,
                "first_season": int(ordered[season_col].min()),
                "last_season": int(ordered[season_col].max()),
                "n_min": float(ordered[n_col].min()),
                "n_max": float(ordered[n_col].max()),
            }
        )

        if len(compressed) < 2:
            continue

        for idx in range(1, len(compressed)):
            prev = compressed.iloc[idx - 1]
            curr = compressed.iloc[idx]
            from_type = str(prev[game_type_col])
            to_type = str(curr[game_type_col])
            if from_type == to_type:
                continue

            from_season = int(prev[season_col])
            to_season = int(curr[season_col])
            row = {
                pitcher_col: pitcher,
                "switch_index": idx,
                "from_type": from_type,
                "to_type": to_type,
                "transition": f"{from_type}->{to_type}",
                "from_season": from_season,
                "to_season": to_season,
                "same_season": from_season == to_season,
                "season_gap": to_season - from_season,
                "from_asof_n": float(prev[n_col]),
                "to_asof_n": float(curr[n_col]),
                "asof_gap": float(curr[n_col] - prev[n_col]),
            }
            if month_col in frame.columns:
                row["from_month"] = int(prev[month_col]) if pd.notna(prev[month_col]) else np.nan
                row["to_month"] = int(curr[month_col]) if pd.notna(curr[month_col]) else np.nan
            if row_id_col and row_id_col in frame.columns:
                row["from_row_id"] = prev[row_id_col]
                row["to_row_id"] = curr[row_id_col]
            switch_rows.append(row)

    switches = pd.DataFrame(switch_rows)
    sequences = pd.DataFrame(sequence_rows)
    if not switches.empty:
        switches = switches.sort_values(["to_season", pitcher_col, "switch_index"]).reset_index(drop=True)
    if not sequences.empty:
        sequences = sequences.sort_values(["switches", pitcher_col], ascending=[False, True]).reset_index(drop=True)
    return switches, sequences


def build_dual_paired(
    frame: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    target_col: str,
    game_type_col: str,
    n_col: str,
) -> pd.DataFrame:
    dual = frame.groupby([pitcher_col, season_col])[game_type_col].nunique()
    dual_keys = dual.loc[dual >= 2].index
    rows: list[dict] = []
    for pitcher, season in dual_keys:
        group = frame.loc[(frame[pitcher_col] == pitcher) & (frame[season_col] == season)]
        r = group.loc[group[game_type_col] == "R"]
        f = group.loc[group[game_type_col] == "F"]
        if r.empty or f.empty:
            continue
        rows.append(
            {
                pitcher_col: pitcher,
                season_col: int(season),
                "r_rows": int(len(r)),
                "f_rows": int(len(f)),
                "r_success_rate": float(r[target_col].mean()),
                "f_success_rate": float(f[target_col].mean()),
                "f_minus_r_success": float(f[target_col].mean() - r[target_col].mean()),
                "r_n_min": float(r[n_col].min()),
                "f_n_min": float(f[n_col].min()),
                "first_observed_type": "R" if r[n_col].min() < f[n_col].min() else "F",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values([season_col, pitcher_col]).reset_index(drop=True)


def _print_table(title: str, frame: pd.DataFrame, max_rows: int = 30) -> None:
    print(f"\n[{title}]")
    if frame.empty:
        print("No rows.")
        return
    print(frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        print(f"... ({len(frame) - max_rows} more rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether the same pitcher appears in both game_type R and F and quantify "
            "R<->F movement patterns using asof_pitcher_n as the within-pitcher ordering axis."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-subdir", default="pitcher_game_type_transitions")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    frame = load_frame(config).copy()

    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col")
    pitcher_col = "pitcher_id"
    game_type_col = "game_type"
    n_col = "asof_pitcher_n"
    month_col = "game_month"

    required = {season_col, target_col, pitcher_col, game_type_col, n_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    frame[n_col] = pd.to_numeric(frame[n_col], errors="raise").astype(float)
    if month_col in frame.columns:
        frame[month_col] = pd.to_numeric(frame[month_col], errors="coerce")
    frame[game_type_col] = _normalize_game_type(frame[game_type_col])

    observed_types = sorted(frame[game_type_col].dropna().unique().tolist())
    unexpected = sorted(set(observed_types) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    output_dir = Path(config["paths"]["output_dir"]) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    presence = build_pitcher_season_presence(
        frame, season_col, target_col, pitcher_col, game_type_col, n_col
    )
    presence.to_csv(output_dir / "pitcher_season_presence.csv", index=False)

    season_state = (
        presence.groupby([season_col, "state"], observed=True)
        .agg(pitcher_seasons=(pitcher_col, "size"), pitchers=(pitcher_col, "nunique"), rows=("rows", "sum"))
        .reset_index()
        .sort_values([season_col, "state"])
    )
    season_state.to_csv(output_dir / "season_state_summary.csv", index=False)

    overall_player_state = (
        frame.groupby(pitcher_col)[game_type_col]
        .agg(lambda s: "RF" if s.nunique() >= 2 else str(s.iloc[0]))
        .rename("overall_state")
        .reset_index()
    )
    overall_state_summary = (
        overall_player_state.groupby("overall_state")
        .agg(pitchers=(pitcher_col, "size"))
        .reset_index()
        .sort_values("overall_state")
    )
    overall_state_summary["share"] = overall_state_summary["pitchers"] / overall_state_summary["pitchers"].sum()
    overall_state_summary.to_csv(output_dir / "overall_player_state_summary.csv", index=False)

    season_transitions = build_adjacent_season_transitions(presence, pitcher_col, season_col)
    season_transitions.to_csv(output_dir / "season_state_transitions.csv", index=False)
    if not season_transitions.empty:
        adjacent_summary = (
            season_transitions.loc[season_transitions["adjacent_season"]]
            .groupby(["from_season", "to_season", "transition"], observed=True)
            .agg(pitchers=(pitcher_col, "nunique"))
            .reset_index()
            .sort_values(["from_season", "to_season", "pitchers"], ascending=[True, True, False])
        )
    else:
        adjacent_summary = pd.DataFrame()
    adjacent_summary.to_csv(output_dir / "adjacent_season_transition_summary.csv", index=False)

    switches, sequences = build_pitch_level_switches(
        frame,
        pitcher_col,
        season_col,
        game_type_col,
        n_col,
        month_col,
        row_id_col,
    )
    switches.to_csv(output_dir / "pitcher_rf_switches.csv", index=False)
    sequences.to_csv(output_dir / "pitcher_rf_sequences.csv", index=False)

    if not switches.empty:
        switch_summary = (
            switches.groupby(["to_season", "transition", "same_season"], observed=True)
            .agg(
                switches=(pitcher_col, "size"),
                pitchers=(pitcher_col, "nunique"),
                median_to_asof_n=("to_asof_n", "median"),
                median_asof_gap=("asof_gap", "median"),
            )
            .reset_index()
            .sort_values(["to_season", "transition", "same_season"])
        )
        if "to_month" in switches.columns:
            month_summary = (
                switches.loc[switches["same_season"]]
                .groupby(["to_season", "to_month", "transition"], observed=True)
                .agg(switches=(pitcher_col, "size"), pitchers=(pitcher_col, "nunique"))
                .reset_index()
                .sort_values(["to_season", "to_month", "transition"])
            )
        else:
            month_summary = pd.DataFrame()
    else:
        switch_summary = pd.DataFrame()
        month_summary = pd.DataFrame()
    switch_summary.to_csv(output_dir / "switch_summary.csv", index=False)
    month_summary.to_csv(output_dir / "same_season_switch_by_month.csv", index=False)

    dual = build_dual_paired(frame, pitcher_col, season_col, target_col, game_type_col, n_col)
    dual.to_csv(output_dir / "dual_pitcher_season_paired.csv", index=False)
    if not dual.empty:
        dual_summary = (
            dual.groupby(season_col)
            .agg(
                dual_pitchers=(pitcher_col, "nunique"),
                median_r_rows=("r_rows", "median"),
                median_f_rows=("f_rows", "median"),
                mean_r_success=("r_success_rate", "mean"),
                mean_f_success=("f_success_rate", "mean"),
                mean_f_minus_r=("f_minus_r_success", "mean"),
                median_f_minus_r=("f_minus_r_success", "median"),
            )
            .reset_index()
        )
    else:
        dual_summary = pd.DataFrame()
    dual_summary.to_csv(output_dir / "dual_pitcher_season_summary.csv", index=False)

    sequence_summary = (
        sequences.groupby("sequence", observed=True)
        .agg(pitchers=(pitcher_col, "size"), median_switches=("switches", "median"))
        .reset_index()
        .sort_values(["pitchers", "sequence"], ascending=[False, True])
        if not sequences.empty
        else pd.DataFrame()
    )
    sequence_summary.to_csv(output_dir / "sequence_summary.csv", index=False)

    print("[Pitcher R/F Transition Analysis]")
    print(f"  rows={len(frame):,}")
    print(f"  pitchers={frame[pitcher_col].nunique():,}")
    print(f"  game_types={observed_types}")
    print("  ordering axis=asof_pitcher_n (row_id/month used only as stable tie-breakers)")

    _print_table("Overall pitcher game_type coverage", overall_state_summary)
    _print_table("Pitcher-season states", season_state, max_rows=40)
    _print_table("Adjacent-season state transitions", adjacent_summary, max_rows=60)
    _print_table("Actual R<->F switches in asof order", switch_summary, max_rows=60)
    _print_table("Same-season R<->F switches by month", month_summary, max_rows=80)
    _print_table("Dual R/F pitcher-season paired target rates", dual_summary, max_rows=20)
    _print_table("Most common compressed R/F sequences", sequence_summary, max_rows=30)

    print("\nInterpretation:")
    print("  RF state means the same pitcher appeared in both R and F within that season.")
    print("  pitcher_rf_switches.csv compresses consecutive pitch-level game_type values after sorting by asof_pitcher_n.")
    print("  F->R / R->F therefore indicate observed movement of the same pitcher between the two domains.")
    print("  same_season=True is the strongest direct evidence that R/F are operational levels rather than fixed player classes.")
    print("  dual_pitcher_season_paired.csv compares R vs F target rates within the same pitcher and same season.")
    print("  This script does not assume R=1st team or F=Futures; that interpretation must be validated from the patterns.")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
