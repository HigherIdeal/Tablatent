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


DATE_CANDIDATES = [
    "game_date",
    "date",
    "game_datetime",
    "game_day",
    "game_dayofmonth",
]


def normalize_game_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def rf_state(series: pd.Series) -> str:
    values = set(series.dropna().astype(str))
    has_r = "R" in values
    has_f = "F" in values
    if has_r and has_f:
        return "RF"
    if has_r:
        return "R"
    if has_f:
        return "F"
    return "NONE"


def mode_with_purity(series: pd.Series) -> tuple[object, float]:
    counts = series.dropna().value_counts()
    if counts.empty:
        return np.nan, np.nan
    value = counts.index[0]
    purity = float(counts.iloc[0] / counts.sum())
    return value, purity


def detect_exact_date(frame: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    for col in DATE_CANDIDATES:
        if col not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[col], errors="coerce")
        coverage = float(parsed.notna().mean())
        if coverage >= 0.95 and parsed.nunique(dropna=True) > 30:
            return col, parsed
    return None, None


def adjacent_violation_rate(
    frame: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    month_col: str,
    n_col: str,
    order_cols: list[str],
    label: str,
) -> dict:
    work = frame.sort_values([pitcher_col, season_col] + order_cols, kind="mergesort")
    same_group = (
        work[pitcher_col].eq(work[pitcher_col].shift())
        & work[season_col].eq(work[season_col].shift())
    )
    month_delta = work[month_col] - work[month_col].shift()
    n_delta = work[n_col] - work[n_col].shift()
    valid = same_group
    pairs = int(valid.sum())
    return {
        "ordering": label,
        "adjacent_pairs": pairs,
        "month_decrease_pairs": int((valid & (month_delta < 0)).sum()),
        "month_decrease_rate": float((valid & (month_delta < 0)).sum() / pairs) if pairs else np.nan,
        "asof_n_decrease_pairs": int((valid & (n_delta < 0)).sum()),
        "asof_n_decrease_rate": float((valid & (n_delta < 0)).sum() / pairs) if pairs else np.nan,
        "asof_n_equal_pairs": int((valid & (n_delta == 0)).sum()),
        "asof_n_equal_rate": float((valid & (n_delta == 0)).sum() / pairs) if pairs else np.nan,
    }


def build_chronology_diagnostics(
    frame: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    month_col: str,
    n_col: str,
    row_id_col: str | None,
    date_col: str | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append(
        adjacent_violation_rate(
            frame,
            pitcher_col,
            season_col,
            month_col,
            n_col,
            ["_source_order"],
            "source_order",
        )
    )

    if row_id_col and "_row_order_numeric" in frame.columns:
        rows.append(
            adjacent_violation_rate(
                frame,
                pitcher_col,
                season_col,
                month_col,
                n_col,
                ["_row_order_numeric", "_source_order"],
                "row_id_numeric",
            )
        )

    rows.append(
        adjacent_violation_rate(
            frame,
            pitcher_col,
            season_col,
            month_col,
            n_col,
            [month_col, "_source_order"],
            "month_then_source",
        )
    )
    rows.append(
        adjacent_violation_rate(
            frame,
            pitcher_col,
            season_col,
            month_col,
            n_col,
            [month_col, n_col, "_source_order"],
            "month_then_asof_n",
        )
    )

    if date_col and "_exact_date" in frame.columns:
        rows.append(
            adjacent_violation_rate(
                frame,
                pitcher_col,
                season_col,
                month_col,
                n_col,
                ["_exact_date", "_source_order"],
                f"exact_date:{date_col}",
            )
        )

    return pd.DataFrame(rows)


def build_month_states(
    frame: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    month_col: str,
    game_type_col: str,
    target_col: str,
    n_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (pitcher, season, month), group in frame.groupby(
        [pitcher_col, season_col, month_col], sort=True, observed=True
    ):
        state = rf_state(group[game_type_col])
        r = group.loc[group[game_type_col] == "R"]
        f = group.loc[group[game_type_col] == "F"]
        rows.append(
            {
                pitcher_col: pitcher,
                season_col: int(season),
                month_col: int(month),
                "state": state,
                "rows": int(len(group)),
                "r_rows": int(len(r)),
                "f_rows": int(len(f)),
                "r_success_rate": float(r[target_col].mean()) if len(r) else np.nan,
                "f_success_rate": float(f[target_col].mean()) if len(f) else np.nan,
                "n_min": float(group[n_col].min()),
                "n_max": float(group[n_col].max()),
            }
        )
    return pd.DataFrame(rows).sort_values([season_col, pitcher_col, month_col]).reset_index(drop=True)


def build_month_transition_tables(
    month_states: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    month_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjacent_rows: list[dict] = []
    clean_rows: list[dict] = []

    for (pitcher, season), group in month_states.groupby([pitcher_col, season_col], sort=True):
        group = group.sort_values(month_col).reset_index(drop=True)
        records = group.to_dict("records")
        for prev, curr in zip(records[:-1], records[1:]):
            adjacent_rows.append(
                {
                    pitcher_col: pitcher,
                    season_col: int(season),
                    "from_month": int(prev[month_col]),
                    "to_month": int(curr[month_col]),
                    "month_gap": int(curr[month_col] - prev[month_col]),
                    "from_state": prev["state"],
                    "to_state": curr["state"],
                    "transition": f"{prev['state']}->{curr['state']}",
                }
            )

        pure = group.loc[group["state"].isin(["R", "F"])].copy()
        if pure.empty:
            continue
        types = pure["state"].astype(str).to_numpy()
        starts = np.r_[True, types[1:] != types[:-1]]
        compressed = pure.loc[starts].reset_index(drop=True)
        if len(compressed) < 2:
            continue
        for idx in range(1, len(compressed)):
            prev = compressed.iloc[idx - 1]
            curr = compressed.iloc[idx]
            clean_rows.append(
                {
                    pitcher_col: pitcher,
                    season_col: int(season),
                    "from_month": int(prev[month_col]),
                    "to_month": int(curr[month_col]),
                    "month_gap": int(curr[month_col] - prev[month_col]),
                    "transition": f"{prev['state']}->{curr['state']}",
                }
            )

    adjacent = pd.DataFrame(adjacent_rows)
    clean = pd.DataFrame(clean_rows)
    return adjacent, clean


def summarize_month_transitions(
    transitions: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    return (
        transitions.groupby([season_col, "transition"], observed=True)
        .agg(
            switches=(pitcher_col, "size"),
            pitchers=(pitcher_col, "nunique"),
            median_month_gap=("month_gap", "median"),
        )
        .reset_index()
        .sort_values([season_col, "switches"], ascending=[True, False])
    )


def build_team_namespace_summary(
    frame: pd.DataFrame,
    season_col: str,
    game_type_col: str,
    team_col: str,
    pitcher_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = (
        frame.groupby([season_col, game_type_col, team_col], observed=True)
        .agg(rows=(pitcher_col, "size"), pitchers=(pitcher_col, "nunique"))
        .reset_index()
        .sort_values([season_col, game_type_col, team_col])
    )

    overlap_rows: list[dict] = []
    for season, group in frame.groupby(season_col, sort=True):
        r_teams = set(group.loc[group[game_type_col] == "R", team_col].dropna().tolist())
        f_teams = set(group.loc[group[game_type_col] == "F", team_col].dropna().tolist())
        overlap = r_teams & f_teams
        overlap_rows.append(
            {
                season_col: int(season),
                "r_teams": len(r_teams),
                "f_teams": len(f_teams),
                "shared_team_ids": len(overlap),
                "r_only_team_ids": len(r_teams - f_teams),
                "f_only_team_ids": len(f_teams - r_teams),
                "jaccard": float(len(overlap) / len(r_teams | f_teams)) if (r_teams | f_teams) else np.nan,
            }
        )
    return detail, pd.DataFrame(overlap_rows)


def build_dual_player_team_mapping(
    frame: pd.DataFrame,
    season_col: str,
    pitcher_col: str,
    game_type_col: str,
    team_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for (pitcher, season), group in frame.groupby([pitcher_col, season_col], sort=True):
        r = group.loc[group[game_type_col] == "R"]
        f = group.loc[group[game_type_col] == "F"]
        if r.empty or f.empty:
            continue
        r_team, r_purity = mode_with_purity(r[team_col])
        f_team, f_purity = mode_with_purity(f[team_col])
        rows.append(
            {
                pitcher_col: pitcher,
                season_col: int(season),
                "r_team": r_team,
                "f_team": f_team,
                "same_team_id": bool(pd.notna(r_team) and pd.notna(f_team) and r_team == f_team),
                "r_team_purity": r_purity,
                "f_team_purity": f_purity,
                "r_rows": int(len(r)),
                "f_rows": int(len(f)),
            }
        )

    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired, pd.DataFrame(), pd.DataFrame()

    mapping = (
        paired.groupby([season_col, "r_team", "f_team"], observed=True)
        .agg(
            dual_pitchers=(pitcher_col, "nunique"),
            mean_r_team_purity=("r_team_purity", "mean"),
            mean_f_team_purity=("f_team_purity", "mean"),
        )
        .reset_index()
    )
    totals = mapping.groupby([season_col, "r_team"])["dual_pitchers"].transform("sum")
    mapping["share_within_r_team"] = mapping["dual_pitchers"] / totals
    mapping = mapping.sort_values(
        [season_col, "r_team", "dual_pitchers"], ascending=[True, True, False]
    ).reset_index(drop=True)

    season_summary = (
        paired.groupby(season_col)
        .agg(
            dual_pitchers=(pitcher_col, "nunique"),
            same_team_id_pitchers=("same_team_id", "sum"),
            same_team_id_rate=("same_team_id", "mean"),
            mean_r_team_purity=("r_team_purity", "mean"),
            mean_f_team_purity=("f_team_purity", "mean"),
        )
        .reset_index()
    )
    return paired, mapping, season_summary


def print_table(title: str, frame: pd.DataFrame, max_rows: int = 80) -> None:
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
            "Test whether game_type R/F behaves like two operational levels using only robust "
            "month-level movement and same-player team-ID mapping. Exact within-month order is "
            "not assumed unless an exact date column exists."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-subdir", default="rf_level_evidence")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    frame = load_frame(config).copy()

    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col")
    pitcher_col = "pitcher_id"
    team_col = "pitcher_team_id"
    game_type_col = "game_type"
    n_col = "asof_pitcher_n"
    month_col = "game_month"

    required = {
        season_col,
        target_col,
        pitcher_col,
        team_col,
        game_type_col,
        n_col,
        month_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    frame[n_col] = pd.to_numeric(frame[n_col], errors="raise").astype(float)
    frame[month_col] = pd.to_numeric(frame[month_col], errors="raise").astype(int)
    frame[game_type_col] = normalize_game_type(frame[game_type_col])

    unexpected = sorted(set(frame[game_type_col].dropna().unique()) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    if row_id_col and row_id_col in frame.columns:
        row_numeric = pd.to_numeric(frame[row_id_col], errors="coerce")
        if float(row_numeric.notna().mean()) >= 0.99:
            frame["_row_order_numeric"] = row_numeric

    date_col, parsed_date = detect_exact_date(frame)
    if date_col is not None and parsed_date is not None:
        frame["_exact_date"] = parsed_date

    output_dir = Path(config["paths"]["output_dir"]) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    chronology = build_chronology_diagnostics(
        frame,
        pitcher_col,
        season_col,
        month_col,
        n_col,
        row_id_col,
        date_col,
    )
    chronology.to_csv(output_dir / "chronology_diagnostics.csv", index=False)

    month_states = build_month_states(
        frame,
        pitcher_col,
        season_col,
        month_col,
        game_type_col,
        target_col,
        n_col,
    )
    month_states.to_csv(output_dir / "pitcher_month_states.csv", index=False)

    month_state_summary = (
        month_states.groupby([season_col, month_col, "state"], observed=True)
        .agg(pitcher_months=(pitcher_col, "size"), pitchers=(pitcher_col, "nunique"), rows=("rows", "sum"))
        .reset_index()
        .sort_values([season_col, month_col, "state"])
    )
    month_state_summary.to_csv(output_dir / "month_state_summary.csv", index=False)

    adjacent, clean = build_month_transition_tables(
        month_states, pitcher_col, season_col, month_col
    )
    adjacent.to_csv(output_dir / "adjacent_month_transitions.csv", index=False)
    clean.to_csv(output_dir / "clean_pure_month_rf_switches.csv", index=False)
    adjacent_summary = summarize_month_transitions(adjacent, pitcher_col, season_col)
    clean_summary = summarize_month_transitions(clean, pitcher_col, season_col)
    adjacent_summary.to_csv(output_dir / "adjacent_month_transition_summary.csv", index=False)
    clean_summary.to_csv(output_dir / "clean_pure_month_rf_switch_summary.csv", index=False)

    team_detail, team_overlap = build_team_namespace_summary(
        frame, season_col, game_type_col, team_col, pitcher_col
    )
    team_detail.to_csv(output_dir / "team_detail.csv", index=False)
    team_overlap.to_csv(output_dir / "team_namespace_overlap.csv", index=False)

    paired, mapping, paired_summary = build_dual_player_team_mapping(
        frame, season_col, pitcher_col, game_type_col, team_col
    )
    paired.to_csv(output_dir / "dual_pitcher_team_pairs.csv", index=False)
    mapping.to_csv(output_dir / "r_to_f_team_mapping.csv", index=False)
    paired_summary.to_csv(output_dir / "dual_pitcher_team_summary.csv", index=False)

    print("[R/F Operational-Level Evidence]")
    print(f"  rows={len(frame):,} pitchers={frame[pitcher_col].nunique():,}")
    print(f"  exact date column detected: {date_col if date_col else 'NONE'}")
    if date_col is None:
        print("  IMPORTANT: train features do not provide a verified exact game date.")
        print("  Therefore within-month F->R/R->F direction is NOT claimed from row/asof ordering.")
        print("  The robust movement analysis below uses chronological MONTH states only.")

    print_table("Chronology candidate diagnostics", chronology)

    rf_months = month_state_summary.loc[month_state_summary["state"] == "RF"]
    print_table("Same pitcher appears in BOTH R and F within the same month", rf_months)

    print_table("Adjacent-month state transitions", adjacent_summary)
    print_table("Clean pure-month F<->R switches", clean_summary)

    print_table("R/F pitcher-team ID namespace overlap", team_overlap)
    print_table("Dual R/F pitcher same-team-ID rate", paired_summary)

    if not mapping.empty:
        best_mapping = mapping.groupby([season_col, "r_team"], sort=False).head(1).copy()
        print_table("Dominant R-team -> F-team mapping from same pitchers", best_mapping, max_rows=100)

    print("\nInterpretation:")
    print("  1) RF within the same month is direct evidence that R/F are not fixed player classes.")
    print("  2) Clean pure-month F->R and R->F transitions establish bidirectional movement without")
    print("     relying on asof_pitcher_n or arbitrary row order inside a month.")
    print("  3) A stable one-to-one-ish R-team <-> F-team mapping through the SAME pitchers strongly")
    print("     supports the hypothesis that R/F are two operational league/team levels.")
    print("  4) This still does not rename R/F as 1st-team/Futures by assumption; the evidence is")
    print("     reported separately from that semantic interpretation.")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
