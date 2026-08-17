#!/usr/bin/env python3
"""Compare shared context + team combinations across train and Trackman."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
COMMON_COLUMNS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
]
TEAM_COLUMNS = ["pitcher_team_id", "batter_team_id"]
TRAIN_COLUMNS = COMMON_COLUMNS + TEAM_COLUMNS
TRACKMAN_COLUMNS = COMMON_COLUMNS + ["pitcher_team", "batter_team"]


def find_file(name: str) -> Path:
    matches = sorted(RAW_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name} under {RAW_DIR}")
    return matches[0]


def combination_counts(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        data.groupby(columns, dropna=False, observed=True)
        .size().rename("row_count").reset_index()
    )


def normalize_for_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Give join keys a common dtype without guessing code mappings."""
    normalized = data.copy()
    for column in normalized.columns:
        normalized[column] = (
            normalized[column]
            .astype("string")
            .fillna("<NA>")
            .str.strip()
        )
    return normalized


def compare(title: str, train: pd.DataFrame, trackman: pd.DataFrame,
            columns: list[str]) -> None:
    train_counts = combination_counts(train, columns)
    trackman_counts = combination_counts(trackman, columns)
    shared = train_counts.merge(
        trackman_counts, on=columns, how="inner",
        suffixes=("_train", "_trackman"),
    )

    train_total = len(train)
    trackman_total = len(trackman)
    train_unique = len(train_counts)
    trackman_unique = len(trackman_counts)
    shared_unique = len(shared)
    train_rows_covered = int(shared["row_count_train"].sum())
    trackman_rows_covered = int(shared["row_count_trackman"].sum())

    train_unique_ratio = shared_unique / train_unique if train_unique else 0.0
    trackman_unique_ratio = shared_unique / trackman_unique if trackman_unique else 0.0
    train_row_ratio = train_rows_covered / train_total if train_total else 0.0
    trackman_row_ratio = trackman_rows_covered / trackman_total if trackman_total else 0.0

    print(f"[{title}]")
    print(f"columns: {', '.join(columns)}")
    print(f"train_total_rows: {train_total:,}")
    print(f"trackman_total_rows: {trackman_total:,}")
    print(f"train_distinct_combinations: {train_unique:,}")
    print(f"trackman_distinct_combinations: {trackman_unique:,}")
    print(f"shared_distinct_combinations: {shared_unique:,}")
    print(f"shared / train_distinct: {train_unique_ratio:.10f} ({train_unique_ratio:.6%})")
    print(f"shared / trackman_distinct: {trackman_unique_ratio:.10f} ({trackman_unique_ratio:.6%})")
    print(
        "train rows belonging to a shared combination: "
        f"{train_rows_covered:,} / {train_total:,} ({train_row_ratio:.6%})"
    )
    print(
        "trackman rows belonging to a shared combination: "
        f"{trackman_rows_covered:,} / {trackman_total:,} ({trackman_row_ratio:.6%})"
    )
    print()


def main() -> None:
    train_path = find_file("train.csv")
    trackman_path = find_file("trackman_history.csv")
    train = pd.read_csv(train_path, usecols=TRAIN_COLUMNS, low_memory=False)
    trackman = pd.read_csv(
        trackman_path, usecols=TRACKMAN_COLUMNS, low_memory=False
    ).rename(columns={
        "pitcher_team": "pitcher_team_id",
        "batter_team": "batter_team_id",
    })
    train = normalize_for_comparison(train)
    trackman = normalize_for_comparison(trackman)

    print(f"train: {train_path}")
    print(f"trackman: {trackman_path}")
    print()
    compare("10 shared context columns", train, trackman, COMMON_COLUMNS)
    compare(
        "10 shared context columns + pitcher/batter teams",
        train, trackman, COMMON_COLUMNS + TEAM_COLUMNS,
    )


if __name__ == "__main__":
    main()
