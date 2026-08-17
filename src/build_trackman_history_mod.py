#!/usr/bin/env python3
"""Create a train-aligned copy of Trackman without mapping any ID system.

Current normalization:
    top_bottom: Top -> T, Bottom -> B
    pitcher_hand/batter_hand: Right -> 2, Left -> 1
    pitcher_team/batter_team: add train-compatible *_team_id columns

All original columns are preserved. Player IDs and row/game IDs remain
unchanged. Team strings remain as provenance alongside the mapped ID columns.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "trackman_history.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "trackman_history_mod.csv"
TOP_BOTTOM_MAP = {"Top": "T", "Bottom": "B", "T": "T", "B": "B"}
HAND_MAP = {"Right": 2, "Left": 1, "1": 1, "2": 2}
TEAM_MAP = {
    "DOO_BEA": 12,
    "MIN_DOO": 12,
    "LG_TWI": 13,
    "MIN_LGT": 13,
    "KIW_HER": 14,
    "MIN_HER": 14,
    "LOT_GIA": 15,
    "MIN_LOT": 15,
    "KIA_TIG": 16,
    "MIN_KIA": 16,
    "HAN_EAG": 17,
    "MIN_HAN": 17,
    "SAM_LIO": 18,
    "MIN_SAM": 18,
    "NC_DIN": 19,
    "MIN_NCD": 19,
    "KT_WIZ": 20,
    "MIN_KTW": 20,
    "SSG_LAN": 21,
    "SK_WYV": 21,
    "MIN_SSG": 21,
    "MIN_SKW": 21,
    "KBO_POL": 22,
    "KBO_ARM": 23,
    "MIN_HAW": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing output file"
    )
    parser.add_argument(
        "--strict-teams",
        action="store_true",
        help="Fail instead of leaving unaudited team strings as nullable IDs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; use --force to replace it")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    total_rows = 0
    unmapped_teams: Counter[str] = Counter()
    try:
        for chunk_index, chunk in enumerate(
            pd.read_csv(source, chunksize=args.chunksize, low_memory=False)
        ):
            if "top_bottom" not in chunk.columns:
                raise KeyError("trackman_history.csv has no top_bottom column")

            observed = set(chunk["top_bottom"].dropna().astype(str).unique())
            unexpected = observed - set(TOP_BOTTOM_MAP)
            if unexpected:
                raise ValueError(
                    f"Unexpected top_bottom values: {sorted(unexpected)}"
                )

            chunk["top_bottom"] = chunk["top_bottom"].map(TOP_BOTTOM_MAP)
            for hand_column in ("pitcher_hand", "batter_hand"):
                if hand_column not in chunk.columns:
                    raise KeyError(f"trackman_history.csv has no {hand_column} column")
                hand_values = chunk[hand_column].dropna().astype(str).str.strip()
                unexpected_hands = set(hand_values.unique()) - set(HAND_MAP)
                if unexpected_hands:
                    raise ValueError(
                        f"Unexpected {hand_column} values: {sorted(unexpected_hands)}"
                    )
                chunk[hand_column] = hand_values.map(HAND_MAP).reindex(chunk.index)

            for team_column, output_column in (
                ("pitcher_team", "pitcher_team_id"),
                ("batter_team", "batter_team_id"),
            ):
                if team_column not in chunk.columns:
                    raise KeyError(f"trackman_history.csv has no {team_column} column")
                team_values = chunk[team_column].astype("string").str.strip()
                unexpected_teams = set(team_values.dropna().unique()) - set(TEAM_MAP)
                if unexpected_teams and args.strict_teams:
                    raise ValueError(
                        f"Unexpected {team_column} values: {sorted(unexpected_teams)}"
                    )
                for team_name, count in team_values.value_counts().items():
                    if team_name not in TEAM_MAP:
                        unmapped_teams[str(team_name)] += int(count)
                chunk[output_column] = team_values.map(TEAM_MAP).astype("Int64")

            chunk.to_csv(
                temporary,
                mode="w" if chunk_index == 0 else "a",
                header=chunk_index == 0,
                index=False,
                encoding="utf-8-sig" if chunk_index == 0 else "utf-8",
            )
            total_rows += len(chunk)
            print(f"processed_rows: {total_rows:,}")

        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"output: {output}")
    print(f"total_rows: {total_rows:,}")
    print("normalized: top_bottom (Top/Bottom -> T/B)")
    print("normalized: pitcher_hand/batter_hand (Right/Left -> 2/1)")
    print("added: pitcher_team_id/batter_team_id from audited game-sequence mapping")
    print("unchanged: team strings, player IDs, row/game IDs")
    if unmapped_teams:
        print("unmapped team strings left as <NA> IDs:")
        for team_name, count in sorted(unmapped_teams.items()):
            print(f"  {team_name}: {count:,}")


if __name__ == "__main__":
    main()
