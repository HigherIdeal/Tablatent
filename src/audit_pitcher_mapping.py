#!/usr/bin/env python3
"""Derive and audit train pitcher-ID to Trackman pitcher-ID candidates.

Only rank-1 games whose context sequence matched exactly in the team-mapping
audit are used. Within each accepted game, train rows are paired with Trackman
pitches by ordinal position after Trackman rows are ordered by pitch_no.

The script scans the large CSV files but never modifies them. It writes compact
evidence tables and labels candidates as accepted, review, or insufficient.
Physical measurements and the prediction target are never used for linkage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from audit_team_mapping import half_code, hand_code, integer, scalar
from build_trackman_history_mod import TEAM_MAP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_TRACKMAN = ROOT / "data" / "raw" / "trackman_history.csv"
DEFAULT_TEAM_AUDIT = ROOT / "outputs" / "team_mapping_audit"
DEFAULT_OUTPUT = ROOT / "outputs" / "pitcher_mapping_audit"

TRAIN_COLUMNS = [
    "row_id",
    "season",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_id",
    "pitcher_hand",
    "pitcher_team_id",
]

TRACKMAN_COLUMNS = [
    "trackman_id",
    "season",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_trackman_id",
    "pitcher_hand",
    "pitcher_team",
]

OUTPUT_FILES = [
    "exact_game_alignment_audit.csv",
    "pitcher_mapping_candidates.csv",
    "accepted_pitcher_mapping.csv",
    "review_pitcher_mapping.csv",
    "pitcher_mapping_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--team-audit-dir", type=Path, default=DEFAULT_TEAM_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--min-pitches", type=int, default=50)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--min-forward-purity", type=float, default=0.98)
    parser.add_argument("--min-reverse-purity", type=float, default=0.98)
    parser.add_argument("--min-top-second-ratio", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def context_token(values: dict[str, Any]) -> str:
    return "|".join(
        [
            scalar(values["inning"]),
            half_code(values["top_bottom"]),
            scalar(values["balls_before"]),
            scalar(values["strikes_before"]),
            scalar(values["outs_before"]),
        ]
    )


def load_exact_matches(path: Path) -> tuple[list[dict[str, str]], int]:
    matches: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["candidate_rank"]) != 1:
                continue
            if float(row["exact_sequence"]) != 1.0:
                continue
            matches.append(
                {
                    "train_game_key": row["train_game_key"],
                    "trackman_game_id": row["trackman_game_id"],
                    "game_date": row["trackman_game_date"],
                    "season": row["season"],
                }
            )

    trackman_counts = Counter(row["trackman_game_id"] for row in matches)
    ambiguous = {game_id for game_id, count in trackman_counts.items() if count > 1}
    filtered = [row for row in matches if row["trackman_game_id"] not in ambiguous]
    return filtered, len(matches) - len(filtered)


def load_segments(path: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["n_rows"] = int(row["n_rows"])
            segments.append(row)
    return segments


def create_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.executescript(
        """
        CREATE TABLE matches (
            train_game_key TEXT PRIMARY KEY,
            trackman_game_id TEXT UNIQUE NOT NULL,
            game_date TEXT,
            season TEXT
        );

        CREATE TABLE train_pitches (
            train_game_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            row_id TEXT,
            season TEXT,
            context TEXT NOT NULL,
            pitcher_id TEXT NOT NULL,
            pitcher_hand TEXT,
            pitcher_team_id TEXT,
            PRIMARY KEY (train_game_key, ordinal)
        );

        CREATE TABLE trackman_pitches (
            trackman_game_id TEXT NOT NULL,
            pitch_no INTEGER,
            source_order INTEGER NOT NULL,
            trackman_id TEXT,
            season TEXT,
            context TEXT NOT NULL,
            pitcher_trackman_id TEXT NOT NULL,
            pitcher_hand TEXT,
            pitcher_team TEXT,
            pitcher_team_id INTEGER
        );
        """
    )
    return connection


def scan_train(
    args: argparse.Namespace,
    connection: sqlite3.Connection,
    segments: list[dict[str, Any]],
    selected_games: set[str],
) -> list[str]:
    errors: list[str] = []
    segment_index = 0
    ordinal = 0
    batch: list[tuple[Any, ...]] = []

    for chunk in pd.read_csv(
        args.train, usecols=TRAIN_COLUMNS, chunksize=args.chunksize, low_memory=False
    ):
        for values in chunk.to_dict(orient="records"):
            if segment_index >= len(segments):
                errors.append("train contains rows beyond the final recorded segment")
                break
            segment = segments[segment_index]
            row_id = scalar(values["row_id"])
            if ordinal == 0 and row_id != segment["first_row_id"]:
                errors.append(
                    f"{segment['game_key']}: first row {row_id} != "
                    f"{segment['first_row_id']}"
                )

            if segment["game_key"] in selected_games:
                batch.append(
                    (
                        segment["game_key"],
                        ordinal,
                        row_id,
                        scalar(values["season"]),
                        context_token(values),
                        scalar(values["pitcher_id"]),
                        hand_code(values["pitcher_hand"]),
                        scalar(values["pitcher_team_id"]),
                    )
                )
            ordinal += 1

            if ordinal == segment["n_rows"]:
                if row_id != segment["last_row_id"]:
                    errors.append(
                        f"{segment['game_key']}: last row {row_id} != "
                        f"{segment['last_row_id']}"
                    )
                segment_index += 1
                ordinal = 0

            if len(batch) >= 50_000:
                connection.executemany(
                    "INSERT INTO train_pitches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                connection.commit()
                batch.clear()
        if errors and segment_index >= len(segments):
            break

    if batch:
        connection.executemany(
            "INSERT INTO train_pitches VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
        )
        connection.commit()
    if segment_index != len(segments) or ordinal != 0:
        errors.append(
            f"segment consumption ended at index={segment_index}, ordinal={ordinal}, "
            f"expected_segments={len(segments)}"
        )
    return errors


def scan_trackman(
    args: argparse.Namespace,
    connection: sqlite3.Connection,
    selected_games: set[str],
) -> None:
    batch: list[tuple[Any, ...]] = []
    source_order = 0
    insert_sql = "INSERT INTO trackman_pitches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

    for chunk in pd.read_csv(
        args.trackman,
        usecols=TRACKMAN_COLUMNS,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        game_ids = chunk["trackman_game_id"].astype("string").str.strip()
        selected = chunk.loc[game_ids.isin(selected_games)]
        for values in selected.to_dict(orient="records"):
            team_string = scalar(values["pitcher_team"])
            batch.append(
                (
                    scalar(values["trackman_game_id"]),
                    integer(values["pitch_no"]),
                    source_order,
                    scalar(values["trackman_id"]),
                    scalar(values["season"]),
                    context_token(values),
                    scalar(values["pitcher_trackman_id"]),
                    hand_code(values["pitcher_hand"]),
                    team_string,
                    TEAM_MAP.get(team_string),
                )
            )
            source_order += 1
            if len(batch) >= 50_000:
                connection.executemany(insert_sql, batch)
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()

    connection.execute(
        "CREATE INDEX trackman_game_pitch_idx ON trackman_pitches "
        "(trackman_game_id, pitch_no, source_order)"
    )
    connection.execute(
        """
        CREATE TABLE trackman_ordered AS
        SELECT
            trackman_game_id,
            ROW_NUMBER() OVER (
                PARTITION BY trackman_game_id
                ORDER BY pitch_no, source_order
            ) - 1 AS ordinal,
            trackman_id,
            season,
            context,
            pitcher_trackman_id,
            pitcher_hand,
            pitcher_team,
            pitcher_team_id
        FROM trackman_pitches
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX trackman_ordered_idx ON trackman_ordered "
        "(trackman_game_id, ordinal)"
    )
    connection.commit()


def audit_games(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        WITH train_counts AS (
            SELECT train_game_key, COUNT(*) AS train_rows
            FROM train_pitches GROUP BY train_game_key
        ),
        trackman_counts AS (
            SELECT trackman_game_id, COUNT(*) AS trackman_rows
            FROM trackman_ordered GROUP BY trackman_game_id
        ),
        aligned AS (
            SELECT
                m.train_game_key,
                COUNT(*) AS aligned_rows,
                SUM(CASE WHEN tr.context = tm.context THEN 0 ELSE 1 END)
                    AS context_mismatches
            FROM matches m
            JOIN train_pitches tr
              ON tr.train_game_key = m.train_game_key
            JOIN trackman_ordered tm
              ON tm.trackman_game_id = m.trackman_game_id
             AND tm.ordinal = tr.ordinal
            GROUP BY m.train_game_key
        )
        SELECT
            m.train_game_key,
            m.trackman_game_id,
            m.game_date,
            m.season,
            COALESCE(tc.train_rows, 0),
            COALESCE(mc.trackman_rows, 0),
            COALESCE(a.aligned_rows, 0),
            COALESCE(a.context_mismatches, 0)
        FROM matches m
        LEFT JOIN train_counts tc ON tc.train_game_key = m.train_game_key
        LEFT JOIN trackman_counts mc ON mc.trackman_game_id = m.trackman_game_id
        LEFT JOIN aligned a ON a.train_game_key = m.train_game_key
        ORDER BY m.train_game_key
    """
    rows = []
    for values in connection.execute(query):
        (
            train_game,
            trackman_game,
            game_date,
            season,
            train_rows,
            trackman_rows,
            aligned_rows,
            mismatches,
        ) = values
        valid = int(
            train_rows == trackman_rows == aligned_rows and mismatches == 0
        )
        rows.append(
            {
                "train_game_key": train_game,
                "trackman_game_id": trackman_game,
                "game_date": game_date,
                "season": season,
                "train_rows": train_rows,
                "trackman_rows": trackman_rows,
                "aligned_rows": aligned_rows,
                "context_mismatches": mismatches,
                "valid_exact_alignment": valid,
            }
        )
    connection.execute("CREATE TABLE valid_games (train_game_key TEXT PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO valid_games VALUES (?)",
        [(row["train_game_key"],) for row in rows if row["valid_exact_alignment"]],
    )
    connection.commit()
    return rows


def pair_evidence(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT
            tr.pitcher_id,
            tm.pitcher_trackman_id,
            COUNT(*) AS aligned_pitches,
            COUNT(DISTINCT tr.train_game_key) AS exact_games,
            COUNT(DISTINCT tr.season) AS seasons,
            SUM(CASE
                WHEN tm.pitcher_team_id IS NOT NULL
                 AND CAST(tm.pitcher_team_id AS TEXT) != tr.pitcher_team_id
                THEN 1 ELSE 0 END) AS team_conflicts,
            SUM(CASE WHEN tm.pitcher_team_id IS NULL THEN 1 ELSE 0 END)
                AS unknown_team_rows,
            SUM(CASE WHEN tm.pitcher_hand != tr.pitcher_hand THEN 1 ELSE 0 END)
                AS hand_mismatches,
            MIN(m.game_date) AS first_date,
            MAX(m.game_date) AS last_date
        FROM valid_games vg
        JOIN matches m ON m.train_game_key = vg.train_game_key
        JOIN train_pitches tr ON tr.train_game_key = vg.train_game_key
        JOIN trackman_ordered tm
          ON tm.trackman_game_id = m.trackman_game_id
         AND tm.ordinal = tr.ordinal
        GROUP BY tr.pitcher_id, tm.pitcher_trackman_id
    """
    columns = [
        "pitcher_id",
        "pitcher_trackman_id",
        "aligned_pitches",
        "exact_games",
        "seasons",
        "team_conflicts",
        "unknown_team_rows",
        "hand_mismatches",
        "first_date",
        "last_date",
    ]
    return [dict(zip(columns, values)) for values in connection.execute(query)]


def classify_candidates(
    evidence: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forward_totals: Counter[str] = Counter()
    reverse_totals: Counter[str] = Counter()
    by_pitcher: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        forward_totals[row["pitcher_id"]] += row["aligned_pitches"]
        reverse_totals[row["pitcher_trackman_id"]] += row["aligned_pitches"]
        by_pitcher[row["pitcher_id"]].append(row)

    top_trackman_counts: Counter[str] = Counter()
    for rows in by_pitcher.values():
        rows.sort(key=lambda item: item["aligned_pitches"], reverse=True)
        top_trackman_counts[rows[0]["pitcher_trackman_id"]] += 1

    candidates: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for pitcher_id, rows in sorted(by_pitcher.items()):
        rows.sort(key=lambda item: item["aligned_pitches"], reverse=True)
        second_pitches = rows[1]["aligned_pitches"] if len(rows) > 1 else 0
        for rank, evidence_row in enumerate(rows, start=1):
            pitches = evidence_row["aligned_pitches"]
            candidate = {
                **evidence_row,
                "candidate_rank": rank,
                "forward_purity": pitches / forward_totals[pitcher_id],
                "reverse_purity": pitches
                / reverse_totals[evidence_row["pitcher_trackman_id"]],
                "hand_match_rate": 1.0
                - evidence_row["hand_mismatches"] / pitches,
            }
            candidates.append(candidate)

        top = candidates[-len(rows)]
        ratio = math.inf if second_pitches == 0 else top["aligned_pitches"] / second_pitches
        collision = top_trackman_counts[top["pitcher_trackman_id"]] > 1
        accepted = (
            top["aligned_pitches"] >= args.min_pitches
            and top["exact_games"] >= args.min_games
            and top["forward_purity"] >= args.min_forward_purity
            and top["reverse_purity"] >= args.min_reverse_purity
            and ratio >= args.min_top_second_ratio
            and top["team_conflicts"] == 0
            and not collision
        )
        review = (
            not accepted
            and top["aligned_pitches"] >= 20
            and top["exact_games"] >= 2
            and top["forward_purity"] >= 0.95
            and top["reverse_purity"] >= 0.95
            and top["team_conflicts"] == 0
            and not collision
        )
        status = "accepted" if accepted else "review" if review else "insufficient"
        mappings.append(
            {
                **top,
                "second_candidate": rows[1]["pitcher_trackman_id"]
                if len(rows) > 1
                else "",
                "second_pitches": second_pitches,
                "top_second_ratio": ratio if math.isfinite(ratio) else "inf",
                "trackman_id_collision": int(collision),
                "status": status,
            }
        )
    return candidates, mappings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else ["empty"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.train = args.train.expanduser().resolve()
    args.trackman = args.trackman.expanduser().resolve()
    args.team_audit_dir = args.team_audit_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    candidate_path = args.team_audit_dir / "game_match_candidates.csv"
    segment_path = args.team_audit_dir / "train_game_segments.csv"
    for path in (args.train, args.trackman, candidate_path, segment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in OUTPUT_FILES if (args.output_dir / name).exists()]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use --force:\n{joined}")

    matches, duplicate_exact_matches = load_exact_matches(candidate_path)
    if not matches:
        raise RuntimeError("No unique rank-1 exact game matches were found")
    segments = load_segments(segment_path)
    selected_train_games = {row["train_game_key"] for row in matches}
    selected_trackman_games = {row["trackman_game_id"] for row in matches}

    database_path = args.output_dir / ".pitcher_mapping_audit.sqlite3"
    connection = create_database(database_path)
    try:
        connection.executemany(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            [
                (
                    row["train_game_key"],
                    row["trackman_game_id"],
                    row["game_date"],
                    row["season"],
                )
                for row in matches
            ],
        )
        connection.commit()

        print(f"stage 1/5: loading {len(matches):,} unique exact game matches")
        print("stage 2/5: scanning train rows and assigning game ordinals")
        segment_errors = scan_train(
            args, connection, segments, selected_train_games
        )
        if segment_errors:
            preview = "\n".join(segment_errors[:20])
            raise RuntimeError(f"Train segmentation audit failed:\n{preview}")

        print("stage 3/5: scanning and externally ordering selected Trackman games")
        scan_trackman(args, connection, selected_trackman_games)

        print("stage 4/5: auditing row alignment and aggregating pitcher-ID evidence")
        game_audit = audit_games(connection)
        evidence = pair_evidence(connection)
        candidates, mappings = classify_candidates(evidence, args)

        print("stage 5/5: writing compact pitcher-mapping audit artifacts")
        write_csv(args.output_dir / "exact_game_alignment_audit.csv", game_audit)
        write_csv(args.output_dir / "pitcher_mapping_candidates.csv", candidates)
        write_csv(
            args.output_dir / "accepted_pitcher_mapping.csv",
            [row for row in mappings if row["status"] == "accepted"],
        )
        write_csv(
            args.output_dir / "review_pitcher_mapping.csv",
            [row for row in mappings if row["status"] != "accepted"],
        )

        status_counts = Counter(row["status"] for row in mappings)
        valid_games = sum(row["valid_exact_alignment"] for row in game_audit)
        summary = {
            "train_path": str(args.train),
            "trackman_path": str(args.trackman),
            "input_rank1_exact_matches": len(matches) + duplicate_exact_matches,
            "excluded_duplicate_trackman_game_matches": duplicate_exact_matches,
            "unique_exact_matches_used": len(matches),
            "valid_exact_row_alignments": valid_games,
            "invalid_exact_row_alignments": len(game_audit) - valid_games,
            "train_pitchers_with_evidence": len(mappings),
            "mapping_status_counts": dict(status_counts),
            "thresholds": {
                "min_pitches": args.min_pitches,
                "min_games": args.min_games,
                "min_forward_purity": args.min_forward_purity,
                "min_reverse_purity": args.min_reverse_purity,
                "min_top_second_ratio": args.min_top_second_ratio,
            },
            "hand_assumption": {"Right": 1, "Left": 2},
            "hand_match_rate_is_diagnostic_only": True,
            "physical_measurements_used_for_mapping": False,
            "target_used_for_mapping": False,
        }
        summary_path = args.output_dir / "pitcher_mapping_summary.json"
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, summary_path)
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()

    print(f"output_dir: {args.output_dir}")
    print("next: inspect pitcher_mapping_summary.json and accepted_pitcher_mapping.csv")


if __name__ == "__main__":
    main()
