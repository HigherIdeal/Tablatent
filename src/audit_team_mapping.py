#!/usr/bin/env python3
"""Audit train team-ID to Trackman team-string mappings.

This script treats anonymous team IDs as symbols in a substitution cipher.  It
reconstructs game-like segments from train row order, fingerprints the shared
pitch-state sequences, compares them with Trackman games, and emits compact
artifacts for a later mapping decision.

The script does not modify either source CSV and does not declare a final team
mapping.  Run it explicitly because it scans both large input files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_TRACKMAN = ROOT / "data" / "raw" / "trackman_history.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "team_mapping_audit"

TRAIN_COLUMNS = [
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

TRACKMAN_COLUMNS = [
    "trackman_id",
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team",
    "batter_team",
]

OUTPUT_FILES = [
    "train_game_segments.csv",
    "trackman_game_profiles.csv",
    "game_match_candidates.csv",
    "team_pair_votes.csv",
    "audit_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--sketch-size", type=int, default=64)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument(
        "--vote-min-score",
        type=float,
        default=0.82,
        help="Minimum best-candidate score allowed to contribute a team vote",
    )
    parser.add_argument(
        "--vote-min-margin",
        type=float,
        default=0.08,
        help="Minimum best-versus-second score margin for a team vote",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scalar(value: Any) -> str:
    """Canonicalize CSV scalars without turning integers into '1.0'."""
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def hand_code(value: Any) -> str:
    normalized = scalar(value)
    return {
        "Right": "1",
        "R": "1",
        "1": "1",
        "Left": "2",
        "L": "2",
        "2": "2",
    }.get(normalized, normalized)


def half_code(value: Any) -> str:
    normalized = scalar(value)
    return {"Top": "T", "Bottom": "B", "T": "T", "B": "B"}.get(
        normalized, normalized
    )


def integer(value: Any) -> int | None:
    normalized = scalar(value)
    if normalized == "<NA>":
        return None
    try:
        return int(float(normalized))
    except ValueError:
        return None


class BottomKSketch:
    """Small deterministic bottom-k sketch of unique sequence shingles."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.heap: list[int] = []
        self.values: set[int] = set()

    def add(self, shingle: str) -> None:
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        if value in self.values:
            return
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, -value)
            self.values.add(value)
            return
        largest = -self.heap[0]
        if value >= largest:
            return
        removed = -heapq.heapreplace(self.heap, -value)
        self.values.remove(removed)
        self.values.add(value)

    def sorted_values(self) -> tuple[int, ...]:
        return tuple(sorted(self.values))


@dataclass
class GameAccumulator:
    source: str
    game_key: str
    sketch_size: int
    shingle_size: int
    season: str = "<NA>"
    month: str = "<NA>"
    dayofweek: str = "<NA>"
    game_date: str = "<NA>"
    first_row_id: str = "<NA>"
    last_row_id: str = "<NA>"
    n_rows: int = 0
    full_digest: Any = field(default_factory=hashlib.sha256)
    state_counts: Counter[str] = field(default_factory=Counter)
    inning_half_counts: Counter[str] = field(default_factory=Counter)
    hand_counts: Counter[str] = field(default_factory=Counter)
    home_team_counts: Counter[str] = field(default_factory=Counter)
    away_team_counts: Counter[str] = field(default_factory=Counter)
    rolling_tokens: deque[str] = field(default_factory=deque)
    sketch: BottomKSketch = field(init=False)
    previous: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.sketch = BottomKSketch(self.sketch_size)
        self.rolling_tokens = deque(maxlen=self.shingle_size)

    def add(self, row: dict[str, Any]) -> None:
        if self.n_rows == 0:
            self.season = scalar(row["season"])
            self.month = scalar(row["game_month"])
            self.dayofweek = scalar(row["game_dayofweek"])
            self.game_date = scalar(row.get("game_date"))
            self.first_row_id = scalar(
                row.get("row_id", row.get("trackman_id", "<NA>"))
            )

        inning = scalar(row["inning"])
        half = half_code(row["top_bottom"])
        balls = scalar(row["balls_before"])
        strikes = scalar(row["strikes_before"])
        outs = scalar(row["outs_before"])
        pitcher_hand = hand_code(row["pitcher_hand"])
        batter_hand = hand_code(row["batter_hand"])
        # Team linkage must not depend on the provisional hand-code mapping.
        # Hand agreement is reported separately, while the sequence fingerprint
        # uses only unambiguous shared game-state fields.
        token = "|".join([inning, half, balls, strikes, outs])

        self.full_digest.update(token.encode("utf-8"))
        self.full_digest.update(b"\n")
        self.state_counts[f"{balls}|{strikes}|{outs}"] += 1
        self.inning_half_counts[f"{inning}|{half}"] += 1
        self.hand_counts[f"{pitcher_hand}|{batter_hand}"] += 1

        self.rolling_tokens.append(token)
        if len(self.rolling_tokens) == self.shingle_size:
            self.sketch.add("\x1f".join(self.rolling_tokens))

        pitcher_team_name = (
            "pitcher_team_id" if self.source == "train" else "pitcher_team"
        )
        batter_team_name = (
            "batter_team_id" if self.source == "train" else "batter_team"
        )
        pitcher_team = scalar(row[pitcher_team_name])
        batter_team = scalar(row[batter_team_name])
        if half == "T":
            home_team, away_team = pitcher_team, batter_team
        elif half == "B":
            home_team, away_team = batter_team, pitcher_team
        else:
            home_team = away_team = "<NA>"
        if home_team != "<NA>":
            self.home_team_counts[home_team] += 1
        if away_team != "<NA>":
            self.away_team_counts[away_team] += 1

        self.last_row_id = scalar(
            row.get("row_id", row.get("trackman_id", "<NA>"))
        )
        self.n_rows += 1
        self.previous = row

    def finish(self, boundary_reason: str) -> dict[str, Any]:
        home_team, home_votes = most_common(self.home_team_counts)
        away_team, away_votes = most_common(self.away_team_counts)
        role_votes = sum(self.home_team_counts.values()) + sum(
            self.away_team_counts.values()
        )
        role_consistency = (
            (home_votes + away_votes) / role_votes if role_votes else 0.0
        )
        return {
            "source": self.source,
            "game_key": self.game_key,
            "season": self.season,
            "game_month": self.month,
            "game_dayofweek": self.dayofweek,
            "game_date": self.game_date,
            "first_row_id": self.first_row_id,
            "last_row_id": self.last_row_id,
            "n_rows": self.n_rows,
            "home_team": home_team,
            "away_team": away_team,
            "role_consistency": role_consistency,
            "boundary_reason": boundary_reason,
            "sequence_sha256": self.full_digest.hexdigest(),
            "state_counts": dict(self.state_counts),
            "inning_half_counts": dict(self.inning_half_counts),
            "hand_counts": dict(self.hand_counts),
            "sketch": self.sketch.sorted_values(),
        }


def most_common(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return "<NA>", 0
    return counter.most_common(1)[0]


def inferred_roles(row: dict[str, Any]) -> tuple[str, str]:
    half = half_code(row["top_bottom"])
    pitcher = scalar(row["pitcher_team_id"])
    batter = scalar(row["batter_team_id"])
    if half == "T":
        return pitcher, batter
    if half == "B":
        return batter, pitcher
    return "<NA>", "<NA>"


def train_boundary(
    previous: dict[str, Any], current: dict[str, Any], game: GameAccumulator
) -> str | None:
    for column in ("season", "game_month", "game_dayofweek"):
        if scalar(previous[column]) != scalar(current[column]):
            return f"{column}_changed"

    prev_inning = integer(previous["inning"])
    curr_inning = integer(current["inning"])
    if prev_inning is not None and curr_inning is not None:
        if curr_inning < prev_inning:
            return "inning_decreased"
        if (
            curr_inning == prev_inning
            and half_code(previous["top_bottom"]) == "B"
            and half_code(current["top_bottom"]) == "T"
        ):
            return "half_reversed"

    for column in ("run_top_before", "run_bot_before"):
        before = integer(previous[column])
        after = integer(current[column])
        if before is not None and after is not None and after < before:
            return f"{column}_decreased"

    home, away = inferred_roles(current)
    current_home, _ = most_common(game.home_team_counts)
    current_away, _ = most_common(game.away_team_counts)
    if (
        current_home != "<NA>"
        and current_away != "<NA>"
        and (home != current_home or away != current_away)
    ):
        return "team_pair_changed"
    return None


def iter_records(path: Path, columns: list[str], chunksize: int) -> Iterable[dict[str, Any]]:
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        yield from chunk.to_dict(orient="records")


def scan_train(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    games: list[dict[str, Any]] = []
    boundaries: Counter[str] = Counter()
    current: GameAccumulator | None = None
    game_number = 0

    for row in iter_records(args.train, TRAIN_COLUMNS, args.chunksize):
        if current is None:
            current = GameAccumulator(
                "train",
                f"train_game_{game_number:06d}",
                args.sketch_size,
                args.shingle_size,
            )
        elif current.previous is not None:
            reason = train_boundary(current.previous, row, current)
            if reason is not None:
                games.append(current.finish(reason))
                boundaries[reason] += 1
                game_number += 1
                current = GameAccumulator(
                    "train",
                    f"train_game_{game_number:06d}",
                    args.sketch_size,
                    args.shingle_size,
                )
        current.add(row)

    if current is not None and current.n_rows:
        games.append(current.finish("end_of_file"))
        boundaries["end_of_file"] += 1
    return games, boundaries


def scan_trackman(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    """Externally sort interleaved Trackman rows by game and pitch number."""
    database_path = args.output_dir / ".trackman_game_sort.sqlite3"
    if database_path.exists():
        database_path.unlink()

    insert_sql = """
        INSERT INTO pitches (
            source_order, game_id, pitch_no, trackman_id, season, game_date,
            game_month, game_dayofweek, inning, top_bottom, balls_before,
            strikes_before, outs_before, pitcher_hand, batter_hand,
            pitcher_team, batter_team
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    select_sql = """
        SELECT
            trackman_id, season, game_date, game_month, game_dayofweek,
            game_id AS trackman_game_id, pitch_no, inning, top_bottom,
            balls_before, strikes_before, outs_before, pitcher_hand,
            batter_hand, pitcher_team, batter_team
        FROM pitches
        ORDER BY game_id, pitch_no, source_order
    """

    input_switches = 0
    observed_games: set[str] = set()
    previous_game_id: str | None = None
    source_order = 0

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute(
            """
            CREATE TABLE pitches (
                source_order INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                pitch_no INTEGER,
                trackman_id TEXT,
                season TEXT,
                game_date TEXT,
                game_month TEXT,
                game_dayofweek TEXT,
                inning TEXT,
                top_bottom TEXT,
                balls_before TEXT,
                strikes_before TEXT,
                outs_before TEXT,
                pitcher_hand TEXT,
                batter_hand TEXT,
                pitcher_team TEXT,
                batter_team TEXT
            )
            """
        )

        batch: list[tuple[Any, ...]] = []
        for row in iter_records(args.trackman, TRACKMAN_COLUMNS, args.chunksize):
            game_id = scalar(row["trackman_game_id"])
            if previous_game_id is not None and game_id != previous_game_id:
                if game_id in observed_games:
                    input_switches += 1
                observed_games.add(previous_game_id)
            previous_game_id = game_id

            batch.append(
                (
                    source_order,
                    game_id,
                    integer(row["pitch_no"]),
                    scalar(row["trackman_id"]),
                    scalar(row["season"]),
                    scalar(row["game_date"]),
                    scalar(row["game_month"]),
                    scalar(row["game_dayofweek"]),
                    scalar(row["inning"]),
                    half_code(row["top_bottom"]),
                    scalar(row["balls_before"]),
                    scalar(row["strikes_before"]),
                    scalar(row["outs_before"]),
                    hand_code(row["pitcher_hand"]),
                    hand_code(row["batter_hand"]),
                    scalar(row["pitcher_team"]),
                    scalar(row["batter_team"]),
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
            "CREATE INDEX pitches_game_pitch_idx "
            "ON pitches (game_id, pitch_no, source_order)"
        )
        connection.commit()

        games: list[dict[str, Any]] = []
        current: GameAccumulator | None = None
        column_names = [description[0] for description in connection.execute(select_sql).description]
        cursor = connection.execute(select_sql)
        for values in cursor:
            row = dict(zip(column_names, values))
            game_id = scalar(row["trackman_game_id"])
            if current is None or current.game_key != game_id:
                if current is not None:
                    games.append(current.finish("game_id_changed"))
                current = GameAccumulator(
                    "trackman",
                    game_id,
                    args.sketch_size,
                    args.shingle_size,
                )
            current.add(row)
        if current is not None and current.n_rows:
            games.append(current.finish("end_of_file"))
        return games, input_switches
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()


def counter_similarity(left: dict[str, int], right: dict[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return 0.0
    keys = set(left) | set(right)
    distance = sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )
    return max(0.0, 1.0 - distance / 2.0)


def sketch_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def game_similarity(train: dict[str, Any], trackman: dict[str, Any]) -> dict[str, float]:
    exact = float(train["sequence_sha256"] == trackman["sequence_sha256"])
    length = math.exp(
        -abs(math.log((train["n_rows"] + 1) / (trackman["n_rows"] + 1)))
    )
    state = counter_similarity(train["state_counts"], trackman["state_counts"])
    inning_half = counter_similarity(
        train["inning_half_counts"], trackman["inning_half_counts"]
    )
    hands = counter_similarity(train["hand_counts"], trackman["hand_counts"])
    shingles = sketch_similarity(train["sketch"], trackman["sketch"])
    score = (
        0.15 * length
        + 0.15 * state
        + 0.15 * inning_half
        + 0.55 * shingles
    )
    if exact:
        score = 1.0
    return {
        "score": score,
        "exact_sequence": exact,
        "length_score": length,
        "state_score": state,
        "inning_half_score": inning_half,
        "hand_score": hands,
        "shingle_score": shingles,
    }


def match_games(
    train_games: list[dict[str, Any]],
    trackman_games: list[dict[str, Any]],
    top_candidates: int,
) -> list[dict[str, Any]]:
    index: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for game in trackman_games:
        index[(game["season"], game["game_month"], game["game_dayofweek"])].append(game)

    matches: list[dict[str, Any]] = []
    for train in train_games:
        key = (train["season"], train["game_month"], train["game_dayofweek"])
        ranked: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        for trackman in index.get(key, []):
            length_ratio = (train["n_rows"] + 1) / (trackman["n_rows"] + 1)
            if not 0.35 <= length_ratio <= 1.65:
                continue
            components = game_similarity(train, trackman)
            ranked.append((components["score"], trackman, components))
        ranked.sort(key=lambda item: item[0], reverse=True)

        for rank, (_, trackman, components) in enumerate(
            ranked[:top_candidates], start=1
        ):
            matches.append(
                {
                    "train_game_key": train["game_key"],
                    "candidate_rank": rank,
                    "trackman_game_id": trackman["game_key"],
                    "season": train["season"],
                    "game_month": train["game_month"],
                    "game_dayofweek": train["game_dayofweek"],
                    "trackman_game_date": trackman["game_date"],
                    "train_home_team_id": train["home_team"],
                    "train_away_team_id": train["away_team"],
                    "trackman_home_team": trackman["home_team"],
                    "trackman_away_team": trackman["away_team"],
                    "train_rows": train["n_rows"],
                    "trackman_rows": trackman["n_rows"],
                    **components,
                }
            )
    return matches


def make_team_votes(
    matches: list[dict[str, Any]], min_score: float, min_margin: float
) -> list[dict[str, Any]]:
    by_game: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        by_game[match["train_game_key"]].append(match)

    votes: defaultdict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"matched_games": 0.0, "vote_weight": 0.0, "exact_games": 0.0}
    )
    for candidates in by_game.values():
        candidates.sort(key=lambda item: int(item["candidate_rank"]))
        best = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = best["score"] - second_score
        if best["score"] < min_score or margin < min_margin:
            continue
        weight = best["score"] * max(margin, 0.01)
        for role in ("home", "away"):
            team_id = best[f"train_{role}_team_id"]
            team_string = best[f"trackman_{role}_team"]
            key = (team_id, team_string, role)
            votes[key]["matched_games"] += 1
            votes[key]["vote_weight"] += weight
            votes[key]["exact_games"] += best["exact_sequence"]

    rows = []
    for (team_id, team_string, role), values in votes.items():
        rows.append(
            {
                "team_id": team_id,
                "team_string": team_string,
                "role": role,
                "matched_games": int(values["matched_games"]),
                "exact_games": int(values["exact_games"]),
                "vote_weight": values["vote_weight"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["team_id"], -row["vote_weight"], row["team_string"]),
    )


def serializable_game(game: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in game.items()
        if key
        not in {
            "state_counts",
            "inning_half_counts",
            "hand_counts",
            "sketch",
        }
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else ["empty"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.train = args.train.expanduser().resolve()
    args.trackman = args.trackman.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    for source in (args.train, args.trackman):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.chunksize <= 0 or args.sketch_size <= 0 or args.shingle_size <= 0:
        raise ValueError("chunk and sketch sizes must be positive")
    if args.top_candidates <= 0:
        raise ValueError("--top-candidates must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in OUTPUT_FILES if (args.output_dir / name).exists()]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use --force:\n{joined}")

    print("stage 1/4: scanning train and reconstructing game-like segments")
    train_games, boundary_counts = scan_train(args)
    print(f"train_game_segments: {len(train_games):,}")

    print("stage 2/4: scanning Trackman and fingerprinting games")
    trackman_games, noncontiguous_games = scan_trackman(args)
    print(f"trackman_games: {len(trackman_games):,}")
    print(f"trackman_input_game_reappearances: {noncontiguous_games:,}")

    print("stage 3/4: ranking candidate game matches")
    matches = match_games(train_games, trackman_games, args.top_candidates)
    votes = make_team_votes(matches, args.vote_min_score, args.vote_min_margin)

    print("stage 4/4: writing compact audit artifacts")
    write_csv(
        args.output_dir / "train_game_segments.csv",
        [serializable_game(game) for game in train_games],
    )
    write_csv(
        args.output_dir / "trackman_game_profiles.csv",
        [serializable_game(game) for game in trackman_games],
    )
    write_csv(args.output_dir / "game_match_candidates.csv", matches)
    write_csv(args.output_dir / "team_pair_votes.csv", votes)

    exact_matches = sum(
        1 for match in matches if match["candidate_rank"] == 1 and match["exact_sequence"]
    )
    accepted_vote_games = len(
        {
            match["train_game_key"]
            for match in matches
            if match["candidate_rank"] == 1
            and match["score"] >= args.vote_min_score
        }
    )
    summary = {
        "train_path": str(args.train),
        "trackman_path": str(args.trackman),
        "hand_assumption": {"Right": 1, "Left": 2},
        "train_game_segments": len(train_games),
        "trackman_games": len(trackman_games),
        "train_boundary_counts": dict(boundary_counts),
        "trackman_input_game_reappearances": noncontiguous_games,
        "candidate_rows": len(matches),
        "rank1_exact_sequence_matches": exact_matches,
        "rank1_games_above_score_threshold": accepted_vote_games,
        "team_vote_rows": len(votes),
        "vote_min_score": args.vote_min_score,
        "vote_min_margin": args.vote_min_margin,
        "note": "Audit candidates are evidence, not a final mapping.",
    }
    summary_path = args.output_dir / "audit_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)

    print(f"output_dir: {args.output_dir}")
    print("next: inspect audit_summary.json, game_match_candidates.csv, and team_pair_votes.csv")


if __name__ == "__main__":
    main()
