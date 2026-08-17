#!/usr/bin/env python3
"""Build leak-safe arsenal tokens and pitch-level auxiliary labels for MoE.

The deployable arsenal for feature season S is aggregated exclusively from
Trackman seasons earlier than S.  Exact train/Trackman row alignments are
written to a separate auxiliary-label table: current-pitch type and physical
measurements must be used as training targets only, never as model inputs.

This is a heavy data-building script.  It scans both raw CSV files and should
be run explicitly by the user.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from build_trackman_history_mod import TEAM_MAP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_TRACKMAN = ROOT / "data" / "raw" / "trackman_history.csv"
DEFAULT_MAPPING = (
    ROOT / "outputs" / "pitcher_mapping_audit" / "accepted_pitcher_mapping.csv"
)
DEFAULT_SEGMENTS = (
    ROOT / "outputs" / "team_mapping_audit" / "train_game_segments.csv"
)
DEFAULT_ALIGNMENTS = (
    ROOT / "outputs" / "pitcher_mapping_audit" / "exact_game_alignment_audit.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pitch_arsenal_moe"

NUMERIC_COLUMNS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
PITCH_GROUP_TO_ID = {name: index for index, name in enumerate(PITCH_GROUPS)}
TARGET_SEASONS = list(range(2019, 2026))

TRACKMAN_PROFILE_COLUMNS = [
    "season",
    "pitcher_trackman_id",
    "pitcher_hand",
    "pitcher_team",
    "pitch_type_group",
    *NUMERIC_COLUMNS,
]
TRACKMAN_ALIGNMENT_COLUMNS = [
    "trackman_id",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitch_type_group",
    *NUMERIC_COLUMNS,
]
TRAIN_ALIGNMENT_COLUMNS = [
    "row_id",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
]

OUTPUT_FILES = [
    "accepted_pitcher_mapping_snapshot.csv",
    "pitcher_arsenal_by_season.parquet",
    "team_hand_arsenal_by_season.parquet",
    "league_hand_arsenal_by_season.parquet",
    "league_arsenal_by_season.parquet",
    "pitcher_arsenal_2025.parquet",
    "fallback_arsenal_2025.parquet",
    "aligned_pitch_auxiliary.parquet",
    "arsenal_feature_manifest.json",
    "build_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--alignments", type=Path, default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required. Install it in the active environment with "
            "`python -m pip install pyarrow`."
        ) from error
    return pa, pq


def scalar(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def context_token(values: dict[str, Any]) -> str:
    half = {"Top": "T", "Bottom": "B", "T": "T", "B": "B"}.get(
        scalar(values["top_bottom"]), scalar(values["top_bottom"])
    )
    return "|".join(
        [
            scalar(values["inning"]),
            half,
            scalar(values["balls_before"]),
            scalar(values["strikes_before"]),
            scalar(values["outs_before"]),
        ]
    )


def normalize_id_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def normalize_hand_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.map(
        {"Right": 2, "R": 2, "2": 2, "Left": 1, "L": 1, "1": 1}
    ).astype("Int64")


def normalize_pitch_group_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.where(values.isin(PITCH_GROUPS[:-1]), "other")


def normalize_pitch_group_value(value: Any) -> str:
    normalized = scalar(value).lower()
    return normalized if normalized in PITCH_GROUPS[:-1] else "other"


def load_mapping(path: Path) -> dict[str, int]:
    frame = pd.read_csv(
        path,
        usecols=["pitcher_id", "pitcher_trackman_id", "status"],
        dtype="string",
    )
    frame = frame.loc[frame["status"].str.strip().eq("accepted")].copy()
    frame["pitcher_id"] = normalize_id_series(frame["pitcher_id"])
    frame["pitcher_trackman_id"] = normalize_id_series(
        frame["pitcher_trackman_id"]
    )
    if frame[["pitcher_id", "pitcher_trackman_id"]].isna().any().any():
        raise ValueError("Accepted mapping contains non-numeric IDs")
    if frame["pitcher_id"].duplicated().any():
        raise ValueError("Accepted mapping contains duplicate train pitcher IDs")
    if frame["pitcher_trackman_id"].duplicated().any():
        raise ValueError("Accepted mapping contains duplicate Trackman pitcher IDs")
    return {
        str(trackman_id): int(train_id)
        for train_id, trackman_id in zip(
            frame["pitcher_id"], frame["pitcher_trackman_id"]
        )
    }


def empty_vector() -> np.ndarray:
    # pitch count followed by metric-specific count, sum, and squared sum.
    return np.zeros(1 + 3 * len(NUMERIC_COLUMNS), dtype=np.float64)


def add_vector(left: np.ndarray, right: np.ndarray) -> None:
    left += right


def update_store(
    frame: pd.DataFrame,
    key_columns: list[str],
    store: dict[tuple[Any, ...], np.ndarray],
) -> None:
    if frame.empty:
        return
    work = frame[key_columns + NUMERIC_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        work[f"{column}__sq"] = work[column] * work[column]
    aggregations: dict[str, tuple[str, str]] = {
        "row_count": (key_columns[0], "size")
    }
    for column in NUMERIC_COLUMNS:
        aggregations[f"{column}__count"] = (column, "count")
        aggregations[f"{column}__sum"] = (column, "sum")
        aggregations[f"{column}__sumsq"] = (f"{column}__sq", "sum")
    grouped = (
        work.groupby(key_columns, dropna=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        values = row._asdict()
        key = tuple(values[column] for column in key_columns)
        vector = store.setdefault(key, empty_vector())
        vector[0] += float(values["row_count"])
        for index, column in enumerate(NUMERIC_COLUMNS):
            offset = 1 + 3 * index
            vector[offset] += float(values[f"{column}__count"])
            vector[offset + 1] += float(values[f"{column}__sum"])
            vector[offset + 2] += float(values[f"{column}__sumsq"])


def vector_mean(vector: np.ndarray, metric_index: int) -> float:
    offset = 1 + 3 * metric_index
    count = vector[offset]
    return vector[offset + 1] / count if count else math.nan


def vector_std(vector: np.ndarray, metric_index: int) -> float:
    offset = 1 + 3 * metric_index
    count = vector[offset]
    if count <= 1:
        return math.nan
    mean = vector[offset + 1] / count
    variance = max(vector[offset + 2] / count - mean * mean, 0.0)
    return math.sqrt(variance)


def arsenal_feature_columns() -> list[str]:
    columns = [
        "ars_pitch_count",
        "ars_pitch_rate",
        "ars_total_pitch_count",
        "ars_seasons_observed",
        "ars_prevseason_pitch_count",
        "ars_prevseason_pitch_rate",
    ]
    for column in NUMERIC_COLUMNS:
        columns.extend(
            [
                f"ars_{column}_mean",
                f"ars_{column}_std",
                f"ars_prevseason_{column}_mean",
                f"ars_prevseason_{column}_delta",
            ]
        )
    return columns


def build_long_profiles(
    store: dict[tuple[Any, ...], np.ndarray], entity_columns: list[str]
) -> pd.DataFrame:
    entity_width = len(entity_columns)
    by_entity: defaultdict[
        tuple[Any, ...], defaultdict[int, dict[str, np.ndarray]]
    ] = defaultdict(lambda: defaultdict(dict))
    for key, vector in store.items():
        entity = tuple(key[:entity_width])
        season = int(key[entity_width])
        group = str(key[entity_width + 1])
        by_entity[entity][season][group] = vector

    rows: list[dict[str, Any]] = []
    for entity, yearly in by_entity.items():
        cumulative: dict[str, np.ndarray] = {}
        observed_seasons: set[int] = set()
        for feature_season in TARGET_SEASONS:
            previous_season = feature_season - 1
            previous = yearly.get(previous_season, {})
            if previous:
                observed_seasons.add(previous_season)
                for group, vector in previous.items():
                    if group not in cumulative:
                        cumulative[group] = empty_vector()
                    add_vector(cumulative[group], vector)
            if not cumulative:
                continue

            total_count = sum(vector[0] for vector in cumulative.values())
            previous_total = sum(vector[0] for vector in previous.values())
            for group_id, group in enumerate(PITCH_GROUPS):
                vector = cumulative.get(group, empty_vector())
                previous_vector = previous.get(group, empty_vector())
                row = {
                    column: value
                    for column, value in zip(entity_columns, entity)
                }
                row.update(
                    {
                        "feature_season": feature_season,
                        "pitch_group": group,
                        "pitch_group_id": group_id,
                        "ars_pitch_count": vector[0],
                        "ars_pitch_rate": vector[0] / total_count
                        if total_count
                        else 0.0,
                        "ars_total_pitch_count": total_count,
                        "ars_seasons_observed": float(len(observed_seasons)),
                        "ars_prevseason_pitch_count": previous_vector[0],
                        "ars_prevseason_pitch_rate": previous_vector[0]
                        / previous_total
                        if previous_total
                        else 0.0,
                    }
                )
                for metric_index, column in enumerate(NUMERIC_COLUMNS):
                    mean = vector_mean(vector, metric_index)
                    previous_mean = vector_mean(previous_vector, metric_index)
                    row[f"ars_{column}_mean"] = mean
                    row[f"ars_{column}_std"] = vector_std(vector, metric_index)
                    row[f"ars_prevseason_{column}_mean"] = previous_mean
                    row[f"ars_prevseason_{column}_delta"] = (
                        previous_mean - mean
                        if not math.isnan(previous_mean) and not math.isnan(mean)
                        else math.nan
                    )
                rows.append(row)

    columns = [
        *entity_columns,
        "feature_season",
        "pitch_group",
        "pitch_group_id",
        *arsenal_feature_columns(),
    ]
    return pd.DataFrame(rows, columns=columns)


def scan_trackman_profiles(
    args: argparse.Namespace, mapping: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    player_store: dict[tuple[Any, ...], np.ndarray] = {}
    team_hand_store: dict[tuple[Any, ...], np.ndarray] = {}
    league_hand_store: dict[tuple[Any, ...], np.ndarray] = {}
    league_store: dict[tuple[Any, ...], np.ndarray] = {}
    mapped_rows = 0
    total_rows = 0
    unmapped_teams: Counter[str] = Counter()

    reader = pd.read_csv(
        args.trackman,
        usecols=TRACKMAN_PROFILE_COLUMNS,
        chunksize=args.chunksize,
        low_memory=False,
    )
    progress = tqdm(reader, desc="Trackman arsenal", unit="chunk")
    for chunk in progress:
        total_rows += len(chunk)
        chunk["source_season"] = pd.to_numeric(
            chunk["season"], errors="coerce"
        ).astype("Int64")
        trackman_ids = normalize_id_series(chunk["pitcher_trackman_id"])
        chunk["train_pitcher_id"] = trackman_ids.map(mapping).astype("Int64")
        mapped_rows += int(chunk["train_pitcher_id"].notna().sum())
        chunk["hand"] = normalize_hand_series(chunk["pitcher_hand"])
        chunk["pitch_group"] = normalize_pitch_group_series(
            chunk["pitch_type_group"]
        )
        team_strings = chunk["pitcher_team"].astype("string").str.strip()
        chunk["team_id"] = team_strings.map(TEAM_MAP).astype("Int64")
        for value, count in team_strings.loc[chunk["team_id"].isna()].value_counts().items():
            unmapped_teams[str(value)] += int(count)
        for column in NUMERIC_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        valid = chunk.loc[chunk["source_season"].notna()].copy()
        update_store(
            valid.loc[valid["train_pitcher_id"].notna()],
            ["train_pitcher_id", "source_season", "pitch_group"],
            player_store,
        )
        update_store(
            valid.loc[valid["team_id"].notna() & valid["hand"].notna()],
            ["team_id", "hand", "source_season", "pitch_group"],
            team_hand_store,
        )
        update_store(
            valid.loc[valid["hand"].notna()],
            ["hand", "source_season", "pitch_group"],
            league_hand_store,
        )
        valid["league_key"] = "league"
        update_store(
            valid,
            ["league_key", "source_season", "pitch_group"],
            league_store,
        )
        progress.set_postfix(rows=f"{total_rows:,}", mapped=f"{mapped_rows:,}")

    player = build_long_profiles(player_store, ["pitcher_id"])
    team_hand = build_long_profiles(
        team_hand_store, ["pitcher_team_id", "pitcher_hand"]
    )
    league_hand = build_long_profiles(league_hand_store, ["pitcher_hand"])
    league = build_long_profiles(league_store, ["league_key"]).drop(
        columns=["league_key"]
    )
    summary = {
        "trackman_rows_scanned": total_rows,
        "trackman_rows_with_accepted_pitcher_mapping": mapped_rows,
        "trackman_mapping_rate": mapped_rows / total_rows if total_rows else 0.0,
        "unmapped_team_occurrences": dict(unmapped_teams),
    }
    return player, team_hand, league_hand, league, summary


def load_valid_alignments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string")
    required = {
        "train_game_key",
        "trackman_game_id",
        "valid_exact_alignment",
        "game_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Alignment audit missing columns: {missing}")
    frame = frame.loc[frame["valid_exact_alignment"].eq("1")].copy()
    if frame.empty:
        raise RuntimeError("No valid exact game alignments were found")
    if frame["train_game_key"].duplicated().any():
        raise ValueError("Duplicate train game keys in exact alignments")
    if frame["trackman_game_id"].duplicated().any():
        raise ValueError("Duplicate Trackman game IDs in exact alignments")
    return frame[["train_game_key", "trackman_game_id", "game_date"]]


def load_segments(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(
        path,
        usecols=["game_key", "first_row_id", "last_row_id", "n_rows"],
        dtype="string",
    )
    frame["n_rows"] = pd.to_numeric(frame["n_rows"], errors="raise").astype(int)
    return frame.to_dict(orient="records")


def create_alignment_database(
    path: Path, alignments: pd.DataFrame
) -> sqlite3.Connection:
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
            game_date TEXT
        );
        CREATE TABLE train_rows (
            train_game_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            row_id TEXT NOT NULL,
            context TEXT NOT NULL,
            PRIMARY KEY (train_game_key, ordinal)
        );
        CREATE TABLE trackman_rows (
            trackman_game_id TEXT NOT NULL,
            pitch_no INTEGER,
            source_order INTEGER NOT NULL,
            context TEXT NOT NULL,
            pitch_group TEXT NOT NULL,
            rel_speed REAL,
            spin_rate REAL,
            induced_vert_break REAL,
            horz_break REAL,
            extension REAL,
            rel_height REAL,
            rel_side REAL,
            zone_speed REAL
        );
        """
    )
    connection.executemany(
        "INSERT INTO matches VALUES (?, ?, ?)",
        list(alignments.itertuples(index=False, name=None)),
    )
    connection.commit()
    return connection


def scan_aligned_train_rows(
    args: argparse.Namespace,
    connection: sqlite3.Connection,
    segments: list[dict[str, Any]],
    selected_games: set[str],
) -> int:
    segment_index = 0
    ordinal = 0
    total_rows = 0
    inserted = 0
    batch: list[tuple[Any, ...]] = []
    reader = pd.read_csv(
        args.train,
        usecols=TRAIN_ALIGNMENT_COLUMNS,
        chunksize=args.chunksize,
        low_memory=False,
    )
    progress = tqdm(reader, desc="Aligned train rows", unit="chunk")
    for chunk in progress:
        for values in chunk.to_dict(orient="records"):
            if segment_index >= len(segments):
                raise RuntimeError("train contains rows beyond recorded segments")
            segment = segments[segment_index]
            row_id = scalar(values["row_id"])
            if ordinal == 0 and row_id != scalar(segment["first_row_id"]):
                raise RuntimeError(
                    f"{segment['game_key']}: first row changed from segment audit"
                )
            if segment["game_key"] in selected_games:
                batch.append(
                    (
                        segment["game_key"],
                        ordinal,
                        row_id,
                        context_token(values),
                    )
                )
                inserted += 1
            ordinal += 1
            total_rows += 1
            if ordinal == int(segment["n_rows"]):
                if row_id != scalar(segment["last_row_id"]):
                    raise RuntimeError(
                        f"{segment['game_key']}: last row changed from segment audit"
                    )
                segment_index += 1
                ordinal = 0
            if len(batch) >= 50_000:
                connection.executemany("INSERT INTO train_rows VALUES (?, ?, ?, ?)", batch)
                connection.commit()
                batch.clear()
        progress.set_postfix(rows=f"{total_rows:,}", selected=f"{inserted:,}")
    if batch:
        connection.executemany("INSERT INTO train_rows VALUES (?, ?, ?, ?)", batch)
        connection.commit()
    if segment_index != len(segments) or ordinal != 0:
        raise RuntimeError(
            f"segment consumption incomplete: index={segment_index}, ordinal={ordinal}"
        )
    return inserted


def scan_aligned_trackman_rows(
    args: argparse.Namespace,
    connection: sqlite3.Connection,
    selected_games: set[str],
) -> int:
    insert_sql = "INSERT INTO trackman_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    batch: list[tuple[Any, ...]] = []
    source_order = 0
    selected_rows = 0
    reader = pd.read_csv(
        args.trackman,
        usecols=TRACKMAN_ALIGNMENT_COLUMNS,
        chunksize=args.chunksize,
        low_memory=False,
    )
    progress = tqdm(reader, desc="Aligned Trackman rows", unit="chunk")
    for chunk in progress:
        game_ids = chunk["trackman_game_id"].astype("string").str.strip()
        selected = chunk.loc[game_ids.isin(selected_games)].copy()
        selected["pitch_no"] = pd.to_numeric(
            selected["pitch_no"], errors="coerce"
        )
        for column in NUMERIC_COLUMNS:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
        for values in selected.to_dict(orient="records"):
            numeric_values = [
                None if pd.isna(values[column]) else float(values[column])
                for column in NUMERIC_COLUMNS
            ]
            pitch_no_value = values["pitch_no"]
            batch.append(
                (
                    scalar(values["trackman_game_id"]),
                    None if pd.isna(pitch_no_value) else int(pitch_no_value),
                    source_order,
                    context_token(values),
                    normalize_pitch_group_value(values["pitch_type_group"]),
                    *numeric_values,
                )
            )
            source_order += 1
            selected_rows += 1
            if len(batch) >= 50_000:
                connection.executemany(insert_sql, batch)
                connection.commit()
                batch.clear()
        progress.set_postfix(selected=f"{selected_rows:,}")
    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()

    connection.execute(
        "CREATE INDEX trackman_game_pitch_idx ON trackman_rows "
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
            context,
            pitch_group,
            rel_speed,
            spin_rate,
            induced_vert_break,
            horz_break,
            extension,
            rel_height,
            rel_side,
            zone_speed
        FROM trackman_rows
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX trackman_ordered_idx ON trackman_ordered "
        "(trackman_game_id, ordinal)"
    )
    connection.commit()
    return selected_rows


def write_auxiliary_parquet(
    connection: sqlite3.Connection, output: Path, pa: Any, pq: Any
) -> dict[str, Any]:
    count_query = """
        SELECT
            COUNT(*),
            SUM(CASE WHEN tr.context = tm.context THEN 0 ELSE 1 END)
        FROM matches m
        JOIN train_rows tr ON tr.train_game_key = m.train_game_key
        JOIN trackman_ordered tm
          ON tm.trackman_game_id = m.trackman_game_id
         AND tm.ordinal = tr.ordinal
    """
    aligned_rows, mismatches = connection.execute(count_query).fetchone()
    aligned_rows = int(aligned_rows or 0)
    mismatches = int(mismatches or 0)
    if not aligned_rows:
        raise RuntimeError("Exact alignment join produced no rows")
    if mismatches:
        raise RuntimeError(f"Exact alignment context mismatches: {mismatches:,}")

    query = """
        SELECT
            tr.row_id,
            m.train_game_key,
            m.trackman_game_id,
            m.game_date,
            tm.pitch_group,
            tm.rel_speed,
            tm.spin_rate,
            tm.induced_vert_break,
            tm.horz_break,
            tm.extension,
            tm.rel_height,
            tm.rel_side,
            tm.zone_speed
        FROM matches m
        JOIN train_rows tr ON tr.train_game_key = m.train_game_key
        JOIN trackman_ordered tm
          ON tm.trackman_game_id = m.trackman_game_id
         AND tm.ordinal = tr.ordinal
        ORDER BY tr.row_id
    """
    columns = [
        "row_id",
        "train_game_key",
        "trackman_game_id",
        "game_date",
        "aux_pitch_group",
        *[f"aux_{column}" for column in NUMERIC_COLUMNS],
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    schema = pa.schema(
        [
            pa.field("row_id", pa.string()),
            pa.field("train_game_key", pa.string()),
            pa.field("trackman_game_id", pa.string()),
            pa.field("game_date", pa.string()),
            pa.field("aux_pitch_group", pa.string()),
            pa.field("aux_pitch_group_id", pa.int8()),
            *[pa.field(f"aux_{column}", pa.float32()) for column in NUMERIC_COLUMNS],
        ]
    )
    writer = pq.ParquetWriter(
        temporary, schema, compression="zstd", use_dictionary=True
    )
    cursor = connection.execute(query)
    written = 0
    progress = tqdm(total=aligned_rows, desc="Auxiliary parquet", unit="row")
    try:
        while True:
            values = cursor.fetchmany(50_000)
            if not values:
                break
            frame = pd.DataFrame.from_records(values, columns=columns)
            frame.insert(
                5,
                "aux_pitch_group_id",
                frame["aux_pitch_group"].map(PITCH_GROUP_TO_ID).astype("int8"),
            )
            for column in NUMERIC_COLUMNS:
                frame[f"aux_{column}"] = pd.to_numeric(
                    frame[f"aux_{column}"], errors="coerce"
                ).astype("float32")
            table = pa.Table.from_pandas(
                frame, schema=schema, preserve_index=False, safe=False
            )
            writer.write_table(table)
            written += len(frame)
            progress.update(len(frame))
    finally:
        progress.close()
        writer.close()
    if written != aligned_rows:
        raise RuntimeError(f"Auxiliary output rows {written:,} != join rows {aligned_rows:,}")
    os.replace(temporary, output)
    group_counts = dict(
        connection.execute(
            "SELECT pitch_group, COUNT(*) FROM trackman_ordered GROUP BY pitch_group"
        ).fetchall()
    )
    return {
        "aligned_auxiliary_rows": written,
        "aligned_context_mismatches": mismatches,
        "auxiliary_pitch_group_counts": group_counts,
    }


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary, path)


def resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "train",
        "trackman",
        "mapping",
        "segments",
        "alignments",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    for path in (
        args.train,
        args.trackman,
        args.mapping,
        args.segments,
        args.alignments,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    pa, pq = require_pyarrow()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        args.output_dir / name
        for name in OUTPUT_FILES
        if (args.output_dir / name).exists()
    ]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use --force:\n{joined}")

    print("stage 1/5: loading accepted pitcher mapping")
    mapping = load_mapping(args.mapping)
    print(f"accepted_pitcher_mappings: {len(mapping):,}")

    print("stage 2/5: building season-lag pitch-group arsenal tokens")
    player, team_hand, league_hand, league, trackman_summary = (
        scan_trackman_profiles(args, mapping)
    )

    print("stage 3/5: writing temporal and frozen-2025 arsenal profiles")
    profile_outputs = [
        (player, "pitcher_arsenal_by_season.parquet"),
        (team_hand, "team_hand_arsenal_by_season.parquet"),
        (league_hand, "league_hand_arsenal_by_season.parquet"),
        (league, "league_arsenal_by_season.parquet"),
    ]
    for frame, name in profile_outputs:
        write_parquet_atomic(frame, args.output_dir / name)
    write_parquet_atomic(
        player.loc[player["feature_season"].eq(2025)],
        args.output_dir / "pitcher_arsenal_2025.parquet",
    )
    frozen_fallback = pd.concat(
        [
            team_hand.loc[team_hand["feature_season"].eq(2025)].assign(
                profile_level="team_hand"
            ),
            league_hand.loc[league_hand["feature_season"].eq(2025)].assign(
                profile_level="league_hand"
            ),
            league.loc[league["feature_season"].eq(2025)].assign(
                profile_level="league"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    write_parquet_atomic(
        frozen_fallback, args.output_dir / "fallback_arsenal_2025.parquet"
    )
    shutil.copy2(
        args.mapping, args.output_dir / "accepted_pitcher_mapping_snapshot.csv"
    )

    print("stage 4/5: reconstructing exact pitch-level auxiliary labels")
    alignments = load_valid_alignments(args.alignments)
    segments = load_segments(args.segments)
    database_path = args.output_dir / ".arsenal_alignment.sqlite3"
    connection = create_alignment_database(database_path, alignments)
    try:
        train_selected = scan_aligned_train_rows(
            args,
            connection,
            segments,
            set(alignments["train_game_key"]),
        )
        trackman_selected = scan_aligned_trackman_rows(
            args, connection, set(alignments["trackman_game_id"])
        )
        auxiliary_summary = write_auxiliary_parquet(
            connection,
            args.output_dir / "aligned_pitch_auxiliary.parquet",
            pa,
            pq,
        )
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()

    print("stage 5/5: writing manifests")
    manifest = {
        "cutoff_rule": "Trackman season < feature_season",
        "pitch_groups": PITCH_GROUPS,
        "pitch_group_to_id": PITCH_GROUP_TO_ID,
        "arsenal_feature_columns": arsenal_feature_columns(),
        "auxiliary_numeric_targets": [
            f"aux_{column}" for column in NUMERIC_COLUMNS
        ],
        "auxiliary_usage_rule": (
            "aux_pitch_group and aux physical values are training targets only; "
            "never model inputs"
        ),
        "hand_mapping": {"Right": 2, "Left": 1},
    }
    (args.output_dir / "arsenal_feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "train_path": str(args.train),
        "trackman_path": str(args.trackman),
        "mapping_path": str(args.mapping),
        "segments_path": str(args.segments),
        "alignments_path": str(args.alignments),
        "output_dir": str(args.output_dir),
        "accepted_pitcher_mapping_count": len(mapping),
        "exact_aligned_games": len(alignments),
        "selected_train_rows": train_selected,
        "selected_trackman_rows": trackman_selected,
        "player_arsenal_rows": len(player),
        "team_hand_arsenal_rows": len(team_hand),
        "league_hand_arsenal_rows": len(league_hand),
        "league_arsenal_rows": len(league),
        "frozen_2025_player_arsenal_rows": int(
            player["feature_season"].eq(2025).sum()
        ),
        **trackman_summary,
        **auxiliary_summary,
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_dir: {args.output_dir}")
    print("next: run scripts/train_pitch_arsenal_moe.py --mode cv")


if __name__ == "__main__":
    main()
