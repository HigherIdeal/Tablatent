#!/usr/bin/env python3
"""Build leak-safe pitcher physical profiles and augment the training table.

For a feature season S, only Trackman rows from seasons strictly earlier than S
are aggregated. The same rule produces a frozen 2025 lookup from 2019-2024
Trackman data, so inference does not require Trackman input at runtime.

The script scans the large CSV files and should be run explicitly by the user.
It never modifies raw inputs and writes a separated artifact bundle under
data/processed/physical_features by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_trackman_history_mod import TEAM_MAP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_TRACKMAN = ROOT / "data" / "raw" / "trackman_history.csv"
DEFAULT_MAPPING = (
    ROOT / "outputs" / "pitcher_mapping_audit" / "accepted_pitcher_mapping.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "physical_features"

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
TARGET_SEASONS = list(range(2019, 2026))

TRACKMAN_COLUMNS = [
    "season",
    "pitcher_trackman_id",
    "pitcher_hand",
    "pitcher_team",
    "pitch_type_group",
    *NUMERIC_COLUMNS,
]

OUTPUT_FILES = [
    "accepted_pitcher_mapping_snapshot.csv",
    "pitcher_physical_profiles_by_season.parquet",
    "team_hand_fallback_profiles_by_season.parquet",
    "league_hand_fallback_profiles_by_season.parquet",
    "league_fallback_profiles_by_season.parquet",
    "pitcher_profiles_2025.parquet",
    "fallback_profiles_2025.parquet",
    "train_with_pitcher_physical.parquet",
    "physical_feature_columns.json",
    "build_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
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
            "pyarrow is required for compact Parquet output. "
            "Install it in the active environment before running this script."
        ) from error
    return pa, pq


def normalize_id_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def normalize_hand_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.map(
        {
            "Right": 2,
            "R": 2,
            "2": 2,
            "Left": 1,
            "L": 1,
            "1": 1,
        }
    ).astype("Int64")


def normalize_pitch_group(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.where(values.isin(PITCH_GROUPS[:-1]), "other")


def load_mapping(path: Path) -> dict[str, int]:
    mapping = pd.read_csv(
        path,
        usecols=["pitcher_id", "pitcher_trackman_id", "status"],
        dtype="string",
    )
    mapping = mapping.loc[mapping["status"].str.strip() == "accepted"].copy()
    mapping["pitcher_id"] = normalize_id_series(mapping["pitcher_id"])
    mapping["pitcher_trackman_id"] = normalize_id_series(
        mapping["pitcher_trackman_id"]
    )
    if mapping["pitcher_id"].isna().any() or mapping["pitcher_trackman_id"].isna().any():
        raise ValueError("Accepted mapping contains non-numeric pitcher IDs")
    if mapping["pitcher_id"].duplicated().any():
        raise ValueError("Accepted mapping contains duplicate train pitcher IDs")
    if mapping["pitcher_trackman_id"].duplicated().any():
        raise ValueError("Accepted mapping contains duplicate Trackman pitcher IDs")
    return {
        str(trackman_id): int(train_id)
        for train_id, trackman_id in zip(
            mapping["pitcher_id"], mapping["pitcher_trackman_id"]
        )
    }


def empty_vector() -> np.ndarray:
    return np.zeros(1 + 3 * len(NUMERIC_COLUMNS), dtype=np.float64)


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


def add_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left += right
    return left


def numeric_mean(vector: np.ndarray, column_index: int) -> float:
    offset = 1 + 3 * column_index
    count = vector[offset]
    return vector[offset + 1] / count if count else math.nan


def numeric_std(vector: np.ndarray, column_index: int) -> float:
    offset = 1 + 3 * column_index
    count = vector[offset]
    if count <= 1:
        return math.nan
    mean = vector[offset + 1] / count
    variance = max(vector[offset + 2] / count - mean * mean, 0.0)
    return math.sqrt(variance)


def feature_columns() -> list[str]:
    columns = ["tm_pitch_count", "tm_seasons_observed"]
    for column in NUMERIC_COLUMNS:
        columns.extend([f"tm_{column}_mean", f"tm_{column}_std"])
    for group in PITCH_GROUPS:
        columns.extend([f"tm_{group}_count", f"tm_{group}_rate"])
        for column in NUMERIC_COLUMNS:
            columns.append(f"tm_{group}_{column}_mean")
    columns.append("tm_prevseason_pitch_count")
    for column in NUMERIC_COLUMNS:
        columns.extend(
            [
                f"tm_prevseason_{column}_mean",
                f"tm_prevseason_{column}_delta",
            ]
        )
    return columns


def profile_values(
    cumulative: dict[str, np.ndarray],
    previous_season: dict[str, np.ndarray],
    seasons_observed: int,
) -> dict[str, float]:
    overall = empty_vector()
    for vector in cumulative.values():
        add_vector(overall, vector)
    previous_overall = empty_vector()
    for vector in previous_season.values():
        add_vector(previous_overall, vector)

    result: dict[str, float] = {
        "tm_pitch_count": overall[0],
        "tm_seasons_observed": float(seasons_observed),
    }
    for index, column in enumerate(NUMERIC_COLUMNS):
        career_mean = numeric_mean(overall, index)
        result[f"tm_{column}_mean"] = career_mean
        result[f"tm_{column}_std"] = numeric_std(overall, index)

    for group in PITCH_GROUPS:
        vector = cumulative.get(group, empty_vector())
        result[f"tm_{group}_count"] = vector[0]
        result[f"tm_{group}_rate"] = (
            vector[0] / overall[0] if overall[0] else math.nan
        )
        for index, column in enumerate(NUMERIC_COLUMNS):
            result[f"tm_{group}_{column}_mean"] = numeric_mean(vector, index)

    result["tm_prevseason_pitch_count"] = previous_overall[0]
    for index, column in enumerate(NUMERIC_COLUMNS):
        previous_mean = numeric_mean(previous_overall, index)
        career_mean = result[f"tm_{column}_mean"]
        result[f"tm_prevseason_{column}_mean"] = previous_mean
        result[f"tm_prevseason_{column}_delta"] = (
            previous_mean - career_mean
            if not math.isnan(previous_mean) and not math.isnan(career_mean)
            else math.nan
        )
    return result


def build_profiles(
    store: dict[tuple[Any, ...], np.ndarray],
    entity_columns: list[str],
) -> pd.DataFrame:
    by_entity: defaultdict[
        tuple[Any, ...], defaultdict[int, dict[str, np.ndarray]]
    ] = defaultdict(lambda: defaultdict(dict))
    entity_width = len(entity_columns)
    for key, vector in store.items():
        entity = tuple(key[:entity_width])
        source_season = int(key[entity_width])
        group = str(key[entity_width + 1])
        by_entity[entity][source_season][group] = vector

    rows: list[dict[str, Any]] = []
    for entity, yearly in by_entity.items():
        cumulative: dict[str, np.ndarray] = {}
        observed_seasons: set[int] = set()
        for target_season in TARGET_SEASONS:
            source_season = target_season - 1
            previous = yearly.get(source_season, {})
            if previous:
                observed_seasons.add(source_season)
                for group, vector in previous.items():
                    if group not in cumulative:
                        cumulative[group] = empty_vector()
                    add_vector(cumulative[group], vector)
            if not cumulative:
                continue
            row = {
                column: value for column, value in zip(entity_columns, entity)
            }
            row["feature_season"] = target_season
            row.update(
                profile_values(cumulative, previous, len(observed_seasons))
            )
            rows.append(row)

    ordered_columns = [*entity_columns, "feature_season", *feature_columns()]
    return pd.DataFrame(rows, columns=ordered_columns)


def scan_trackman(
    args: argparse.Namespace, mapping: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    player_store: dict[tuple[Any, ...], np.ndarray] = {}
    team_hand_store: dict[tuple[Any, ...], np.ndarray] = {}
    league_hand_store: dict[tuple[Any, ...], np.ndarray] = {}
    league_store: dict[tuple[Any, ...], np.ndarray] = {}
    total_rows = 0
    mapped_rows = 0
    unmapped_team_counts: Counter[str] = Counter()
    hand_counts: Counter[str] = Counter()

    for chunk in pd.read_csv(
        args.trackman,
        usecols=TRACKMAN_COLUMNS,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk["source_season"] = pd.to_numeric(chunk["season"], errors="coerce").astype(
            "Int64"
        )
        trackman_ids = normalize_id_series(chunk["pitcher_trackman_id"])
        chunk["train_pitcher_id"] = trackman_ids.map(mapping).astype("Int64")
        mapped_rows += int(chunk["train_pitcher_id"].notna().sum())
        chunk["hand"] = normalize_hand_series(chunk["pitcher_hand"])
        chunk["pitch_group"] = normalize_pitch_group(chunk["pitch_type_group"])
        team_strings = chunk["pitcher_team"].astype("string").str.strip()
        chunk["team_id"] = team_strings.map(TEAM_MAP).astype("Int64")

        for value, count in team_strings.loc[chunk["team_id"].isna()].value_counts().items():
            unmapped_team_counts[str(value)] += int(count)
        for value, count in chunk["pitcher_hand"].astype("string").value_counts().items():
            hand_counts[str(value)] += int(count)
        for column in NUMERIC_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        valid_common = chunk.loc[chunk["source_season"].notna()].copy()
        update_store(
            valid_common.loc[valid_common["train_pitcher_id"].notna()],
            ["train_pitcher_id", "source_season", "pitch_group"],
            player_store,
        )
        update_store(
            valid_common.loc[
                valid_common["team_id"].notna() & valid_common["hand"].notna()
            ],
            ["team_id", "hand", "source_season", "pitch_group"],
            team_hand_store,
        )
        update_store(
            valid_common.loc[valid_common["hand"].notna()],
            ["hand", "source_season", "pitch_group"],
            league_hand_store,
        )
        valid_common["league_key"] = "league"
        update_store(
            valid_common,
            ["league_key", "source_season", "pitch_group"],
            league_store,
        )
        print(f"trackman_rows_scanned: {total_rows:,}")

    player = build_profiles(player_store, ["pitcher_id"])
    team_hand = build_profiles(team_hand_store, ["pitcher_team_id", "pitcher_hand"])
    league_hand = build_profiles(league_hand_store, ["pitcher_hand"])
    league = build_profiles(league_store, ["league_key"]).drop(
        columns=["league_key"]
    )
    diagnostics = {
        "trackman_rows_scanned": total_rows,
        "trackman_rows_with_accepted_pitcher_mapping": mapped_rows,
        "trackman_row_mapping_rate": mapped_rows / total_rows if total_rows else 0.0,
        "unmapped_team_occurrences": dict(unmapped_team_counts),
        "raw_pitcher_hand_counts": dict(hand_counts),
    }
    return player, team_hand, league_hand, league, diagnostics


def indexed(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.duplicated(keys).any():
        raise ValueError(f"Profile table has duplicate keys: {keys}")
    return frame.set_index(keys)[feature_columns()].sort_index()


def lookup(
    table: pd.DataFrame, key_arrays: list[pd.Series], key_names: list[str]
) -> pd.DataFrame:
    if len(key_names) == 1:
        index = pd.Index(key_arrays[0].to_numpy(), name=key_names[0])
    else:
        index = pd.MultiIndex.from_arrays(
            [array.to_numpy() for array in key_arrays], names=key_names
        )
    return table.reindex(index).reset_index(drop=True)


def augment_train(
    args: argparse.Namespace,
    player_profiles: pd.DataFrame,
    team_hand_profiles: pd.DataFrame,
    league_hand_profiles: pd.DataFrame,
    league_profiles: pd.DataFrame,
    pa: Any,
    pq: Any,
) -> dict[str, Any]:
    features = feature_columns()
    player_index = indexed(player_profiles, ["pitcher_id", "feature_season"])
    team_hand_index = indexed(
        team_hand_profiles,
        ["pitcher_team_id", "pitcher_hand", "feature_season"],
    )
    league_hand_index = indexed(
        league_hand_profiles, ["pitcher_hand", "feature_season"]
    )
    league_index = indexed(league_profiles, ["feature_season"])

    output = args.output_dir / "train_with_pitcher_physical.parquet"
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = None
    total_rows = 0
    source_counts: Counter[str] = Counter()

    try:
        for chunk in pd.read_csv(args.train, chunksize=args.chunksize, low_memory=False):
            pitcher_ids = pd.to_numeric(chunk["pitcher_id"], errors="coerce").astype(
                "Int64"
            )
            team_ids = pd.to_numeric(chunk["pitcher_team_id"], errors="coerce").astype(
                "Int64"
            )
            hands = pd.to_numeric(chunk["pitcher_hand"], errors="coerce").astype("Int64")
            seasons = pd.to_numeric(chunk["season"], errors="coerce").astype("Int64")

            player_values = lookup(
                player_index,
                [pitcher_ids, seasons],
                ["pitcher_id", "feature_season"],
            )
            team_values = lookup(
                team_hand_index,
                [team_ids, hands, seasons],
                ["pitcher_team_id", "pitcher_hand", "feature_season"],
            )
            hand_values = lookup(
                league_hand_index,
                [hands, seasons],
                ["pitcher_hand", "feature_season"],
            )
            league_values = lookup(
                league_index, [seasons], ["feature_season"]
            )

            player_available = player_values["tm_pitch_count"].notna()
            team_available = team_values["tm_pitch_count"].notna()
            hand_available = hand_values["tm_pitch_count"].notna()
            source = pd.Series("missing", index=chunk.index, dtype="string")
            source.loc[hand_available.to_numpy()] = "league_hand"
            source.loc[team_available.to_numpy()] = "team_hand"
            source.loc[player_available.to_numpy()] = "player"
            league_available = league_values["tm_pitch_count"].notna()
            source.loc[
                (source == "missing").to_numpy() & league_available.to_numpy()
            ] = "league"

            resolved = player_values.combine_first(team_values)
            resolved = resolved.combine_first(hand_values)
            resolved = resolved.combine_first(league_values)
            resolved.index = chunk.index
            for column in features:
                chunk[column] = resolved[column].astype("float64")
            chunk["tm_player_profile_available"] = player_available.to_numpy().astype(
                "int8"
            )
            chunk["tm_profile_source"] = source.to_numpy()

            source_counts.update(str(value) for value in source)
            total_rows += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            else:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            print(f"train_rows_augmented: {total_rows:,}")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("train.csv contained no rows")
    os.replace(temporary, output)
    return {
        "train_rows_augmented": total_rows,
        "profile_source_counts": dict(source_counts),
        "physical_feature_count": len(features) + 2,
    }


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.train = args.train.expanduser().resolve()
    args.trackman = args.trackman.expanduser().resolve()
    args.mapping = args.mapping.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.train, args.trackman, args.mapping):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    pa, pq = require_pyarrow()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in OUTPUT_FILES if (args.output_dir / name).exists()]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use --force:\n{joined}")

    print("stage 1/4: loading accepted pitcher mapping")
    mapping = load_mapping(args.mapping)
    print(f"accepted_pitcher_mappings: {len(mapping):,}")

    print("stage 2/4: scanning Trackman and building season-lag profiles")
    (
        player_profiles,
        team_hand_profiles,
        league_hand_profiles,
        league_profiles,
        trackman_summary,
    ) = scan_trackman(args, mapping)

    print("stage 3/4: writing profile and frozen 2025 inference artifacts")
    profile_outputs = [
        (player_profiles, "pitcher_physical_profiles_by_season.parquet"),
        (team_hand_profiles, "team_hand_fallback_profiles_by_season.parquet"),
        (league_hand_profiles, "league_hand_fallback_profiles_by_season.parquet"),
        (league_profiles, "league_fallback_profiles_by_season.parquet"),
    ]
    for frame, name in profile_outputs:
        write_parquet_atomic(frame, args.output_dir / name)
    write_parquet_atomic(
        player_profiles.loc[player_profiles["feature_season"] == 2025],
        args.output_dir / "pitcher_profiles_2025.parquet",
    )
    frozen_fallback = pd.concat(
        [
            team_hand_profiles.loc[team_hand_profiles["feature_season"] == 2025].assign(
                profile_level="team_hand"
            ),
            league_hand_profiles.loc[
                league_hand_profiles["feature_season"] == 2025
            ].assign(profile_level="league_hand"),
            league_profiles.loc[league_profiles["feature_season"] == 2025].assign(
                profile_level="league"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    write_parquet_atomic(
        frozen_fallback, args.output_dir / "fallback_profiles_2025.parquet"
    )
    shutil.copy2(
        args.mapping,
        args.output_dir / "accepted_pitcher_mapping_snapshot.csv",
    )

    print("stage 4/4: augmenting train with player and fallback physical profiles")
    train_summary = augment_train(
        args,
        player_profiles,
        team_hand_profiles,
        league_hand_profiles,
        league_profiles,
        pa,
        pq,
    )

    feature_manifest = {
        "feature_columns": feature_columns(),
        "metadata_columns": [
            "tm_player_profile_available",
            "tm_profile_source",
        ],
        "cutoff_rule": "Trackman season < feature_season",
        "hand_mapping": {"Right": 2, "Left": 1},
    }
    (args.output_dir / "physical_feature_columns.json").write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "train_path": str(args.train),
        "trackman_path": str(args.trackman),
        "mapping_path": str(args.mapping),
        "output_dir": str(args.output_dir),
        "accepted_pitcher_mapping_count": len(mapping),
        "cutoff_rule": "Trackman season < feature_season",
        "target_feature_seasons": TARGET_SEASONS,
        "player_profile_rows": len(player_profiles),
        "team_hand_profile_rows": len(team_hand_profiles),
        "league_hand_profile_rows": len(league_hand_profiles),
        "league_profile_rows": len(league_profiles),
        "inference_2025_player_profiles": int(
            (player_profiles["feature_season"] == 2025).sum()
        ),
        **trackman_summary,
        **train_summary,
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_dir: {args.output_dir}")
    print("runtime inference artifact: pitcher_profiles_2025.parquet")
    print("augmented training table: train_with_pitcher_physical.parquet")


if __name__ == "__main__":
    main()
