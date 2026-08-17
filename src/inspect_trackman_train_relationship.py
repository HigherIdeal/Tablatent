#!/usr/bin/env python3
"""Produce an intuitive, non-modeling audit of train/Trackman relationships.

The report describes possible links; it never asserts that similarly encoded
anonymous IDs refer to the same entity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ID_HINTS = ("id", "pitcher", "batter", "team", "player")
TIME_HINTS = ("date", "year", "season", "month", "day", "game")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--train", type=Path, help="Explicit train.csv path")
    parser.add_argument("--trackman", type=Path, help="Explicit trackman_history.csv path")
    parser.add_argument(
        "--rows",
        type=int,
        default=200_000,
        help="Rows read from each table for the intuitive audit; 0 reads all rows",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "trackman_relationship_audit",
    )
    return parser.parse_args()


def find_one(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename), key=lambda p: (len(p.parts), p.as_posix()))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    if len(matches) > 1:
        print(f"[warning] multiple {filename} files found; using {matches[0]}")
    return matches[0]


def read_table(path: Path, rows: int) -> pd.DataFrame:
    print(f"[read] {path} ({'all rows' if rows == 0 else f'first {rows:,} rows'})")
    return pd.read_csv(path, nrows=None if rows == 0 else rows, low_memory=False)


def scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def column_profile(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        examples = [scalar(value) for value in series.dropna().drop_duplicates().head(5)]
        rows.append(
            {
                "column": name,
                "dtype": str(series.dtype),
                "non_null": non_null,
                "null_rate": float(1 - non_null / max(len(frame), 1)),
                "nunique_sample": int(series.nunique(dropna=True)),
                "examples": examples,
            }
        )
    return rows


def comparable_values(series: pd.Series) -> set[str]:
    values = series.dropna().astype("string").str.strip()
    return set(values[values.ne("")].unique().tolist())


def overlap(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    left_values = comparable_values(left)
    right_values = comparable_values(right)
    shared = left_values & right_values
    union = left_values | right_values
    return {
        "left_unique": len(left_values),
        "right_unique": len(right_values),
        "shared_unique": len(shared),
        "left_coverage": len(shared) / max(len(left_values), 1),
        "right_coverage": len(shared) / max(len(right_values), 1),
        "jaccard": len(shared) / max(len(union), 1),
        "shared_examples": sorted(shared)[:10],
    }


def candidate_pairs(train: pd.DataFrame, trackman: pd.DataFrame) -> list[dict[str, Any]]:
    train_candidates = [
        name for name in train.columns if any(hint in name.lower() for hint in ID_HINTS)
    ]
    track_candidates = [
        name for name in trackman.columns if any(hint in name.lower() for hint in ID_HINTS)
    ]
    rows = []
    for left_name in train_candidates:
        for right_name in track_candidates:
            stats = overlap(train[left_name], trackman[right_name])
            if stats["shared_unique"]:
                rows.append(
                    {
                        "train_column": left_name,
                        "trackman_column": right_name,
                        **stats,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (row["jaccard"], row["shared_unique"]),
        reverse=True,
    )


def common_column_audit(
    train: pd.DataFrame, trackman: pd.DataFrame
) -> list[dict[str, Any]]:
    common = sorted(set(train.columns) & set(trackman.columns))
    return [
        {"column": name, **overlap(train[name], trackman[name])}
        for name in common
    ]


def hinted_columns(frame: pd.DataFrame, hints: tuple[str, ...]) -> list[str]:
    return [name for name in frame.columns if any(hint in name.lower() for hint in hints)]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No candidates found._\n"
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            cells.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body]) + "\n"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = args.train.resolve() if args.train else find_one(data_dir, "train.csv")
    trackman_path = (
        args.trackman.resolve()
        if args.trackman
        else find_one(data_dir, "trackman_history.csv")
    )
    train = read_table(train_path, args.rows)
    trackman = read_table(trackman_path, args.rows)

    same_names = common_column_audit(train, trackman)
    pairs = candidate_pairs(train, trackman)
    result = {
        "warning": (
            "Value overlap is only a linkage hypothesis. Anonymous ID equality does not "
            "prove that the two tables use the same identifier system."
        ),
        "sample_rows_requested": args.rows,
        "train": {
            "path": str(train_path),
            "rows_loaded": len(train),
            "columns": len(train.columns),
            "id_like_columns": hinted_columns(train, ID_HINTS),
            "time_like_columns": hinted_columns(train, TIME_HINTS),
            "profile": column_profile(train),
        },
        "trackman": {
            "path": str(trackman_path),
            "rows_loaded": len(trackman),
            "columns": len(trackman.columns),
            "id_like_columns": hinted_columns(trackman, ID_HINTS),
            "time_like_columns": hinted_columns(trackman, TIME_HINTS),
            "profile": column_profile(trackman),
        },
        "same_name_column_overlap": same_names,
        "id_like_candidate_pairs": pairs,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relationship_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = [
        "# Train and Trackman: intuitive relationship audit\n",
        "값의 겹침은 연결 가설일 뿐이며, 익명 ID가 같다는 사실만으로 동일 선수임을 의미하지 않는다.\n",
        f"- train sample: {len(train):,} rows x {len(train.columns)} columns",
        f"- Trackman sample: {len(trackman):,} rows x {len(trackman.columns)} columns\n",
        "## Same-name columns\n",
        markdown_table(
            same_names,
            ["column", "left_unique", "right_unique", "shared_unique", "jaccard"],
        ),
        "## ID-like candidate pairs\n",
        markdown_table(
            pairs[:30],
            [
                "train_column",
                "trackman_column",
                "left_unique",
                "right_unique",
                "shared_unique",
                "left_coverage",
                "right_coverage",
                "jaccard",
            ],
        ),
        "## Time-like columns\n",
        f"- train: {', '.join(result['train']['time_like_columns']) or '(none)'}",
        f"- Trackman: {', '.join(result['trackman']['time_like_columns']) or '(none)'}\n",
    ]
    (output_dir / "relationship_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"[write] {output_dir / 'relationship_audit.md'}")
    print(f"[write] {output_dir / 'relationship_audit.json'}")


if __name__ == "__main__":
    main()
