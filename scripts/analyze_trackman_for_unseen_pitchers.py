from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_frame
from src.utils import load_config


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def detect_columns(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    out = []
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in tokens):
            out.append(column)
    return out


def column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(frame)
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if n else 0.0
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()),
            "nunique": int(series.nunique(dropna=True)),
            "numeric_fraction": numeric_fraction,
        }
        if numeric_fraction >= 0.95:
            valid = numeric.dropna()
            row.update(
                {
                    "mean": float(valid.mean()) if len(valid) else np.nan,
                    "std": float(valid.std()) if len(valid) else np.nan,
                    "min": float(valid.min()) if len(valid) else np.nan,
                    "max": float(valid.max()) if len(valid) else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit trackman_history.csv before building unseen-pitcher features. "
            "Reports schema, candidate identity/time columns, train overlap, season coverage, "
            "and whether Trackman can be used only as training-time privileged information."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trackman", default="data/raw/trackman_history.csv")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    train = load_frame(config)
    trackman_path = ROOT / args.trackman
    if not trackman_path.exists():
        raise FileNotFoundError(
            f"{trackman_path} not found. Pass --trackman with the actual trackman_history.csv path."
        )

    tm = pd.read_csv(trackman_path, low_memory=False)
    columns = tm.columns.tolist()
    train_columns = train.columns.tolist()

    pitcher_candidates = detect_columns(columns, ("pitcher", "player", "thrower"))
    season_candidates = detect_columns(columns, ("season", "year"))
    time_candidates = detect_columns(
        columns,
        ("date", "time", "game", "month", "day", "pitch_no", "pitchno", "pa", "inning"),
    )
    physical_candidates = detect_columns(
        columns,
        (
            "speed",
            "velo",
            "spin",
            "break",
            "movement",
            "release",
            "extension",
            "height",
            "side",
            "angle",
            "pitch_type",
            "pitchtype",
        ),
    )
    exact_overlap = sorted(set(columns) & set(train_columns))

    output_dir = Path(config["paths"]["output_dir"]) / "trackman_unseen_pitcher_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = column_profile(tm)
    profile.to_csv(output_dir / "column_profile.csv", index=False)

    report: dict = {
        "trackman_path": str(trackman_path),
        "trackman_rows": int(len(tm)),
        "trackman_columns": int(tm.shape[1]),
        "column_names": columns,
        "exact_columns_shared_with_train": exact_overlap,
        "pitcher_id_candidates": pitcher_candidates,
        "season_candidates": season_candidates,
        "time_order_candidates": time_candidates,
        "physical_feature_candidates": physical_candidates,
        "train_rows": int(len(train)),
        "train_pitcher_count": int(train["pitcher_id"].nunique(dropna=True)) if "pitcher_id" in train else None,
    }

    print(
        f"[Trackman Audit] rows={len(tm):,}, columns={tm.shape[1]}, file={trackman_path}"
    )
    print(f"  exact train-column overlap: {exact_overlap}")
    print(f"  pitcher candidates: {pitcher_candidates}")
    print(f"  season candidates: {season_candidates}")
    print(f"  time/order candidates: {time_candidates}")
    print(f"  physical candidates: {physical_candidates}")

    if "pitcher_id" in tm.columns and "pitcher_id" in train.columns:
        tm_ids = set(normalize(tm["pitcher_id"]).unique())
        train_ids = set(normalize(train["pitcher_id"]).unique())
        common = tm_ids & train_ids
        report.update(
            {
                "trackman_pitcher_count": len(tm_ids),
                "common_pitcher_count": len(common),
                "train_pitcher_coverage_fraction": float(len(common) / max(1, len(train_ids))),
                "trackman_pitcher_overlap_fraction": float(len(common) / max(1, len(tm_ids))),
            }
        )
        print(
            f"  pitcher overlap: common={len(common):,}, "
            f"train coverage={len(common) / max(1, len(train_ids)):.3f}, "
            f"Trackman overlap={len(common) / max(1, len(tm_ids)):.3f}"
        )

        pitcher_counts = tm.groupby("pitcher_id", dropna=False).size().rename("trackman_rows").reset_index()
        pitcher_counts.to_csv(output_dir / "pitcher_trackman_counts.csv", index=False)
    else:
        print("  exact pitcher_id is not shared; inspect candidate identity columns before any join.")

    season_col = None
    if "season" in tm.columns:
        season_col = "season"
    elif season_candidates:
        season_col = season_candidates[0]

    if season_col is not None:
        season_numeric = pd.to_numeric(tm[season_col], errors="coerce")
        season_summary = (
            pd.DataFrame({"season": season_numeric})
            .dropna()
            .assign(season=lambda x: x["season"].astype(int))
            .groupby("season", as_index=False)
            .size()
            .rename(columns={"size": "rows"})
        )
        season_summary.to_csv(output_dir / "season_coverage.csv", index=False)
        report["season_values"] = season_summary["season"].tolist()
        print("\n[Season coverage]")
        print(season_summary.to_string(index=False))

    if "pitcher_id" in tm.columns and season_col is not None:
        tmp = tm[["pitcher_id", season_col]].copy()
        tmp[season_col] = pd.to_numeric(tmp[season_col], errors="coerce")
        tmp = tmp.dropna(subset=[season_col])
        tmp[season_col] = tmp[season_col].astype(int)
        coverage = (
            tmp.groupby(season_col)
            .agg(rows=("pitcher_id", "size"), pitchers=("pitcher_id", "nunique"))
            .reset_index()
            .rename(columns={season_col: "season"})
        )
        coverage.to_csv(output_dir / "season_pitcher_coverage.csv", index=False)

    with (output_dir / "audit.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[Next decision]")
    print("  1) If pitcher_id/time alignment is reliable, Trackman can build leakage-safe TRAINING-TIME teacher features.")
    print("  2) Under the unseen-test-pitcher assumption, do not rely on pitcher_id -> Trackman lookup at inference.")
    print("  3) Prefer teacher/distillation or population-level physical-state relations that can be reconstructed from official as-of/context inputs.")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
