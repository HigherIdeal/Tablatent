from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/train.pkl"
DEFAULT_OUTPUT = ROOT / "outputs/game_type_experience"
DEFAULT_EDGES = [0, 1, 51, 201, 1001, 4001, np.inf]
DEFAULT_LABELS = ["P0_0", "P1_1_50", "P2_51_200", "P3_201_1000", "P4_1001_4000", "P5_4001_plus"]


def experience_bin(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="raise")
    if (numeric < 0).any():
        raise ValueError("as-of experience counts must be non-negative")
    return pd.cut(numeric, bins=edges, labels=labels, right=False, include_lowest=True)


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in frame.columns]
    return frame.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit game_type effects conditional on pitcher/batter experience.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-cell-rows", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    expected = [
        output / "cell_stats.csv",
        output / "conditional_game_type_effects.csv",
        output / "season_stability.csv",
        output / "game_type_composition.csv",
        output / "summary.md",
    ]
    if any(path.exists() for path in expected) and not args.force:
        raise FileExistsError(f"{output} already contains results; pass --force to replace them")

    columns = [
        "season", "game_type", "pitcher_id", "batter_id", "control_success",
        "asof_pitcher_n", "asof_batter_n", "asof_pitcher_success_rate",
        "asof_batter_success_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    ]
    print("stage 1/3: loading analysis columns")
    if args.input.suffix == ".parquet":
        frame = pd.read_parquet(args.input, columns=columns)
    else:
        frame = pd.read_pickle(args.input)[columns]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")

    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").fillna("<MISSING>")
    frame["pitcher_exp"] = experience_bin(frame["asof_pitcher_n"], DEFAULT_EDGES, DEFAULT_LABELS)
    frame["batter_exp"] = experience_bin(frame["asof_batter_n"], DEFAULT_EDGES, DEFAULT_LABELS)

    keys = ["season", "pitcher_exp", "batter_exp", "game_type"]
    print("stage 2/3: aggregating conditional cells and temporal stability")
    cells = flatten_columns(
        frame.groupby(keys, observed=True).agg(
            row_count=("control_success", "size"),
            pitcher_count=("pitcher_id", "nunique"),
            batter_count=("batter_id", "nunique"),
            control_success_mean=("control_success", "mean"),
            asof_pitcher_n_median=("asof_pitcher_n", "median"),
            asof_batter_n_median=("asof_batter_n", "median"),
            asof_pitcher_success_rate_mean=("asof_pitcher_success_rate", "mean"),
            asof_batter_success_rate_mean=("asof_batter_success_rate", "mean"),
            pitcher_prev1_mean=("asof_pitcher_prev1_game_success_rate", "mean"),
            pitcher_prev3_mean=("asof_pitcher_prev3_game_success_rate", "mean"),
            pitcher_prev5_mean=("asof_pitcher_prev5_game_success_rate", "mean"),
        )
    )
    cells["low_support"] = cells["row_count"] < args.min_cell_rows

    baseline_keys = ["season", "pitcher_exp", "batter_exp"]
    cells["target_sum"] = cells["control_success_mean"] * cells["row_count"]
    baseline = cells.groupby(baseline_keys, observed=True)[["target_sum", "row_count"]].sum().reset_index()
    baseline["experience_cell_rate"] = baseline["target_sum"] / baseline["row_count"]
    baseline = baseline[baseline_keys + ["experience_cell_rate"]]
    effects = cells.merge(baseline, on=["season", "pitcher_exp", "batter_exp"], validate="many_to_one")
    effects["game_type_effect"] = effects["control_success_mean"] - effects["experience_cell_rate"]
    effects["season_rate_change"] = effects.groupby(
        ["pitcher_exp", "batter_exp", "game_type"], observed=True
    )["control_success_mean"].diff()

    stable = effects.loc[~effects["low_support"]].copy()
    stability = flatten_columns(
        stable.groupby(["pitcher_exp", "batter_exp", "game_type"], observed=True).agg(
            seasons=("season", "nunique"),
            total_rows=("row_count", "sum"),
            mean_rate=("control_success_mean", "mean"),
            min_rate=("control_success_mean", "min"),
            max_rate=("control_success_mean", "max"),
            rate_std=("control_success_mean", "std"),
            mean_effect=("game_type_effect", "mean"),
            min_effect=("game_type_effect", "min"),
            max_effect=("game_type_effect", "max"),
            effect_std=("game_type_effect", "std"),
        )
    )
    stability["rate_range"] = stability["max_rate"] - stability["min_rate"]
    stability["effect_sign_stable"] = (stability["min_effect"] > 0) | (stability["max_effect"] < 0)
    stability = stability.sort_values(["effect_sign_stable", "seasons", "total_rows"], ascending=False)

    composition = flatten_columns(
        frame.groupby(["season", "game_type", "pitcher_exp", "batter_exp"], observed=True)
        .agg(row_count=("control_success", "size"), target_rate=("control_success", "mean"))
    )
    totals = composition.groupby(["season", "game_type"])["row_count"].transform("sum")
    composition["within_game_type_share"] = composition["row_count"] / totals

    output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(expected[0], index=False)
    effects.to_csv(expected[1], index=False)
    stability.to_csv(expected[2], index=False)
    composition.to_csv(expected[3], index=False)

    reliable = effects.loc[~effects["low_support"]]
    top = reliable.reindex(reliable["game_type_effect"].abs().sort_values(ascending=False).index).head(15)
    report = [
        "# Game type x experience audit",
        "",
        f"- rows: {len(frame):,}",
        f"- seasons: {frame['season'].min()}-{frame['season'].max()}",
        f"- game types: {frame['game_type'].nunique()}",
        f"- conditional cells: {len(cells):,}",
        f"- reliable cells (rows >= {args.min_cell_rows:,}): {len(reliable):,}",
        f"- stable combinations present in >= 3 seasons: {(stability['seasons'] >= 3).sum():,}",
        "",
        "## Largest conditional effects among reliable cells",
        "",
        "```text",
        top[["season", "pitcher_exp", "batter_exp", "game_type", "row_count", "control_success_mean", "experience_cell_rate", "game_type_effect"]].to_string(index=False),
        "```",
        "",
        "`game_type_effect` is the game-type rate minus the same season/pitcher-experience/batter-experience baseline.",
    ]
    expected[4].write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps({
        "input": str(args.input.resolve()),
        "experience_bins": dict(zip(DEFAULT_LABELS, ["0", "1-50", "51-200", "201-1000", "1001-4000", "4001+"])),
        "min_cell_rows": args.min_cell_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("stage 3/3: writing compact audit artifacts")
    print(f"output_dir: {output}")
    print("next: inspect summary.md and season_stability.csv")


if __name__ == "__main__":
    main()
