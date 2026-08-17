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


DEFAULT_CUTS = "0,10,50,200,1000,4000"
MISSING_BAND = "<MISSING>"


def parse_cuts(text: str) -> list[int]:
    cuts = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not cuts:
        raise ValueError("Experience cuts must not be empty.")
    if cuts[0] != 0:
        raise ValueError("The first experience cut must be 0 so exact cold-start rows form their own band.")
    if any(value < 0 for value in cuts):
        raise ValueError(f"Experience cuts must be non-negative: {cuts}")
    if cuts != sorted(set(cuts)):
        raise ValueError(f"Experience cuts must be strictly increasing and unique: {cuts}")
    return cuts


def make_band_labels(cuts: list[int], prefix: str) -> list[str]:
    labels = [f"{prefix}00_n0"]
    lower = 1
    for index, upper in enumerate(cuts[1:], start=1):
        labels.append(f"{prefix}{index:02d}_n{lower}_{upper}")
        lower = upper + 1
    labels.append(f"{prefix}{len(cuts):02d}_n{lower}_plus")
    return labels


def assign_experience_band(series: pd.Series, cuts: list[int], prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    negative = numeric < 0
    if negative.any():
        sample = numeric.loc[negative].head(5).tolist()
        raise ValueError(f"Negative experience counts found in {series.name}: {sample}")

    labels = make_band_labels(cuts, prefix)
    result = pd.Series(MISSING_BAND, index=series.index, dtype="string")

    result.loc[numeric == 0] = labels[0]
    lower = 1
    for label, upper in zip(labels[1:-1], cuts[1:]):
        mask = numeric.between(lower, upper, inclusive="both")
        result.loc[mask] = label
        lower = upper + 1
    result.loc[numeric >= lower] = labels[-1]

    ordered = labels + [MISSING_BAND]
    return result.astype(pd.CategoricalDtype(categories=ordered, ordered=True))


def marginal_summary(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    band_col: str,
    n_col: str,
    id_col: str,
    entity: str,
) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "rows": (target_col, "size"),
        "success_rate": (target_col, "mean"),
        "n_mean": (n_col, "mean"),
        "n_median": (n_col, "median"),
        "n_min": (n_col, "min"),
        "n_max": (n_col, "max"),
    }
    if id_col in frame.columns:
        aggregations["unique_players"] = (id_col, "nunique")

    out = (
        frame.groupby([season_col, band_col], observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    season_rows = frame.groupby(season_col).size().rename("season_rows")
    out = out.merge(season_rows, on=season_col, how="left")
    out["row_share"] = out["rows"] / out["season_rows"]
    out.insert(0, "entity", entity)
    return out.sort_values([season_col, band_col]).reset_index(drop=True)


def axis_game_type_summary(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    band_col: str,
    game_type_col: str,
) -> pd.DataFrame:
    out = (
        frame.groupby([season_col, band_col, game_type_col], observed=True)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
    )
    baseline = (
        frame.groupby([season_col, band_col], observed=True)
        .agg(band_rows=(target_col, "size"), band_baseline=(target_col, "mean"))
        .reset_index()
    )
    out = out.merge(baseline, on=[season_col, band_col], how="left")
    out["share_within_band"] = out["rows"] / out["band_rows"]
    out["game_type_effect"] = out["success_rate"] - out["band_baseline"]
    return out.sort_values([season_col, band_col, game_type_col]).reset_index(drop=True)


def build_joint_summary(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    pitcher_band_col: str,
    batter_band_col: str,
    pitcher_n_col: str,
    batter_n_col: str,
) -> pd.DataFrame:
    out = (
        frame.groupby([season_col, pitcher_band_col, batter_band_col], observed=True)
        .agg(
            rows=(target_col, "size"),
            success_rate=(target_col, "mean"),
            pitcher_n_mean=(pitcher_n_col, "mean"),
            pitcher_n_median=(pitcher_n_col, "median"),
            batter_n_mean=(batter_n_col, "mean"),
            batter_n_median=(batter_n_col, "median"),
        )
        .reset_index()
    )
    season_rows = frame.groupby(season_col).size().rename("season_rows")
    out = out.merge(season_rows, on=season_col, how="left")
    out["cluster_share"] = out["rows"] / out["season_rows"]
    return out.sort_values([season_col, pitcher_band_col, batter_band_col]).reset_index(drop=True)


def build_joint_game_type_summary(
    frame: pd.DataFrame,
    joint: pd.DataFrame,
    season_col: str,
    target_col: str,
    pitcher_band_col: str,
    batter_band_col: str,
    game_type_col: str,
    min_cell_rows: int,
) -> pd.DataFrame:
    keys = [season_col, pitcher_band_col, batter_band_col]
    out = (
        frame.groupby(keys + [game_type_col], observed=True)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
    )
    baseline = joint[keys + ["rows", "success_rate", "cluster_share"]].rename(
        columns={"rows": "joint_rows", "success_rate": "joint_baseline"}
    )
    out = out.merge(baseline, on=keys, how="left")
    out["share_within_joint"] = out["rows"] / out["joint_rows"]
    out["game_type_effect"] = out["success_rate"] - out["joint_baseline"]
    out["supported"] = out["rows"] >= min_cell_rows
    return out.sort_values(keys + [game_type_col]).reset_index(drop=True)


def build_composition_decomposition(
    frame: pd.DataFrame,
    joint_game_type: pd.DataFrame,
    season_col: str,
    target_col: str,
    pitcher_band_col: str,
    batter_band_col: str,
    game_type_col: str,
    min_cell_rows: int,
) -> pd.DataFrame:
    season_summary = (
        frame.groupby(season_col)
        .agg(season_rows=(target_col, "size"), season_rate=(target_col, "mean"))
        .reset_index()
    )
    raw = (
        frame.groupby([season_col, game_type_col], observed=True)
        .agg(raw_rows=(target_col, "size"), raw_rate=(target_col, "mean"))
        .reset_index()
    )
    raw = raw.merge(season_summary, on=season_col, how="left")
    raw["game_type_share"] = raw["raw_rows"] / raw["season_rows"]
    raw["raw_effect_vs_season"] = raw["raw_rate"] - raw["season_rate"]

    rows: list[dict] = []
    for (season, game_type), group in joint_game_type.groupby(
        [season_col, game_type_col], observed=True, sort=True
    ):
        supported = group.loc[group["rows"] >= min_cell_rows].copy()
        coverage = float(supported["cluster_share"].sum())
        if supported.empty or coverage <= 0.0:
            standardized_rate = np.nan
            standardized_effect = np.nan
            supported_clusters = 0
        else:
            weights = supported["cluster_share"].to_numpy(dtype=np.float64)
            rates = supported["success_rate"].to_numpy(dtype=np.float64)
            standardized_rate = float(np.sum(weights * rates) / np.sum(weights))
            supported_clusters = int(len(supported))
            season_rate = float(
                season_summary.loc[season_summary[season_col] == season, "season_rate"].iloc[0]
            )
            standardized_effect = standardized_rate - season_rate

        rows.append(
            {
                season_col: int(season),
                game_type_col: game_type,
                "standardized_rate": standardized_rate,
                "standardized_effect_vs_season": standardized_effect,
                "support_coverage": coverage,
                "supported_joint_clusters": supported_clusters,
            }
        )

    standardized = pd.DataFrame(rows)
    out = raw.merge(standardized, on=[season_col, game_type_col], how="left")
    out["composition_shift"] = out["raw_rate"] - out["standardized_rate"]
    out["effect_change_after_standardization"] = (
        out["raw_effect_vs_season"] - out["standardized_effect_vs_season"]
    )
    return out.sort_values([season_col, game_type_col]).reset_index(drop=True)


def build_effect_heterogeneity(
    joint_game_type: pd.DataFrame,
    season_col: str,
    game_type_col: str,
    min_cell_rows: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (season, game_type), group in joint_game_type.groupby(
        [season_col, game_type_col], observed=True, sort=True
    ):
        supported = group.loc[group["rows"] >= min_cell_rows].copy()
        if supported.empty:
            continue
        weights = supported["rows"].to_numpy(dtype=np.float64)
        effects = supported["game_type_effect"].to_numpy(dtype=np.float64)
        weight_sum = float(weights.sum())
        weighted_mean = float(np.sum(weights * effects) / weight_sum)
        weighted_var = float(np.sum(weights * (effects - weighted_mean) ** 2) / weight_sum)
        rows.append(
            {
                season_col: int(season),
                game_type_col: game_type,
                "supported_joint_clusters": int(len(supported)),
                "supported_rows": int(weight_sum),
                "weighted_mean_effect": weighted_mean,
                "weighted_std_effect": float(np.sqrt(max(0.0, weighted_var))),
                "min_effect": float(effects.min()),
                "max_effect": float(effects.max()),
                "effect_range": float(effects.max() - effects.min()),
            }
        )
    return pd.DataFrame(rows).sort_values([season_col, game_type_col]).reset_index(drop=True)


def build_yoy_joint_game_type(
    joint_game_type: pd.DataFrame,
    season_col: str,
    pitcher_band_col: str,
    batter_band_col: str,
    game_type_col: str,
) -> pd.DataFrame:
    keys = [pitcher_band_col, batter_band_col, game_type_col]
    out = joint_game_type.copy().sort_values(keys + [season_col]).reset_index(drop=True)
    grouped = out.groupby(keys, observed=True, sort=False)
    out["previous_season"] = grouped[season_col].shift(1)
    out["previous_success_rate"] = grouped["success_rate"].shift(1)
    out["previous_game_type_effect"] = grouped["game_type_effect"].shift(1)
    out["season_gap"] = out[season_col] - out["previous_season"]
    out["delta_success_rate"] = out["success_rate"] - out["previous_success_rate"]
    out["delta_game_type_effect"] = out["game_type_effect"] - out["previous_game_type_effect"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze game_type conditional effects after splitting each pitch by the amount of "
            "pitcher/batter history available at that exact as-of moment."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pitcher-cuts", default=DEFAULT_CUTS)
    parser.add_argument("--batter-cuts", default=DEFAULT_CUTS)
    parser.add_argument("--min-cell-rows", type=int, default=500)
    parser.add_argument("--output-subdir", default="experience_game_type")
    args = parser.parse_args()

    if args.min_cell_rows < 1:
        raise ValueError("--min-cell-rows must be >= 1")

    config = load_config(ROOT / args.config)
    frame = load_frame(config).copy()

    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    pitcher_n_col = "asof_pitcher_n"
    batter_n_col = "asof_batter_n"
    pitcher_band_col = "pitcher_experience_band"
    batter_band_col = "batter_experience_band"
    game_type_col = "game_type"

    required = {
        season_col,
        target_col,
        pitcher_n_col,
        batter_n_col,
        game_type_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(np.float64)
    frame[pitcher_n_col] = pd.to_numeric(frame[pitcher_n_col], errors="coerce")
    frame[batter_n_col] = pd.to_numeric(frame[batter_n_col], errors="coerce")
    frame[game_type_col] = frame[game_type_col].astype("string").fillna("<MISSING>").astype(str)

    pitcher_cuts = parse_cuts(args.pitcher_cuts)
    batter_cuts = parse_cuts(args.batter_cuts)
    frame[pitcher_band_col] = assign_experience_band(frame[pitcher_n_col], pitcher_cuts, "P")
    frame[batter_band_col] = assign_experience_band(frame[batter_n_col], batter_cuts, "B")

    output_dir = Path(config["paths"]["output_dir"]) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    band_rows: list[dict] = []
    for entity, cuts, prefix in [
        ("pitcher", pitcher_cuts, "P"),
        ("batter", batter_cuts, "B"),
    ]:
        for label in make_band_labels(cuts, prefix):
            band_rows.append({"entity": entity, "band": label})
    pd.DataFrame(band_rows).to_csv(output_dir / "band_definitions.csv", index=False)

    pitcher_marginal = marginal_summary(
        frame,
        season_col,
        target_col,
        pitcher_band_col,
        pitcher_n_col,
        "pitcher_id",
        "pitcher",
    )
    batter_marginal = marginal_summary(
        frame,
        season_col,
        target_col,
        batter_band_col,
        batter_n_col,
        "batter_id",
        "batter",
    )
    marginal = pd.concat([pitcher_marginal, batter_marginal], ignore_index=True)
    marginal.to_csv(output_dir / "experience_marginal.csv", index=False)

    pitcher_game_type = axis_game_type_summary(
        frame, season_col, target_col, pitcher_band_col, game_type_col
    )
    batter_game_type = axis_game_type_summary(
        frame, season_col, target_col, batter_band_col, game_type_col
    )
    pitcher_game_type.to_csv(output_dir / "pitcher_experience_game_type.csv", index=False)
    batter_game_type.to_csv(output_dir / "batter_experience_game_type.csv", index=False)

    joint = build_joint_summary(
        frame,
        season_col,
        target_col,
        pitcher_band_col,
        batter_band_col,
        pitcher_n_col,
        batter_n_col,
    )
    joint.to_csv(output_dir / "joint_experience.csv", index=False)

    joint_game_type = build_joint_game_type_summary(
        frame,
        joint,
        season_col,
        target_col,
        pitcher_band_col,
        batter_band_col,
        game_type_col,
        args.min_cell_rows,
    )
    joint_game_type.to_csv(output_dir / "joint_experience_game_type.csv", index=False)

    composition = build_composition_decomposition(
        frame,
        joint_game_type,
        season_col,
        target_col,
        pitcher_band_col,
        batter_band_col,
        game_type_col,
        args.min_cell_rows,
    )
    composition.to_csv(output_dir / "game_type_composition_decomposition.csv", index=False)

    heterogeneity = build_effect_heterogeneity(
        joint_game_type, season_col, game_type_col, args.min_cell_rows
    )
    heterogeneity.to_csv(output_dir / "game_type_effect_heterogeneity.csv", index=False)

    yoy = build_yoy_joint_game_type(
        joint_game_type,
        season_col,
        pitcher_band_col,
        batter_band_col,
        game_type_col,
    )
    yoy.to_csv(output_dir / "joint_game_type_yoy.csv", index=False)

    print("[Experience x Game-Type] experience bands")
    print(f"  pitcher cuts={pitcher_cuts}")
    print(f"  batter cuts={batter_cuts}")
    print(f"  min supported cell rows={args.min_cell_rows}")

    print("\n[Pitcher experience by season]")
    pitcher_display = pitcher_marginal[
        [season_col, pitcher_band_col, "rows", "row_share", "success_rate", "n_median"]
    ].copy()
    print(
        pitcher_display.to_string(
            index=False,
            formatters={
                "row_share": "{:.4f}".format,
                "success_rate": "{:.6f}".format,
                "n_median": "{:.1f}".format,
            },
        )
    )

    print("\n[Game-type composition shift after fixing joint experience mix]")
    composition_display = composition[
        [
            season_col,
            game_type_col,
            "raw_rows",
            "raw_rate",
            "standardized_rate",
            "composition_shift",
            "support_coverage",
        ]
    ].copy()
    print(
        composition_display.to_string(
            index=False,
            formatters={
                "raw_rate": "{:.6f}".format,
                "standardized_rate": "{:.6f}".format,
                "composition_shift": "{:+.6f}".format,
                "support_coverage": "{:.4f}".format,
            },
        )
    )

    print("\n[Game-type effect heterogeneity across joint experience clusters]")
    if heterogeneity.empty:
        print("No supported cells. Lower --min-cell-rows for exploratory analysis.")
    else:
        print(
            heterogeneity.to_string(
                index=False,
                formatters={
                    "weighted_mean_effect": "{:+.6f}".format,
                    "weighted_std_effect": "{:.6f}".format,
                    "min_effect": "{:+.6f}".format,
                    "max_effect": "{:+.6f}".format,
                    "effect_range": "{:.6f}".format,
                },
            )
        )

    print("\nInterpretation:")
    print("  game_type_effect = success_rate(game_type | experience band) - band baseline")
    print("  composition_shift = raw game_type rate - rate after fixing season experience mix")
    print("    positive: raw game_type rate is inflated by a more favorable experience mix")
    print("    negative: raw game_type rate is depressed by a less favorable experience mix")
    print("  weighted_std_effect/effect_range quantify game_type x experience interaction")
    print("  bands are exploratory probes, not final cluster boundaries")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
