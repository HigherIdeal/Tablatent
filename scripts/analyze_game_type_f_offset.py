from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_frame
from src.utils import load_config, save_json


DEFAULT_OLD_YEARS = (2019, 2020, 2021, 2022)
DEFAULT_NEW_YEARS = (2023, 2024)


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one year is required")
    return values


def parse_delta_grid(value: str) -> np.ndarray:
    text = value.strip()
    if ":" not in text:
        values = np.array([float(item.strip()) for item in text.split(",") if item.strip()])
    else:
        parts = [float(item.strip()) for item in text.split(":")]
        if len(parts) != 3:
            raise ValueError("Delta range must be start:stop:step, e.g. 0:0.30:0.005")
        start, stop, step = parts
        if step <= 0:
            raise ValueError("Delta step must be > 0")
        count = int(math.floor((stop - start) / step + 1e-12)) + 1
        values = start + step * np.arange(count, dtype=np.float64)
        if values[-1] < stop - 1e-10:
            values = np.append(values, stop)
    if values.size == 0:
        raise ValueError("Delta grid is empty")
    return np.unique(np.round(values.astype(np.float64), 10))


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.average(np.square(values), weights=weights)))


def weighted_mae(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.average(np.abs(values), weights=weights))


def rate_table(
    frame: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    game_type_col: str,
) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols + [game_type_col], observed=True)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
    )
    rate = grouped.pivot(index=group_cols, columns=game_type_col, values="success_rate")
    rows = grouped.pivot(index=group_cols, columns=game_type_col, values="rows")
    needed = {"F", "R"}
    if not needed.issubset(set(rate.columns)):
        return pd.DataFrame()
    result = pd.DataFrame(index=rate.index)
    result["f_rate"] = rate["F"]
    result["r_rate"] = rate["R"]
    result["f_rows"] = rows["F"]
    result["r_rows"] = rows["R"]
    result = result.dropna().reset_index()
    return result


def add_experience_bucket(frame: pd.DataFrame) -> None:
    if "asof_pitcher_n" not in frame.columns:
        return
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
    frame["_pitcher_exp_bucket"] = pd.cut(
        n,
        bins=[-np.inf, 499, 1499, 2999, 5999, np.inf],
        labels=["0-499", "500-1499", "1500-2999", "3000-5999", "6000+"],
        ordered=True,
    ).astype("string").fillna("<MISSING>")


def build_subgroup_pairs(
    frame: pd.DataFrame,
    old_years: list[int],
    new_years: list[int],
    season_col: str,
    target_col: str,
    game_type_col: str,
    min_group_rows: int,
) -> pd.DataFrame:
    families: list[tuple[str, list[str]]] = []
    candidates = [
        ("pitcher_team", ["pitcher_team_id"]),
        ("month", ["game_month"]),
        ("count", ["balls_before", "strikes_before"]),
        ("handedness", ["pitcher_hand", "batter_hand"]),
        ("inning", ["inning"]),
        ("pitcher_experience", ["_pitcher_exp_bucket"]),
    ]
    for name, cols in candidates:
        if all(col in frame.columns for col in cols):
            families.append((name, cols))

    old_frame = frame.loc[frame[season_col].isin(old_years)]
    all_pairs: list[pd.DataFrame] = []

    for family, group_cols in families:
        old = rate_table(old_frame, group_cols, target_col, game_type_col)
        if old.empty:
            continue
        old = old.rename(
            columns={
                "f_rate": "old_f_rate",
                "r_rate": "old_r_rate",
                "f_rows": "old_f_rows",
                "r_rows": "old_r_rows",
            }
        )

        for year in new_years:
            new_frame = frame.loc[frame[season_col].eq(year)]
            new = rate_table(new_frame, group_cols, target_col, game_type_col)
            if new.empty:
                continue
            new = new.rename(
                columns={
                    "f_rate": "new_f_rate",
                    "r_rate": "new_r_rate",
                    "f_rows": "new_f_rows",
                    "r_rows": "new_r_rows",
                }
            )
            pair = old.merge(new, on=group_cols, how="inner")
            if pair.empty:
                continue
            required_rows = ["old_f_rows", "old_r_rows", "new_f_rows", "new_r_rows"]
            pair = pair.loc[(pair[required_rows] >= min_group_rows).all(axis=1)].copy()
            if pair.empty:
                continue
            pair["family"] = family
            pair["validation_year"] = int(year)
            pair["old_effect"] = pair["old_f_rate"] - pair["old_r_rate"]
            pair["new_effect"] = pair["new_f_rate"] - pair["new_r_rate"]
            pair["raw_residual"] = pair["new_effect"] - pair["old_effect"]
            pair["weight"] = pair[required_rows].min(axis=1).astype(float)
            pair["group_key"] = pair[group_cols].astype(str).agg("|".join, axis=1)
            all_pairs.append(
                pair[
                    [
                        "family",
                        "validation_year",
                        "group_key",
                        "old_f_rate",
                        "old_r_rate",
                        "new_f_rate",
                        "new_r_rate",
                        "old_effect",
                        "new_effect",
                        "raw_residual",
                        "weight",
                        *required_rows,
                    ]
                ]
            )

    if not all_pairs:
        return pd.DataFrame()
    return pd.concat(all_pairs, ignore_index=True)


def evaluate_delta(
    delta: float,
    season_effects: pd.DataFrame,
    old_reference_effect: float,
    subgroup_pairs: pd.DataFrame,
    years: list[int] | None = None,
) -> dict:
    if years is None:
        season_slice = season_effects.copy()
        subgroup_slice = subgroup_pairs.copy()
    else:
        season_slice = season_effects.loc[season_effects["season"].isin(years)].copy()
        subgroup_slice = subgroup_pairs.loc[subgroup_pairs["validation_year"].isin(years)].copy()

    global_residual = season_slice["raw_effect"].to_numpy(np.float64) + delta - old_reference_effect
    global_rmse = float(np.sqrt(np.mean(np.square(global_residual))))
    global_mae = float(np.mean(np.abs(global_residual)))

    if subgroup_slice.empty:
        subgroup_rmse = np.nan
        subgroup_mae = np.nan
        out_of_bounds_groups = 0
    else:
        residual = subgroup_slice["raw_residual"].to_numpy(np.float64) + delta
        weights = subgroup_slice["weight"].to_numpy(np.float64)
        subgroup_rmse = weighted_rmse(residual, weights)
        subgroup_mae = weighted_mae(residual, weights)
        corrected_f = subgroup_slice["new_f_rate"].to_numpy(np.float64) + delta
        out_of_bounds_groups = int(((corrected_f < 0.0) | (corrected_f > 1.0)).sum())

    primary = subgroup_rmse if np.isfinite(subgroup_rmse) else global_rmse
    return {
        "delta": float(delta),
        "global_effect_rmse": global_rmse,
        "global_effect_mae": global_mae,
        "subgroup_effect_rmse": subgroup_rmse,
        "subgroup_effect_mae": subgroup_mae,
        "out_of_bounds_subgroups": out_of_bounds_groups,
        "objective_rmse": float(primary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a constant additive correction applied only to game_type=F in 2023+ "
            "restores the 2019-2022 F-vs-R relationship. This is a diagnostic on rates, not a "
            "label rewrite or a deployable prediction rule."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--old-years", default=",".join(map(str, DEFAULT_OLD_YEARS)))
    parser.add_argument("--new-years", default=",".join(map(str, DEFAULT_NEW_YEARS)))
    parser.add_argument("--deltas", default="0:0.30:0.005")
    parser.add_argument("--min-group-rows", type=int, default=100)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    frame = load_frame(config).copy()
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    game_type_col = "game_type"

    old_years = parse_ints(args.old_years)
    new_years = parse_ints(args.new_years)
    deltas = parse_delta_grid(args.deltas)

    required = {season_col, target_col, game_type_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(np.float64)
    frame[game_type_col] = frame[game_type_col].astype("string").fillna("<MISSING>").astype(str)
    add_experience_bucket(frame)

    available = set(frame[season_col].unique().tolist())
    absent = sorted((set(old_years) | set(new_years)) - available)
    if absent:
        raise ValueError(f"Requested seasons are missing from data: {absent}")

    output_dir = Path(config["paths"]["output_dir"]) / "game_type_f_offset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Season-level F/R effects.
    grouped = (
        frame.loc[frame[season_col].isin(old_years + new_years)]
        .groupby([season_col, game_type_col], observed=True)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
    )
    rate = grouped.pivot(index=season_col, columns=game_type_col, values="success_rate")
    rows = grouped.pivot(index=season_col, columns=game_type_col, values="rows")
    if not {"F", "R"}.issubset(set(rate.columns)):
        raise ValueError(f"Expected game_type values F and R; found {sorted(rate.columns.astype(str))}")

    season_all = pd.DataFrame(
        {
            "season": rate.index.astype(int),
            "f_rate": rate["F"].to_numpy(float),
            "r_rate": rate["R"].to_numpy(float),
            "f_rows": rows["F"].to_numpy(int),
            "r_rows": rows["R"].to_numpy(int),
        }
    ).sort_values("season")
    season_all["raw_effect"] = season_all["f_rate"] - season_all["r_rate"]
    old_reference_effect = float(
        season_all.loc[season_all["season"].isin(old_years), "raw_effect"].mean()
    )
    season_new = season_all.loc[season_all["season"].isin(new_years)].copy()

    subgroup_pairs = build_subgroup_pairs(
        frame=frame,
        old_years=old_years,
        new_years=new_years,
        season_col=season_col,
        target_col=target_col,
        game_type_col=game_type_col,
        min_group_rows=args.min_group_rows,
    )

    # Common delta: exactly the user's hypothesis -- the same additive correction
    # is applied to every F rate in both 2023 and 2024; R is untouched.
    sweep = pd.DataFrame(
        [evaluate_delta(delta, season_new, old_reference_effect, subgroup_pairs) for delta in deltas]
    )
    best_common = sweep.sort_values(["objective_rmse", "global_effect_rmse", "delta"]).iloc[0]
    best_common_delta = float(best_common["delta"])

    zero_eval = evaluate_delta(0.0, season_new, old_reference_effect, subgroup_pairs)
    raw_rmse = float(zero_eval["objective_rmse"])
    best_rmse = float(best_common["objective_rmse"])
    explained = 0.0 if raw_rmse <= 0 else 1.0 - (best_rmse * best_rmse) / (raw_rmse * raw_rmse)

    # Also estimate each year independently. The common-offset hypothesis is more
    # credible if 2023 and 2024 independently prefer nearly the same delta.
    per_year_rows: list[dict] = []
    best_year_rows: list[dict] = []
    for year in new_years:
        year_sweep = pd.DataFrame(
            [
                {
                    "validation_year": int(year),
                    **evaluate_delta(
                        delta,
                        season_new,
                        old_reference_effect,
                        subgroup_pairs,
                        years=[year],
                    ),
                }
                for delta in deltas
            ]
        )
        per_year_rows.extend(year_sweep.to_dict("records"))
        best = year_sweep.sort_values(["objective_rmse", "global_effect_rmse", "delta"]).iloc[0].to_dict()
        best_year_rows.append(best)

    per_year_sweep = pd.DataFrame(per_year_rows)
    best_by_year = pd.DataFrame(best_year_rows).sort_values("validation_year")

    corrected_season = season_all.copy()
    corrected_season["era"] = np.where(corrected_season["season"].isin(old_years), "old", "new")
    corrected_season["correction_delta"] = np.where(
        corrected_season["season"].isin(new_years), best_common_delta, 0.0
    )
    corrected_season["corrected_f_rate"] = corrected_season["f_rate"] + corrected_season["correction_delta"]
    corrected_season["corrected_effect"] = corrected_season["corrected_f_rate"] - corrected_season["r_rate"]
    corrected_season["effect_error_vs_old_reference"] = (
        corrected_season["corrected_effect"] - old_reference_effect
    )

    if not subgroup_pairs.empty:
        subgroup_best = subgroup_pairs.copy()
        subgroup_best["correction_delta"] = best_common_delta
        subgroup_best["corrected_new_f_rate"] = subgroup_best["new_f_rate"] + best_common_delta
        subgroup_best["corrected_new_effect"] = subgroup_best["new_effect"] + best_common_delta
        subgroup_best["corrected_residual"] = subgroup_best["raw_residual"] + best_common_delta
        subgroup_best["corrected_f_out_of_bounds"] = (
            (subgroup_best["corrected_new_f_rate"] < 0.0)
            | (subgroup_best["corrected_new_f_rate"] > 1.0)
        )
    else:
        subgroup_best = subgroup_pairs.copy()

    family_summary_rows: list[dict] = []
    if not subgroup_pairs.empty:
        for (family, year), part in subgroup_pairs.groupby(["family", "validation_year"], observed=True):
            weights = part["weight"].to_numpy(float)
            raw_residual = part["raw_residual"].to_numpy(float)
            corrected_residual = raw_residual + best_common_delta
            raw_family_rmse = weighted_rmse(raw_residual, weights)
            corrected_family_rmse = weighted_rmse(corrected_residual, weights)
            family_explained = (
                0.0
                if raw_family_rmse <= 0
                else 1.0 - (corrected_family_rmse**2) / (raw_family_rmse**2)
            )
            family_summary_rows.append(
                {
                    "family": family,
                    "validation_year": int(year),
                    "groups": int(len(part)),
                    "raw_weighted_mean_residual": float(np.average(raw_residual, weights=weights)),
                    "raw_weighted_rmse": raw_family_rmse,
                    "corrected_weighted_mean_residual": float(
                        np.average(corrected_residual, weights=weights)
                    ),
                    "corrected_weighted_rmse": corrected_family_rmse,
                    "constant_offset_explained_fraction": float(family_explained),
                }
            )
    family_summary = pd.DataFrame(family_summary_rows)

    # Save all diagnostics before printing.
    season_all.to_csv(output_dir / "season_raw_effects.csv", index=False)
    corrected_season.to_csv(output_dir / "season_corrected_effects.csv", index=False)
    sweep.to_csv(output_dir / "common_delta_sweep.csv", index=False)
    per_year_sweep.to_csv(output_dir / "per_year_delta_sweep.csv", index=False)
    best_by_year.to_csv(output_dir / "best_delta_by_year.csv", index=False)
    subgroup_pairs.to_csv(output_dir / "subgroup_pairs_raw.csv", index=False)
    subgroup_best.to_csv(output_dir / "subgroup_pairs_best_common.csv", index=False)
    family_summary.to_csv(output_dir / "family_summary_best_common.csv", index=False)

    save_json(
        {
            "old_years": old_years,
            "new_years": new_years,
            "delta_grid": deltas.tolist(),
            "min_group_rows": int(args.min_group_rows),
            "old_reference_effect_macro_mean": old_reference_effect,
            "best_common_delta": best_common_delta,
            "objective_rmse_raw": raw_rmse,
            "objective_rmse_best_common": best_rmse,
            "constant_offset_explained_fraction": float(explained),
            "subgroup_pair_count": int(len(subgroup_pairs)),
            "important_note": (
                "Diagnostic only. The script adds delta to aggregated F success rates for 2023/2024. "
                "It does not modify binary training labels and does not use this correction for inference."
            ),
        },
        output_dir / "run_config.json",
    )

    print(
        f"[F-only Offset] old_years={old_years}, new_years={new_years}, "
        f"deltas={deltas[0]:.3f}..{deltas[-1]:.3f} ({len(deltas)} values), "
        f"min_group_rows={args.min_group_rows}"
    )
    print("[F-only Offset] correction is applied ONLY to F success rates in new years; R is unchanged")
    print("[F-only Offset] diagnostic only: raw binary control_success labels are never rewritten")

    print("\n[Raw season effects] F - R")
    print(
        season_all.to_string(
            index=False,
            formatters={
                "f_rate": "{:.6f}".format,
                "r_rate": "{:.6f}".format,
                "raw_effect": "{:+.6f}".format,
            },
        )
    )
    print(f"\nOld-era macro reference effect ({old_years[0]}-{old_years[-1]}): {old_reference_effect:+.6f}")

    print("\n[Best common F-only additive correction]")
    print(f"  delta                         = {best_common_delta:+.4f}")
    print(f"  objective RMSE: raw -> best  = {raw_rmse:.6f} -> {best_rmse:.6f}")
    print(f"  explained fraction           = {explained:.3f}")
    print(f"  subgroup pairs               = {len(subgroup_pairs):,}")
    print(f"  out-of-bounds subgroups      = {int(best_common['out_of_bounds_subgroups'])}")

    print("\n[Corrected season effects using the SAME delta for all new-year F rows]")
    cols = ["season", "f_rate", "r_rate", "raw_effect", "correction_delta", "corrected_f_rate", "corrected_effect"]
    print(
        corrected_season[cols].to_string(
            index=False,
            formatters={
                "f_rate": "{:.6f}".format,
                "r_rate": "{:.6f}".format,
                "raw_effect": "{:+.6f}".format,
                "correction_delta": "{:+.4f}".format,
                "corrected_f_rate": "{:.6f}".format,
                "corrected_effect": "{:+.6f}".format,
            },
        )
    )

    print("\n[Independent best delta by new year]")
    print(
        best_by_year[
            ["validation_year", "delta", "objective_rmse", "global_effect_rmse", "subgroup_effect_rmse"]
        ].to_string(
            index=False,
            formatters={
                "delta": "{:+.4f}".format,
                "objective_rmse": "{:.6f}".format,
                "global_effect_rmse": "{:.6f}".format,
                "subgroup_effect_rmse": "{:.6f}".format,
            },
        )
    )

    if len(best_by_year) >= 2:
        spread = float(best_by_year["delta"].max() - best_by_year["delta"].min())
        print(f"  preferred-delta spread across new years = {spread:.4f}")

    if not family_summary.empty:
        print("\n[Subgroup consistency at best common delta]")
        print(
            family_summary.to_string(
                index=False,
                formatters={
                    "raw_weighted_mean_residual": "{:+.6f}".format,
                    "raw_weighted_rmse": "{:.6f}".format,
                    "corrected_weighted_mean_residual": "{:+.6f}".format,
                    "corrected_weighted_rmse": "{:.6f}".format,
                    "constant_offset_explained_fraction": "{:.3f}".format,
                },
            )
        )

    print("\nInterpretation:")
    print("  - Similar independent best deltas for 2023 and 2024 support a common measurement-offset hypothesis.")
    print("  - A large explained fraction means a constant F-only shift removes much of the old-vs-new mismatch.")
    print("  - Large residual RMSE after correction means the change is not just a constant offset.")
    print("  - Any out-of-bounds corrected F rates are evidence against a literal additive probability model.")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
