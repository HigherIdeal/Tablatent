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


HISTORY_FEATURES = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]

KEY_COMPARISONS = [
    ("F", 2022, 2023),
    ("F", 2023, 2024),
    ("R", 2022, 2023),
    ("R", 2023, 2024),
]


def _normalize_game_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return np.nan
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _fixed_quantile_edges(series: pd.Series, bins: int) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(x) == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if len(edges) < 2:
        value = float(edges[0])
        return np.array([-np.inf, value, np.inf], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def build_season_domain_summary(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str,
) -> pd.DataFrame:
    out = (
        frame.groupby([season_col, game_type_col], observed=True)
        .agg(
            rows=(target_col, "size"),
            pitchers=("pitcher_id", "nunique"),
            batters=("batter_id", "nunique"),
            success_rate=(target_col, "mean"),
            pitcher_n_median=("asof_pitcher_n", "median"),
            batter_n_median=("asof_batter_n", "median"),
        )
        .reset_index()
        .sort_values([season_col, game_type_col])
    )
    return out


def build_monthly_summary(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str,
) -> pd.DataFrame:
    if "game_month" not in frame.columns:
        return pd.DataFrame()
    out = (
        frame.groupby([season_col, "game_month", game_type_col], observed=True)
        .agg(
            rows=(target_col, "size"),
            pitchers=("pitcher_id", "nunique"),
            success_rate=(target_col, "mean"),
        )
        .reset_index()
        .sort_values([season_col, "game_month", game_type_col])
    )
    return out


def build_count_state_summary(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str,
) -> pd.DataFrame:
    needed = {"balls_before", "strikes_before"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    work["count_state"] = (
        pd.to_numeric(work["balls_before"], errors="coerce").astype("Int64").astype(str)
        + "-"
        + pd.to_numeric(work["strikes_before"], errors="coerce").astype("Int64").astype(str)
    )
    out = (
        work.groupby([season_col, game_type_col, "count_state"], observed=True)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
        .sort_values([season_col, game_type_col, "count_state"])
    )
    return out


def build_history_response_curves(
    frame: pd.DataFrame,
    features: list[str],
    season_col: str,
    target_col: str,
    game_type_col: str,
    bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict] = []
    drift_rows: list[dict] = []

    for feature in features:
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        edges = _fixed_quantile_edges(numeric, bins)
        if len(edges) < 2:
            continue
        bucket = pd.cut(numeric, bins=edges, include_lowest=True, duplicates="drop")
        work = frame[[season_col, game_type_col, target_col]].copy()
        work["feature_bin"] = bucket.astype("string")
        work["feature_value"] = numeric

        grouped = (
            work.dropna(subset=["feature_bin"])
            .groupby([season_col, game_type_col, "feature_bin"], observed=True)
            .agg(
                rows=(target_col, "size"),
                feature_mean=("feature_value", "mean"),
                target_rate=(target_col, "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "feature", feature)
        curve_rows.extend(grouped.to_dict("records"))

        for game_type, year_a, year_b in KEY_COMPARISONS:
            a = grouped.loc[
                (grouped[season_col] == year_a) & (grouped[game_type_col] == game_type),
                ["feature_bin", "rows", "target_rate"],
            ].rename(columns={"rows": "rows_a", "target_rate": "rate_a"})
            b = grouped.loc[
                (grouped[season_col] == year_b) & (grouped[game_type_col] == game_type),
                ["feature_bin", "rows", "target_rate"],
            ].rename(columns={"rows": "rows_b", "target_rate": "rate_b"})
            merged = a.merge(b, on="feature_bin", how="inner")
            if merged.empty:
                continue
            weights = np.minimum(
                merged["rows_a"].to_numpy(dtype=np.float64),
                merged["rows_b"].to_numpy(dtype=np.float64),
            )
            valid = weights > 0
            if not valid.any():
                continue
            weights = weights[valid]
            diff = (
                merged.loc[valid, "rate_b"].to_numpy(dtype=np.float64)
                - merged.loc[valid, "rate_a"].to_numpy(dtype=np.float64)
            )
            drift_rows.append(
                {
                    "feature": feature,
                    "game_type": game_type,
                    "year_a": year_a,
                    "year_b": year_b,
                    "shared_bins": int(valid.sum()),
                    "support_rows": int(weights.sum()),
                    "weighted_mean_delta": float(np.average(diff, weights=weights)),
                    "weighted_mae_delta": float(np.average(np.abs(diff), weights=weights)),
                    "weighted_rmse_delta": float(np.sqrt(np.average(diff**2, weights=weights))),
                    "curve_corr": _safe_corr(merged.loc[valid, "rate_a"], merged.loc[valid, "rate_b"]),
                }
            )

    curves = pd.DataFrame(curve_rows)
    drift = pd.DataFrame(drift_rows)
    if not drift.empty:
        drift = drift.sort_values(["game_type", "year_a", "year_b", "weighted_rmse_delta"], ascending=[True, True, True, False])
    return curves, drift


def build_same_pitcher_year_shift(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    summary_rows: list[dict] = []
    for game_type, year_a, year_b in KEY_COMPARISONS:
        a = (
            frame.loc[(frame[game_type_col] == game_type) & (frame[season_col] == year_a)]
            .groupby("pitcher_id")
            .agg(rows_a=(target_col, "size"), rate_a=(target_col, "mean"), n_a=("asof_pitcher_n", "median"))
            .reset_index()
        )
        b = (
            frame.loc[(frame[game_type_col] == game_type) & (frame[season_col] == year_b)]
            .groupby("pitcher_id")
            .agg(rows_b=(target_col, "size"), rate_b=(target_col, "mean"), n_b=("asof_pitcher_n", "median"))
            .reset_index()
        )
        paired = a.merge(b, on="pitcher_id", how="inner")
        if paired.empty:
            continue
        paired["game_type"] = game_type
        paired["year_a"] = year_a
        paired["year_b"] = year_b
        paired["delta_rate"] = paired["rate_b"] - paired["rate_a"]
        rows.extend(paired.to_dict("records"))
        weights = np.minimum(paired["rows_a"], paired["rows_b"]).to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "game_type": game_type,
                "year_a": year_a,
                "year_b": year_b,
                "paired_pitchers": int(len(paired)),
                "median_rows_a": float(paired["rows_a"].median()),
                "median_rows_b": float(paired["rows_b"].median()),
                "mean_delta_rate": float(paired["delta_rate"].mean()),
                "median_delta_rate": float(paired["delta_rate"].median()),
                "weighted_delta_rate": float(np.average(paired["delta_rate"], weights=weights)) if weights.sum() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def build_team_shift(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        frame.groupby([season_col, game_type_col, "pitcher_team_id"], observed=True)
        .agg(rows=(target_col, "size"), pitchers=("pitcher_id", "nunique"), success_rate=(target_col, "mean"))
        .reset_index()
    )
    rows: list[dict] = []
    for game_type, year_a, year_b in KEY_COMPARISONS:
        a = base.loc[(base[game_type_col] == game_type) & (base[season_col] == year_a)].rename(
            columns={"rows": "rows_a", "pitchers": "pitchers_a", "success_rate": "rate_a"}
        )
        b = base.loc[(base[game_type_col] == game_type) & (base[season_col] == year_b)].rename(
            columns={"rows": "rows_b", "pitchers": "pitchers_b", "success_rate": "rate_b"}
        )
        merged = a[["pitcher_team_id", "rows_a", "pitchers_a", "rate_a"]].merge(
            b[["pitcher_team_id", "rows_b", "pitchers_b", "rate_b"]], on="pitcher_team_id", how="inner"
        )
        if merged.empty:
            continue
        merged["game_type"] = game_type
        merged["year_a"] = year_a
        merged["year_b"] = year_b
        merged["delta_rate"] = merged["rate_b"] - merged["rate_a"]
        rows.extend(merged.to_dict("records"))
    shifts = pd.DataFrame(rows)
    if shifts.empty:
        return base, pd.DataFrame()
    summary = (
        shifts.groupby(["game_type", "year_a", "year_b"], observed=True)
        .agg(
            shared_teams=("pitcher_team_id", "nunique"),
            mean_team_delta=("delta_rate", "mean"),
            median_team_delta=("delta_rate", "median"),
            min_team_delta=("delta_rate", "min"),
            max_team_delta=("delta_rate", "max"),
        )
        .reset_index()
    )
    return base, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether the 2023 F and 2024 R changes look like broad label/regime shifts. "
            "This is an observational diagnostic only and does not claim ABS causality."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-subdir", default="abs_regime_shift")
    args = parser.parse_args()

    if args.bins < 4:
        raise ValueError("--bins must be >= 4")

    config = load_config(ROOT / args.config)
    frame = load_frame(config).copy()
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    game_type_col = "game_type"

    required = {
        season_col,
        target_col,
        game_type_col,
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "asof_pitcher_n",
        "asof_batter_n",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    frame[game_type_col] = _normalize_game_type(frame[game_type_col])
    for col in ["asof_pitcher_n", "asof_batter_n"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "game_month" in frame.columns:
        frame["game_month"] = pd.to_numeric(frame["game_month"], errors="coerce").astype("Int64")

    features = [col for col in HISTORY_FEATURES if col in frame.columns]
    if not features:
        raise ValueError("No configured historical-rate features were found.")

    output_dir = Path(config["paths"]["output_dir"]) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    season_domain = build_season_domain_summary(frame, season_col, target_col, game_type_col)
    season_domain.to_csv(output_dir / "season_domain_summary.csv", index=False)

    monthly = build_monthly_summary(frame, season_col, target_col, game_type_col)
    monthly.to_csv(output_dir / "monthly_domain_summary.csv", index=False)

    count_state = build_count_state_summary(frame, season_col, target_col, game_type_col)
    count_state.to_csv(output_dir / "count_state_summary.csv", index=False)

    curves, curve_drift = build_history_response_curves(
        frame, features, season_col, target_col, game_type_col, args.bins
    )
    curves.to_csv(output_dir / "history_feature_response_curves.csv", index=False)
    curve_drift.to_csv(output_dir / "history_feature_curve_drift.csv", index=False)

    paired, paired_summary = build_same_pitcher_year_shift(frame, season_col, target_col, game_type_col)
    paired.to_csv(output_dir / "same_pitcher_year_shift.csv", index=False)
    paired_summary.to_csv(output_dir / "same_pitcher_year_shift_summary.csv", index=False)

    team_base, team_shift_summary = build_team_shift(frame, season_col, target_col, game_type_col)
    team_base.to_csv(output_dir / "team_season_domain_summary.csv", index=False)
    team_shift_summary.to_csv(output_dir / "team_shift_summary.csv", index=False)

    print("[ABS / Regime-Shift Diagnostic]")
    print("  observational only: control_success is NOT assumed to be called-strike outcome")
    print("  purpose: test whether 2023 F / 2024 R changes are broad across players, teams, contexts, and history states")
    print(f"  history features={features}")

    print("\n[Season x game_type target rates]")
    print(season_domain.to_string(index=False, formatters={"success_rate": "{:.6f}".format}))

    print("\n[Same-pitcher adjacent-year shifts]")
    if paired_summary.empty:
        print("No matched pitcher comparisons.")
    else:
        print(
            paired_summary.to_string(
                index=False,
                formatters={
                    "mean_delta_rate": "{:+.6f}".format,
                    "median_delta_rate": "{:+.6f}".format,
                    "weighted_delta_rate": "{:+.6f}".format,
                },
            )
        )

    print("\n[Team-level shifts]")
    if team_shift_summary.empty:
        print("No shared-team comparisons.")
    else:
        print(
            team_shift_summary.to_string(
                index=False,
                formatters={
                    "mean_team_delta": "{:+.6f}".format,
                    "median_team_delta": "{:+.6f}".format,
                    "min_team_delta": "{:+.6f}".format,
                    "max_team_delta": "{:+.6f}".format,
                },
            )
        )

    print("\n[Historical-feature response-curve drift: largest first within each comparison]")
    if curve_drift.empty:
        print("No response-curve comparisons.")
    else:
        display = curve_drift.copy()
        print(
            display.to_string(
                index=False,
                formatters={
                    "weighted_mean_delta": "{:+.6f}".format,
                    "weighted_mae_delta": "{:.6f}".format,
                    "weighted_rmse_delta": "{:.6f}".format,
                    "curve_corr": lambda x: "nan" if pd.isna(x) else f"{x:.4f}",
                },
            )
        )

    if not monthly.empty:
        print("\n[F monthly rates around 2022-2024]")
        display = monthly.loc[(monthly[game_type_col] == "F") & (monthly[season_col].isin([2022, 2023, 2024]))]
        print(display.to_string(index=False, formatters={"success_rate": "{:.6f}".format}))

    print("\nInterpretation guide:")
    print("  - Broad F 2022->2023 drops for the SAME pitchers and most teams => composition is unlikely to explain the shift.")
    print("  - Large response-curve drift means the mapping from historical pitcher state to target changed, not just the marginal mix.")
    print("  - A synchronized monthly/team-wide change is more consistent with a common operational/label regime change than player development.")
    print("  - If only a subset of teams/months moves, investigate venue/system coverage rather than assuming one global change.")
    print("  - ABS may be a plausible external mechanism, but this script cannot prove causality because control_success is not documented as an ABS call.")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
