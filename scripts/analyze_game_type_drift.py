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


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def competition_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    baseline = float(y.mean() * (1.0 - y.mean()))
    value = 1.0 - brier(y, p) / baseline
    return float(max(0.0, 100000.0 * value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose season-to-season drift in game_type without training CatBoost."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--min-val-year", type=int, default=2020)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    frame = load_frame(config).copy()

    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    game_type_col = "game_type"

    required = {season_col, target_col, game_type_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(np.float64)
    frame[game_type_col] = (
        frame[game_type_col].astype("string").fillna("<MISSING>").astype(str)
    )

    output_dir = Path(config["paths"]["output_dir"]) / "game_type_drift"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Raw season x game_type distribution and target rate.
    grouped = (
        frame.groupby([season_col, game_type_col], observed=True)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
    )
    season_rows = frame.groupby(season_col).size().rename("season_rows")
    grouped = grouped.merge(season_rows, on=season_col, how="left")
    grouped["share"] = grouped["rows"] / grouped["season_rows"]
    grouped = grouped.sort_values([season_col, game_type_col]).reset_index(drop=True)
    grouped.to_csv(output_dir / "season_game_type.csv", index=False)

    season_summary = (
        frame.groupby(season_col)
        .agg(rows=(target_col, "size"), success_rate=(target_col, "mean"))
        .reset_index()
        .sort_values(season_col)
    )
    season_summary.to_csv(output_dir / "season_summary.csv", index=False)

    share_pivot = grouped.pivot(index=game_type_col, columns=season_col, values="share")
    rate_pivot = grouped.pivot(index=game_type_col, columns=season_col, values="success_rate")
    share_pivot.to_csv(output_dir / "share_by_season.csv")
    rate_pivot.to_csv(output_dir / "success_rate_by_season.csv")

    # 2) Forward validation with game_type alone.
    # Compare a historical global prior against historical game_type means. This
    # directly answers whether game_type adds temporally stable information.
    years = sorted(frame[season_col].unique().tolist())
    forward_rows: list[dict] = []

    for val_year in years:
        if val_year < args.min_val_year:
            continue
        train = frame.loc[frame[season_col] < val_year]
        valid = frame.loc[frame[season_col] == val_year]
        if train.empty or valid.empty:
            continue

        y_train = train[target_col].to_numpy(np.float64)
        y_valid = valid[target_col].to_numpy(np.float64)
        train_prior = float(y_train.mean())
        val_rate = float(y_valid.mean())

        rate_map = train.groupby(game_type_col, observed=True)[target_col].mean()
        p_prior = np.full(len(valid), train_prior, dtype=np.float64)
        p_game_type = (
            valid[game_type_col].map(rate_map).fillna(train_prior).to_numpy(np.float64)
        )

        # Diagnostic only: preserve the historical game_type deviation but rebase
        # the intercept to the actual validation-year mean. This uses validation
        # labels and is never a deployable prediction. It separates category-effect
        # drift from global target-rate drift.
        effect_map = rate_map - train_prior
        p_rebased = (
            val_rate
            + valid[game_type_col].map(effect_map).fillna(0.0).to_numpy(np.float64)
        )
        p_rebased = np.clip(p_rebased, 0.0, 1.0)
        p_oracle_mean = np.full(len(valid), val_rate, dtype=np.float64)

        b_prior = brier(y_valid, p_prior)
        b_type = brier(y_valid, p_game_type)
        b_oracle = brier(y_valid, p_oracle_mean)
        b_rebased = brier(y_valid, p_rebased)

        unseen = ~valid[game_type_col].isin(rate_map.index)
        forward_rows.append(
            {
                "validation_year": int(val_year),
                "train_rows": int(len(train)),
                "val_rows": int(len(valid)),
                "train_prior": train_prior,
                "val_rate": val_rate,
                "prior_shift": val_rate - train_prior,
                "game_types_train": int(rate_map.size),
                "game_types_val": int(valid[game_type_col].nunique()),
                "unseen_val_rows": int(unseen.sum()),
                "brier_train_prior": b_prior,
                "brier_game_type": b_type,
                "delta_game_type_vs_train_prior": b_type - b_prior,
                "score_game_type": competition_score(y_valid, p_game_type),
                "brier_oracle_val_mean": b_oracle,
                "brier_rebased_game_type": b_rebased,
                "delta_rebased_vs_oracle_mean": b_rebased - b_oracle,
            }
        )

    forward = pd.DataFrame(forward_rows)
    forward.to_csv(output_dir / "forward_validation.csv", index=False)

    print("[Game-Type Drift] season summary")
    print(season_summary.to_string(index=False, formatters={"success_rate": "{:.6f}".format}))

    print("\n[Game-Type Drift] season x game_type")
    display = grouped[[season_col, game_type_col, "rows", "share", "success_rate"]].copy()
    print(
        display.to_string(
            index=False,
            formatters={"share": "{:.4f}".format, "success_rate": "{:.6f}".format},
        )
    )

    print("\n[Game-Type Drift] forward validation")
    if forward.empty:
        print("No valid forward folds.")
    else:
        cols = [
            "validation_year",
            "train_prior",
            "val_rate",
            "prior_shift",
            "brier_train_prior",
            "brier_game_type",
            "delta_game_type_vs_train_prior",
            "score_game_type",
            "delta_rebased_vs_oracle_mean",
        ]
        print(
            forward[cols].to_string(
                index=False,
                formatters={
                    "train_prior": "{:.6f}".format,
                    "val_rate": "{:.6f}".format,
                    "prior_shift": "{:+.6f}".format,
                    "brier_train_prior": "{:.8f}".format,
                    "brier_game_type": "{:.8f}".format,
                    "delta_game_type_vs_train_prior": "{:+.8f}".format,
                    "score_game_type": "{:.2f}".format,
                    "delta_rebased_vs_oracle_mean": "{:+.8f}".format,
                },
            )
        )

    print("\nInterpretation:")
    print("  delta_game_type_vs_train_prior < 0 : historical game_type signal helps next year")
    print("  delta_game_type_vs_train_prior > 0 : historical game_type signal hurts next year")
    print("  delta_rebased_vs_oracle_mean isolates game_type effect after removing global rate drift")
    print("  rebased metric is diagnostic only; it uses validation target mean and must not be used for inference")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
