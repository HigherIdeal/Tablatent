from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_catboost_ablation as core
import run_unseen_pitcher_validation as base
from src.canonical_features import (
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.utils import load_config, save_json, seed_everything


FEATURES = list(CANONICAL_FEATURES)
EXPERIENCE_BINS = [-1, 100, 500, 1000, 2000, 5000, np.inf]
EXPERIENCE_LABELS = ["0-100", "101-500", "501-1000", "1001-2000", "2001-5000", "5000+"]


def assign_balanced_pitcher_groups(train: pd.DataFrame, valid: pd.DataFrame, pitchers: set[str], n_groups: int) -> dict[str, int]:
    history = train.loc[train["_pitcher_key"].isin(pitchers), "_pitcher_key"].value_counts()
    valid_rows = valid.loc[valid["_pitcher_key"].isin(pitchers), "_pitcher_key"].value_counts()
    items = []
    for pitcher in pitchers:
        items.append((pitcher, int(history.get(pitcher, 0)), int(valid_rows.get(pitcher, 0))))
    items.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)

    hist_load = [0] * n_groups
    valid_load = [0] * n_groups
    mapping: dict[str, int] = {}
    for pitcher, hist_n, valid_n in items:
        group = min(range(n_groups), key=lambda g: (hist_load[g], valid_load[g], g))
        mapping[pitcher] = group
        hist_load[group] += hist_n
        valid_load[group] += valid_n
    return mapping


def matched_size_control(
    train_full: pd.DataFrame,
    heldout_pitchers: set[str],
    purge_train: pd.DataFrame,
    season_col: str,
    seed: int,
) -> pd.DataFrame:
    """Match purge-train row count season-by-season while KEEPING heldout-pitcher history."""
    rng = np.random.default_rng(seed)
    pieces = []
    for year, year_full in train_full.groupby(season_col, sort=True):
        target_n = int((purge_train[season_col] == year).sum())
        held = year_full.loc[year_full["_pitcher_key"].isin(heldout_pitchers)]
        other = year_full.loc[~year_full["_pitcher_key"].isin(heldout_pitchers)]
        if len(held) > target_n:
            raise ValueError(
                f"Cannot build matched control for season={year}: heldout history={len(held)} > target={target_n}"
            )
        need = target_n - len(held)
        if need > len(other):
            raise ValueError(f"Not enough non-heldout rows for matched control in season={year}")
        if need:
            chosen = rng.choice(other.index.to_numpy(), size=need, replace=False)
            sampled = other.loc[chosen]
            pieces.append(pd.concat([held, sampled], axis=0))
        else:
            pieces.append(held)
    out = pd.concat(pieces, axis=0).sort_index().copy()
    if len(out) != len(purge_train):
        raise AssertionError((len(out), len(purge_train)))
    return out


def cohort_summary(valid: pd.DataFrame, train_pitchers: set[str], target: str) -> pd.DataFrame:
    work = valid.copy()
    work["cohort"] = np.where(work["_pitcher_key"].isin(train_pitchers), "seen", "natural_unseen")
    asof_n = pd.to_numeric(work["asof_pitcher_n"], errors="coerce")
    work["_asof_n"] = asof_n
    work["_target"] = pd.to_numeric(work[target], errors="raise")
    work["_is_f"] = work["game_type"].astype("string").fillna("<MISSING>").astype(str).eq("F").astype(float)

    rows = []
    for cohort, part in work.groupby("cohort", sort=True):
        rows.append(
            {
                "cohort": cohort,
                "rows": int(len(part)),
                "pitchers": int(part["_pitcher_key"].nunique()),
                "target_rate": float(part["_target"].mean()),
                "asof_pitcher_n_mean": float(part["_asof_n"].mean()),
                "asof_pitcher_n_median": float(part["_asof_n"].median()),
                "asof_pitcher_n_p25": float(part["_asof_n"].quantile(0.25)),
                "asof_pitcher_n_p75": float(part["_asof_n"].quantile(0.75)),
                "game_type_F_share": float(part["_is_f"].mean()),
                "pitcher_success_rate_mean": float(pd.to_numeric(part["asof_pitcher_success_rate"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def bucket_metrics(valid: pd.DataFrame, prediction: np.ndarray, train_pitchers: set[str], target: str) -> pd.DataFrame:
    work = valid[["_pitcher_key", "asof_pitcher_n", target]].copy()
    work["prediction"] = np.asarray(prediction, dtype=np.float64)
    work["cohort"] = np.where(work["_pitcher_key"].isin(train_pitchers), "seen", "natural_unseen")
    work["experience_bucket"] = pd.cut(
        pd.to_numeric(work["asof_pitcher_n"], errors="coerce"),
        bins=EXPERIENCE_BINS,
        labels=EXPERIENCE_LABELS,
        include_lowest=True,
    ).astype("string")

    rows = []
    for (cohort, bucket), part in work.groupby(["cohort", "experience_bucket"], observed=True, sort=False):
        if len(part) < 100:
            continue
        y = pd.to_numeric(part[target], errors="raise").to_numpy(np.float64)
        p = part["prediction"].to_numpy(np.float64)
        m = core.metrics(y, p)
        rows.append(
            {
                "cohort": str(cohort),
                "experience_bucket": str(bucket),
                "rows": int(len(part)),
                "pitchers": int(part["_pitcher_key"].nunique()),
                "target_rate": float(y.mean()),
                **m,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled unseen-pitcher diagnostic. Partition 2024 seen pitchers into balanced groups, "
            "compare the same validation rows under full-history training, group-history purge, and a "
            "season-matched random-size control that keeps the group's own history."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if args.groups < 2:
        raise ValueError("--groups must be >= 2")

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame = load_frame(config).copy()
    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id, "pitcher_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["_pitcher_key"] = base.normalize_id(frame["pitcher_id"])
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    train_full = frame.loc[frame[season] < args.validation_year].copy()
    valid = frame.loc[frame[season].eq(args.validation_year)].copy()
    train_pitchers = set(train_full["_pitcher_key"].unique())
    valid_pitchers = set(valid["_pitcher_key"].unique())
    seen_pitchers = train_pitchers & valid_pitchers
    natural_unseen = valid_pitchers - train_pitchers
    if not seen_pitchers:
        raise ValueError("No seen pitchers in validation year")

    params = base.catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)

    output_dir = Path(config["paths"]["output_dir"]) / "unseen_pitcher_exposure_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Unseen-Pitcher Exposure Diagnostic] validation={args.validation_year}, groups={args.groups}, "
        f"iterations={args.iterations}, task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print(
        f"  train={len(train_full):,} rows/{len(train_pitchers):,} pitchers; "
        f"valid={len(valid):,} rows/{len(valid_pitchers):,} pitchers"
    )
    print(
        f"  seen={len(seen_pitchers):,} pitchers; natural_unseen={len(natural_unseen):,} pitchers"
    )

    print("\n[1] Fit one full-history reference model")
    p_full = base.fit_predict(train_full, valid, target, FEATURES, params)

    cohort_df = cohort_summary(valid, train_pitchers, target)
    bucket_df = bucket_metrics(valid, p_full, train_pitchers, target)
    cohort_df.to_csv(output_dir / "cohort_composition.csv", index=False)
    bucket_df.to_csv(output_dir / "experience_bucket_metrics.csv", index=False)

    group_map = assign_balanced_pitcher_groups(train_full, valid, seen_pitchers, args.groups)
    fold_rows = []
    group_rows = []
    stitched_full = []
    stitched_purge = []
    stitched_control = []
    stitched_y = []

    for group in range(args.groups):
        heldout = {pitcher for pitcher, g in group_map.items() if g == group}
        eval_mask = valid["_pitcher_key"].isin(heldout).to_numpy()
        eval_frame = valid.loc[eval_mask]
        y_eval = y_valid[eval_mask]
        p_eval_full = p_full[eval_mask]

        purge_train = train_full.loc[~train_full["_pitcher_key"].isin(heldout)].copy()
        removed_rows = len(train_full) - len(purge_train)
        control_train = matched_size_control(
            train_full,
            heldout_pitchers=heldout,
            purge_train=purge_train,
            season_col=season,
            seed=seed + 1000 + group,
        )

        print(
            f"[{group + 2}/{args.groups + 1}] group={group} pitchers={len(heldout):,}, "
            f"eval_rows={len(eval_frame):,}, removed_history={removed_rows:,}, train_rows={len(purge_train):,}"
        )

        p_purge = base.fit_predict(purge_train, eval_frame, target, FEATURES, params)
        p_control = base.fit_predict(control_train, eval_frame, target, FEATURES, params)

        m_full = core.metrics(y_eval, p_eval_full)
        m_purge = core.metrics(y_eval, p_purge)
        m_control = core.metrics(y_eval, p_control)

        row = {
            "group": group,
            "pitchers": int(len(heldout)),
            "eval_rows": int(len(eval_frame)),
            "removed_history_rows": int(removed_rows),
            "purge_train_rows": int(len(purge_train)),
            "control_train_rows": int(len(control_train)),
            "full_brier": float(m_full["brier"]),
            "purge_brier": float(m_purge["brier"]),
            "matched_control_brier": float(m_control["brier"]),
            "full_auc": float(m_full["auc"]),
            "purge_auc": float(m_purge["auc"]),
            "matched_control_auc": float(m_control["auc"]),
        }
        row["purge_minus_full"] = row["purge_brier"] - row["full_brier"]
        row["control_minus_full"] = row["matched_control_brier"] - row["full_brier"]
        row["purge_minus_control"] = row["purge_brier"] - row["matched_control_brier"]
        fold_rows.append(row)

        group_rows.extend(
            {
                "pitcher_id": pitcher,
                "group": group,
                "history_rows": int((train_full["_pitcher_key"] == pitcher).sum()),
                "validation_rows": int((valid["_pitcher_key"] == pitcher).sum()),
            }
            for pitcher in sorted(heldout)
        )

        stitched_full.append(p_eval_full)
        stitched_purge.append(p_purge)
        stitched_control.append(p_control)
        stitched_y.append(y_eval)
        del purge_train, control_train, p_purge, p_control
        gc.collect()

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / "fold_results.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "pitcher_groups.csv", index=False)

    yy = np.concatenate(stitched_y)
    pp_full = np.concatenate(stitched_full)
    pp_purge = np.concatenate(stitched_purge)
    pp_control = np.concatenate(stitched_control)
    agg = []
    for name, pp in [
        ("full_history", pp_full),
        ("purged_same_pitchers", pp_purge),
        ("matched_size_keeps_pitcher_history", pp_control),
    ]:
        agg.append({"protocol": name, **core.metrics(yy, pp)})
    aggregate = pd.DataFrame(agg)
    full_brier = float(aggregate.loc[aggregate["protocol"].eq("full_history"), "brier"].iloc[0])
    aggregate["delta_brier_vs_full"] = aggregate["brier"] - full_brier
    aggregate.to_csv(output_dir / "aggregate_seen_pitcher_results.csv", index=False)

    save_json(
        {
            "validation_year": int(args.validation_year),
            "groups": int(args.groups),
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices,
            "features": FEATURES,
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "interpretation": {
                "purge_minus_full": "effect of removing the evaluated pitchers' own historical rows plus reduced train size",
                "control_minus_full": "effect of reducing train size to the same season-by-season counts while keeping evaluated-pitcher history",
                "purge_minus_control": "best estimate here of pitcher-exposure/distribution effect beyond train-size loss; pitcher_id itself is not a canonical feature",
            },
        },
        output_dir / "run_config.json",
    )

    print("\n[Cohort composition]")
    print(cohort_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n[Controlled exposure folds: lower Brier is better]")
    print(
        folds[
            [
                "group",
                "pitchers",
                "eval_rows",
                "removed_history_rows",
                "full_brier",
                "purge_brier",
                "matched_control_brier",
                "purge_minus_full",
                "control_minus_full",
                "purge_minus_control",
            ]
        ].to_string(
            index=False,
            formatters={
                "full_brier": "{:.8f}".format,
                "purge_brier": "{:.8f}".format,
                "matched_control_brier": "{:.8f}".format,
                "purge_minus_full": "{:+.8f}".format,
                "control_minus_full": "{:+.8f}".format,
                "purge_minus_control": "{:+.8f}".format,
            },
        )
    )

    print("\n[Aggregate over all seen 2024 pitchers]")
    print(
        aggregate[["protocol", "brier", "competition_score", "auc", "prediction_std", "delta_brier_vs_full"]].to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "competition_score": "{:.2f}".format,
                "auc": "{:.5f}".format,
                "prediction_std": "{:.5f}".format,
                "delta_brier_vs_full": "{:+.8f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
