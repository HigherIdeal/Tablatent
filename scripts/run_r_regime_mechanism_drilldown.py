from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_regime_atlas as atlas
import run_regime_candidate_robustness as robust
from src.utils import load_config, save_json

YEARS = np.asarray([2019, 2020, 2021, 2022, 2023, 2024], dtype=np.int16)
OLD_IDX = np.asarray([0, 1, 2, 3], dtype=np.int8)
RECENT_IDX = np.asarray([4, 5], dtype=np.int8)
BIN_SETTINGS = [4, 6, 8]

# These are the two candidates that survived the R/F split and bin-robustness
# screen.  This script does not train a model; it exposes the actual within-R
# mechanism by season and checks whether the 2022->2023 break persists in 2024.
CANDIDATES: dict[str, tuple[str, str | None]] = {
    "fastball_rate_x_batter_hand": ("asof_pitcher_fastball_rate", "batter_hand"),
    "eng_ps_recent_range_135": ("eng_ps_recent_range_135", None),
}


def _era_mean(counts: np.ndarray, effects: np.ndarray, idx: np.ndarray) -> np.ndarray:
    c = counts[idx].sum(axis=0)
    weighted = np.nansum(counts[idx] * effects[idx], axis=0)
    return np.divide(weighted, c, out=np.full_like(weighted, np.nan), where=c > 0)


def _mechanism_table(
    counts: np.ndarray,
    effects: np.ndarray,
    *,
    min_old_count: int,
    min_recent_count: int,
) -> pd.DataFrame:
    old_count = counts[OLD_IDX].sum(axis=0)
    recent_count = counts[RECENT_IDX].sum(axis=0)
    old_mean = _era_mean(counts, effects, OLD_IDX)
    recent_mean = _era_mean(counts, effects, RECENT_IDX)
    support = (
        (old_count >= min_old_count)
        & (recent_count >= min_recent_count)
        & np.isfinite(old_mean)
        & np.isfinite(recent_mean)
    )

    rows = []
    for group in np.where(support)[0]:
        e19, e20, e21, e22, e23, e24 = effects[:, group]
        c19, c20, c21, c22, c23, c24 = counts[:, group]
        break_22_23 = e23 - e22 if np.isfinite(e22) and np.isfinite(e23) else np.nan
        post_change = e24 - e23 if np.isfinite(e23) and np.isfinite(e24) else np.nan
        regime_delta = recent_mean[group] - old_mean[group]
        # 1 means 2024 stayed at the 2023 side of the old-era mean; 0 means it
        # crossed back.  This is intentionally sign-based and easy to interpret.
        if np.isfinite(e24) and np.isfinite(old_mean[group]) and abs(regime_delta) >= 0.001:
            persistent_side = float(np.sign(e24 - old_mean[group]) == np.sign(regime_delta))
        else:
            persistent_side = np.nan
        rows.append(
            {
                "group_code": int(group),
                "old_count": int(old_count[group]),
                "recent_count": int(recent_count[group]),
                "old_mean": float(old_mean[group]),
                "recent_mean": float(recent_mean[group]),
                "regime_delta": float(regime_delta),
                "effect_2019": float(e19),
                "effect_2020": float(e20),
                "effect_2021": float(e21),
                "effect_2022": float(e22),
                "effect_2023": float(e23),
                "effect_2024": float(e24),
                "break_22_23": float(break_22_23),
                "change_23_24": float(post_change),
                "persistent_side_2024": persistent_side,
                "count_2019": int(c19),
                "count_2020": int(c20),
                "count_2021": int(c21),
                "count_2022": int(c22),
                "count_2023": int(c23),
                "count_2024": int(c24),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_regime_delta"] = result["regime_delta"].abs()
        result["abs_break_22_23"] = result["break_22_23"].abs()
        result = result.sort_values(["abs_regime_delta", "abs_break_22_23"], ascending=False)
    return result


def _group_labels(
    *,
    edges: np.ndarray,
    n_numeric_groups: int,
    category_levels: list[str] | None,
) -> dict[int, str]:
    real_bins = n_numeric_groups - 1  # final numeric code is <MISSING>
    numeric_labels = []
    for i in range(real_bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        close = "]" if i == real_bins - 1 else ")"
        numeric_labels.append(f"Q{i + 1}[{left:.4f},{right:.4f}{close}")
    numeric_labels.append("<MISSING>")

    labels: dict[int, str] = {}
    if category_levels is None:
        for i, label in enumerate(numeric_labels):
            labels[i] = label
        return labels

    n_cat = len(category_levels)
    for numeric_code, numeric_label in enumerate(numeric_labels):
        for cat_code, level in enumerate(category_levels):
            combined = numeric_code * n_cat + cat_code
            labels[combined] = f"{numeric_label} | batter_hand={level}"
    return labels


def _summary(table: pd.DataFrame) -> dict[str, float | int]:
    if table.empty:
        return {
            "supported_groups": 0,
            "weighted_abs_regime_delta": np.nan,
            "weighted_abs_break_22_23": np.nan,
            "weighted_abs_change_23_24": np.nan,
            "persistent_side_share_2024": np.nan,
        }
    weights = np.minimum(
        table["old_count"].to_numpy(np.float64),
        table["recent_count"].to_numpy(np.float64),
    )
    persistent = table["persistent_side_2024"].to_numpy(np.float64)
    finite_persistent = np.isfinite(persistent)
    return {
        "supported_groups": int(len(table)),
        "weighted_abs_regime_delta": float(np.average(np.abs(table["regime_delta"]), weights=weights)),
        "weighted_abs_break_22_23": float(np.average(np.abs(table["break_22_23"]), weights=weights)),
        "weighted_abs_change_23_24": float(np.average(np.abs(table["change_23_24"]), weights=weights)),
        "persistent_side_share_2024": (
            float(np.average(persistent[finite_persistent], weights=weights[finite_persistent]))
            if finite_persistent.any()
            else np.nan
        ),
    }


def _same_player_delta_correlation(full: pd.DataFrame, same: pd.DataFrame) -> float:
    if full.empty or same.empty:
        return float("nan")
    merged = full[["group_code", "regime_delta"]].merge(
        same[["group_code", "regime_delta"]], on="group_code", suffixes=("_full", "_same")
    )
    if len(merged) < 3:
        return float("nan")
    a = merged["regime_delta_full"].to_numpy(np.float64)
    b = merged["regime_delta_same"].to_numpy(np.float64)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _print_top(table: pd.DataFrame, top_groups: int) -> None:
    if table.empty:
        print("  no supported groups")
        return
    cols = [
        "group",
        "old_mean",
        "effect_2022",
        "effect_2023",
        "effect_2024",
        "break_22_23",
        "change_23_24",
        "regime_delta",
        "persistent_side_2024",
        "old_count",
        "recent_count",
    ]
    display = table.sort_values("abs_regime_delta", ascending=False).head(top_groups)[cols].copy()
    fmt = {
        col: (lambda x: f"{float(x):+.4f}" if pd.notna(x) else "nan")
        for col in [
            "old_mean",
            "effect_2022",
            "effect_2023",
            "effect_2024",
            "break_22_23",
            "change_23_24",
            "regime_delta",
        ]
    }
    fmt["persistent_side_2024"] = lambda x: "Y" if float(x) == 1.0 else ("N" if float(x) == 0.0 else "-")
    print(display.to_string(index=False, formatters=fmt))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R-only mechanism drilldown for the two robust non-game_type regime candidates. "
            "Reports actual within-R season-centered effects, the 2022->2023 break, 2024 persistence, "
            "bin robustness, and same-pitcher preservation. CuPy is used automatically when available."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "cupy"])
    parser.add_argument("--min-old-count", type=int, default=500)
    parser.add_argument("--min-recent-count", type=int, default=300)
    parser.add_argument("--same-player-min-old-count", type=int, default=150)
    parser.add_argument("--same-player-min-recent-count", type=int, default=100)
    parser.add_argument("--same-player-min-old-seasons", type=int, default=2)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs/r_regime_mechanism_drilldown")
    args = parser.parse_args()

    xp, backend = robust._select_backend(args.backend)
    config = load_config(ROOT / args.config)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, _ = atlas.recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)

    required = {"game_type", "pitcher_id", target_col, season_col}
    required |= {column for numeric, categorical in CANDIDATES.values() for column in (numeric, categorical) if column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    same_mask_pd, same_stats = atlas._same_player_mask(
        frame,
        pitcher_col="pitcher_id",
        season_col=season_col,
        min_old_seasons=args.same_player_min_old_seasons,
    )

    year_map = {year: idx for idx, year in enumerate(YEARS.tolist())}
    year_idx_np = frame[season_col].map(year_map).fillna(-1).to_numpy(np.int16)
    r_mask_np = frame["game_type"].astype(str).eq("R").to_numpy(bool)
    y_np = pd.to_numeric(frame[target_col], errors="raise").to_numpy(np.float64)
    same_np = same_mask_pd.to_numpy(bool)

    year_idx = xp.asarray(year_idx_np)
    y = xp.asarray(y_np)
    r_mask = xp.asarray(r_mask_np)
    same_mask = xp.asarray(same_np) & r_mask

    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("[R-only Regime Mechanism Drilldown]")
    print(f"  backend             : {backend.upper()}")
    print("  training            : NONE")
    print("  game_type           : R only")
    print("  effects             : success rate centered by R-season mean")
    print(f"  candidates          : {list(CANDIDATES)}")
    print(f"  bin settings        : {BIN_SETTINGS}")
    print(
        f"  same-player cohort  : {same_stats['pitchers']:,} pitchers / "
        f"{same_stats['rows']:,} all-type rows ({same_stats['row_fraction']:.1%})"
    )

    all_tables = []
    summary_rows = []
    metadata = {}

    for candidate, (numeric_col, categorical_col) in CANDIDATES.items():
        print(f"\n[{candidate}]")
        metadata[candidate] = {}
        numeric_values = xp.asarray(pd.to_numeric(frame[numeric_col], errors="coerce").to_numpy(np.float64))
        cat_codes = None
        cat_levels = None
        if categorical_col:
            cat_codes, cat_levels = robust._categorical_codes(frame[categorical_col], xp)

        for bins in BIN_SETTINGS:
            numeric_codes, edges, n_numeric_groups = robust._numeric_codes(numeric_values, bins, xp)
            group_codes, n_groups = robust._combine_codes(
                numeric_codes,
                n_numeric_groups,
                cat_codes,
                cat_levels,
                xp,
            )
            labels = _group_labels(
                edges=edges,
                n_numeric_groups=n_numeric_groups,
                category_levels=cat_levels,
            )

            full_counts, full_effects = robust._profile_from_codes(
                response=y,
                year_idx=year_idx,
                group_codes=group_codes,
                n_groups=n_groups,
                mask=r_mask,
                center_by_season=True,
                xp=xp,
            )
            same_counts, same_effects = robust._profile_from_codes(
                response=y,
                year_idx=year_idx,
                group_codes=group_codes,
                n_groups=n_groups,
                mask=same_mask,
                center_by_season=True,
                xp=xp,
            )

            full_table = _mechanism_table(
                full_counts,
                full_effects,
                min_old_count=args.min_old_count,
                min_recent_count=args.min_recent_count,
            )
            same_table = _mechanism_table(
                same_counts,
                same_effects,
                min_old_count=args.same_player_min_old_count,
                min_recent_count=args.same_player_min_recent_count,
            )
            if not full_table.empty:
                full_table.insert(0, "candidate", candidate)
                full_table.insert(1, "bins", bins)
                full_table.insert(2, "cohort", "all_R")
                full_table["group"] = full_table["group_code"].map(labels)
                all_tables.append(full_table)
            if not same_table.empty:
                same_table.insert(0, "candidate", candidate)
                same_table.insert(1, "bins", bins)
                same_table.insert(2, "cohort", "same_player_R")
                same_table["group"] = same_table["group_code"].map(labels)
                all_tables.append(same_table)

            stats = _summary(full_table)
            stats.update(
                {
                    "candidate": candidate,
                    "bins": bins,
                    "same_player_delta_correlation": _same_player_delta_correlation(full_table, same_table),
                }
            )
            summary_rows.append(stats)
            metadata[candidate][str(bins)] = {
                "numeric_column": numeric_col,
                "categorical_column": categorical_col,
                "edges": edges.tolist(),
                "category_levels": cat_levels,
            }
            print(
                f"  bins={bins} supported={stats['supported_groups']:2d} "
                f"regimeDelta={stats['weighted_abs_regime_delta']:.5f} "
                f"break22_23={stats['weighted_abs_break_22_23']:.5f} "
                f"postMove23_24={stats['weighted_abs_change_23_24']:.5f} "
                f"persist={stats['persistent_side_share_2024']:.2f} "
                f"sameCorr={stats['same_player_delta_correlation']:.2f}"
            )
            if bins == 6:
                _print_top(full_table, args.top_groups)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "summary.csv", index=False)
    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(out / "group_mechanisms.csv", index=False)
    save_json(
        {
            "backend": backend,
            "game_type": "R",
            "bin_settings": BIN_SETTINGS,
            "same_player_cohort": same_stats,
            "candidate_metadata": metadata,
            "notes": [
                "Effects are centered by the R-only season target mean, so they expose within-R conditional structure.",
                "break_22_23 is effect_2023 - effect_2022; regime_delta compares 2023-2024 with 2019-2022.",
                "persistent_side_2024 asks whether 2024 remains on the same side of the old-era mean as the post-2023 shift.",
                "Same-player correlation checks whether groupwise regime deltas are reproduced among pitchers spanning old and recent eras.",
            ],
        },
        out / "run_metadata.json",
    )

    print("\n[Mechanism Summary]")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
