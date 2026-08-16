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
from src.utils import load_config, save_json

YEARS = np.asarray([2019, 2020, 2021, 2022, 2023, 2024], dtype=np.int16)
OLD_IDX = np.asarray([0, 1, 2, 3], dtype=np.int8)
RECENT_IDX = np.asarray([4, 5], dtype=np.int8)
CHANGE_YEARS = [2021, 2022, 2023]
BIN_SETTINGS = [4, 6, 8]

CANDIDATES: dict[str, tuple[str, str | None]] = {
    "fastball_rate_x_batter_hand": ("asof_pitcher_fastball_rate", "batter_hand"),
    "breaking_rate_x_batter_hand": ("asof_pitcher_breaking_rate", "batter_hand"),
    "eng_ps_recent_range_135": ("eng_ps_recent_range_135", None),
}


def _select_backend(requested: str):
    requested = requested.lower()
    if requested not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be one of auto/numpy/cupy")
    if requested in {"auto", "cupy"}:
        try:
            import cupy as cp

            _ = cp.asarray([1], dtype=cp.int8).sum().item()
            return cp, "cupy"
        except Exception as exc:
            if requested == "cupy":
                raise RuntimeError(f"CuPy backend requested but unavailable: {exc}") from exc
    return np, "numpy"


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return np.asarray(value)


def _numeric_codes(values, bins: int, xp):
    finite = xp.isfinite(values)
    finite_values = values[finite]
    if int(finite_values.size) == 0:
        raise ValueError("numeric candidate has no finite values")
    edges = xp.unique(xp.quantile(finite_values, xp.linspace(0.0, 1.0, bins + 1)))
    if int(edges.size) < 3:
        raise ValueError("numeric candidate has insufficient unique quantile edges")
    real_bins = int(edges.size) - 1
    codes = xp.full(values.shape, real_bins, dtype=xp.int16)  # final code is missing
    codes[finite] = xp.searchsorted(edges[1:-1], values[finite], side="right").astype(xp.int16)
    return codes, _to_numpy(edges).astype(float), real_bins + 1


def _categorical_codes(series: pd.Series, xp):
    tokens = series.astype("string").fillna("<MISSING>").astype(str)
    codes, levels = pd.factorize(tokens, sort=True)
    return xp.asarray(codes, dtype=xp.int16), [str(x) for x in levels.tolist()]


def _combine_codes(numeric_codes, n_numeric_groups: int, categorical_codes, levels, xp):
    if categorical_codes is None:
        return numeric_codes.astype(xp.int32), n_numeric_groups
    n_cat = len(levels)
    codes = numeric_codes.astype(xp.int32) * n_cat + categorical_codes.astype(xp.int32)
    return codes, n_numeric_groups * n_cat


def _game_type_controlled_residual(y, year_idx, gt_codes, n_gt: int, xp):
    key = year_idx.astype(xp.int32) * n_gt + gt_codes.astype(xp.int32)
    counts = xp.bincount(key, minlength=6 * n_gt).astype(xp.float64)
    sums = xp.bincount(key, weights=y.astype(xp.float64), minlength=6 * n_gt)
    means = xp.divide(sums, counts, out=xp.zeros_like(sums), where=counts > 0)
    return y - means[key]


def _profile_from_codes(*, response, year_idx, group_codes, n_groups: int, mask, center_by_season: bool, xp):
    valid = mask & (year_idx >= 0) & (year_idx < 6) & (group_codes >= 0)
    flat = year_idx[valid].astype(xp.int64) * n_groups + group_codes[valid].astype(xp.int64)
    resp = response[valid].astype(xp.float64)
    counts = xp.bincount(flat, minlength=6 * n_groups).reshape(6, n_groups).astype(xp.float64)
    sums = xp.bincount(flat, weights=resp, minlength=6 * n_groups).reshape(6, n_groups)
    means = xp.divide(sums, counts, out=xp.full_like(sums, xp.nan), where=counts > 0)
    if center_by_season:
        season_count = counts.sum(axis=1)
        season_sum = sums.sum(axis=1)
        season_mean = xp.divide(
            season_sum,
            season_count,
            out=xp.zeros_like(season_sum),
            where=season_count > 0,
        )
        means = means - season_mean[:, None]
    return _to_numpy(counts), _to_numpy(means)


def _era_aggregate(counts: np.ndarray, effects: np.ndarray, idx: np.ndarray):
    era_count = counts[idx].sum(axis=0)
    weighted = np.nansum(counts[idx] * effects[idx], axis=0)
    era_effect = np.divide(
        weighted,
        era_count,
        out=np.full_like(weighted, np.nan),
        where=era_count > 0,
    )
    return era_count, era_effect


def _internal_rmse(counts: np.ndarray, effects: np.ndarray, idx: np.ndarray, era_effect: np.ndarray) -> float:
    c = counts[idx]
    diff = effects[idx] - era_effect[None, :]
    mask = np.isfinite(diff) & (c > 0)
    denom = float(c[mask].sum())
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(c[mask] * diff[mask] ** 2) / denom))


def _era_shift(
    counts: np.ndarray,
    effects: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    *,
    min_era_count: int,
    min_effect_for_flip: float = 0.002,
) -> dict:
    left_count, left_effect = _era_aggregate(counts, effects, left_idx)
    right_count, right_effect = _era_aggregate(counts, effects, right_idx)
    support = (
        (left_count >= min_era_count)
        & (right_count >= min_era_count)
        & np.isfinite(left_effect)
        & np.isfinite(right_effect)
    )
    if not support.any():
        return {
            "supported_groups": 0,
            "shift_rmse": np.nan,
            "changepoint_ratio": np.nan,
            "sign_flip_rate": np.nan,
            "left_effect": left_effect,
            "right_effect": right_effect,
            "support": support,
        }

    weight = np.minimum(left_count[support], right_count[support])
    delta = right_effect[support] - left_effect[support]
    shift = float(np.sqrt(np.average(delta**2, weights=weight)))
    left_internal = _internal_rmse(counts, effects, left_idx, left_effect)
    right_internal = _internal_rmse(counts, effects, right_idx, right_effect)
    floor = max(
        left_internal if np.isfinite(left_internal) else 0.0,
        right_internal if np.isfinite(right_internal) else 0.0,
        0.002,
    )
    strong = (
        (np.abs(left_effect[support]) >= min_effect_for_flip)
        & (np.abs(right_effect[support]) >= min_effect_for_flip)
    )
    flips = strong & (left_effect[support] * right_effect[support] < 0)
    strong_weight = float(weight[strong].sum())
    flip_rate = float(weight[flips].sum() / strong_weight) if strong_weight > 0 else 0.0
    return {
        "supported_groups": int(support.sum()),
        "shift_rmse": shift,
        "changepoint_ratio": float(shift / floor),
        "sign_flip_rate": flip_rate,
        "left_effect": left_effect,
        "right_effect": right_effect,
        "support": support,
    }


def _best_changepoint(counts: np.ndarray, effects: np.ndarray, min_era_count: int):
    rows = []
    for change_year in CHANGE_YEARS:
        left = np.where(YEARS < change_year)[0]
        right = np.where(YEARS >= change_year)[0]
        metrics = _era_shift(counts, effects, left, right, min_era_count=min_era_count)
        rows.append((change_year, float(metrics["changepoint_ratio"]), float(metrics["shift_rmse"])))
    valid = [row for row in rows if np.isfinite(row[1])]
    if not valid:
        return np.nan, np.nan
    best = max(valid, key=lambda row: (row[1], row[2]))
    return int(best[0]), float(best[1])


def _recent_direction_consistency(counts: np.ndarray, effects: np.ndarray, min_abs_effect: float = 0.001) -> float:
    e23, e24 = effects[4], effects[5]
    weight = np.minimum(counts[4], counts[5])
    strong = (
        np.isfinite(e23)
        & np.isfinite(e24)
        & (weight > 0)
        & (np.abs(e23) >= min_abs_effect)
        & (np.abs(e24) >= min_abs_effect)
    )
    if not strong.any():
        return float("nan")
    same = np.sign(e23[strong]) == np.sign(e24[strong])
    return float(np.average(same.astype(float), weights=weight[strong]))


def _same_player_metrics(full: dict, same: dict):
    full_shift = float(full["shift_rmse"])
    same_shift = float(same["shift_rmse"])
    preservation = (
        same_shift / full_shift
        if np.isfinite(full_shift) and full_shift > 0 and np.isfinite(same_shift)
        else np.nan
    )
    support = np.asarray(full["support"]) & np.asarray(same["support"])
    if support.sum() < 3:
        return preservation, np.nan
    full_delta = np.asarray(full["right_effect"])[support] - np.asarray(full["left_effect"])[support]
    same_delta = np.asarray(same["right_effect"])[support] - np.asarray(same["left_effect"])[support]
    if np.std(full_delta) == 0 or np.std(same_delta) == 0:
        return preservation, np.nan
    return preservation, float(np.corrcoef(full_delta, same_delta)[0, 1])


def _candidate_verdict(rows: pd.DataFrame) -> dict[str, object]:
    candidate = str(rows["candidate"].iloc[0])
    result: dict[str, object] = {"candidate": candidate}
    flags: dict[str, bool] = {}
    for gt in ["R", "F"]:
        part = rows.loc[rows["game_type"].eq(gt)]
        if part.empty:
            flags[gt] = False
            continue
        median_shift = float(part["shift_2023_rmse"].median())
        median_ratio = float(part["changepoint_ratio_2023"].median())
        cp_share = float(part["best_change_year"].eq(2023).mean())
        recent = float(part["recent_direction_consistency"].median())
        same_corr = float(part["same_player_delta_correlation"].median())
        result[f"{gt}_median_shift"] = median_shift
        result[f"{gt}_median_ratio"] = median_ratio
        result[f"{gt}_cp2023_share"] = cp_share
        result[f"{gt}_median_recent_consistency"] = recent
        result[f"{gt}_median_same_corr"] = same_corr
        flags[gt] = (
            median_shift >= 0.004
            and median_ratio >= 1.2
            and cp_share >= 2.0 / 3.0
            and recent >= 0.55
            and (not np.isfinite(same_corr) or same_corr >= 0.45)
        )

    if flags.get("R") and flags.get("F"):
        result["verdict"] = "ROBUST_IN_BOTH_R_F"
    elif flags.get("R"):
        result["verdict"] = "ROBUST_R_SPECIFIC"
    elif flags.get("F"):
        result["verdict"] = "ROBUST_F_SPECIFIC"
    else:
        result["verdict"] = "BIN_SENSITIVE_OR_MIXED"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robustness validation for the strongest non-game_type temporal candidates. "
            "Runs R-only and F-only analyses separately and repeats 4/6/8 target-independent quantile bins. "
            "CuPy is used automatically when available; otherwise a vectorized NumPy bincount backend is used."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "cupy"])
    parser.add_argument("--min-era-count", type=int, default=300)
    parser.add_argument("--same-player-min-era-count", type=int, default=100)
    parser.add_argument("--same-player-min-old-seasons", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/regime_candidate_robustness")
    args = parser.parse_args()

    xp, backend = _select_backend(args.backend)
    config = load_config(ROOT / args.config)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, _ = atlas.recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)

    required = {"game_type", "pitcher_id", target_col, season_col} | {
        column
        for pair in CANDIDATES.values()
        for column in pair
        if column is not None
    }
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
    year_idx = xp.asarray(frame[season_col].map(year_map).fillna(-1).to_numpy(np.int16))
    y = xp.asarray(pd.to_numeric(frame[target_col], errors="raise").to_numpy(np.float64))
    gt_tokens = frame["game_type"].astype("string").fillna("<MISSING>").astype(str)
    gt_codes_np, gt_level_index = pd.factorize(gt_tokens, sort=True)
    gt_levels = [str(level) for level in gt_level_index.tolist()]
    if "R" not in gt_levels or "F" not in gt_levels:
        raise ValueError(f"Expected game_type levels R/F, got {gt_levels}")
    gt_codes = xp.asarray(gt_codes_np.astype(np.int16))
    same_mask = xp.asarray(same_mask_pd.to_numpy(bool))
    all_mask = xp.ones(len(frame), dtype=bool)
    gt_masks = {level: gt_codes == gt_levels.index(level) for level in ["R", "F"]}
    residual = _game_type_controlled_residual(y, year_idx, gt_codes, len(gt_levels), xp)

    print("[Regime Candidate Robustness]")
    print(f"  backend             : {backend.upper()}")
    if backend == "cupy":
        try:
            props = xp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"GPU")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            print(f"  device              : {name}")
        except Exception:
            pass
    print("  training            : NONE")
    print("  robustness checks   : R-only / F-only + global quantile bins 4/6/8")
    print(f"  game_type levels    : {gt_levels}")
    print(
        f"  same-player cohort  : {same_stats['pitchers']:,} pitchers / {same_stats['rows']:,} rows "
        f"({same_stats['row_fraction']:.1%})"
    )

    rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for candidate, (numeric_col, categorical_col) in CANDIDATES.items():
        numeric_values = xp.asarray(pd.to_numeric(frame[numeric_col], errors="coerce").to_numpy(np.float64))
        if categorical_col is not None:
            cat_codes, cat_levels = _categorical_codes(frame[categorical_col], xp)
        else:
            cat_codes, cat_levels = None, None
        print(f"\n[{candidate}]")
        metadata[candidate] = {}

        for bins in BIN_SETTINGS:
            numeric_codes, edges, n_numeric_groups = _numeric_codes(numeric_values, bins, xp)
            group_codes, n_groups = _combine_codes(
                numeric_codes,
                n_numeric_groups,
                cat_codes,
                cat_levels,
                xp,
            )
            metadata[candidate][str(bins)] = {
                "numeric_column": numeric_col,
                "edges": edges.tolist(),
                "categorical_column": categorical_col,
                "categorical_levels": cat_levels,
                "group_count": int(n_groups),
            }

            for gt in ["ALL", "R", "F"]:
                if gt == "ALL":
                    mask, response, center = all_mask, residual, False
                else:
                    mask, response, center = gt_masks[gt], y, True

                full_counts, full_effects = _profile_from_codes(
                    response=response,
                    year_idx=year_idx,
                    group_codes=group_codes,
                    n_groups=n_groups,
                    mask=mask,
                    center_by_season=center,
                    xp=xp,
                )
                same_counts, same_effects = _profile_from_codes(
                    response=response,
                    year_idx=year_idx,
                    group_codes=group_codes,
                    n_groups=n_groups,
                    mask=mask & same_mask,
                    center_by_season=center,
                    xp=xp,
                )
                full = _era_shift(
                    full_counts,
                    full_effects,
                    OLD_IDX,
                    RECENT_IDX,
                    min_era_count=args.min_era_count,
                )
                same = _era_shift(
                    same_counts,
                    same_effects,
                    OLD_IDX,
                    RECENT_IDX,
                    min_era_count=args.same_player_min_era_count,
                )
                best_year, best_ratio = _best_changepoint(
                    full_counts,
                    full_effects,
                    min_era_count=args.min_era_count,
                )
                same_pres, same_corr = _same_player_metrics(full, same)
                recent = _recent_direction_consistency(full_counts, full_effects)
                row = {
                    "candidate": candidate,
                    "bins": bins,
                    "game_type": gt,
                    "supported_groups": int(full["supported_groups"]),
                    "shift_2023_rmse": float(full["shift_rmse"]),
                    "changepoint_ratio_2023": float(full["changepoint_ratio"]),
                    "sign_flip_rate_2023": float(full["sign_flip_rate"]),
                    "best_change_year": best_year,
                    "best_changepoint_ratio": best_ratio,
                    "recent_direction_consistency": recent,
                    "same_player_shift_preservation": same_pres,
                    "same_player_delta_correlation": same_corr,
                }
                rows.append(row)
                print(
                    f"  bins={bins} gt={gt:<3} "
                    f"shift={row['shift_2023_rmse']:.5f} "
                    f"ratio={row['changepoint_ratio_2023']:.2f} "
                    f"flip={row['sign_flip_rate_2023']:.2f} "
                    f"cp={row['best_change_year']} recent={row['recent_direction_consistency']:.2f} "
                    f"sameCorr={row['same_player_delta_correlation']:.2f}"
                )

    detail = pd.DataFrame(rows)
    verdicts = pd.DataFrame([
        _candidate_verdict(part)
        for _, part in detail.groupby("candidate", sort=False)
    ])
    rank = {
        "ROBUST_IN_BOTH_R_F": 0,
        "ROBUST_R_SPECIFIC": 1,
        "ROBUST_F_SPECIFIC": 2,
        "BIN_SENSITIVE_OR_MIXED": 3,
    }
    verdicts["_rank"] = verdicts["verdict"].map(rank).fillna(9)
    verdicts = verdicts.sort_values(
        ["_rank", "R_median_shift", "F_median_shift"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns="_rank")

    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "robustness_detail.csv", index=False)
    verdicts.to_csv(out / "candidate_verdicts.csv", index=False)
    save_json(
        {
            "backend": backend,
            "bin_settings": BIN_SETTINGS,
            "same_player_cohort": same_stats,
            "bin_metadata": metadata,
            "notes": [
                "ALL uses y - E[y | season, game_type], matching the prior controlled analysis.",
                "R/F are analyzed independently and centered within season, so R/F mixture cannot create their shift.",
                "Global target-independent 4/6/8 quantile bins test sensitivity to arbitrary bin boundaries.",
                "Verdicts are screening labels, not causal claims.",
            ],
        },
        out / "run_config.json",
    )

    print("\n[Candidate Verdicts]")
    print(verdicts.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
