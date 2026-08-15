from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.data import load_frame
from src.utils import load_config
import run_trackman_mapping_feasibility as base


DEFAULT_CONTEXT = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
]

MAIN_MIX_COLS = [
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _split_counts(counts: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = counts["count"].to_numpy(np.int64)
    a = rng.binomial(n=n, p=0.5)
    b = n - a
    left = counts.copy()
    right = counts.copy()
    left["count"] = a
    right["count"] = b
    left = left[left["count"] > 0].reset_index(drop=True)
    right = right[right["count"] > 0].reset_index(drop=True)
    return left, right


def _totals_from_counts(counts: pd.DataFrame, season_col: str, pitcher_col: str) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame(columns=[season_col, pitcher_col, "rows"])
    return (
        counts.groupby([season_col, pitcher_col], sort=False)["count"]
        .sum()
        .rename("rows")
        .reset_index()
    )


def _context_similarity(
    main_counts: pd.DataFrame,
    main_totals: pd.DataFrame,
    tm_counts: pd.DataFrame,
    tm_totals: pd.DataFrame,
    *,
    main_pitcher_col: str,
    tm_pitcher_col: str,
    hash_dim: int,
) -> tuple[list[str], list[str], np.ndarray]:
    main_ids = sorted(main_totals[main_pitcher_col].astype(str).unique().tolist())
    tm_ids = sorted(tm_totals[tm_pitcher_col].astype(str).unique().tolist())
    if not main_ids or not tm_ids:
        return main_ids, tm_ids, np.empty((len(main_ids), len(tm_ids)), dtype=np.float64)
    main_matrix = base._dense_hashed_matrix(main_counts, main_ids, main_pitcher_col, hash_dim=hash_dim)
    tm_matrix = base._dense_hashed_matrix(tm_counts, tm_ids, tm_pitcher_col, hash_dim=hash_dim)
    return main_ids, tm_ids, base._tfidf_cosine(main_matrix, tm_matrix)


def _hungarian_map(main_ids: list[str], tm_ids: list[str], score: np.ndarray, *, season: int) -> pd.DataFrame:
    if score.size == 0:
        return pd.DataFrame()
    safe = np.nan_to_num(score, nan=-1e6, neginf=-1e6, posinf=1e6)
    row_idx, col_idx = linear_sum_assignment(-safe)
    local_best = np.argmax(safe, axis=1)
    rows = []
    for r, c in zip(row_idx, col_idx):
        row = safe[r]
        order = np.argsort(-row)
        best_local = int(order[0])
        second_local = int(order[1]) if len(order) > 1 else best_local
        rows.append(
            {
                "season": int(season),
                "main_pitcher_id": str(main_ids[r]),
                "best_trackman_id": str(tm_ids[c]),
                "score": float(safe[r, c]),
                "local_best_score": float(row[best_local]),
                "local_second_score": float(row[second_local]),
                "local_margin": float(row[best_local] - row[second_local]),
                "is_local_best": bool(c == local_best[r]),
            }
        )
    return pd.DataFrame(rows)


def _latest_main_pitchmix(frame: pd.DataFrame, *, pitcher_col: str, season_col: str) -> pd.DataFrame:
    needed = [pitcher_col, season_col, "asof_pitcher_n"] + MAIN_MIX_COLS
    if any(c not in frame.columns for c in needed):
        return pd.DataFrame(columns=[season_col, pitcher_col, "fastball", "breaking", "offspeed"])
    x = frame[needed].copy()
    x["asof_pitcher_n"] = pd.to_numeric(x["asof_pitcher_n"], errors="coerce")
    for c in MAIN_MIX_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.sort_values([season_col, pitcher_col, "asof_pitcher_n"], kind="stable")
    x = x.groupby([season_col, pitcher_col], sort=False).tail(1)
    out = x[[season_col, pitcher_col] + MAIN_MIX_COLS].rename(
        columns={
            MAIN_MIX_COLS[0]: "fastball",
            MAIN_MIX_COLS[1]: "breaking",
            MAIN_MIX_COLS[2]: "offspeed",
        }
    )
    return out.reset_index(drop=True)


def _canonical_pitch_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    if s in {"F", "FB", "FAST", "FASTBALL"} or "FASTBALL" in s:
        return "fastball"
    if s in {"B", "BRK", "BREAKING", "BREAKINGBALL"} or "BREAK" in s:
        return "breaking"
    if s in {"O", "OS", "OFF", "OFFSPEED"} or "OFFSPEED" in s:
        return "offspeed"
    return None


def _trackman_pitchmix(frame: pd.DataFrame, *, pitcher_col: str, season_col: str) -> tuple[pd.DataFrame, float]:
    if "pitch_type_group" not in frame.columns:
        return pd.DataFrame(columns=[season_col, pitcher_col, "fastball", "breaking", "offspeed"]), 0.0
    x = frame[[season_col, pitcher_col, "pitch_type_group"]].copy()
    x["canonical"] = x["pitch_type_group"].map(_canonical_pitch_group)
    recognized = float(x["canonical"].notna().mean()) if len(x) else 0.0
    x = x.dropna(subset=["canonical"])
    if x.empty:
        return pd.DataFrame(columns=[season_col, pitcher_col, "fastball", "breaking", "offspeed"]), recognized
    counts = (
        x.groupby([season_col, pitcher_col, "canonical"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    for c in ["fastball", "breaking", "offspeed"]:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["fastball", "breaking", "offspeed"]].astype(float)
    denom = counts.sum(axis=1).replace(0, np.nan)
    rates = counts.div(denom, axis=0).reset_index()
    return rates, recognized


def _mix_similarity_matrix(
    main_ids: list[str],
    tm_ids: list[str],
    main_mix: pd.DataFrame,
    tm_mix: pd.DataFrame,
    *,
    main_pitcher_col: str,
    tm_pitcher_col: str,
) -> np.ndarray:
    m = main_mix.set_index(main_pitcher_col)[["fastball", "breaking", "offspeed"]] if not main_mix.empty else pd.DataFrame()
    t = tm_mix.set_index(tm_pitcher_col)[["fastball", "breaking", "offspeed"]] if not tm_mix.empty else pd.DataFrame()
    out = np.full((len(main_ids), len(tm_ids)), np.nan, dtype=np.float64)
    if m.empty or t.empty:
        return out
    for i, mid in enumerate(main_ids):
        if mid not in m.index:
            continue
        a = np.asarray(m.loc[mid], dtype=float)
        if a.ndim > 1:
            a = a[-1]
        if not np.all(np.isfinite(a)):
            continue
        asum = float(a.sum())
        if asum <= 0:
            continue
        a = a / asum
        for j, tid in enumerate(tm_ids):
            if tid not in t.index:
                continue
            b = np.asarray(t.loc[tid], dtype=float)
            if b.ndim > 1:
                b = b[-1]
            if not np.all(np.isfinite(b)):
                continue
            bsum = float(b.sum())
            if bsum <= 0:
                continue
            b = b / bsum
            out[i, j] = 1.0 - 0.5 * float(np.abs(a - b).sum())
    return out


def _legacy_score(context: np.ndarray, pitchmix: np.ndarray) -> np.ndarray:
    out = context.astype(np.float64, copy=True)
    mask = np.isfinite(pitchmix)
    out[mask] = 0.85 * context[mask] + 0.15 * pitchmix[mask]
    return out


def _mapping_agreement(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, float]:
    if a.empty or b.empty:
        return 0, np.nan
    x = a[["main_pitcher_id", "best_trackman_id"]].merge(
        b[["main_pitcher_id", "best_trackman_id"]], on="main_pitcher_id", suffixes=("_a", "_b")
    )
    if x.empty:
        return 0, np.nan
    return int(len(x)), float((x["best_trackman_id_a"] == x["best_trackman_id_b"]).mean())


def _cross_season_consistency(maps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, g in maps.groupby("main_pitcher_id", sort=False):
        if len(g) < 2:
            continue
        vc = g["best_trackman_id"].value_counts()
        mode_count = int(vc.iloc[0])
        rows.append(
            {
                "main_pitcher_id": str(mid),
                "seasons": int(len(g)),
                "mode_trackman_id": str(vc.index[0]),
                "mode_seasons": mode_count,
                "consistency": float(mode_count / len(g)),
                "all_same": bool(mode_count == len(g)),
            }
        )
    return pd.DataFrame(rows)


def _permutation_cross_season_null(maps: pd.DataFrame, *, repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_cols = ["season", "main_pitcher_id", "best_trackman_id"]
    base_df = maps[base_cols].copy()
    scores = []
    for _ in range(repetitions):
        pieces = []
        for _, g in base_df.groupby("season", sort=False):
            z = g.copy()
            z["best_trackman_id"] = rng.permutation(z["best_trackman_id"].to_numpy())
            pieces.append(z)
        shuffled = pd.concat(pieces, ignore_index=True)
        c = _cross_season_consistency(shuffled)
        scores.append(float(c["consistency"].mean()) if not c.empty else np.nan)
    return np.asarray(scores, dtype=float)


def _matched_pitchmix_vs_null(
    mapping: pd.DataFrame,
    main_ids: list[str],
    tm_ids: list[str],
    pitchmix: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    main_pos = {v: i for i, v in enumerate(main_ids)}
    tm_pos = {v: i for i, v in enumerate(tm_ids)}
    values = []
    valid_rows = []
    for _, row in mapping.iterrows():
        i = main_pos.get(str(row["main_pitcher_id"]))
        j = tm_pos.get(str(row["best_trackman_id"]))
        if i is None or j is None or not np.isfinite(pitchmix[i, j]):
            continue
        values.append(float(pitchmix[i, j]))
        valid_rows.append(i)
    if not values or not tm_ids:
        return {"n": 0, "matched_mean": np.nan, "null_mean": np.nan, "null_p95": np.nan, "excess_vs_null_mean": np.nan}
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(repetitions):
        js = rng.integers(0, len(tm_ids), size=len(valid_rows))
        vals = [pitchmix[i, j] for i, j in zip(valid_rows, js) if np.isfinite(pitchmix[i, j])]
        if vals:
            null.append(float(np.mean(vals)))
    null_arr = np.asarray(null, dtype=float)
    matched = float(np.mean(values))
    return {
        "n": int(len(values)),
        "matched_mean": matched,
        "null_mean": float(np.nanmean(null_arr)) if len(null_arr) else np.nan,
        "null_p95": float(np.nanpercentile(null_arr, 95)) if len(null_arr) else np.nan,
        "excess_vs_null_mean": matched - float(np.nanmean(null_arr)) if len(null_arr) else np.nan,
    }


def _summary_verdict(summary: dict) -> str:
    signals = 0
    if summary.get("cross_season_excess_vs_null95", -np.inf) >= 0.08:
        signals += 1
    if summary.get("mean_split_same_pair_rate", -np.inf) >= 0.20:
        signals += 1
    if summary.get("mean_context_map_pitchmix_excess", -np.inf) >= 0.08:
        signals += 1
    if signals >= 2:
        return "CORRESPONDENCE_SIGNAL_WORTH_USING"
    if signals == 1:
        return "SOME_SIGNAL_BUT_NOT_RELIABLE_IDENTITY_MAPPING"
    return "NO_RELIABLE_CORRESPONDENCE_SIGNAL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only reproduction of the project's earlier Trackman mapping idea. "
            "It does not invent a new mapper and does not assume anonymous IDs are the same person."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trackman", default="data/raw/trackman_history.csv")
    parser.add_argument("--trackman-pitcher-col", default=None)
    parser.add_argument("--context-cols", default=None, help="comma-separated override")
    parser.add_argument("--hash-dim", type=int, default=2048)
    parser.add_argument("--null-repetitions", type=int, default=200)
    parser.add_argument("--output-dir", default="outputs/legacy_mapping_validation")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    seed = int(config.get("seed", 42))
    season_col = config["data"]["season_col"]
    main_pitcher_col = "pitcher_id"
    train = load_frame(config).copy()
    tm_path = ROOT / args.trackman
    if not tm_path.exists():
        raise FileNotFoundError(f"Trackman file not found: {tm_path}")
    tm = pd.read_csv(tm_path, low_memory=False)
    tm_pitcher_col = args.trackman_pitcher_col or base._detect_trackman_pitcher_column(tm.columns.tolist())
    tm_season_col = base._detect_season_column(tm.columns.tolist(), preferred=season_col)

    train[main_pitcher_col] = _normalize_id(train[main_pitcher_col])
    tm[tm_pitcher_col] = _normalize_id(tm[tm_pitcher_col])
    train[season_col] = pd.to_numeric(train[season_col], errors="coerce")
    tm[tm_season_col] = pd.to_numeric(tm[tm_season_col], errors="coerce")
    train = train.dropna(subset=[season_col])
    tm = tm.dropna(subset=[tm_season_col])
    train[season_col] = train[season_col].astype(int)
    tm[tm_season_col] = tm[tm_season_col].astype(int)

    if args.context_cols:
        context_cols = [c.strip() for c in args.context_cols.split(",") if c.strip()]
    else:
        context_cols = [c for c in DEFAULT_CONTEXT if c in train.columns and c in tm.columns]
    if len(context_cols) < 5:
        raise RuntimeError(f"Too few shared context columns: {context_cols}")

    common_seasons = sorted(set(train[season_col].unique()) & set(tm[tm_season_col].unique()))
    main_mix_all = _latest_main_pitchmix(train, pitcher_col=main_pitcher_col, season_col=season_col)
    tm_mix_all, recognized_mix_fraction = _trackman_pitchmix(tm, pitcher_col=tm_pitcher_col, season_col=tm_season_col)

    print("[Legacy Mapping Validation]")
    print(f"  common seasons       : {common_seasons}")
    print(f"  context columns      : {context_cols}")
    print("  legacy score         : 0.85 * context cosine + 0.15 * pitch-mix similarity")
    print("  assignment           : global Hungarian one-to-one")
    print("  hand encoding        : NOT used as a filter; undocumented main 1/2 coding is kept out of the evidence")
    print(f"  Trackman mix coverage: {recognized_mix_fraction:.3f}")
    print("  validation only      : split reproducibility + cross-season persistence + independent pitchmix check")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    context_maps = []
    legacy_maps = []
    season_rows = []

    for season_idx, season in enumerate(common_seasons):
        main_s = train[train[season_col] == season]
        tm_s = tm[tm[tm_season_col] == season]
        main_counts, main_totals = base._make_profile_table(
            main_s, pitcher_col=main_pitcher_col, season_col=season_col, context_columns=context_cols
        )
        tm_counts, tm_totals = base._make_profile_table(
            tm_s, pitcher_col=tm_pitcher_col, season_col=tm_season_col, context_columns=context_cols
        )
        main_ids, tm_ids, context = _context_similarity(
            main_counts, main_totals, tm_counts, tm_totals,
            main_pitcher_col=main_pitcher_col, tm_pitcher_col=tm_pitcher_col, hash_dim=args.hash_dim,
        )
        main_mix = main_mix_all[main_mix_all[season_col] == season].drop(columns=[season_col], errors="ignore")
        tm_mix = tm_mix_all[tm_mix_all[tm_season_col] == season].drop(columns=[tm_season_col], errors="ignore")
        mix = _mix_similarity_matrix(
            main_ids, tm_ids, main_mix, tm_mix,
            main_pitcher_col=main_pitcher_col, tm_pitcher_col=tm_pitcher_col,
        )
        legacy = _legacy_score(context, mix)
        cmap = _hungarian_map(main_ids, tm_ids, context, season=season)
        lmap = _hungarian_map(main_ids, tm_ids, legacy, season=season)
        context_maps.append(cmap)
        legacy_maps.append(lmap)

        ma, mb = _split_counts(main_counts, seed=seed + 1000 + season_idx)
        ta, tb = _split_counts(tm_counts, seed=seed + 2000 + season_idx)
        mta = _totals_from_counts(ma, season_col, main_pitcher_col)
        mtb = _totals_from_counts(mb, season_col, main_pitcher_col)
        tta = _totals_from_counts(ta, tm_season_col, tm_pitcher_col)
        ttb = _totals_from_counts(tb, tm_season_col, tm_pitcher_col)
        a_main, a_tm, a_score = _context_similarity(
            ma, mta, ta, tta, main_pitcher_col=main_pitcher_col, tm_pitcher_col=tm_pitcher_col, hash_dim=args.hash_dim
        )
        b_main, b_tm, b_score = _context_similarity(
            mb, mtb, tb, ttb, main_pitcher_col=main_pitcher_col, tm_pitcher_col=tm_pitcher_col, hash_dim=args.hash_dim
        )
        amap = _hungarian_map(a_main, a_tm, a_score, season=season)
        bmap = _hungarian_map(b_main, b_tm, b_score, season=season)
        split_n, split_rate = _mapping_agreement(amap, bmap)
        pitch_check = _matched_pitchmix_vs_null(
            cmap, main_ids, tm_ids, mix,
            repetitions=args.null_repetitions, seed=seed + 3000 + season_idx,
        )

        season_rows.append(
            {
                "season": int(season),
                "main_pitchers": len(main_ids),
                "trackman_pitchers": len(tm_ids),
                "context_median_score": float(cmap["score"].median()) if not cmap.empty else np.nan,
                "context_local_best_fraction": float(cmap["is_local_best"].mean()) if not cmap.empty else np.nan,
                "context_median_margin": float(cmap["local_margin"].median()) if not cmap.empty else np.nan,
                "legacy_median_score": float(lmap["score"].median()) if not lmap.empty else np.nan,
                "legacy_local_best_fraction": float(lmap["is_local_best"].mean()) if not lmap.empty else np.nan,
                "split_common_pitchers": split_n,
                "split_same_pair_rate": split_rate,
                "context_map_pitchmix_n": pitch_check["n"],
                "context_map_pitchmix_mean": pitch_check["matched_mean"],
                "pitchmix_null_mean": pitch_check["null_mean"],
                "pitchmix_null_p95": pitch_check["null_p95"],
                "pitchmix_excess_vs_null_mean": pitch_check["excess_vs_null_mean"],
            }
        )
        print(
            f"[{season}] context={season_rows[-1]['context_median_score']:.3f} "
            f"localbest={season_rows[-1]['context_local_best_fraction']:.3f} "
            f"split_same={split_rate:.3f} "
            f"pitchmix_excess={pitch_check['excess_vs_null_mean']:.3f}"
        )

    context_all = pd.concat(context_maps, ignore_index=True) if context_maps else pd.DataFrame()
    legacy_all = pd.concat(legacy_maps, ignore_index=True) if legacy_maps else pd.DataFrame()
    season_df = pd.DataFrame(season_rows)
    context_cross = _cross_season_consistency(context_all)
    legacy_cross = _cross_season_consistency(legacy_all)
    null = _permutation_cross_season_null(
        legacy_all, repetitions=args.null_repetitions, seed=seed + 9000
    ) if not legacy_all.empty else np.asarray([], dtype=float)

    actual_cross = float(legacy_cross["consistency"].mean()) if not legacy_cross.empty else np.nan
    null_mean = float(np.nanmean(null)) if len(null) else np.nan
    null_p95 = float(np.nanpercentile(null, 95)) if len(null) else np.nan
    summary = {
        "common_seasons": [int(x) for x in common_seasons],
        "context_columns": context_cols,
        "trackman_pitch_type_recognized_fraction": recognized_mix_fraction,
        "mean_split_same_pair_rate": float(season_df["split_same_pair_rate"].mean()) if not season_df.empty else np.nan,
        "mean_context_map_pitchmix_excess": float(season_df["pitchmix_excess_vs_null_mean"].mean()) if not season_df.empty else np.nan,
        "context_cross_season_mean_consistency": float(context_cross["consistency"].mean()) if not context_cross.empty else np.nan,
        "legacy_cross_season_mean_consistency": actual_cross,
        "cross_season_null_mean": null_mean,
        "cross_season_null_p95": null_p95,
        "cross_season_excess_vs_null95": actual_cross - null_p95 if np.isfinite(actual_cross) and np.isfinite(null_p95) else np.nan,
        "legacy_cross_season_all_same_fraction": float(legacy_cross["all_same"].mean()) if not legacy_cross.empty else np.nan,
    }
    summary["verdict"] = _summary_verdict(summary)

    season_df.to_csv(output_dir / "season_validation.csv", index=False)
    context_all.to_csv(output_dir / "context_only_mapping.csv", index=False)
    legacy_all.to_csv(output_dir / "legacy_85_15_mapping.csv", index=False)
    context_cross.to_csv(output_dir / "context_cross_season.csv", index=False)
    legacy_cross.to_csv(output_dir / "legacy_cross_season.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[Summary]")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nInterpretation:")
    print("  - split_same_pair_rate: same mapping survives an independent half-sample of pitch contexts")
    print("  - pitchmix_excess: context-only mapping independently predicts Trackman pitch-mix similarity")
    print("  - cross-season excess: legacy Hungarian pairing persists across seasons above a permuted null")
    print("  These are plausibility checks only; they are not ground-truth identity labels.")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
