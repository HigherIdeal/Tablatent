from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_frame
from src.utils import load_config


DEFAULT_CONTEXT_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
]

NUMERIC_CONTEXT_COLUMNS = {
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
}


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _canonical_context(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column]
        if column in NUMERIC_CONTEXT_COLUMNS:
            values = pd.to_numeric(series, errors="coerce")
            rounded = values.round(6)
            out[column] = rounded.astype("Float64").astype("string").fillna("<MISSING>")
        else:
            out[column] = (
                series.astype("string")
                .str.strip()
                .str.upper()
                .fillna("<MISSING>")
            )
    return out


def _context_tokens(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    canonical = _canonical_context(frame, columns)
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(np.uint64)


def _detect_trackman_pitcher_column(columns: list[str]) -> str:
    preferred = [
        "pitcher_trackman_id",
        "trackman_pitcher_id",
        "pitcher_id",
    ]
    for column in preferred:
        if column in columns:
            return column
    candidates = [
        column
        for column in columns
        if "pitcher" in column.lower() and "id" in column.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "Could not uniquely detect Trackman pitcher id column. "
        f"Candidates={candidates}. Pass --trackman-pitcher-col explicitly."
    )


def _detect_season_column(columns: list[str], preferred: str = "season") -> str:
    if preferred in columns:
        return preferred
    candidates = [c for c in columns if c.lower() in {"season", "year"}]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Could not detect season column from {candidates}")


def _make_profile_table(
    frame: pd.DataFrame,
    *,
    pitcher_col: str,
    season_col: str,
    context_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame[[pitcher_col, season_col] + context_columns].copy()
    work[pitcher_col] = _normalize_id(work[pitcher_col])
    work[season_col] = pd.to_numeric(work[season_col], errors="coerce")
    work = work.dropna(subset=[season_col])
    work[season_col] = work[season_col].astype(int)
    work["context_token"] = _context_tokens(work, context_columns)

    counts = (
        work.groupby([season_col, pitcher_col, "context_token"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        counts.groupby([season_col, pitcher_col], sort=False)["count"]
        .sum()
        .rename("rows")
        .reset_index()
    )
    return counts, totals


def _dense_hashed_matrix(
    counts: pd.DataFrame,
    ids: list[str],
    pitcher_col: str,
    *,
    hash_dim: int,
) -> np.ndarray:
    id_to_row = {pitcher_id: idx for idx, pitcher_id in enumerate(ids)}
    matrix = np.zeros((len(ids), hash_dim), dtype=np.float32)
    if counts.empty or not ids:
        return matrix

    rows = counts[pitcher_col].map(id_to_row).to_numpy()
    valid = pd.notna(rows)
    rows = rows[valid].astype(np.int64)
    buckets = (
        counts.loc[valid, "context_token"].to_numpy(np.uint64) % np.uint64(hash_dim)
    ).astype(np.int64)
    values = counts.loc[valid, "count"].to_numpy(np.float32)
    np.add.at(matrix, (rows, buckets), values)
    return matrix


def _tfidf_cosine(main_matrix: np.ndarray, tm_matrix: np.ndarray) -> np.ndarray:
    stacked_presence = np.vstack([main_matrix > 0, tm_matrix > 0])
    document_frequency = stacked_presence.sum(axis=0).astype(np.float64)
    n_documents = stacked_presence.shape[0]
    idf = np.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0

    main = np.log1p(main_matrix.astype(np.float64)) * idf
    tm = np.log1p(tm_matrix.astype(np.float64)) * idf
    main_norm = np.linalg.norm(main, axis=1, keepdims=True)
    tm_norm = np.linalg.norm(tm, axis=1, keepdims=True)
    main /= np.maximum(main_norm, 1e-12)
    tm /= np.maximum(tm_norm, 1e-12)
    return main @ tm.T


def _exact_overlap(
    main_counts: dict[int, int],
    tm_counts: dict[int, int],
) -> tuple[float, float, float]:
    if not main_counts or not tm_counts:
        return 0.0, 0.0, 0.0
    if len(main_counts) > len(tm_counts):
        small, large = tm_counts, main_counts
    else:
        small, large = main_counts, tm_counts
    intersection = 0
    for token, count in small.items():
        other = large.get(token)
        if other is not None:
            intersection += min(int(count), int(other))

    main_total = float(sum(main_counts.values()))
    tm_total = float(sum(tm_counts.values()))
    min_total = min(main_total, tm_total)
    union = main_total + tm_total - intersection
    overlap_coefficient = float(intersection / min_total) if min_total > 0 else 0.0
    jaccard = float(intersection / union) if union > 0 else 0.0
    count_ratio = float(min(main_total, tm_total) / max(main_total, tm_total)) if max(main_total, tm_total) > 0 else 0.0
    return overlap_coefficient, jaccard, count_ratio


def _profile_dicts(
    counts: pd.DataFrame,
    pitcher_col: str,
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for pitcher_id, group in counts.groupby(pitcher_col, sort=False):
        result[str(pitcher_id)] = {
            int(token): int(count)
            for token, count in zip(group["context_token"], group["count"])
        }
    return result


def _score_one_season(
    main_counts: pd.DataFrame,
    main_totals: pd.DataFrame,
    tm_counts: pd.DataFrame,
    tm_totals: pd.DataFrame,
    *,
    main_pitcher_col: str,
    tm_pitcher_col: str,
    hash_dim: int,
    top_k: int,
) -> pd.DataFrame:
    main_ids = sorted(main_totals[main_pitcher_col].astype(str).unique().tolist())
    tm_ids = sorted(tm_totals[tm_pitcher_col].astype(str).unique().tolist())
    if not main_ids or not tm_ids:
        return pd.DataFrame()

    main_matrix = _dense_hashed_matrix(main_counts, main_ids, main_pitcher_col, hash_dim=hash_dim)
    tm_matrix = _dense_hashed_matrix(tm_counts, tm_ids, tm_pitcher_col, hash_dim=hash_dim)
    cosine = _tfidf_cosine(main_matrix, tm_matrix)

    main_profiles = _profile_dicts(main_counts, main_pitcher_col)
    tm_profiles = _profile_dicts(tm_counts, tm_pitcher_col)
    reverse_best = np.argmax(cosine, axis=0)
    k = min(top_k, len(tm_ids))

    rows: list[dict] = []
    for main_idx, main_id in enumerate(main_ids):
        if k == len(tm_ids):
            candidate_indices = np.argsort(-cosine[main_idx])[:k]
        else:
            partial = np.argpartition(-cosine[main_idx], kth=k - 1)[:k]
            candidate_indices = partial[np.argsort(-cosine[main_idx, partial])]

        candidate_rows = []
        for tm_idx in candidate_indices:
            tm_id = tm_ids[int(tm_idx)]
            overlap, jaccard, count_ratio = _exact_overlap(
                main_profiles.get(main_id, {}), tm_profiles.get(tm_id, {})
            )
            coarse = float(cosine[main_idx, tm_idx])
            combined = 0.50 * coarse + 0.35 * overlap + 0.15 * jaccard
            candidate_rows.append(
                {
                    "tm_id": tm_id,
                    "tm_idx": int(tm_idx),
                    "coarse_cosine": coarse,
                    "overlap_coefficient": overlap,
                    "weighted_jaccard": jaccard,
                    "count_ratio": count_ratio,
                    "combined_score": combined,
                }
            )
        candidate_rows.sort(key=lambda x: x["combined_score"], reverse=True)
        best = candidate_rows[0]
        second = candidate_rows[1] if len(candidate_rows) > 1 else None
        second_score = float(second["combined_score"]) if second else float("nan")
        rows.append(
            {
                "main_pitcher_id": main_id,
                "best_trackman_id": best["tm_id"],
                "best_score": float(best["combined_score"]),
                "second_score": second_score,
                "margin": float(best["combined_score"] - second_score) if second else float("nan"),
                "coarse_cosine": float(best["coarse_cosine"]),
                "overlap_coefficient": float(best["overlap_coefficient"]),
                "weighted_jaccard": float(best["weighted_jaccard"]),
                "count_ratio": float(best["count_ratio"]),
                "coarse_mutual_nearest": bool(reverse_best[best["tm_idx"]] == main_idx),
            }
        )
    return pd.DataFrame(rows)


def _cross_season_consistency(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for main_id, group in matches.groupby("main_pitcher_id", sort=False):
        if len(group) < 2:
            continue
        counts = group["best_trackman_id"].value_counts()
        mode_id = str(counts.index[0])
        mode_count = int(counts.iloc[0])
        rows.append(
            {
                "main_pitcher_id": str(main_id),
                "seasons": int(len(group)),
                "mode_trackman_id": mode_id,
                "mode_seasons": mode_count,
                "consistency": float(mode_count / len(group)),
                "all_same": bool(mode_count == len(group)),
                "mean_best_score": float(group["best_score"].mean()),
                "mean_margin": float(group["margin"].mean()),
                "mutual_fraction": float(group["coarse_mutual_nearest"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _null_consistency(
    matches: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = matches[["season", "main_pitcher_id", "best_trackman_id"]].copy()
    scores = []
    for _ in range(repetitions):
        shuffled_parts = []
        for _, group in base.groupby("season", sort=False):
            temp = group.copy()
            temp["best_trackman_id"] = rng.permutation(temp["best_trackman_id"].to_numpy())
            shuffled_parts.append(temp)
        shuffled = pd.concat(shuffled_parts, ignore_index=True)
        consistency = _cross_season_consistency(shuffled)
        if consistency.empty:
            scores.append(np.nan)
        else:
            scores.append(float(consistency["consistency"].mean()))
    return np.asarray(scores, dtype=np.float64)


def _verdict(summary: dict) -> str:
    actual = summary.get("cross_season_mean_consistency")
    null95 = summary.get("null_consistency_p95")
    mutual = summary.get("mutual_fraction")
    median_margin = summary.get("median_margin")
    if all(v is not None and np.isfinite(v) for v in [actual, null95, mutual, median_margin]):
        excess = float(actual - null95)
        if excess >= 0.20 and mutual >= 0.45 and median_margin >= 0.05:
            return "STRONG_IDENTITY_LIKE_SIGNAL"
        if excess >= 0.08 and mutual >= 0.25 and median_margin >= 0.02:
            return "SUGGESTIVE_SIGNAL"
    return "WEAK_OR_AMBIGUOUS_SIGNAL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Side experiment: test whether anonymous main pitcher IDs and Trackman pitcher IDs show "
            "identity-like correspondence through repeated pitch-context fingerprints. This does NOT "
            "assume the two namespaces identify the same people."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trackman", default="data/raw/trackman_history.csv")
    parser.add_argument("--trackman-pitcher-col", default=None)
    parser.add_argument("--context-cols", default=None, help="comma-separated override")
    parser.add_argument("--hash-dim", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--null-repetitions", type=int, default=200)
    parser.add_argument("--output-dir", default="outputs/trackman_mapping_feasibility")
    args = parser.parse_args()

    if args.hash_dim < 256:
        raise ValueError("--hash-dim must be >= 256")
    if args.top_k < 2:
        raise ValueError("--top-k must be >= 2")

    config = load_config(ROOT / args.config)
    seed = int(config.get("seed", 42))
    train = load_frame(config)
    trackman_path = ROOT / args.trackman
    if not trackman_path.exists():
        raise FileNotFoundError(f"Trackman file not found: {trackman_path}")
    tm = pd.read_csv(trackman_path, low_memory=False)

    main_pitcher_col = "pitcher_id"
    main_season_col = config["data"]["season_col"]
    tm_pitcher_col = args.trackman_pitcher_col or _detect_trackman_pitcher_column(tm.columns.tolist())
    tm_season_col = _detect_season_column(tm.columns.tolist(), preferred=main_season_col)

    if args.context_cols:
        requested = [c.strip() for c in args.context_cols.split(",") if c.strip()]
        missing = [c for c in requested if c not in train.columns or c not in tm.columns]
        if missing:
            raise RuntimeError(f"Requested context columns are not shared: {missing}")
        context_columns = requested
    else:
        context_columns = [
            c for c in DEFAULT_CONTEXT_COLUMNS if c in train.columns and c in tm.columns
        ]
    if len(context_columns) < 5:
        raise RuntimeError(
            "Too few shared context columns for a meaningful fingerprint. "
            f"Found {context_columns}. Use --context-cols only if definitions are known to align."
        )

    train = train.copy()
    tm = tm.copy()
    train[main_pitcher_col] = _normalize_id(train[main_pitcher_col])
    tm[tm_pitcher_col] = _normalize_id(tm[tm_pitcher_col])
    train[main_season_col] = pd.to_numeric(train[main_season_col], errors="coerce")
    tm[tm_season_col] = pd.to_numeric(tm[tm_season_col], errors="coerce")
    train = train.dropna(subset=[main_season_col])
    tm = tm.dropna(subset=[tm_season_col])
    train[main_season_col] = train[main_season_col].astype(int)
    tm[tm_season_col] = tm[tm_season_col].astype(int)

    common_seasons = sorted(set(train[main_season_col].unique()) & set(tm[tm_season_col].unique()))
    if not common_seasons:
        raise RuntimeError("No common seasons between main train and Trackman")

    print("[Trackman Mapping Feasibility]")
    print(f"  main pitcher id    : {main_pitcher_col}")
    print(f"  Trackman pitcher id: {tm_pitcher_col}")
    print(f"  common seasons     : {common_seasons}")
    print(f"  context fingerprint: {context_columns}")
    print(f"  hash_dim={args.hash_dim}, exact rerank top_k={args.top_k}")
    print("  IMPORTANT: this is a correspondence test, not an assumed identity join.\n")

    all_matches = []
    diagnostics = []
    for season in common_seasons:
        main_season = train.loc[train[main_season_col].eq(season)].copy()
        tm_season = tm.loc[tm[tm_season_col].eq(season)].copy()
        if main_season.empty or tm_season.empty:
            continue

        main_counts, main_totals = _make_profile_table(
            main_season,
            pitcher_col=main_pitcher_col,
            season_col=main_season_col,
            context_columns=context_columns,
        )
        tm_counts, tm_totals = _make_profile_table(
            tm_season.rename(columns={tm_season_col: main_season_col}),
            pitcher_col=tm_pitcher_col,
            season_col=main_season_col,
            context_columns=context_columns,
        )
        matches = _score_one_season(
            main_counts,
            main_totals,
            tm_counts,
            tm_totals,
            main_pitcher_col=main_pitcher_col,
            tm_pitcher_col=tm_pitcher_col,
            hash_dim=args.hash_dim,
            top_k=args.top_k,
        )
        if matches.empty:
            continue
        matches.insert(0, "season", season)
        all_matches.append(matches)
        diagnostic = {
            "season": int(season),
            "main_pitchers": int(len(main_totals)),
            "trackman_pitchers": int(len(tm_totals)),
            "median_best_score": float(matches["best_score"].median()),
            "median_margin": float(matches["margin"].median()),
            "median_overlap": float(matches["overlap_coefficient"].median()),
            "mutual_fraction": float(matches["coarse_mutual_nearest"].mean()),
        }
        diagnostics.append(diagnostic)
        print(
            f"[{season}] main={diagnostic['main_pitchers']:,} tm={diagnostic['trackman_pitchers']:,} "
            f"best={diagnostic['median_best_score']:.3f} margin={diagnostic['median_margin']:.3f} "
            f"overlap={diagnostic['median_overlap']:.3f} mutual={diagnostic['mutual_fraction']:.3f}"
        )

    if not all_matches:
        raise RuntimeError("No season produced usable matches")

    matches_df = pd.concat(all_matches, ignore_index=True)
    consistency_df = _cross_season_consistency(matches_df)
    null_scores = _null_consistency(
        matches_df,
        repetitions=args.null_repetitions,
        seed=seed,
    )
    finite_null = null_scores[np.isfinite(null_scores)]

    summary = {
        "context_columns": context_columns,
        "main_pitcher_column": main_pitcher_col,
        "trackman_pitcher_column": tm_pitcher_col,
        "common_seasons": common_seasons,
        "pitcher_season_matches": int(len(matches_df)),
        "median_best_score": float(matches_df["best_score"].median()),
        "median_margin": float(matches_df["margin"].median()),
        "median_overlap_coefficient": float(matches_df["overlap_coefficient"].median()),
        "median_weighted_jaccard": float(matches_df["weighted_jaccard"].median()),
        "mutual_fraction": float(matches_df["coarse_mutual_nearest"].mean()),
        "multi_season_main_pitchers": int(len(consistency_df)),
        "cross_season_mean_consistency": float(consistency_df["consistency"].mean()) if not consistency_df.empty else None,
        "cross_season_all_same_fraction": float(consistency_df["all_same"].mean()) if not consistency_df.empty else None,
        "null_consistency_mean": float(np.mean(finite_null)) if len(finite_null) else None,
        "null_consistency_p95": float(np.quantile(finite_null, 0.95)) if len(finite_null) else None,
        "null_consistency_p99": float(np.quantile(finite_null, 0.99)) if len(finite_null) else None,
        "season_diagnostics": diagnostics,
    }
    summary["verdict"] = _verdict(summary)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(output_dir / "pitcher_season_best_matches.csv", index=False)
    consistency_df.to_csv(output_dir / "cross_season_consistency.csv", index=False)
    pd.DataFrame({"null_mean_consistency": null_scores}).to_csv(
        output_dir / "null_consistency_distribution.csv", index=False
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Overall]")
    print(f"  median best score       : {summary['median_best_score']:.4f}")
    print(f"  median best-second gap  : {summary['median_margin']:.4f}")
    print(f"  median exact overlap    : {summary['median_overlap_coefficient']:.4f}")
    print(f"  mutual nearest fraction : {summary['mutual_fraction']:.4f}")
    if summary["cross_season_mean_consistency"] is not None:
        print(f"  cross-season consistency: {summary['cross_season_mean_consistency']:.4f}")
        print(f"  null mean / p95         : {summary['null_consistency_mean']:.4f} / {summary['null_consistency_p95']:.4f}")
        print(f"  all-seasons-same frac   : {summary['cross_season_all_same_fraction']:.4f}")
    print(f"\n[Verdict] {summary['verdict']}")
    print("  STRONG/SUGGESTIVE means identity-like correspondence is plausible, not proven.")
    print("  WEAK means do not build a player-level Trackman join from this fingerprint.")
    print(f"\nSaved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
