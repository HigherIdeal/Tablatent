from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_asof_state_engineering as asof_core
from src.canonical_features import CANONICAL_CATEGORICAL, CANONICAL_FEATURES
from src.utils import load_config, save_json, seed_everything


# Discovery intentionally excludes direct time labels and entity IDs.  The point is
# to ask whether the *distribution of baseball/context/history signals* separates
# into persistent states, not whether an HMM can rediscover the calendar or roster.
EXCLUDE_FROM_DISCOVERY = {
    "season",
    "game_month",
    "game_dayofweek",
    "pitcher_team_id",
    "batter_team_id",
    "game_type",  # blind primary analysis; add back only with --include-game-type
}


def parse_ints(value: str) -> list[int]:
    out = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("empty integer list")
    return out


def _token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _numeric_monthly(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    x = pd.to_numeric(frame[col], errors="coerce")
    tmp = pd.DataFrame({"season": frame["season"], "game_month": frame["game_month"], "x": x})
    g = tmp.groupby(["season", "game_month"], sort=True, observed=True)["x"]
    return pd.DataFrame(
        {
            f"{col}__mean": g.mean(),
            f"{col}__std": g.std(ddof=0),
            f"{col}__median": g.median(),
        }
    )


def _categorical_monthly(frame: pd.DataFrame, col: str, max_levels: int) -> pd.DataFrame:
    token = _token(frame[col])
    keep = token.value_counts().head(max_levels).index
    mapped = token.where(token.isin(keep), "<OTHER>")
    tmp = pd.DataFrame(
        {
            "season": frame["season"].to_numpy(),
            "game_month": frame["game_month"].to_numpy(),
            "value": mapped.to_numpy(),
        }
    )
    table = pd.crosstab(
        [tmp["season"], tmp["game_month"]], tmp["value"], normalize="index"
    ).sort_index()
    table.columns = [f"{col}__share__{c}" for c in table.columns]
    return table


def build_deduplicated_monthly(
    frame: pd.DataFrame,
    *,
    include_game_type: bool,
    max_cat_levels: int,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Build one observation per in-season month from canonical, de-duplicated signals.

    The HMM is fit as six independent season sequences using ``lengths``.  This
    prevents 2019-10 -> 2020-05 and 2023-10 -> 2024-03 from being treated as one
    ordinary one-step transition.
    """
    base = list(CANONICAL_FEATURES) + list(asof_core.SUCCESS_STATE)
    if include_game_type:
        excluded = EXCLUDE_FROM_DISCOVERY - {"game_type"}
    else:
        excluded = set(EXCLUDE_FROM_DISCOVERY)
    features = [f for f in base if f not in excluded]
    features = list(dict.fromkeys(features))

    work = frame[["season", "game_month"] + features].copy()
    work["season"] = pd.to_numeric(work["season"], errors="raise").astype(int)
    work["game_month"] = pd.to_numeric(work["game_month"], errors="raise").astype(int)

    categorical = set(CANONICAL_CATEGORICAL)
    if include_game_type:
        categorical.add("game_type")

    blocks: list[pd.DataFrame] = []
    for col in features:
        if col in categorical:
            blocks.append(_categorical_monthly(work, col, max_cat_levels))
        else:
            blocks.append(_numeric_monthly(work, col))

    monthly = pd.concat(blocks, axis=1).sort_index()
    monthly = monthly.replace([np.inf, -np.inf], np.nan)
    monthly = monthly.loc[:, monthly.notna().any(axis=0)]
    monthly = monthly.fillna(monthly.median(numeric_only=True)).fillna(0.0)
    std = monthly.std(axis=0, ddof=0)
    monthly = monthly.loc[:, std.gt(1e-12)]

    # Ensure each season is a contiguous independent sequence.
    seasons = monthly.index.get_level_values("season").to_numpy(int)
    lengths = [int(np.count_nonzero(seasons == year)) for year in sorted(np.unique(seasons))]
    if sum(lengths) != len(monthly):
        raise RuntimeError("HMM lengths do not cover all monthly observations")
    return monthly, features, lengths


def hmm_parameter_count(k: int, d: int, covariance_type: str) -> int:
    n = (k - 1) + k * (k - 1) + k * d
    if covariance_type == "diag":
        n += k * d
    elif covariance_type == "full":
        n += k * d * (d + 1) // 2
    else:
        raise ValueError(covariance_type)
    return int(n)


def state_feature_importance(monthly: pd.DataFrame, state: np.ndarray) -> pd.DataFrame:
    x = monthly.to_numpy(np.float64)
    overall = x.mean(axis=0)
    total = ((x - overall[None, :]) ** 2).mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    rows_by_state: dict[int, int] = {}
    means: dict[int, np.ndarray] = {}
    for s in np.unique(state):
        mask = state == s
        rows_by_state[int(s)] = int(mask.sum())
        means[int(s)] = x[mask].mean(axis=0)
        between += float(mask.mean()) * (means[int(s)] - overall) ** 2
    ratio = np.divide(between, total, out=np.zeros_like(between), where=total > 1e-12)
    rows = []
    for j, feature in enumerate(monthly.columns):
        row: dict[str, object] = {
            "feature": feature,
            "between_state_variance_ratio": float(ratio[j]),
            "global_mean": float(overall[j]),
            "global_std": float(np.sqrt(total[j])),
        }
        for s in sorted(means):
            row[f"state_{s}_mean"] = float(means[s][j])
            row[f"state_{s}_months"] = rows_by_state[s]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("between_state_variance_ratio", ascending=False)


def summarize_path(index: pd.MultiIndex, state: np.ndarray) -> dict[str, float | int]:
    season = index.get_level_values("season").to_numpy(int)
    month = index.get_level_values("game_month").to_numpy(int)
    latest = int(season.max())
    prev = latest - 1
    latest_state = state[season == latest]
    if len(latest_state):
        values, counts = np.unique(latest_state, return_counts=True)
        dominant = int(values[np.argmax(counts)])
        latest_purity = float(counts.max() / counts.sum())
    else:
        dominant = -1
        latest_purity = float("nan")
    prev_state = state[season == prev]
    prev_overlap = float(np.mean(prev_state == dominant)) if len(prev_state) else float("nan")

    within_changes = 0
    for year in sorted(np.unique(season)):
        s = state[season == year]
        if len(s) > 1:
            within_changes += int(np.count_nonzero(s[1:] != s[:-1]))
    return {
        "latest_season": latest,
        "latest_dominant_state": dominant,
        "latest_state_purity": latest_purity,
        "previous_season_overlap_with_latest_state": prev_overlap,
        "within_season_state_changes": within_changes,
        "first_year_month": int(season[0] * 100 + month[0]),
        "last_year_month": int(season[-1] * 100 + month[-1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Robustness suite for train-only latent-regime discovery. Uses canonical de-duplicated signals, "
            "splits HMM sequences at season boundaries, and tests state-count/PCA/seed stability."
        )
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--states", default="2,3")
    p.add_argument("--pca-dims", default="3,5,8")
    p.add_argument("--seeds", default="11,29,47,71,97")
    p.add_argument("--covariance-type", choices=["diag", "full"], default="diag")
    p.add_argument("--hmm-iterations", type=int, default=500)
    p.add_argument("--max-cat-levels", type=int, default=8)
    p.add_argument("--include-game-type", action="store_true")
    p.add_argument("--output-dir", default="outputs/latent_regime_robustness")
    args = p.parse_args()

    states_grid = parse_ints(args.states)
    pca_dims = parse_ints(args.pca_dims)
    seeds = parse_ints(args.seeds)
    if min(states_grid) < 2:
        raise ValueError("states must be >= 2")

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise RuntimeError("Install hmmlearn>=0.3.2 before running this experiment") from exc

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target_col = config["data"]["target_col"]
    frame, invariant_check = recent_core.prepare_frame(config)
    monthly, source_features, lengths = build_deduplicated_monthly(
        frame,
        include_game_type=bool(args.include_game_type),
        max_cat_levels=args.max_cat_levels,
    )

    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    monthly.reset_index().to_csv(outdir / "monthly_features.csv", index=False)

    scaler = StandardScaler()
    xs = scaler.fit_transform(monthly.to_numpy(np.float64))
    fit_rows: list[dict[str, object]] = []
    paths: dict[str, np.ndarray] = {}
    best_for_setting: dict[tuple[int, int], tuple[float, object, np.ndarray, np.ndarray, float]] = {}

    for requested_dim in pca_dims:
        dim = max(1, min(requested_dim, len(monthly) - 1, monthly.shape[1]))
        pca = PCA(n_components=dim, random_state=0)
        z = pca.fit_transform(xs)
        explained = float(pca.explained_variance_ratio_.sum())
        for k in states_grid:
            for seed in seeds:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type=args.covariance_type,
                    n_iter=args.hmm_iterations,
                    tol=1e-4,
                    random_state=seed,
                    min_covar=1e-4,
                )
                error = ""
                try:
                    model.fit(z, lengths=lengths)
                    ll = float(model.score(z, lengths=lengths))
                    params = hmm_parameter_count(k, dim, args.covariance_type)
                    bic = float(-2.0 * ll + params * math.log(max(len(z), 2)))
                    state = np.asarray(model.predict(z, lengths=lengths), dtype=int)
                    posterior = np.asarray(model.predict_proba(z, lengths=lengths), dtype=np.float64)
                    converged = bool(getattr(model.monitor_, "converged", True))
                    path_summary = summarize_path(monthly.index, state)
                    key_name = f"pca{dim}_k{k}_seed{seed}"
                    paths[key_name] = state
                    fit_rows.append(
                        {
                            "pca_dim": dim,
                            "states": k,
                            "seed": seed,
                            "explained_var": explained,
                            "log_likelihood": ll,
                            "bic": bic,
                            "converged": converged,
                            "mean_max_posterior": float(posterior.max(axis=1).mean()),
                            **path_summary,
                            "error": error,
                        }
                    )
                    setting = (dim, k)
                    if setting not in best_for_setting or bic < best_for_setting[setting][0]:
                        best_for_setting[setting] = (bic, model, state, posterior, explained)
                except Exception as exc:
                    fit_rows.append(
                        {
                            "pca_dim": dim,
                            "states": k,
                            "seed": seed,
                            "explained_var": explained,
                            "log_likelihood": np.nan,
                            "bic": np.inf,
                            "converged": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    fits = pd.DataFrame(fit_rows).sort_values(["pca_dim", "states", "bic", "seed"])
    fits.to_csv(outdir / "all_fits.csv", index=False)

    # Label-invariant stability of the inferred partitions.  ARI avoids the state
    # label-switching problem (state 0 in one HMM need not equal state 0 in another).
    ari_rows = []
    keys = sorted(paths)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ari_rows.append({"run_a": a, "run_b": b, "ari": adjusted_rand_score(paths[a], paths[b])})
    ari = pd.DataFrame(ari_rows)
    ari.to_csv(outdir / "partition_stability_ari.csv", index=False)

    # Save the best seed separately *within* each (PCA dimension, state count).
    # BIC values across different PCA dimensions are not treated as directly
    # comparable because they are likelihoods in different transformed spaces.
    best_rows = []
    target_month = (
        frame.assign(
            season=pd.to_numeric(frame["season"], errors="raise").astype(int),
            game_month=pd.to_numeric(frame["game_month"], errors="raise").astype(int),
            _target=pd.to_numeric(frame[target_col], errors="coerce"),
        )
        .groupby(["season", "game_month"], sort=True)["_target"]
        .agg(["size", "mean"])
        .rename(columns={"size": "rows", "mean": "target_rate"})
        .reset_index()
    )

    for (dim, k), (bic, model, state, posterior, explained) in sorted(best_for_setting.items()):
        run_name = f"pca{dim}_k{k}"
        run_dir = outdir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        path = monthly.reset_index()[["season", "game_month"]].copy()
        path["state"] = state
        for s in range(k):
            path[f"p_state_{s}"] = posterior[:, s]
        path["max_posterior"] = posterior.max(axis=1)
        # Do not flag the first month of a new season as a transition: seasons are
        # independent HMM sequences in this robustness run.
        path["state_changed_within_season"] = False
        for year in sorted(path["season"].unique()):
            idx = path.index[path["season"].eq(year)].to_numpy()
            if len(idx) > 1:
                path.loc[idx[1:], "state_changed_within_season"] = state[idx[1:]] != state[idx[:-1]]
        path.to_csv(run_dir / "monthly_regime_posterior.csv", index=False)
        importance = state_feature_importance(monthly, state)
        importance.to_csv(run_dir / "feature_regime_importance.csv", index=False)
        transition = pd.DataFrame(
            np.asarray(model.transmat_, dtype=np.float64),
            index=[f"from_{s}" for s in range(k)],
            columns=[f"to_{s}" for s in range(k)],
        )
        transition.to_csv(run_dir / "transition_matrix.csv")
        path.merge(target_month, on=["season", "game_month"], how="left").to_csv(
            run_dir / "target_diagnostic_posthoc.csv", index=False
        )
        row = {
            "pca_dim": dim,
            "states": k,
            "best_bic_within_setting": bic,
            "explained_var": explained,
            **summarize_path(monthly.index, state),
            "mean_max_posterior": float(posterior.max(axis=1).mean()),
            "top_feature": str(importance.iloc[0]["feature"]),
        }
        best_rows.append(row)

    best = pd.DataFrame(best_rows).sort_values(["pca_dim", "states"])
    best.to_csv(outdir / "best_by_setting.csv", index=False)

    metadata = {
        "experiment": "latent_regime_robustness",
        "train_only_discovery": True,
        "include_game_type": bool(args.include_game_type),
        "target_used_for_hmm": False,
        "season_sequences_split_with_lengths": True,
        "source_features": source_features,
        "monthly_feature_count": int(monthly.shape[1]),
        "months": int(len(monthly)),
        "lengths": lengths,
        "states_grid": states_grid,
        "pca_dims": pca_dims,
        "seeds": seeds,
        "canonical_invariant_check": invariant_check,
        "note": "Compare BIC only within the same PCA dimension; use ARI and path stability across dimensions.",
    }
    save_json(metadata, outdir / "metadata.json")

    print("[Latent Regime Robustness]")
    print(f"  months={len(monthly)} lengths={lengths} monthly_features={monthly.shape[1]}")
    print(f"  include_game_type={args.include_game_type} states={states_grid} pca_dims={pca_dims} seeds={seeds}")
    print("  season boundaries are independent HMM sequences; target never fits/selects states")
    print("\n[Best seed within each PCA/state setting]")
    show_cols = [
        "pca_dim", "states", "best_bic_within_setting", "explained_var",
        "latest_state_purity", "previous_season_overlap_with_latest_state",
        "within_season_state_changes", "mean_max_posterior", "top_feature",
    ]
    print(best[show_cols].to_string(index=False, formatters={
        "best_bic_within_setting": "{:.3f}".format,
        "explained_var": "{:.4f}".format,
        "latest_state_purity": "{:.3f}".format,
        "previous_season_overlap_with_latest_state": "{:.3f}".format,
        "mean_max_posterior": "{:.3f}".format,
    }))
    if not ari.empty:
        print(f"\nPartition ARI: mean={ari['ari'].mean():.3f} median={ari['ari'].median():.3f} min={ari['ari'].min():.3f}")
    print("\nInterpretation guardrail: a 2024-only state is credible only if it recurs across K/PCA/seeds; this script does not turn HMM states into test-time features.")
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
