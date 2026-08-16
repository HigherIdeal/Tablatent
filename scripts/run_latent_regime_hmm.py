from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
from src.utils import load_config, save_json, seed_everything


EXCLUDE_ALWAYS = {
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "control_success",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}


def _token(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("<MISSING>").astype(str)


def _month_key(frame: pd.DataFrame) -> pd.Series:
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    month = pd.to_numeric(frame["game_month"], errors="raise").astype(int)
    return season * 100 + month


def _safe_std(x: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=0)) if len(a) else np.nan


def _numeric_monthly(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    work = frame[["_month", col]].copy()
    work[col] = pd.to_numeric(work[col], errors="coerce")
    g = work.groupby("_month", sort=True, observed=True)[col]
    out = pd.DataFrame({
        f"{col}__mean": g.mean(),
        f"{col}__std": g.apply(_safe_std),
        f"{col}__median": g.median(),
    })
    return out


def _categorical_monthly(frame: pd.DataFrame, col: str, max_levels: int) -> pd.DataFrame:
    tok = _token(frame[col])
    counts = tok.value_counts(dropna=False)
    keep = counts.head(max_levels).index.tolist()
    mapped = tok.where(tok.isin(keep), "<OTHER>")
    tmp = pd.DataFrame({"_month": frame["_month"].to_numpy(), "_value": mapped.to_numpy()})
    tab = pd.crosstab(tmp["_month"], tmp["_value"], normalize="index")
    tab = tab.reindex(sorted(frame["_month"].unique()), fill_value=0.0)
    tab.columns = [f"{col}__share__{str(c)}" for c in tab.columns]
    p = tab.to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
    denom = math.log(max(p.shape[1], 2))
    tab[f"{col}__entropy"] = entropy / denom
    return tab


def build_monthly_matrix(
    frame: pd.DataFrame,
    *,
    include_game_type: bool,
    max_cat_levels: int,
) -> tuple[pd.DataFrame, list[str]]:
    work = frame.copy()
    work["_month"] = _month_key(work)

    numeric_cols: list[str] = []
    cat_cols: list[str] = []
    for col in work.columns:
        if col.startswith("_") or col in EXCLUDE_ALWAYS:
            continue
        if col == "game_type" and not include_game_type:
            continue
        s = work[col]
        numeric = pd.to_numeric(s, errors="coerce")
        finite_ratio = float(np.isfinite(numeric.to_numpy(np.float64)).mean())
        if finite_ratio >= 0.95:
            numeric_cols.append(col)
        else:
            nunique = int(_token(s).nunique())
            if 1 < nunique <= max(2 * max_cat_levels, 32):
                cat_cols.append(col)

    blocks: list[pd.DataFrame] = []
    for col in numeric_cols:
        blocks.append(_numeric_monthly(work, col))
    for col in cat_cols:
        blocks.append(_categorical_monthly(work, col, max_cat_levels=max_cat_levels))

    if not blocks:
        raise RuntimeError("no usable monthly features")
    monthly = pd.concat(blocks, axis=1).sort_index()
    monthly.index.name = "year_month"
    monthly = monthly.replace([np.inf, -np.inf], np.nan)
    monthly = monthly.loc[:, monthly.notna().any(axis=0)]
    monthly = monthly.fillna(monthly.median(numeric_only=True)).fillna(0.0)

    # Remove constants before scaling/PCA.
    std = monthly.std(axis=0, ddof=0)
    monthly = monthly.loc[:, std.gt(1e-12)]
    return monthly, list(monthly.columns)


def _hmm_parameter_count(k: int, d: int, covariance_type: str) -> int:
    # start probabilities + transition matrix + means + covariance parameters.
    n = (k - 1) + k * (k - 1) + k * d
    if covariance_type == "diag":
        n += k * d
    elif covariance_type == "full":
        n += k * d * (d + 1) // 2
    else:
        raise ValueError(covariance_type)
    return int(n)


def fit_hmm_grid(
    z: np.ndarray,
    *,
    states_grid: list[int],
    seeds: list[int],
    covariance_type: str,
    n_iter: int,
) -> tuple[object, pd.DataFrame]:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise RuntimeError(
            "hmmlearn is required. Install once with: pip install hmmlearn>=0.3.2"
        ) from exc

    rows: list[dict[str, object]] = []
    best_model = None
    best_bic = float("inf")
    n, d = z.shape
    for k in states_grid:
        for seed in seeds:
            model = GaussianHMM(
                n_components=k,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=1e-4,
                random_state=seed,
                min_covar=1e-4,
            )
            try:
                model.fit(z)
                ll = float(model.score(z))
                n_params = _hmm_parameter_count(k, d, covariance_type)
                bic = float(-2.0 * ll + n_params * math.log(max(n, 2)))
                converged = bool(getattr(model.monitor_, "converged", True))
                rows.append({
                    "states": k,
                    "seed": seed,
                    "log_likelihood": ll,
                    "bic": bic,
                    "converged": converged,
                })
                if np.isfinite(bic) and bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception as exc:  # diagnostic grid should continue
                rows.append({
                    "states": k,
                    "seed": seed,
                    "log_likelihood": np.nan,
                    "bic": np.inf,
                    "converged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    if best_model is None:
        raise RuntimeError("all HMM fits failed")
    return best_model, pd.DataFrame(rows).sort_values(["bic", "states", "seed"]).reset_index(drop=True)


def _feature_importance(monthly: pd.DataFrame, state: np.ndarray) -> pd.DataFrame:
    x = monthly.to_numpy(np.float64)
    global_mean = x.mean(axis=0)
    total_var = ((x - global_mean[None, :]) ** 2).mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    unique = np.unique(state)
    state_means: dict[int, np.ndarray] = {}
    for s in unique:
        mask = state == s
        mean_s = x[mask].mean(axis=0)
        state_means[int(s)] = mean_s
        between += float(mask.mean()) * (mean_s - global_mean) ** 2
    ratio = np.divide(between, total_var, out=np.zeros_like(between), where=total_var > 1e-12)
    rows: list[dict[str, object]] = []
    for j, name in enumerate(monthly.columns):
        row: dict[str, object] = {
            "feature": name,
            "between_state_variance_ratio": float(ratio[j]),
            "global_mean": float(global_mean[j]),
            "global_std": float(np.sqrt(total_var[j])),
        }
        for s in unique:
            row[f"state_{int(s)}_mean"] = float(state_means[int(s)][j])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("between_state_variance_ratio", ascending=False).reset_index(drop=True)


def _target_diagnostic(frame: pd.DataFrame, posterior: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Diagnostic only: inferred states come from X, not y. Do not use this table to refit the HMM.
    work = frame[["season", "game_month", target_col]].copy()
    work["year_month"] = _month_key(work)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    month_y = work.groupby("year_month", sort=True)[target_col].agg(["size", "mean"]).reset_index()
    month_y = month_y.rename(columns={"size": "rows", "mean": "target_rate"})
    out = posterior.merge(month_y, on="year_month", how="left", validate="one_to_one")
    return out


def run_one(
    frame: pd.DataFrame,
    *,
    name: str,
    include_game_type: bool,
    outdir: Path,
    states_grid: list[int],
    seeds: list[int],
    pca_dim: int,
    max_cat_levels: int,
    covariance_type: str,
    n_iter: int,
    target_col: str,
) -> dict[str, object]:
    monthly, source_features = build_monthly_matrix(
        frame,
        include_game_type=include_game_type,
        max_cat_levels=max_cat_levels,
    )
    scaler = StandardScaler()
    xs = scaler.fit_transform(monthly.to_numpy(np.float64))
    dim = max(1, min(pca_dim, xs.shape[0] - 1, xs.shape[1]))
    pca = PCA(n_components=dim, random_state=0)
    z = pca.fit_transform(xs)

    model, selection = fit_hmm_grid(
        z,
        states_grid=states_grid,
        seeds=seeds,
        covariance_type=covariance_type,
        n_iter=n_iter,
    )
    posterior = np.asarray(model.predict_proba(z), dtype=np.float64)
    state = np.asarray(model.predict(z), dtype=int)

    run_dir = outdir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(run_dir / "model_selection.csv", index=False)
    monthly.reset_index().to_csv(run_dir / "monthly_features.csv", index=False)

    posterior_df = pd.DataFrame({"year_month": monthly.index.to_numpy(int), "state": state})
    for s in range(posterior.shape[1]):
        posterior_df[f"p_state_{s}"] = posterior[:, s]
    posterior_df["max_posterior"] = posterior.max(axis=1)
    posterior_df["state_changed"] = np.r_[False, state[1:] != state[:-1]]
    posterior_df.to_csv(run_dir / "monthly_regime_posterior.csv", index=False)

    importance = _feature_importance(monthly, state)
    importance.to_csv(run_dir / "feature_regime_importance.csv", index=False)

    transition = pd.DataFrame(
        np.asarray(model.transmat_, dtype=np.float64),
        index=[f"from_{i}" for i in range(model.n_components)],
        columns=[f"to_{i}" for i in range(model.n_components)],
    )
    transition.to_csv(run_dir / "transition_matrix.csv")

    diag = _target_diagnostic(frame, posterior_df, target_col)
    diag.to_csv(run_dir / "target_diagnostic.csv", index=False)

    # Explicitly inspect the known breakpoint neighborhood without using it to fit/select.
    around = posterior_df.loc[posterior_df["year_month"].between(202210, 202310)].copy()
    around.to_csv(run_dir / "breakpoint_neighborhood.csv", index=False)

    top = importance.head(20)
    print(f"\n[{name}] include_game_type={include_game_type}")
    print(
        f"  months={len(monthly)} raw_month_features={len(source_features)} pca_dim={dim} "
        f"states={model.n_components} explained_var={pca.explained_variance_ratio_.sum():.4f}"
    )
    print("  model selection (top 5 BIC)")
    print(selection.head(5).to_string(index=False, formatters={"log_likelihood": "{:.3f}".format, "bic": "{:.3f}".format}))
    print("  inferred state path")
    compact = posterior_df[["year_month", "state", "max_posterior", "state_changed"]]
    print(compact.to_string(index=False, formatters={"max_posterior": "{:.3f}".format}))
    print("  top regime-sensitive monthly summaries")
    print(top[["feature", "between_state_variance_ratio"]].to_string(index=False, formatters={"between_state_variance_ratio": "{:.4f}".format}))
    print("  transition matrix")
    print(transition.to_string(float_format=lambda v: f"{v:.4f}"))

    return {
        "name": name,
        "include_game_type": include_game_type,
        "months": int(len(monthly)),
        "raw_month_features": int(len(source_features)),
        "pca_dim": int(dim),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "selected_states": int(model.n_components),
        "selected_bic": float(selection.iloc[0]["bic"]),
        "state_changes": int(posterior_df["state_changed"].sum()),
        "top_features": top["feature"].tolist(),
    }


def parse_ints(value: str) -> list[int]:
    out = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("empty integer list")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Unsupervised monthly latent-regime discovery with Gaussian HMM. "
            "Primary run excludes game_type so a 2023-like transition must be supported by other features."
        )
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--states", default="2,3,4")
    p.add_argument("--seeds", default="11,29,47,71,97")
    p.add_argument("--pca-dim", type=int, default=8)
    p.add_argument("--max-cat-levels", type=int, default=8)
    p.add_argument("--covariance-type", choices=["diag", "full"], default="diag")
    p.add_argument("--hmm-iterations", type=int, default=500)
    p.add_argument("--output-dir", default="outputs/latent_regime_hmm")
    p.add_argument("--only-blind", action="store_true")
    args = p.parse_args()

    states = parse_ints(args.states)
    seeds = parse_ints(args.seeds)
    if min(states) < 2:
        raise ValueError("states must be >=2")
    if args.pca_dim <= 0 or args.max_cat_levels <= 0 or args.hmm_iterations <= 0:
        raise ValueError("positive dimensions/iterations required")

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target_col = config["data"]["target_col"]
    frame, invariant_check = recent_core.prepare_frame(config)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_month"] = pd.to_numeric(frame["game_month"], errors="raise").astype(int)

    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Latent Regime HMM]")
    print(f"  rows={len(frame):,} months={frame[['season','game_month']].drop_duplicates().shape[0]}")
    print(f"  states_grid={states} seeds={seeds} pca_dim={args.pca_dim} covariance={args.covariance_type}")
    print("  PRIMARY/blind run excludes game_type and control_success from HMM fitting")
    print("  target is used only for a post-hoc diagnostic table, never to fit/select regimes")

    summaries = [
        run_one(
            frame,
            name="blind_no_game_type",
            include_game_type=False,
            outdir=outdir,
            states_grid=states,
            seeds=seeds,
            pca_dim=args.pca_dim,
            max_cat_levels=args.max_cat_levels,
            covariance_type=args.covariance_type,
            n_iter=args.hmm_iterations,
            target_col=target_col,
        )
    ]
    if not args.only_blind:
        summaries.append(
            run_one(
                frame,
                name="with_game_type",
                include_game_type=True,
                outdir=outdir,
                states_grid=states,
                seeds=seeds,
                pca_dim=args.pca_dim,
                max_cat_levels=args.max_cat_levels,
                covariance_type=args.covariance_type,
                n_iter=args.hmm_iterations,
                target_col=target_col,
            )
        )

    save_json(
        {
            "experiment": "latent_regime_hmm",
            "states_grid": states,
            "seeds": seeds,
            "pca_dim_requested": args.pca_dim,
            "covariance_type": args.covariance_type,
            "summaries": summaries,
            "invariant_check": invariant_check,
            "guardrails": [
                "control_success excluded from HMM fitting and model selection",
                "primary blind run excludes game_type",
                "target_diagnostic is post-hoc only",
                "breakpoint neighborhood is inspected after fitting, not used to choose states",
            ],
        },
        outdir / "metadata.json",
    )
    pd.DataFrame(summaries).drop(columns=["top_features"]).to_csv(outdir / "run_summary.csv", index=False)
    print(f"\nSaved: {outdir}")


if __name__ == "__main__":
    main()
