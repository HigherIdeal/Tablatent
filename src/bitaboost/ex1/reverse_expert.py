from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.baseline import _params, _prepare_x
from bitaboost.config import load_config
from bitaboost.features import AUX_NAMES, prepare
from bitaboost.metrics import brier, summary
from bitaboost.runtime import configure_cuda, configure_warnings, log, stage

PROFILE_NAMES = ("success", *AUX_NAMES)
ID_FEATURES = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}


def _load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[3]
        path = root / path
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError("experiment configuration root must be a mapping")
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(Path(__file__).resolve().parents[3])
    return cfg


def _experiment_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_repo_root"]) / path


def _season_pitcher_profiles(
    frame: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    target_col: str,
    prior_strength: float,
) -> pd.DataFrame:
    """Build season-frozen pitcher traits from labels available in training seasons only."""
    base = frame[[season_col, pitcher_col, target_col]].reset_index(drop=True).copy()
    aux = aux.reset_index(drop=True)
    for name in AUX_NAMES:
        base[name] = pd.to_numeric(aux[name], errors="coerce")
    base[target_col] = pd.to_numeric(base[target_col], errors="coerce")

    rows: list[pd.DataFrame] = []
    for season, block in base.groupby(season_col, sort=True):
        season = int(season)
        out = block[[pitcher_col]].drop_duplicates().reset_index(drop=True)
        out[season_col] = season

        pairs = ((target_col, "success"), *[(x, x) for x in AUX_NAMES])
        for source_name, output_name in pairs:
            values = pd.to_numeric(block[source_name], errors="coerce")
            global_mean = float(values.mean())
            stats = (
                pd.DataFrame({pitcher_col: block[pitcher_col].to_numpy(), "v": values.to_numpy()})
                .groupby(pitcher_col, dropna=False)["v"]
                .agg(["sum", "count"])
                .reset_index()
            )
            stats[f"past_{output_name}_rate"] = (
                stats["sum"] + prior_strength * global_mean
            ) / (stats["count"] + prior_strength)
            stats = stats[[pitcher_col, f"past_{output_name}_rate"]]
            out = out.merge(stats, on=pitcher_col, how="left", sort=False)

        counts = (
            block.groupby(pitcher_col, dropna=False)[target_col]
            .count()
            .rename("past_pitch_count")
            .reset_index()
        )
        out = out.merge(counts, on=pitcher_col, how="left", sort=False)
        rows.append(out)

    profiles = pd.concat(rows, ignore_index=True)
    profiles["target_season"] = profiles[season_col].astype(int) + 1
    return profiles


def _attach_previous_season_targets(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
) -> pd.DataFrame:
    target_cols = [f"past_{name}_rate" for name in PROFILE_NAMES]
    lookup = profiles[[pitcher_col, "target_season", "past_pitch_count", *target_cols]].copy()
    lookup = lookup.rename(columns={"target_season": season_col})
    keys = frame[[season_col, pitcher_col]].reset_index(drop=False).rename(columns={"index": "_row"})
    joined = keys.merge(lookup, on=[season_col, pitcher_col], how="left", sort=False)
    joined = joined.sort_values("_row", kind="stable").set_index("_row")
    return joined[["past_pitch_count", *target_cols]].reindex(range(len(frame)))


def _variant_features(features: list[str], variant: str) -> list[str]:
    if variant == "full":
        return list(features)
    if variant == "structural":
        return [name for name in features if name not in ID_FEATURES]
    raise ValueError(f"unknown variant: {variant}")


def _load_frozen_baseline(path: Path, y: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"frozen baseline predictions not found: {path}. "
            "Run the stable baseline once; EX1 never retrains the forward model."
        )
    with np.load(path, allow_pickle=False) as data:
        pred = np.asarray(data["pred"], dtype=np.float64)
        y_ref = np.asarray(data["y"], dtype=np.float64)
        gt_ref = np.asarray(data["gt"]).astype(str)
    if len(pred) != len(y) or not np.array_equal(y_ref, y):
        raise RuntimeError("frozen baseline predictions do not match the 2024 validation rows")
    if not np.array_equal(gt_ref, gt.astype(str)):
        raise RuntimeError("frozen baseline game_type order does not match validation rows")
    return pred


def _best_grid_blend(
    y: np.ndarray,
    base: np.ndarray,
    expert: np.ndarray,
    alphas: list[float],
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, float]]:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    best_alpha = 0.0
    best_pred = base.copy()
    best_brier = brier(y[mask], base[mask])
    for alpha in alphas:
        candidate = np.clip((1.0 - alpha) * base + alpha * expert, 0.0, 1.0)
        score = brier(y[mask], candidate[mask])
        if score < best_brier:
            best_alpha = float(alpha)
            best_pred = candidate
            best_brier = score
    return best_alpha, best_pred, summary(y[mask], best_pred[mask])


def _domain_metrics(y: np.ndarray, p: np.ndarray, gt: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for domain in ("R", "F"):
        mask = gt.astype(str) == domain
        if mask.any():
            result[domain] = summary(y[mask], p[mask])
    return result


def _independence_audit(model, valid: pd.DataFrame, features: list[str], sample_rows: int) -> float:
    from catboost import Pool

    n = min(int(sample_rows), len(valid))
    if n <= 0:
        return 0.0
    sample = valid.iloc[:n].copy()
    x_batch, cats = _prepare_x(sample, features)
    batch = np.asarray(model.predict(Pool(x_batch, cat_features=cats, feature_names=features)))[:, 0]
    single = []
    for i in range(n):
        row = sample.iloc[[i]].copy()
        x_one, cats_one = _prepare_x(row, features)
        value = np.asarray(
            model.predict(Pool(x_one, cat_features=cats_one, feature_names=features))
        )[0, 0]
        single.append(value)
    return float(np.max(np.abs(batch - np.asarray(single, dtype=np.float64))))


def run(config_path: str | Path) -> dict[str, Any]:
    from catboost import CatBoostRegressor, Pool

    exp = _load_experiment_config(config_path)
    base_cfg = load_config(_experiment_path(exp, exp["baseline_config"]))
    configure_cuda(base_cfg)
    configure_warnings(base_cfg)

    out_dir = _experiment_path(exp, exp["output_dir"])
    model_dir = _experiment_path(exp, exp["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with stage("EX1 prepare SAFE feature state"):
        data = prepare(base_cfg)
        frame = data.frame.reset_index(drop=True)
        aux = data.aux.reset_index(drop=True)

    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = exp.get("pitcher_col", "pitcher_id")
    valid_season = int(base_cfg["data"]["validation_season"])
    train_seasons = [int(x) for x in exp["train_seasons"]]
    target_cols = [f"past_{name}_rate" for name in PROFILE_NAMES]

    with stage("EX1 construct reverse targets from prior seasons"):
        profiles = _season_pitcher_profiles(
            frame,
            aux,
            season_col=season_col,
            pitcher_col=pitcher_col,
            target_col=target_col,
            prior_strength=float(exp["profile"]["prior_strength"]),
        )
        reverse_targets = _attach_previous_season_targets(
            frame,
            profiles,
            season_col=season_col,
            pitcher_col=pitcher_col,
        )

    valid_mask = frame[season_col].eq(valid_season).to_numpy()
    valid = frame.loc[valid_mask].reset_index(drop=True)
    y = frame.loc[valid_mask, target_col].to_numpy(np.float64)
    gt = frame.loc[valid_mask, "game_type"].astype(str).to_numpy()
    baseline_path = _experiment_path(exp, exp["baseline_predictions"])
    frozen = _load_frozen_baseline(baseline_path, y, gt)

    train_mask = frame[season_col].isin(train_seasons).to_numpy()
    target_ok = reverse_targets[target_cols].notna().all(axis=1).to_numpy()
    min_rows = int(exp["profile"].get("min_prior_rows", 1))
    target_ok &= reverse_targets["past_pitch_count"].fillna(0).to_numpy() >= min_rows
    fit_mask = train_mask & target_ok
    if not fit_mask.any():
        raise RuntimeError("EX1 reverse training set is empty")

    variants = [str(x) for x in exp.get("variants", ["structural", "full"])]
    alpha_grid = [float(x) for x in exp["blend_alpha_grid"]]
    results: dict[str, Any] = {
        "experiment": "EX1 reverse future-to-past expert with frozen SAFE forward predictor",
        "baseline": summary(y, frozen),
        "baseline_domain": _domain_metrics(y, frozen, gt),
        "rows": {
            "reverse_train": int(fit_mask.sum()),
            "valid": int(valid_mask.sum()),
            "reverse_target_coverage_2024": float(
                reverse_targets.loc[valid_mask, target_cols].notna().all(axis=1).mean()
            ),
        },
        "variants": {},
    }

    for variant in variants:
        features = _variant_features(data.feature_sets[exp.get("feature_set", "rich")], variant)
        train = frame.loc[fit_mask].reset_index(drop=True)
        labels = reverse_targets.loc[fit_mask, target_cols].to_numpy(np.float32)

        x, cats = _prepare_x(train, features)
        xv, _ = _prepare_x(valid, features)

        if bool(exp["reverse_model"].get("equal_pitcher_season_weight", True)):
            group_sizes = train.groupby([season_col, pitcher_col], dropna=False)[pitcher_col].transform("size")
            weights = (1.0 / group_sizes.to_numpy(np.float64)).astype(np.float32)
            weights *= float(len(weights) / weights.sum())
        else:
            weights = np.ones(len(train), dtype=np.float32)

        params = _params(base_cfg, "MultiRMSE")
        for key, value in exp["reverse_model"].items():
            if key != "equal_pitcher_season_weight":
                params[key] = value
        params["iterations"] = int(exp["reverse_model"].get("iterations", params["iterations"]))

        with stage(f"EX1 train reverse expert [{variant}]"):
            pool = Pool(x, labels, weight=weights, cat_features=cats, feature_names=features)
            valid_pool = Pool(xv, cat_features=cats, feature_names=features)
            model = CatBoostRegressor(**params).fit(pool)
            pred_profile = np.clip(np.asarray(model.predict(valid_pool)), 0.0, 1.0)

        model_path = model_dir / f"reverse_{variant}.cbm"
        model.save_model(str(model_path))
        expert = pred_profile[:, 0].astype(np.float64)

        reverse_truth = reverse_targets.loc[valid_mask, target_cols].reset_index(drop=True)
        reverse_eval_mask = reverse_truth.notna().all(axis=1).to_numpy()
        reverse_mse: dict[str, float] = {}
        if reverse_eval_mask.any():
            truth = reverse_truth.loc[reverse_eval_mask].to_numpy(np.float64)
            pred_eval = pred_profile[reverse_eval_mask]
            for i, name in enumerate(PROFILE_NAMES):
                reverse_mse[name] = float(np.mean((truth[:, i] - pred_eval[:, i]) ** 2))

        fixed_blends: dict[str, dict[str, float]] = {}
        for alpha in alpha_grid:
            candidate = np.clip((1.0 - alpha) * frozen + alpha * expert, 0.0, 1.0)
            fixed_blends[f"{alpha:.4f}"] = summary(y, candidate)

        best_alpha, best_overall, best_overall_metric = _best_grid_blend(
            y, frozen, expert, alpha_grid
        )
        domain_alpha: dict[str, float] = {}
        domain_blend = frozen.copy()
        for domain in ("R", "F"):
            mask = gt == domain
            alpha, _, _ = _best_grid_blend(y, frozen, expert, alpha_grid, mask=mask)
            domain_alpha[domain] = alpha
            domain_blend[mask] = np.clip(
                (1.0 - alpha) * frozen[mask] + alpha * expert[mask], 0.0, 1.0
            )

        direction = expert - frozen
        residual = y - frozen
        corr = float(np.corrcoef(direction, residual)[0, 1]) if np.std(direction) > 0 else 0.0
        independence_max_diff = _independence_audit(
            model,
            valid,
            features,
            int(exp.get("independence_audit_rows", 16)),
        )

        results["variants"][variant] = {
            "feature_count": len(features),
            "reverse_reconstruction_mse": reverse_mse,
            "expert_success": summary(y, expert),
            "expert_success_domain": _domain_metrics(y, expert, gt),
            "direction_vs_frozen_residual_corr": corr,
            "fixed_blends": fixed_blends,
            "diagnostic_best_overall": {
                "alpha": best_alpha,
                "metrics": best_overall_metric,
            },
            "diagnostic_best_domain": {
                "alpha": domain_alpha,
                "metrics": summary(y, domain_blend),
                "domain": _domain_metrics(y, domain_blend, gt),
            },
            "row_independence_max_abs_diff": independence_max_diff,
            "model_path": str(model_path.relative_to(Path(exp["_repo_root"]))),
        }

        np.savez_compressed(
            out_dir / f"predictions_{variant}.npz",
            y=y,
            gt=gt,
            frozen=frozen,
            reverse_success=expert,
            reverse_profile=pred_profile,
            best_overall=best_overall,
            best_domain=domain_blend,
        )

        log(
            f"[EX1:{variant}] frozen={results['baseline']['brier']:.12f} "
            f"reverse={results['variants'][variant]['expert_success']['brier']:.12f} "
            f"best={best_overall_metric['brier']:.12f} alpha={best_alpha:.3f} "
            f"row_diff={independence_max_diff:.3e}"
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "resolved_experiment.json").write_text(
        json.dumps({k: v for k, v in exp.items() if not k.startswith("_")}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return results
