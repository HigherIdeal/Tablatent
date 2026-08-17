#!/usr/bin/env python3
"""Run staged physical-state CatBoost ablations on existing leak-safe profiles.

The expensive Trackman scan is intentionally not repeated. This script consumes
the season-safe augmented Parquet and fallback tables already produced by
``src/build_physical_augmented_datasets.py``. Run stages in order: raw, shrink,
then pca. Model fitting is performed only when the user executes this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_physical_regime_suite import (  # noqa: E402
    BASE_FEATURES,
    FOLDS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    SUCCESS_STATE,
    TARGET,
    add_derived_features,
    brier,
    fit_predict,
    fold_masks,
    raw_score,
)


DEFAULT_DATA = (
    ROOT / "data" / "processed" / "physical_features"
    / "train_with_pitcher_physical.parquet"
)
DEFAULT_PROFILES = ROOT / "data" / "processed" / "physical_features"
DEFAULT_REFERENCE_CACHE = ROOT / "outputs" / "physical_regime_suite" / "cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "physical_state_distillation"

MEASURES = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
COMPACT_VALUES = [
    *(f"tm_{measure}_{stat}" for measure in MEASURES for stat in ("mean", "std")),
    "tm_fastball_rate",
    "tm_breaking_rate",
    "tm_offspeed_rate",
]
SUPPORT_FEATURES = [
    "tm_pitch_count",
    "tm_seasons_observed",
    "tm_player_profile_available",
    "tm_profile_source",
]
COMPACT_FEATURES = [*SUPPORT_FEATURES, *COMPACT_VALUES]
SHRINKAGE_DEFAULTS = (50.0, 100.0, 300.0, 500.0, 1000.0)
PCA_DEFAULTS = (4, 6, 8)
REFERENCE_EXPERTS = (
    "old_full",
    "old_recent",
    "physical_full",
    "physical_recent",
    "r_fast",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--experiments",
        default="all",
        help="all or a comma-separated subset of raw,shrink,pca",
    )
    parser.add_argument("--shrinkage-values", default="50,100,300,500,1000")
    parser.add_argument("--pca-components", default="4,6,8")
    parser.add_argument("--pca-shrinkage", type=float, default=300.0)
    parser.add_argument("--alpha-values", default="0.2")
    parser.add_argument("--lambda-values", default="0.5,0.75,1.0")
    parser.add_argument("--beta-r", type=float, default=0.10)
    parser.add_argument("--high-support", type=float, default=300.0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def csv_floats(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def csv_ints(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("PCA components must be positive integers")
    return values


def selected_experiments(value: str) -> list[str]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    allowed = {"raw", "shrink", "pca"}
    if requested == {"all"}:
        requested = allowed
    unknown = sorted(requested - allowed)
    if not requested or unknown:
        raise ValueError(f"Experiments must be from {sorted(allowed)}; unknown={unknown}")
    return [name for name in ("raw", "shrink", "pca") if name in requested]


def required_columns() -> list[str]:
    derived = {PITCHER_TEAM_WIN_EXPECTANCY, *SUCCESS_STATE}
    columns = (set(BASE_FEATURES) - derived) | set(COMPACT_FEATURES)
    columns |= {
        TARGET,
        "row_id",
        "pitcher_id",
        "home_win_expectancy",
        "away_win_expectancy",
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    }
    return sorted(columns)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=required_columns())
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("row_id must be complete and unique")
    add_derived_features(frame)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_month"] = pd.to_numeric(
        frame["game_month"], errors="raise"
    ).astype(int)
    return frame.sort_values(
        ["season", "game_month", "row_id"], kind="stable"
    ).reset_index(drop=True)


def profile_lookup(
    path: Path,
    key_columns: list[str],
    key_values: list[pd.Series],
) -> pd.DataFrame:
    table = pd.read_parquet(path, columns=[*key_columns, *COMPACT_VALUES])
    if table.duplicated(key_columns).any():
        raise ValueError(f"Duplicate fallback keys in {path.name}: {key_columns}")
    table = table.set_index(key_columns)[COMPACT_VALUES]
    if len(key_columns) == 1:
        index = pd.Index(key_values[0].to_numpy(), name=key_columns[0])
    else:
        index = pd.MultiIndex.from_arrays(
            [value.to_numpy() for value in key_values], names=key_columns
        )
    return table.reindex(index).reset_index(drop=True).astype(np.float32)


def fallback_reference(frame: pd.DataFrame, profiles_dir: Path) -> pd.DataFrame:
    season = pd.to_numeric(frame["season"], errors="raise").astype("Int64")
    team = pd.to_numeric(frame["pitcher_team_id"], errors="coerce").astype("Int64")
    hand = pd.to_numeric(frame["pitcher_hand"], errors="coerce").astype("Int64")
    team_hand = profile_lookup(
        profiles_dir / "team_hand_fallback_profiles_by_season.parquet",
        ["pitcher_team_id", "pitcher_hand", "feature_season"],
        [team, hand, season],
    )
    league_hand = profile_lookup(
        profiles_dir / "league_hand_fallback_profiles_by_season.parquet",
        ["pitcher_hand", "feature_season"],
        [hand, season],
    )
    league = profile_lookup(
        profiles_dir / "league_fallback_profiles_by_season.parquet",
        ["feature_season"],
        [season],
    )
    reference = team_hand.combine_first(league_hand).combine_first(league)
    reference.index = frame.index
    return reference


def shrink_columns(value: float) -> list[str]:
    tag = f"{value:g}".replace(".", "p")
    return [f"sd{tag}_{column}" for column in COMPACT_VALUES]


def add_shrunk_state(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    shrinkage: float,
) -> list[str]:
    output = shrink_columns(shrinkage)
    support = pd.to_numeric(frame["tm_pitch_count"], errors="coerce").fillna(0.0)
    weight = (support / (support + shrinkage)).astype(np.float32)
    player = frame["tm_profile_source"].astype(str).eq("player")
    for source, target in zip(COMPACT_VALUES, output):
        observed = pd.to_numeric(frame[source], errors="coerce").astype(np.float32)
        prior = reference[source].combine_first(observed)
        shrunk = weight * observed + (1.0 - weight) * prior
        frame[target] = observed.where(~player, shrunk).astype(np.float32)
    return output


def fit_pca_state(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    state_columns: list[str],
    components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    profile_rows = train.loc[
        train["tm_player_profile_available"].eq(1),
        ["pitcher_id", "season", *state_columns],
    ].drop_duplicates(["pitcher_id", "season"])
    fit = profile_rows[state_columns].to_numpy(np.float64)
    if len(fit) <= components:
        raise ValueError(f"Only {len(fit)} PCA profile rows for {components} components")
    median = np.asarray(
        [np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
         for column in fit.T],
        dtype=np.float64,
    )
    fit = np.where(np.isfinite(fit), fit, median)
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = (fit - mean) / scale
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    basis = right[:components]
    variance = singular * singular
    ratio = variance[:components] / variance.sum()

    def transform(source: pd.DataFrame) -> np.ndarray:
        values = source[state_columns].to_numpy(np.float64)
        values = np.where(np.isfinite(values), values, median)
        return (((values - mean) / scale) @ basis.T).astype(np.float32)

    artifacts = {
        "feature_names": np.asarray(state_columns),
        "median": median,
        "mean": mean,
        "scale": scale,
        "components": basis,
        "explained_variance_ratio": ratio,
    }
    return transform(train), transform(valid), artifacts


def load_prediction(path: Path, expected_rows: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as payload:
            prediction = payload["prediction"]
        if len(prediction) == expected_rows:
            return prediction.astype(np.float64)
    except Exception:
        pass
    return None


def reference_prediction(cache_dir: Path, fold: str, expert: str) -> np.ndarray:
    matches = list(cache_dir.glob(f"*__{fold}__{expert}.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one reference cache for {fold}/{expert}, found {len(matches)}"
        )
    with np.load(matches[0]) as payload:
        return payload["prediction"].astype(np.float64)


def cache_signature(args: argparse.Namespace) -> str:
    stat = args.data.stat()
    payload = {
        "data": str(args.data),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "iterations": args.iterations,
        "seed": args.seed,
        "task_type": args.task_type,
        "base_features": BASE_FEATURES,
        "compact_values": COMPACT_VALUES,
        "pca_shrinkage": args.pca_shrinkage,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def subgroup_metrics(
    valid: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    high_support: float,
) -> dict[str, float | int]:
    game_type = valid["game_type"].astype(str).to_numpy()
    mapped = valid["tm_profile_source"].astype(str).eq("player").to_numpy()
    support = pd.to_numeric(valid["tm_pitch_count"], errors="coerce").fillna(0).to_numpy()
    masks = {
        "r": game_type == "R",
        "f": game_type == "F",
        "mapped": mapped,
        "fallback": ~mapped,
        "high_support": mapped & (support >= high_support),
        "low_support": mapped & (support < high_support),
    }
    output: dict[str, float | int] = {}
    for name, mask in masks.items():
        output[f"{name}_rows"] = int(mask.sum())
        output[f"{name}_brier"] = brier(target, prediction, mask)
    return output


def weighted_summary(rows: pd.DataFrame) -> pd.DataFrame:
    fold_weights = {fold.name: fold.weight for fold in FOLDS}
    output: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["candidate", "alpha_recent", "lambda_physical"], sort=False
    ):
        weights = np.asarray([fold_weights[name] for name in group["fold"]], dtype=float)
        weights /= weights.sum()
        row: dict[str, Any] = {
            "candidate": keys[0],
            "alpha_recent": keys[1],
            "lambda_physical": keys[2],
            "weighted_brier": float(np.dot(weights, group["brier"])),
            "weighted_score": float(np.dot(weights, group["score"])),
            "worst_brier": float(group["brier"].max()),
            "improved_folds": int(group["delta_vs_core"].lt(0).sum()),
        }
        for name in ("r", "f", "mapped", "fallback", "high_support", "low_support"):
            row[f"weighted_{name}_brier"] = float(
                np.dot(weights, group[f"{name}_brier"])
            )
        output.append(row)
    return pd.DataFrame(output).sort_values(
        ["weighted_brier", "worst_brier"]
    ).reset_index(drop=True)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    experiments: list[str],
    coverage: dict[str, int],
    command: str,
) -> None:
    best = summary.iloc[0]
    columns = [
        "candidate", "alpha_recent", "lambda_physical", "weighted_brier",
        "weighted_score", "weighted_r_brier", "weighted_f_brier",
        "improved_folds",
    ]
    top = summary.loc[:, columns].head(12)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    table = [header, divider]
    for row in top.itertuples(index=False):
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    text = f"""# Physical State Distillation Ablation

## Implemented

Experiments: {', '.join(experiments)}. Current core CatBoost caches are the reference.
Physical source mapping: `rel_speed`, `spin_rate`, `induced_vert_break`,
`horz_break`, `extension`, `rel_height`, `rel_side`, `zone_speed`; pitch mix
uses `pitch_type_group` rates.

## Leakage controls

- Input profiles obey `Trackman season < feature_season`.
- Fallback tables are keyed by `feature_season`.
- PCA is fitted separately for each fold and full/recent training horizon using
  training-side unique accepted pitcher-season profiles only.
- `pitcher_id` is used only to deduplicate PCA fitting rows, never as a model feature.

## Coverage

{json.dumps(coverage, ensure_ascii=False)}

## Results

{chr(10).join(table)}

Best candidate: `{best['candidate']}`; weighted Brier={best['weighted_brier']:.9f};
score={best['weighted_score']:.3f}; improved folds={int(best['improved_folds'])}/3.

## Deferred by stop gate

Recent300/EMA, cold-start learned inference, and final convex optimization are
not run until the preceding physical-state ablation improves consistently.

## Reproduce

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for name in ("data", "profiles_dir", "reference_cache", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    experiments = selected_experiments(args.experiments)
    shrinkages = csv_floats(args.shrinkage_values)
    pca_sizes = csv_ints(args.pca_components)
    alpha_values = csv_floats(args.alpha_values)
    physical_weights = csv_floats(args.lambda_values)
    if any(value <= 0 for value in shrinkages) or args.pca_shrinkage <= 0:
        raise ValueError("Shrinkage must be positive")
    if max(pca_sizes) > len(COMPACT_VALUES):
        raise ValueError(f"PCA components cannot exceed {len(COMPACT_VALUES)}")
    if args.iterations <= 0 or args.thread_count <= 0:
        raise ValueError("Iterations and thread count must be positive")
    if any(not 0 <= value <= 1 for value in [*alpha_values, *physical_weights, args.beta_r]):
        raise ValueError("Blend weights must be in [0, 1]")
    for path in (args.data, args.profiles_dir, args.reference_cache):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    pca_dir = args.output_dir / "pca"
    cache_dir.mkdir(exist_ok=True)
    pca_dir.mkdir(exist_ok=True)
    signature = cache_signature(args)

    print("stage 1/4: loading compact season-safe data")
    frame = load_frame(args.data)
    print(f"rows: {len(frame):,} compact_values: {len(COMPACT_VALUES)}")

    reference: pd.DataFrame | None = None
    needed_shrinkages: set[float] = set()
    if "shrink" in experiments:
        needed_shrinkages.update(shrinkages)
    if "pca" in experiments:
        needed_shrinkages.add(args.pca_shrinkage)
    if needed_shrinkages:
        print("stage 2/4: resolving season-aware team/hand reference profiles")
        reference = fallback_reference(frame, args.profiles_dir)
    else:
        print("stage 2/4: no shrinkage reference required")

    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {
        fold.name: {} for fold in FOLDS
    }
    reference_data: dict[str, dict[str, np.ndarray]] = {}
    for fold in FOLDS:
        full_mask, recent_mask, valid_mask = fold_masks(frame, fold)
        reference_data[fold.name] = {
            expert: reference_prediction(args.reference_cache, fold.name, expert)
            for expert in REFERENCE_EXPERTS
        }
        expected = int(valid_mask.sum())
        if any(len(value) != expected for value in reference_data[fold.name].values()):
            raise ValueError(f"Reference cache row mismatch for {fold.name}")

    flat_candidates: list[tuple[str, list[str]]] = []
    if "raw" in experiments:
        flat_candidates.append(("compact_raw", COMPACT_VALUES))
    if "shrink" in experiments:
        for value in shrinkages:
            if reference is None:
                raise RuntimeError("Missing fallback reference")
            columns = add_shrunk_state(frame, reference, value)
            flat_candidates.append((f"compact_shrink_{value:g}", columns))

    model_count = len(FOLDS) * 2 * (
        len(flat_candidates) + (len(pca_sizes) + 1 if "pca" in experiments else 0)
    )
    pca_diagnostics: list[dict[str, Any]] = []
    print(f"stage 3/4: fitting/loading {model_count:,} candidate experts")
    with tqdm(total=model_count, desc="Physical state experts", unit="model", dynamic_ncols=True) as progress:
        for fold in FOLDS:
            full_mask, recent_mask, valid_mask = fold_masks(frame, fold)
            valid = frame.loc[valid_mask]
            for horizon, train_mask in (("full", full_mask), ("recent", recent_mask)):
                for candidate, state_columns in flat_candidates:
                    cache = cache_dir / f"{signature}__{fold.name}__{candidate}__{horizon}.npz"
                    prediction = None if args.no_resume else load_prediction(cache, len(valid))
                    progress.set_description_str(f"{fold.name} / {candidate} / {horizon}")
                    progress.set_postfix_str("cache hit" if prediction is not None else "fit")
                    if prediction is None:
                        features = [*BASE_FEATURES, *SUPPORT_FEATURES, *state_columns]
                        prediction = fit_predict(
                            frame.loc[train_mask], valid, features, args
                        )
                        np.savez_compressed(cache, prediction=prediction.astype(np.float32))
                    predictions[fold.name].setdefault(candidate, {})[horizon] = prediction
                    progress.update(1)

                if "pca" not in experiments:
                    continue
                if reference is None:
                    raise RuntimeError("Missing fallback reference")
                pca_state = shrink_columns(args.pca_shrinkage)
                if not set(pca_state).issubset(frame.columns):
                    add_shrunk_state(frame, reference, args.pca_shrinkage)
                pca_candidates = [f"pca_{size}" for size in pca_sizes] + ["raw_pca_6"]
                misses = {
                    candidate: cache_dir / f"{signature}__{fold.name}__{candidate}__{horizon}.npz"
                    for candidate in pca_candidates
                }
                loaded = {
                    candidate: None if args.no_resume else load_prediction(path, len(valid))
                    for candidate, path in misses.items()
                }
                artifact_path = pca_dir / f"{signature}__{fold.name}__{horizon}.npz"
                if any(value is None for value in loaded.values()) or not artifact_path.exists():
                    train = frame.loc[train_mask].copy()
                    valid_pca = valid.copy()
                    max_components = max(max(pca_sizes), 6)
                    train_pc, valid_pc, artifacts = fit_pca_state(
                        train, valid_pca, pca_state, max_components
                    )
                    pc_columns = [f"phys_pc{index + 1}" for index in range(max_components)]
                    train.loc[:, pc_columns] = train_pc
                    valid_pca.loc[:, pc_columns] = valid_pc
                    np.savez_compressed(
                        artifact_path, **artifacts,
                    )
                    for candidate in pca_candidates:
                        if loaded[candidate] is not None:
                            continue
                        size = 6 if candidate == "raw_pca_6" else int(candidate.split("_")[1])
                        state_features = pc_columns[:size]
                        if candidate == "raw_pca_6":
                            state_features = [*COMPACT_VALUES, *state_features]
                        features = [*BASE_FEATURES, *SUPPORT_FEATURES, *state_features]
                        loaded[candidate] = fit_predict(train, valid_pca, features, args)
                        np.savez_compressed(
                            misses[candidate],
                            prediction=loaded[candidate].astype(np.float32),
                        )
                    del train, valid_pca, train_pc, valid_pc
                    gc.collect()
                with np.load(artifact_path) as payload:
                    ratios = payload["explained_variance_ratio"]
                for index, ratio in enumerate(ratios, start=1):
                    pca_diagnostics.append(
                        {
                            "fold": fold.name,
                            "horizon": horizon,
                            "component": index,
                            "explained_variance_ratio": float(ratio),
                            "cumulative_explained_variance": float(ratios[:index].sum()),
                        }
                    )
                for candidate, prediction in loaded.items():
                    if prediction is None:
                        raise RuntimeError(f"Missing prediction for {candidate}")
                    predictions[fold.name].setdefault(candidate, {})[horizon] = prediction
                    progress.set_description_str(f"{fold.name} / {candidate} / {horizon}")
                    progress.set_postfix_str("done")
                    progress.update(1)

    print("stage 4/4: evaluating against the current core reference")
    reference_by_fold: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    candidates = sorted({name for values in predictions.values() for name in values})
    for fold in FOLDS:
        _, _, valid_mask = fold_masks(frame, fold)
        valid = frame.loc[valid_mask]
        target = pd.to_numeric(valid[TARGET], errors="raise").to_numpy(np.float64)
        refs = reference_data[fold.name]
        is_r = valid["game_type"].astype(str).eq("R").to_numpy()
        old_reference = 0.8 * refs["old_full"] + 0.2 * refs["old_recent"]
        core_reference = 0.8 * refs["physical_full"] + 0.2 * refs["physical_recent"]
        core_reference = 0.25 * old_reference + 0.75 * core_reference
        core_reference[is_r] = (
            0.9 * core_reference[is_r] + 0.1 * refs["r_fast"][is_r]
        )
        reference_by_fold[fold.name] = brier(target, core_reference)
        rows.append(
            {
                "fold": fold.name,
                "candidate": "core_reference",
                "alpha_recent": 0.2,
                "lambda_physical": 0.75,
                "brier": reference_by_fold[fold.name],
                "score": raw_score(target, core_reference),
                **subgroup_metrics(
                    valid, target, core_reference, args.high_support
                ),
                "delta_vs_core": 0.0,
            }
        )

        for candidate in candidates:
            for alpha in alpha_values:
                old = (1.0 - alpha) * refs["old_full"] + alpha * refs["old_recent"]
                physical = (
                    (1.0 - alpha) * predictions[fold.name][candidate]["full"]
                    + alpha * predictions[fold.name][candidate]["recent"]
                )
                for physical_weight in physical_weights:
                    final = (1.0 - physical_weight) * old + physical_weight * physical
                    final[is_r] = (
                        (1.0 - args.beta_r) * final[is_r]
                        + args.beta_r * refs["r_fast"][is_r]
                    )
                    row = {
                        "fold": fold.name,
                        "candidate": candidate,
                        "alpha_recent": alpha,
                        "lambda_physical": physical_weight,
                        "brier": brier(target, final),
                        "score": raw_score(target, final),
                        **subgroup_metrics(
                            valid, target, final, args.high_support
                        ),
                    }
                    row["delta_vs_core"] = row["brier"] - reference_by_fold[fold.name]
                    rows.append(row)

    fold_results = pd.DataFrame(rows)
    summary = weighted_summary(fold_results)
    tag = "-".join(experiments)
    fold_results.to_csv(args.output_dir / f"{tag}_fold_results.csv", index=False)
    summary.to_csv(args.output_dir / f"{tag}_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": COMPACT_VALUES,
            "missing_rate": [float(frame[column].isna().mean()) for column in COMPACT_VALUES],
        }
    ).to_csv(args.output_dir / f"{tag}_feature_missingness.csv", index=False)
    if pca_diagnostics:
        pd.DataFrame(pca_diagnostics).to_csv(
            args.output_dir / f"{tag}_pca_explained_variance.csv", index=False
        )
    coverage = {
        str(key): int(value)
        for key, value in frame["tm_profile_source"].astype(str).value_counts().items()
    }
    config = {
        "data": str(args.data),
        "experiments": experiments,
        "compact_values": COMPACT_VALUES,
        "source_columns": {
            "velocity": "rel_speed",
            "spin": "spin_rate",
            "vertical_break": "induced_vert_break",
            "horizontal_break": "horz_break",
            "extension": "extension",
            "release_height": "rel_height",
            "release_side": "rel_side",
            "plate_speed": "zone_speed",
            "pitch_mix": "pitch_type_group",
        },
        "cutoff_rule": "Trackman season < feature_season",
        "shrinkage_values": shrinkages,
        "pca_components": pca_sizes,
        "pca_shrinkage": args.pca_shrinkage,
        "alpha_values": alpha_values,
        "lambda_values": physical_weights,
        "beta_r": args.beta_r,
        "reference": "existing core physical CatBoost, alpha=.2 lambda=.75 beta_r=.1",
    }
    (args.output_dir / f"{tag}_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    command = (
        "CUDA_VISIBLE_DEVICES=2 python scripts/run_physical_state_distillation.py "
        f"--experiments {','.join(experiments)} --task-type GPU --devices 0 "
        f"--iterations {args.iterations}"
    )
    write_report(
        args.output_dir / f"{tag}_report.md", summary, experiments, coverage, command
    )
    print(summary.head(12).to_string(index=False))
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
