from __future__ import annotations

import gc
import itertools
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitaboost.config import load_config
from bitaboost.ex1.backward_suite import _prepare_minimal_frame
from bitaboost.ex1.pitcher_season_backward import _season_profiles
from bitaboost.ex2.hypothesis_backward import HISTORY_PREFIXES, ID_FEATURES, _base_features, _prepare_x
from bitaboost.night.common import (
    CampaignTimer,
    TrialRecorder,
    attach_history_mode,
    atomic_write_json,
    atomic_write_text,
    classification_metrics,
    grouped_metrics,
    load_yaml,
    objective_from_folds,
    read_jsonl,
    resolve_path,
    shrunk_trait,
    utc_timestamp,
    weighted_profile_lookup,
)


ALL_TRAITS = ("success", "ball", "reverse", "strike", "middle")
STABLE_POOL = ("ball", "reverse", "strike")


def _configure_warnings() -> None:
    import warnings

    try:
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    except Exception:
        pass


def _prepare_frame(exp: dict[str, Any]) -> tuple[pd.DataFrame, str, str, str]:
    base_cfg = load_config(resolve_path(exp["baseline_config"]))
    _configure_warnings()
    frame, aux = _prepare_minimal_frame(base_cfg)
    frame = frame.reset_index(drop=True)
    aux = aux.reset_index(drop=True)
    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    pitcher_col = str(exp.get("pitcher_col", "pitcher_id"))
    profiles = _season_profiles(
        frame,
        aux,
        season_col=season_col,
        pitcher_col=pitcher_col,
        target_col=target_col,
    )
    current_seasons = sorted(
        int(x)
        for x in pd.to_numeric(frame[season_col], errors="raise").astype(int).unique().tolist()
        if int(x) >= 2020
    )
    for mode in exp.get("history_modes", ["prev", "recent2", "career"]):
        lookup = weighted_profile_lookup(
            profiles,
            season_col=season_col,
            pitcher_col=pitcher_col,
            traits=ALL_TRAITS,
            mode=str(mode),
            current_seasons=current_seasons,
        )
        frame = attach_history_mode(
            frame,
            lookup,
            season_col=season_col,
            pitcher_col=pitcher_col,
            traits=ALL_TRAITS,
            mode=str(mode),
        )
    season_values = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame = frame.loc[season_values >= 2020].reset_index(drop=True)
    return frame, season_col, target_col, pitcher_col


def _feature_kind_base(kind: str, base: list[str]) -> list[str]:
    if kind == "trait_only":
        return []
    if kind == "context":
        return [
            f
            for f in base
            if f not in ID_FEATURES and not any(f.startswith(prefix) for prefix in HISTORY_PREFIXES)
        ]
    if kind == "history_no_id":
        return [f for f in base if f not in ID_FEATURES]
    if kind == "full":
        return list(base)
    raise ValueError(f"unknown feature_kind={kind}")


def _materialize_trial_features(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    base_features: list[str],
) -> list[str]:
    mode = str(cfg["history_mode"])
    k = float(cfg["reliability_k"])
    traits = [str(x) for x in cfg["traits"]]
    n = pd.to_numeric(frame[f"hist_{mode}_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    reliability = n / (n + k)
    frame["night_hist_log_n"] = np.log1p(n).astype(np.float32)
    frame["night_hist_reliability"] = reliability.astype(np.float32)
    frame["night_hist_has"] = (n > 0).astype(np.float32)
    synthetic = ["night_hist_log_n", "night_hist_reliability", "night_hist_has"]
    for trait in traits:
        name = f"night_trait_{trait}"
        if cfg.get("profile_value", "shrunk") == "raw":
            values = pd.to_numeric(frame[f"hist_{mode}_raw_{trait}"], errors="coerce").to_numpy(np.float64)
        else:
            values = shrunk_trait(frame, mode=mode, trait=trait, k=k)
        frame[name] = np.asarray(values, dtype=np.float32)
        synthetic.append(name)

    if bool(cfg.get("deviation", False)):
        candidates = {
            "success": "asof_pitcher_success_rate",
            "middle": "asof_pitcher_middle_rate",
            "ball": "asof_pitcher_ball_rate",
            "reverse": "asof_pitcher_reverse_rate",
            "strike": "asof_pitcher_strike_rate",
        }
        for trait in traits:
            source = candidates.get(trait)
            stable_name = f"night_trait_{trait}"
            if source and source in frame.columns:
                current = pd.to_numeric(frame[source], errors="coerce").to_numpy(np.float64)
                stable = pd.to_numeric(frame[stable_name], errors="coerce").to_numpy(np.float64)
                name = f"night_dev_{trait}"
                frame[name] = (current - stable).astype(np.float32)
                synthetic.append(name)

    features = _feature_kind_base(str(cfg["feature_kind"]), base_features) + synthetic
    features = list(dict.fromkeys(features))
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise RuntimeError(f"night GPU2 missing features: {missing[:10]}")
    return features


def _temporal_weights(train: pd.DataFrame, season_col: str, test_season: int, cfg: dict[str, Any]) -> np.ndarray:
    mode = str(cfg.get("temporal_weight", "equal"))
    season = pd.to_numeric(train[season_col], errors="raise").astype(int).to_numpy()
    if mode == "equal":
        weight = np.ones(len(train), dtype=np.float64)
    elif mode == "decay":
        decay = float(cfg.get("decay", 0.72))
        age = np.maximum((int(test_season) - 1) - season, 0)
        weight = np.power(decay, age, dtype=np.float64)
    elif mode == "latest_boost":
        boost = float(cfg.get("latest_boost", 2.0))
        latest = int(test_season) - 1
        weight = np.where(season == latest, boost, 1.0).astype(np.float64)
    elif mode == "recent_window":
        window = int(cfg.get("window", 3))
        weight = (season >= int(test_season) - window).astype(np.float64)
        weight = np.maximum(weight, 1e-6)
    else:
        raise ValueError(f"unknown temporal_weight={mode}")
    weight *= float(len(weight) / np.sum(weight))
    return weight.astype(np.float32)


def _catboost_params(cfg: dict[str, Any], *, loss_kind: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "iterations": int(cfg.get("iterations", 400)),
        "learning_rate": float(cfg.get("learning_rate", 0.035)),
        "depth": int(cfg.get("depth", 7)),
        "l2_leaf_reg": float(cfg.get("l2_leaf_reg", 10.0)),
        "random_strength": float(cfg.get("random_strength", 0.5)),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": float(cfg.get("bagging_temperature", 0.5)),
        "random_seed": int(cfg.get("seed", 42)),
        "task_type": "GPU",
        "devices": "0",
        "allow_writing_files": False,
        "logging_level": "Silent",
    }
    if loss_kind == "logloss":
        params["loss_function"] = "Logloss"
    elif loss_kind == "rmse":
        params["loss_function"] = "RMSE"
    else:
        raise ValueError(loss_kind)
    return params


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    y_train: np.ndarray,
    weights: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    x_train, cats = _prepare_x(train, features)
    x_test, _ = _prepare_x(test, features)
    loss_kind = str(cfg.get("loss", "logloss"))
    params = _catboost_params(cfg, loss_kind=loss_kind)
    train_pool = Pool(
        x_train,
        label=y_train,
        weight=weights,
        cat_features=cats,
        feature_names=features,
    )
    test_pool = Pool(x_test, cat_features=cats, feature_names=features)
    if loss_kind == "logloss":
        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        pred = np.asarray(model.predict_proba(test_pool)[:, 1], dtype=np.float64)
    else:
        model = CatBoostRegressor(**params)
        model.fit(train_pool)
        pred = np.asarray(model.predict(test_pool), dtype=np.float64)
    pred = np.clip(pred, 0.0, 1.0)
    del model, train_pool, test_pool, x_train, x_test
    gc.collect()
    return pred


def _config_key(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _initial_trials(exp: dict[str, Any]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    trait_sets: list[tuple[str, ...]] = []
    for r in range(0, len(STABLE_POOL) + 1):
        for subset in itertools.combinations(STABLE_POOL, r):
            trait_sets.append(("success", *subset))
    # Negative controls are deliberately sparse rather than multiplying the full grid.
    trait_sets += [("success", "middle"), ("success", "ball", "middle")]
    modes = [str(x) for x in exp.get("history_modes", ["prev", "recent2", "career"])]
    ks = [float(x) for x in exp.get("reliability_grid", [50, 100, 200, 500, 1000])]

    # Phase A: isolate trait/profile/reliability effects quickly.
    for traits in trait_sets:
        for mode in modes:
            for k in ks:
                trials.append(
                    {
                        "traits": list(traits),
                        "history_mode": mode,
                        "reliability_k": k,
                        "profile_value": "shrunk",
                        "feature_kind": "trait_only",
                        "deviation": False,
                        "temporal_weight": "equal",
                        "loss": "logloss",
                        "iterations": 300,
                        "learning_rate": 0.04,
                        "depth": 6,
                        "l2_leaf_reg": 10.0,
                        "seed": 42,
                    }
                )

    # Phase B: structurally richer probes centered on the EX4 stable set.
    rich_trait_sets = [
        ["success", "ball"],
        ["success", "strike"],
        ["success", "reverse"],
        ["success", "ball", "strike"],
        ["success", "ball", "reverse"],
        ["success", "reverse", "strike"],
        ["success", "ball", "reverse", "strike"],
    ]
    for traits in rich_trait_sets:
        for mode in modes:
            for feature_kind in ("context", "history_no_id"):
                for temporal_weight in ("equal", "decay", "latest_boost"):
                    trials.append(
                        {
                            "traits": traits,
                            "history_mode": mode,
                            "reliability_k": 200.0,
                            "profile_value": "shrunk",
                            "feature_kind": feature_kind,
                            "deviation": True,
                            "temporal_weight": temporal_weight,
                            "decay": 0.72,
                            "latest_boost": 2.0,
                            "loss": "logloss",
                            "iterations": 450,
                            "learning_rate": 0.035,
                            "depth": 7,
                            "l2_leaf_reg": 12.0,
                            "seed": 42,
                        }
                    )
    return trials


def _refinement_trials(best_cfgs: list[dict[str, Any]], round_index: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seeds = [19, 42, 73, 101, 211, 509]
    capacities = [
        (600, 6, 8.0, 0.035),
        (700, 7, 12.0, 0.03),
        (900, 8, 16.0, 0.025),
        (1200, 7, 20.0, 0.02),
    ]
    temporal = [
        ("equal", {}),
        ("decay", {"decay": 0.60}),
        ("decay", {"decay": 0.80}),
        ("latest_boost", {"latest_boost": 1.5}),
        ("latest_boost", {"latest_boost": 3.0}),
        ("recent_window", {"window": 2}),
        ("recent_window", {"window": 3}),
    ]
    losses = ["logloss", "rmse"]
    for i, base in enumerate(best_cfgs[:12]):
        for j in range(4):
            cfg = dict(base)
            cap = capacities[(round_index + i + j) % len(capacities)]
            tw, extra = temporal[(round_index * 3 + i + j) % len(temporal)]
            cfg.update(
                {
                    "iterations": cap[0],
                    "depth": cap[1],
                    "l2_leaf_reg": cap[2],
                    "learning_rate": cap[3],
                    "seed": seeds[(round_index + i * 2 + j) % len(seeds)],
                    "temporal_weight": tw,
                    "loss": losses[(round_index + i + j) % len(losses)],
                }
            )
            cfg.update(extra)
            out.append(cfg)
    return out


def _top_configs(trials_path: Path, n: int = 12) -> list[dict[str, Any]]:
    rows = [x for x in read_jsonl(trials_path) if x.get("status") == "ok"]
    rows.sort(key=lambda x: float(x.get("objective", float("inf"))))
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            continue
        # Keep structural diversity; seed/capacity differences do not define a family.
        family = {
            k: cfg.get(k)
            for k in (
                "traits",
                "history_mode",
                "reliability_k",
                "profile_value",
                "feature_kind",
                "deviation",
                "temporal_weight",
                "loss",
            )
        }
        key = _config_key(family)
        if key in seen:
            continue
        seen.add(key)
        configs.append(dict(cfg))
        if len(configs) >= n:
            break
    return configs


def _write_leaderboard(output_dir: Path, limit: int = 25) -> None:
    rows = [x for x in read_jsonl(output_dir / "trials.jsonl") if x.get("status") == "ok"]
    rows.sort(key=lambda x: float(x.get("objective", float("inf"))))
    lines = [
        "# GPU2 structural search leaderboard",
        "",
        "| rank | trial | objective | weighted Brier | std | 2022 | 2023 | 2024 | family |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:limit], start=1):
        folds = row.get("folds", {})
        cfg = row.get("config", {})
        family = f"{cfg.get('history_mode')}/{'+'.join(cfg.get('traits', []))}/{cfg.get('feature_kind')}/{cfg.get('temporal_weight')}/{cfg.get('loss')}"
        def fb(s: int) -> float:
            return float((folds.get(str(s)) or {}).get("brier", float("nan")))
        lines.append(
            f"| {rank} | `{row.get('trial_id')}` | {float(row.get('objective', float('nan'))):.9f} | "
            f"{float(row.get('weighted_brier', float('nan'))):.9f} | {float(row.get('std_brier', float('nan'))):.7f} | "
            f"{fb(2022):.8f} | {fb(2023):.8f} | {fb(2024):.8f} | `{family}` |"
        )
    atomic_write_text(output_dir / "leaderboard.md", "\n".join(lines) + "\n")


def run(config_path: str | Path, *, hours: float, expected_gpu: int = 2) -> dict[str, Any]:
    # GPU isolation must be set before CatBoost is imported by a trial.
    from bitaboost.night.common import ensure_worker_gpu

    ensure_worker_gpu(expected_gpu)
    exp = load_yaml(config_path)
    worker_cfg = dict(exp.get("gpu2", {}))
    output_dir = resolve_path(worker_cfg.get("output_dir", "outputs/night_20260818/gpu2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = CampaignTimer(hours=hours, reserve_minutes=float(exp.get("reserve_minutes", 20)))
    recorder = TrialRecorder(
        output_dir,
        worker="gpu2_structure",
        timer=timer,
        stability_penalty=float(exp.get("stability_penalty", 0.25)),
    )
    recorder.heartbeat(phase="prepare")
    print(f"[GPU2] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} logical_catboost=0", flush=True)
    print(f"[GPU2] campaign_hours={hours:.2f} output={output_dir}", flush=True)

    t0 = time.time()
    frame, season_col, target_col, pitcher_col = _prepare_frame(exp)
    base_features = _base_features()
    prepare_seconds = time.time() - t0
    atomic_write_json(
        output_dir / "campaign_meta.json",
        {
            "worker": "gpu2_structure",
            "started_utc": utc_timestamp(),
            "hours": float(hours),
            "physical_gpu": int(expected_gpu),
            "logical_device": 0,
            "rows": int(len(frame)),
            "prepare_seconds": prepare_seconds,
            "safe_contract": "all evaluation-row features are row-local or frozen from seasons before the row season",
        },
    )
    recorder.heartbeat(phase="search", extra={"rows": int(len(frame)), "prepare_seconds": prepare_seconds})

    fold_seasons = [int(x) for x in exp.get("fold_seasons", [2022, 2023, 2024])]
    stability_penalty = float(exp.get("stability_penalty", 0.25))
    completed = read_jsonl(recorder.trials_path)
    seen = {
        _config_key(row["config"])
        for row in completed
        if isinstance(row.get("config"), dict)
    }
    queue = _initial_trials(exp)
    trial_counter = len(completed)
    refinement_round = 0

    while timer.searching():
        while queue and timer.searching():
            cfg = queue.pop(0)
            key = _config_key(cfg)
            if key in seen:
                continue
            seen.add(key)
            trial_counter += 1
            trial_id = f"g2_{trial_counter:05d}"
            recorder.heartbeat(phase="trial", trial_id=trial_id, extra={"config": cfg})
            started = time.time()
            try:
                features = _materialize_trial_features(frame, cfg, base_features)
                fold_metrics: dict[int, dict[str, Any]] = {}
                fold_groups: dict[str, Any] = {}
                seasons_all = pd.to_numeric(frame[season_col], errors="raise").astype(int)
                for test_season in fold_seasons:
                    train_mask = seasons_all < test_season
                    test_mask = seasons_all == test_season
                    train = frame.loc[train_mask]
                    test = frame.loc[test_mask]
                    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8)
                    y_test = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.int8)
                    weights = _temporal_weights(train, season_col, test_season, cfg)
                    pred = _fit_predict(
                        train,
                        test,
                        features=features,
                        y_train=y_train,
                        weights=weights,
                        cfg=cfg,
                    )
                    metrics = classification_metrics(y_test, pred)
                    prior = float(np.average(y_train.astype(np.float64), weights=weights.astype(np.float64)))
                    metrics["prior_brier"] = float(np.mean((y_test - prior) ** 2))
                    metrics["delta_vs_prior"] = metrics["brier"] - metrics["prior_brier"]
                    mode = str(cfg["history_mode"])
                    prior_n = pd.to_numeric(test[f"hist_{mode}_n"], errors="coerce").fillna(0).to_numpy(np.float64)
                    fold_metrics[test_season] = metrics
                    fold_groups[str(test_season)] = grouped_metrics(y_test, pred, prior_n)
                    del train, test, y_train, y_test, weights, pred
                    gc.collect()
                objective = objective_from_folds(fold_metrics, stability_penalty=stability_penalty)
                trial = {
                    "trial_id": trial_id,
                    "status": "ok",
                    "config": cfg,
                    "feature_count": len(features),
                    "folds": {str(k): v for k, v in fold_metrics.items()},
                    "experience_groups": fold_groups,
                    "runtime_seconds": time.time() - started,
                    **objective,
                }
                recorder.record(trial)
                print(
                    f"[GPU2:{trial_id}] obj={trial['objective']:.9f} weighted={trial['weighted_brier']:.9f} "
                    f"std={trial['std_brier']:.7f} features={len(features)} cfg={cfg}",
                    flush=True,
                )
            except Exception as exc:
                trial = {
                    "trial_id": trial_id,
                    "status": "error",
                    "config": cfg,
                    "runtime_seconds": time.time() - started,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=20),
                    "objective": float("inf"),
                }
                recorder.record(trial)
                print(f"[GPU2:{trial_id}] ERROR {exc!r}", flush=True)
            if trial_counter % 3 == 0:
                _write_leaderboard(output_dir)

        if not timer.searching():
            break
        best_cfgs = _top_configs(recorder.trials_path, n=12)
        if not best_cfgs:
            best_cfgs = [
                {
                    "traits": ["success", "ball"],
                    "history_mode": "prev",
                    "reliability_k": 200.0,
                    "profile_value": "shrunk",
                    "feature_kind": "history_no_id",
                    "deviation": True,
                    "temporal_weight": "equal",
                    "loss": "logloss",
                    "iterations": 600,
                    "learning_rate": 0.03,
                    "depth": 7,
                    "l2_leaf_reg": 12.0,
                    "seed": 42,
                }
            ]
        queue = _refinement_trials(best_cfgs, refinement_round)
        refinement_round += 1
        # Avoid a tight loop if every generated config has already been seen.
        if all(_config_key(cfg) in seen for cfg in queue):
            for cfg in queue:
                cfg["seed"] = int(cfg.get("seed", 42)) + 1009 * refinement_round

    recorder.heartbeat(phase="finalize")
    _write_leaderboard(output_dir, limit=50)
    rows = [x for x in read_jsonl(recorder.trials_path) if x.get("status") == "ok"]
    rows.sort(key=lambda x: float(x.get("objective", float("inf"))))
    final = {
        "worker": "gpu2_structure",
        "completed_utc": utc_timestamp(),
        "elapsed_seconds": timer.total_elapsed(),
        "trials_ok": len(rows),
        "trials_total": len(read_jsonl(recorder.trials_path)),
        "best": rows[0] if rows else None,
        "top10": rows[:10],
    }
    atomic_write_json(output_dir / "final_summary.json", final)
    recorder.heartbeat(phase="complete", extra={"trials_ok": len(rows)})
    return final
