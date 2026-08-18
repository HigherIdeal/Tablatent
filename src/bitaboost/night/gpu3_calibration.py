from __future__ import annotations

import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitaboost.ex2.hypothesis_backward import HISTORY_PREFIXES, ID_FEATURES, _base_features, _prepare_x
from bitaboost.night.common import (
    CampaignTimer,
    TrialRecorder,
    atomic_write_json,
    atomic_write_text,
    brier,
    classification_metrics,
    grouped_metrics,
    load_yaml,
    logit,
    objective_from_folds,
    read_jsonl,
    resolve_path,
    shrunk_trait,
    sigmoid,
    utc_timestamp,
)
from bitaboost.night.gpu2_structure import (
    ALL_TRAITS,
    _catboost_params,
    _prepare_frame,
    _temporal_weights,
)


FOLDS = (2022, 2023, 2024)


def _base_feature_set(kind: str, base: list[str]) -> list[str]:
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
    raise ValueError(kind)


def _fit_base_fold(
    frame: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    test_season: int,
    features: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    train = frame.loc[season < test_season]
    test = frame.loc[season == test_season]
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8)
    y_test = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.int8)
    weights = _temporal_weights(train, season_col, test_season, cfg)
    x_train, cats = _prepare_x(train, features)
    x_test, _ = _prepare_x(test, features)
    params = _catboost_params(cfg, loss_kind=str(cfg.get("loss", "logloss")))
    train_pool = Pool(
        x_train,
        label=y_train,
        weight=weights,
        cat_features=cats,
        feature_names=features,
    )
    test_pool = Pool(x_test, cat_features=cats, feature_names=features)
    if str(cfg.get("loss", "logloss")) == "logloss":
        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        pred = np.asarray(model.predict_proba(test_pool)[:, 1], dtype=np.float64)
    else:
        model = CatBoostRegressor(**params)
        model.fit(train_pool)
        pred = np.asarray(model.predict(test_pool), dtype=np.float64)
    pred = np.clip(pred, 0.0, 1.0)
    row_id = test["row_id"].astype(str).to_numpy(dtype="U") if "row_id" in test.columns else np.arange(len(test)).astype(str)
    indices = test.index.to_numpy(np.int64)
    del model, train_pool, test_pool, x_train, x_test, train, test
    gc.collect()
    return pred, y_test, row_id, indices


def _base_configs(exp: dict[str, Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for feature_kind in ("history_no_id", "full", "context"):
        for temporal, extra in (
            ("equal", {}),
            ("decay", {"decay": 0.72}),
            ("latest_boost", {"latest_boost": 2.0}),
        ):
            configs.append(
                {
                    "feature_kind": feature_kind,
                    "temporal_weight": temporal,
                    **extra,
                    "loss": "logloss",
                    "iterations": 650 if feature_kind != "context" else 500,
                    "learning_rate": 0.03,
                    "depth": 8 if feature_kind != "context" else 7,
                    "l2_leaf_reg": 12.0,
                    "random_strength": 0.5,
                    "bagging_temperature": 0.5,
                    "seed": 42,
                }
            )
    for feature_kind in ("history_no_id", "full"):
        configs.append(
            {
                "feature_kind": feature_kind,
                "temporal_weight": "equal",
                "loss": "rmse",
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 14.0,
                "random_strength": 0.5,
                "bagging_temperature": 0.5,
                "seed": 42,
            }
        )
    return configs


def _base_key(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _save_base_artifact(
    path: Path,
    *,
    seasons: list[np.ndarray],
    ys: list[np.ndarray],
    preds: list[np.ndarray],
    row_ids: list[np.ndarray],
    indices: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        season=np.concatenate(seasons).astype(np.int16),
        y=np.concatenate(ys).astype(np.int8),
        pred=np.concatenate(preds).astype(np.float32),
        row_id=np.concatenate(row_ids).astype("U"),
        frame_index=np.concatenate(indices).astype(np.int64),
    )


def _load_base_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _affine_predict(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(float(a) * logit(pred) + float(b))


def _fit_affine(y: np.ndarray, pred: np.ndarray, *, mode: str) -> tuple[float, float, float]:
    if len(y) == 0:
        return 1.0, 0.0, float("nan")
    if mode == "identity":
        return 1.0, 0.0, brier(y, pred)
    if mode == "temperature":
        a_grid = [0.55, 0.70, 0.82, 0.92, 1.0, 1.08, 1.18, 1.35, 1.60]
        b_grid = [0.0]
    elif mode == "intercept":
        a_grid = [1.0]
        b_grid = [-0.30, -0.20, -0.12, -0.07, -0.03, 0.0, 0.03, 0.07, 0.12, 0.20, 0.30]
    elif mode == "affine":
        a_grid = [0.55, 0.70, 0.82, 0.92, 1.0, 1.08, 1.18, 1.35, 1.60]
        b_grid = [-0.25, -0.15, -0.08, -0.03, 0.0, 0.03, 0.08, 0.15, 0.25]
    else:
        raise ValueError(mode)
    best = (1.0, 0.0, float("inf"))
    z = logit(pred)
    for a in a_grid:
        for bb in b_grid:
            p = sigmoid(float(a) * z + float(bb))
            score = brier(y, p)
            if score < best[2]:
                best = (float(a), float(bb), score)
    # Small local refinement avoids making the overnight result depend on a coarse grid.
    a0, b0, _ = best
    for da in (-0.08, -0.04, 0.0, 0.04, 0.08):
        for db in (-0.04, -0.02, 0.0, 0.02, 0.04):
            a = max(0.15, a0 + da)
            bb = b0 + db
            p = sigmoid(a * z + bb)
            score = brier(y, p)
            if score < best[2]:
                best = (float(a), float(bb), score)
    return best


def _group_labels(frame: pd.DataFrame, indices: np.ndarray, pred: np.ndarray, group_kind: str) -> np.ndarray:
    block = frame.loc[indices]
    if group_kind == "experience":
        n = pd.to_numeric(block["hist_prev_n"], errors="coerce").fillna(0).to_numpy(np.float64)
        return np.where(
            n <= 0,
            "no_prior",
            np.where(n < 50, "lt50", np.where(n < 200, "50_199", np.where(n < 500, "200_499", "ge500"))),
        ).astype(str)
    if group_kind == "game_type":
        if "game_type" not in block.columns:
            return np.repeat("all", len(block))
        return block["game_type"].astype("string").fillna("<MISSING>").astype(str).to_numpy()
    if group_kind == "confidence":
        d = np.abs(np.asarray(pred, dtype=np.float64) - 0.5)
        return np.where(d < 0.02, "c0", np.where(d < 0.05, "c1", np.where(d < 0.10, "c2", "c3"))).astype(str)
    raise ValueError(group_kind)


def _domain_affine(
    history_y: np.ndarray,
    history_p: np.ndarray,
    history_groups: np.ndarray,
    test_p: np.ndarray,
    test_groups: np.ndarray,
    *,
    min_rows: int,
    shrink_k: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    ga, gb, _ = _fit_affine(history_y, history_p, mode="affine")
    global_pred = _affine_predict(test_p, ga, gb)
    output = global_pred.copy()
    params: dict[str, Any] = {"global": {"a": ga, "b": gb}, "groups": {}}
    for group in sorted(set(str(x) for x in history_groups.tolist())):
        mask_h = history_groups == group
        mask_t = test_groups == group
        count = int(mask_h.sum())
        if count < int(min_rows) or not mask_t.any():
            continue
        a, bb, score = _fit_affine(history_y[mask_h], history_p[mask_h], mode="affine")
        r = count / (count + float(shrink_k))
        aa = r * a + (1.0 - r) * ga
        b2 = r * bb + (1.0 - r) * gb
        output[mask_t] = _affine_predict(test_p[mask_t], aa, b2)
        params["groups"][group] = {
            "rows": count,
            "raw_a": a,
            "raw_b": bb,
            "shrunk_a": aa,
            "shrunk_b": b2,
            "history_brier": score,
        }
    return output, params


def _residual_features(
    frame: pd.DataFrame,
    indices: np.ndarray,
    base_pred: np.ndarray,
    *,
    k: float,
    feature_kind: str,
) -> tuple[pd.DataFrame, list[str]]:
    block = frame.loc[indices].copy()
    p = np.clip(np.asarray(base_pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    block["night_base_pred"] = p.astype(np.float32)
    block["night_base_logit"] = logit(p).astype(np.float32)
    block["night_base_confidence"] = np.abs(p - 0.5).astype(np.float32)
    n = pd.to_numeric(block["hist_prev_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    r = n / (n + float(k))
    block["night_stable_log_n"] = np.log1p(n).astype(np.float32)
    block["night_stable_reliability"] = r.astype(np.float32)
    stable = ["success", "ball", "reverse", "strike"]
    for trait in stable:
        block[f"night_stable_{trait}"] = shrunk_trait(block, mode="prev", trait=trait, k=k).astype(np.float32)
    features = [
        "night_base_pred",
        "night_base_logit",
        "night_base_confidence",
        "night_stable_log_n",
        "night_stable_reliability",
        *[f"night_stable_{x}" for x in stable],
    ]
    if feature_kind == "stable":
        pass
    elif feature_kind == "stable_context":
        base = _base_features()
        context = [
            f
            for f in base
            if f not in ID_FEATURES and not any(f.startswith(prefix) for prefix in HISTORY_PREFIXES)
        ]
        features = list(dict.fromkeys(features + context))
    elif feature_kind == "stable_history":
        base = [f for f in _base_features() if f not in ID_FEATURES]
        features = list(dict.fromkeys(features + base))
    else:
        raise ValueError(feature_kind)
    x, cats = _prepare_x(block, features)
    return x, cats


def _fit_residual_corrector(
    frame: pd.DataFrame,
    *,
    history_idx: np.ndarray,
    history_y: np.ndarray,
    history_p: np.ndarray,
    test_idx: np.ndarray,
    test_p: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    xh, cats = _residual_features(
        frame,
        history_idx,
        history_p,
        k=float(cfg.get("reliability_k", 200.0)),
        feature_kind=str(cfg.get("feature_kind", "stable")),
    )
    xt, _ = _residual_features(
        frame,
        test_idx,
        test_p,
        k=float(cfg.get("reliability_k", 200.0)),
        feature_kind=str(cfg.get("feature_kind", "stable")),
    )
    params = {
        "iterations": int(cfg.get("iterations", 350)),
        "learning_rate": float(cfg.get("learning_rate", 0.035)),
        "depth": int(cfg.get("depth", 6)),
        "l2_leaf_reg": float(cfg.get("l2_leaf_reg", 16.0)),
        "random_strength": float(cfg.get("random_strength", 0.5)),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "loss_function": "Logloss",
        "random_seed": int(cfg.get("seed", 42)),
        "task_type": "GPU",
        "devices": "0",
        "allow_writing_files": False,
        "logging_level": "Silent",
    }
    model = CatBoostClassifier(**params)
    model.fit(Pool(xh, history_y, cat_features=cats, feature_names=list(xh.columns)))
    pred = np.asarray(
        model.predict_proba(Pool(xt, cat_features=cats, feature_names=list(xt.columns)))[:, 1],
        dtype=np.float64,
    )
    del model, xh, xt
    gc.collect()
    return np.clip(pred, 0.0, 1.0)


def _evaluate_calibration_method(
    artifact: dict[str, np.ndarray],
    frame: pd.DataFrame,
    *,
    method: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    season = artifact["season"].astype(int)
    y = artifact["y"].astype(np.int8)
    base = artifact["pred"].astype(np.float64)
    indices = artifact["frame_index"].astype(np.int64)
    fold_metrics: dict[int, dict[str, Any]] = {}
    details: dict[str, Any] = {}
    kind = str(method["kind"])
    for test_season in FOLDS:
        mt = season == test_season
        mh = season < test_season
        y_test = y[mt]
        p_test = base[mt]
        if kind == "identity" or not mh.any():
            pred = p_test.copy()
            detail = {"fallback": "identity" if not mh.any() else None}
        elif kind in {"temperature", "intercept", "affine"}:
            a, bb, hist_score = _fit_affine(y[mh], base[mh], mode=kind)
            pred = _affine_predict(p_test, a, bb)
            detail = {"a": a, "b": bb, "history_brier": hist_score}
        elif kind == "domain_affine":
            group_kind = str(method["group_kind"])
            gh = _group_labels(frame, indices[mh], base[mh], group_kind)
            gt = _group_labels(frame, indices[mt], base[mt], group_kind)
            pred, detail = _domain_affine(
                y[mh],
                base[mh],
                gh,
                p_test,
                gt,
                min_rows=int(method.get("min_rows", 10000)),
                shrink_k=float(method.get("shrink_k", 50000.0)),
            )
        elif kind == "residual":
            pred = _fit_residual_corrector(
                frame,
                history_idx=indices[mh],
                history_y=y[mh],
                history_p=base[mh],
                test_idx=indices[mt],
                test_p=p_test,
                cfg=method,
            )
            detail = {k: v for k, v in method.items() if k != "kind"}
        else:
            raise ValueError(kind)
        metrics = classification_metrics(y_test, pred)
        base_metrics = classification_metrics(y_test, p_test)
        metrics["base_brier"] = base_metrics["brier"]
        metrics["delta_vs_base"] = metrics["brier"] - base_metrics["brier"]
        prior_n = pd.to_numeric(frame.loc[indices[mt], "hist_prev_n"], errors="coerce").fillna(0).to_numpy(np.float64)
        metrics["experience_groups"] = grouped_metrics(y_test, pred, prior_n)
        fold_metrics[test_season] = metrics
        details[str(test_season)] = detail
    return fold_metrics, details


def _method_queue() -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = [
        {"kind": "identity"},
        {"kind": "temperature"},
        {"kind": "intercept"},
        {"kind": "affine"},
    ]
    for group in ("experience", "game_type", "confidence"):
        for min_rows, shrink_k in ((5000, 20000.0), (15000, 50000.0), (30000, 100000.0)):
            methods.append(
                {
                    "kind": "domain_affine",
                    "group_kind": group,
                    "min_rows": min_rows,
                    "shrink_k": shrink_k,
                }
            )
    for feature_kind in ("stable", "stable_context", "stable_history"):
        for k in (50.0, 200.0, 500.0, 1000.0):
            for depth in (5, 6, 7):
                methods.append(
                    {
                        "kind": "residual",
                        "feature_kind": feature_kind,
                        "reliability_k": k,
                        "iterations": 300 if feature_kind == "stable" else 450,
                        "learning_rate": 0.035,
                        "depth": depth,
                        "l2_leaf_reg": 16.0 if depth <= 6 else 24.0,
                        "seed": 42,
                    }
                )
    return methods


def _write_leaderboard(output_dir: Path, limit: int = 35) -> None:
    rows = [x for x in read_jsonl(output_dir / "trials.jsonl") if x.get("status") == "ok"]
    rows.sort(key=lambda x: float(x.get("objective", float("inf"))))
    lines = [
        "# GPU3 calibration / stacking leaderboard",
        "",
        "| rank | trial | objective | weighted Brier | 2022 | 2023 | 2024 | base | method |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(rows[:limit], start=1):
        folds = row.get("folds", {})
        def fb(s: int) -> float:
            return float((folds.get(str(s)) or {}).get("brier", float("nan")))
        lines.append(
            f"| {rank} | `{row.get('trial_id')}` | {float(row.get('objective', float('nan'))):.9f} | "
            f"{float(row.get('weighted_brier', float('nan'))):.9f} | {fb(2022):.8f} | {fb(2023):.8f} | {fb(2024):.8f} | "
            f"`{row.get('base_id')}` | `{json.dumps(row.get('method', {}), sort_keys=True)}` |"
        )
    atomic_write_text(output_dir / "leaderboard.md", "\n".join(lines) + "\n")


def _build_base_artifacts(
    frame: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    output_dir: Path,
    timer: CampaignTimer,
    recorder: TrialRecorder,
    exp: dict[str, Any],
) -> list[dict[str, Any]]:
    base_features_all = _base_features()
    base_rows: list[dict[str, Any]] = []
    manifest_path = output_dir / "base_manifest.json"
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                valid = [x for x in loaded if resolve_path(x["artifact_path"]).exists()]
                if valid:
                    return valid
        except Exception:
            pass

    for i, cfg in enumerate(_base_configs(exp), start=1):
        if not timer.searching():
            break
        base_id = f"base_{i:02d}"
        recorder.heartbeat(phase="base_model", trial_id=base_id, extra={"config": cfg})
        features = _base_feature_set(str(cfg["feature_kind"]), base_features_all)
        ys: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        seasons: list[np.ndarray] = []
        row_ids: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        fold_metrics: dict[int, dict[str, Any]] = {}
        started = time.time()
        try:
            for test_season in FOLDS:
                pred, y_test, row_id, idx = _fit_base_fold(
                    frame,
                    season_col=season_col,
                    target_col=target_col,
                    test_season=test_season,
                    features=features,
                    cfg=cfg,
                )
                ys.append(y_test)
                preds.append(pred)
                seasons.append(np.full(len(y_test), test_season, dtype=np.int16))
                row_ids.append(row_id)
                indices.append(idx)
                fold_metrics[test_season] = classification_metrics(y_test, pred)
                print(
                    f"[GPU3:{base_id}:{test_season}] brier={fold_metrics[test_season]['brier']:.8f} "
                    f"auc={fold_metrics[test_season]['auc']:.5f}",
                    flush=True,
                )
            objective = objective_from_folds(fold_metrics, stability_penalty=float(exp.get("stability_penalty", 0.25)))
            artifact = output_dir / "base_predictions" / f"{base_id}.npz"
            _save_base_artifact(
                artifact,
                seasons=seasons,
                ys=ys,
                preds=preds,
                row_ids=row_ids,
                indices=indices,
            )
            row = {
                "base_id": base_id,
                "config": cfg,
                "features": features,
                "feature_count": len(features),
                "folds": {str(k): v for k, v in fold_metrics.items()},
                "runtime_seconds": time.time() - started,
                "artifact_path": str(artifact.relative_to(resolve_path("."))),
                **objective,
            }
            base_rows.append(row)
            atomic_write_json(manifest_path, base_rows)
        except Exception as exc:
            print(f"[GPU3:{base_id}] ERROR {exc!r}\n{traceback.format_exc(limit=10)}", flush=True)
    base_rows.sort(key=lambda x: float(x.get("objective", float("inf"))))
    atomic_write_json(manifest_path, base_rows)
    return base_rows


def _refine_base_config(base: dict[str, Any], round_index: int) -> dict[str, Any]:
    cfg = dict(base)
    seeds = [19, 73, 101, 211, 509, 1021]
    cfg["seed"] = seeds[round_index % len(seeds)]
    cfg["iterations"] = [700, 900, 1100, 1400][round_index % 4]
    cfg["depth"] = [7, 8, 9][round_index % 3]
    cfg["l2_leaf_reg"] = [10.0, 16.0, 24.0, 36.0][round_index % 4]
    cfg["learning_rate"] = [0.03, 0.025, 0.02][round_index % 3]
    temporal = ["equal", "decay", "latest_boost"]
    cfg["temporal_weight"] = temporal[round_index % len(temporal)]
    if cfg["temporal_weight"] == "decay":
        cfg["decay"] = [0.60, 0.72, 0.82][round_index % 3]
    if cfg["temporal_weight"] == "latest_boost":
        cfg["latest_boost"] = [1.5, 2.0, 3.0][round_index % 3]
    return cfg


def run(config_path: str | Path, *, hours: float, expected_gpu: int = 3) -> dict[str, Any]:
    from bitaboost.night.common import ensure_worker_gpu

    ensure_worker_gpu(expected_gpu)
    exp = load_yaml(config_path)
    worker_cfg = dict(exp.get("gpu3", {}))
    output_dir = resolve_path(worker_cfg.get("output_dir", "outputs/night_20260818/gpu3"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = CampaignTimer(hours=hours, reserve_minutes=float(exp.get("reserve_minutes", 20)))
    recorder = TrialRecorder(
        output_dir,
        worker="gpu3_calibration",
        timer=timer,
        stability_penalty=float(exp.get("stability_penalty", 0.25)),
    )
    recorder.heartbeat(phase="prepare")
    print(f"[GPU3] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} logical_catboost=0", flush=True)
    print(f"[GPU3] campaign_hours={hours:.2f} output={output_dir}", flush=True)

    t0 = time.time()
    frame, season_col, target_col, _ = _prepare_frame(exp)
    prepare_seconds = time.time() - t0
    atomic_write_json(
        output_dir / "campaign_meta.json",
        {
            "worker": "gpu3_calibration",
            "started_utc": utc_timestamp(),
            "hours": float(hours),
            "physical_gpu": int(expected_gpu),
            "logical_device": 0,
            "rows": int(len(frame)),
            "prepare_seconds": prepare_seconds,
            "safe_contract": "calibration/stacking for fold s is fitted only from OOF predictions of folds before s",
            "safe982_note": "frozen SAFE982 is not tuned on 2024 because earlier SAFE OOF vectors are unavailable",
        },
    )

    base_rows = _build_base_artifacts(
        frame,
        season_col=season_col,
        target_col=target_col,
        output_dir=output_dir,
        timer=timer,
        recorder=recorder,
        exp=exp,
    )
    if not base_rows:
        raise RuntimeError("GPU3 produced no base OOF artifacts")
    base_rows.sort(key=lambda x: float(x["objective"]))
    atomic_write_text(
        output_dir / "base_leaderboard.md",
        "# Base OOF models\n\n"
        + "\n".join(
            f"- `{x['base_id']}` obj={x['objective']:.9f} weighted={x['weighted_brier']:.9f} config=`{json.dumps(x['config'], sort_keys=True)}`"
            for x in base_rows
        )
        + "\n",
    )

    methods = _method_queue()
    completed = read_jsonl(recorder.trials_path)
    seen = {
        json.dumps({"base_id": x.get("base_id"), "method": x.get("method")}, sort_keys=True)
        for x in completed
        if isinstance(x.get("method"), dict)
    }
    trial_counter = len(completed)
    round_index = 0

    # Repeatedly work on the strongest base families.  Once the initial method grid is
    # exhausted, seed/capacity refinements of the best base are generated until the
    # search deadline so the requested wall time is actually used.
    while timer.searching():
        ranked_bases = sorted(base_rows, key=lambda x: float(x["objective"]))[: min(4, len(base_rows))]
        made_progress = False
        for base_row in ranked_bases:
            if not timer.searching():
                break
            artifact_path = resolve_path(base_row["artifact_path"])
            artifact = _load_base_artifact(artifact_path)
            for method in methods:
                if not timer.searching():
                    break
                signature = json.dumps({"base_id": base_row["base_id"], "method": method}, sort_keys=True)
                if signature in seen:
                    continue
                seen.add(signature)
                made_progress = True
                trial_counter += 1
                trial_id = f"g3_{trial_counter:05d}"
                recorder.heartbeat(
                    phase="calibration_trial",
                    trial_id=trial_id,
                    extra={"base_id": base_row["base_id"], "method": method},
                )
                started = time.time()
                try:
                    folds, details = _evaluate_calibration_method(artifact, frame, method=method)
                    objective = objective_from_folds(
                        folds,
                        stability_penalty=float(exp.get("stability_penalty", 0.25)),
                    )
                    row = {
                        "trial_id": trial_id,
                        "status": "ok",
                        "base_id": base_row["base_id"],
                        "base_config": base_row["config"],
                        "method": method,
                        "config": {"base": base_row["config"], "method": method},
                        "folds": {str(k): v for k, v in folds.items()},
                        "fit_details": details,
                        "runtime_seconds": time.time() - started,
                        **objective,
                    }
                    recorder.record(row)
                    print(
                        f"[GPU3:{trial_id}] base={base_row['base_id']} method={method} "
                        f"obj={row['objective']:.9f} weighted={row['weighted_brier']:.9f}",
                        flush=True,
                    )
                except Exception as exc:
                    recorder.record(
                        {
                            "trial_id": trial_id,
                            "status": "error",
                            "base_id": base_row["base_id"],
                            "base_config": base_row["config"],
                            "method": method,
                            "config": {"base": base_row["config"], "method": method},
                            "runtime_seconds": time.time() - started,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=20),
                            "objective": float("inf"),
                        }
                    )
                    print(f"[GPU3:{trial_id}] ERROR {exc!r}", flush=True)
                if trial_counter % 3 == 0:
                    _write_leaderboard(output_dir)
            del artifact
            gc.collect()

        if not timer.searching():
            break
        if made_progress:
            continue

        # Method grid exhausted: add one refined rolling base model and then run the
        # complete calibration/stacking suite on it.
        best_base_cfg = dict(base_rows[0]["config"])
        cfg = _refine_base_config(best_base_cfg, round_index)
        round_index += 1
        base_id = f"refine_{round_index:03d}"
        features = _base_feature_set(str(cfg["feature_kind"]), _base_features())
        ys: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        seasons: list[np.ndarray] = []
        row_ids: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        folds: dict[int, dict[str, Any]] = {}
        try:
            for test_season in FOLDS:
                pred, yy, rid, idx = _fit_base_fold(
                    frame,
                    season_col=season_col,
                    target_col=target_col,
                    test_season=test_season,
                    features=features,
                    cfg=cfg,
                )
                preds.append(pred)
                ys.append(yy)
                seasons.append(np.full(len(yy), test_season, dtype=np.int16))
                row_ids.append(rid)
                indices.append(idx)
                folds[test_season] = classification_metrics(yy, pred)
            obj = objective_from_folds(folds, stability_penalty=float(exp.get("stability_penalty", 0.25)))
            artifact_path = output_dir / "base_predictions" / f"{base_id}.npz"
            _save_base_artifact(
                artifact_path,
                seasons=seasons,
                ys=ys,
                preds=preds,
                row_ids=row_ids,
                indices=indices,
            )
            new_row = {
                "base_id": base_id,
                "config": cfg,
                "features": features,
                "feature_count": len(features),
                "folds": {str(k): v for k, v in folds.items()},
                "artifact_path": str(artifact_path.relative_to(resolve_path("."))),
                **obj,
            }
            base_rows.append(new_row)
            base_rows.sort(key=lambda x: float(x["objective"]))
            atomic_write_json(output_dir / "base_manifest.json", base_rows)
        except Exception as exc:
            print(f"[GPU3:{base_id}] base refinement ERROR {exc!r}", flush=True)

    recorder.heartbeat(phase="finalize")
    _write_leaderboard(output_dir, limit=60)
    rows = [x for x in read_jsonl(recorder.trials_path) if x.get("status") == "ok"]
    rows.sort(key=lambda x: float(x.get("objective", float("inf"))))
    final = {
        "worker": "gpu3_calibration",
        "completed_utc": utc_timestamp(),
        "elapsed_seconds": timer.total_elapsed(),
        "trials_ok": len(rows),
        "trials_total": len(read_jsonl(recorder.trials_path)),
        "best": rows[0] if rows else None,
        "top10": rows[:10],
        "base_top5": sorted(base_rows, key=lambda x: float(x["objective"]))[:5],
        "safe982_policy": "not calibrated against 2024 labels; no earlier SAFE OOF vectors available",
    }
    atomic_write_json(output_dir / "final_summary.json", final)
    recorder.heartbeat(phase="complete", extra={"trials_ok": len(rows)})
    return final
