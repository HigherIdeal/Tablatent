from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd

from .config import resolve_path
from .ensemble import build_final, build_mixed, build_safe_core
from .features import AUX_NAMES, auxiliary_targets, prepare
from .legacy import activate
from .metrics import summary
from .references import audit_if_available
from .runtime import log, stage


def _params(cfg: dict, loss: str) -> dict:
    p = dict(cfg["catboost"])
    p["loss_function"] = loss
    p["task_type"] = "GPU"
    p["devices"] = str(cfg["runtime"]["catboost_device"])
    p.pop("verbose", None)
    p["logging_level"] = "Silent"
    if cfg["runtime"].get("gpu_ram_part") is not None:
        p["gpu_ram_part"] = float(cfg["runtime"]["gpu_ram_part"])
    return p


def _prepare_x(frame: pd.DataFrame, features: list[str]):
    activate()
    import run_regime_feature_prediction_suite as regime_core

    return regime_core.prepare_x(frame, features)


def _save(model, out, name: str, enabled: bool) -> None:
    if enabled:
        directory = out / "models"
        directory.mkdir(parents=True, exist_ok=True)
        model.save_model(str(directory / f"{name}.cbm"))


def _joint_mapping(train, keep, class_index, success, nclasses):
    q = np.zeros((2, nclasses), np.float64)
    gt = train.loc[keep, "game_type"].astype(str).to_numpy()
    season = train.loc[keep, "season"].to_numpy(int)
    for gi, dom in enumerate(("R", "F")):
        dm = gt == dom
        if dom == "F" and np.any(dm & (season >= 2023)):
            dm &= season >= 2023
        fallback = float(success[dm].mean()) if dm.any() else float(success.mean())
        for ci in range(nclasses):
            cm = dm & (class_index == ci)
            q[gi, ci] = float(success[cm].mean()) if cm.any() else fallback
    return q


def train(cfg: dict) -> dict:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    activate()
    import run_offset_residual_boosting as offset_core

    out = resolve_path(cfg, cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_models = bool(cfg["output"].get("save_models", False))

    data = prepare(cfg)
    frame = data.frame
    tr = frame.loc[data.train_mask].reset_index(drop=True)
    va = frame.loc[data.valid_mask].reset_index(drop=True)

    # Exact recovered target scopes:
    # - frozen profiles + standalone reverse/middle heads: full-frame reconstruction
    # - direct/joint/hurdle/structured: recompute after the <2024 split
    aux_full_tr = data.aux.loc[data.train_mask].reset_index(drop=True)
    aux_train = auxiliary_targets(tr)
    y = data.y_valid
    gt = data.gt_valid
    comp: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    with stage("rich models: direct + aux heads + joint"):
        features = data.feature_sets["rich"]
        x, cats = _prepare_x(tr, features)
        xv, _ = _prepare_x(va, features)
        vp = Pool(xv, cat_features=cats, feature_names=features)

        keep = aux_train[list(AUX_NAMES)].notna().all(axis=1).to_numpy()
        success = tr.loc[keep, "control_success"].to_numpy(np.float32)
        av = aux_train.loc[keep, list(AUX_NAMES)].to_numpy(np.float32)

        repeats = int(cfg["recipe"]["direct"]["success_repeats"])
        labels = np.column_stack([*[success] * repeats, av])
        f_weight = float(cfg["recipe"]["direct"]["f_weight"])
        weights = np.where(
            tr.loc[keep, "game_type"].astype(str).to_numpy() == "F", f_weight, 1.0
        ).astype(np.float32)
        pool = Pool(
            x.loc[keep], labels, weight=weights, cat_features=cats, feature_names=features
        )
        model = CatBoostRegressor(**_params(cfg, "MultiRMSE")).fit(pool)
        comp["direct"] = np.clip(
            model.predict(vp, ntree_end=int(cfg["recipe"]["direct"]["tree"])), 0, 1
        )[:, 0].astype(np.float64)
        _save(model, out, "direct_multi", save_models)
        del model, pool, labels, weights
        gc.collect()

        # These heads came from run_aux_head_classifiers.py, whose labels were
        # reconstructed on the full frame before selecting <2024 training rows.
        for head, key in (("reverse", "reverse_tree"), ("middle", "middle_tree")):
            head_keep = aux_full_tr[head].notna().to_numpy()
            pool = Pool(
                x.loc[head_keep],
                aux_full_tr.loc[head_keep, head].to_numpy(np.float32),
                cat_features=cats,
                feature_names=features,
            )
            model = CatBoostClassifier(**_params(cfg, "Logloss")).fit(pool)
            name = "reverse600" if head == "reverse" else "middle400"
            comp[name] = model.predict_proba(
                vp, ntree_end=int(cfg["recipe"]["aux_heads"][key])
            )[:, 1].astype(np.float64)
            _save(model, out, f"aux_{head}", save_models)
            del model, pool
            gc.collect()

        jav = aux_train.loc[keep, list(AUX_NAMES)].to_numpy(np.int8)
        codes = jav @ (1 << np.arange(len(AUX_NAMES), dtype=np.int16))
        classes = np.unique(codes)
        class_index = np.searchsorted(classes, codes)
        joint_fw = float(cfg["recipe"]["joint"]["f_weight"])
        weights = np.where(
            tr.loc[keep, "game_type"].astype(str).to_numpy() == "F", joint_fw, 1.0
        ).astype(np.float32)
        pool = Pool(
            x.loc[keep], class_index, weight=weights, cat_features=cats, feature_names=features
        )
        model = CatBoostClassifier(**_params(cfg, "MultiClass")).fit(pool)
        prob = model.predict_proba(vp, ntree_end=int(cfg["recipe"]["joint"]["tree"]))
        q = _joint_mapping(tr, keep, class_index, success, len(classes))
        comp["joint"] = np.sum(prob * q[(gt == "F").astype(np.int8)], axis=1)
        _save(model, out, "joint", save_models)
        del model, pool, prob, jav, codes, class_index, weights, x, xv, vp
        gc.collect()
    timing["rich_models_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with stage("gate + conditional"):
        features = data.feature_sets["hurdle"]
        x, cats = _prepare_x(tr, features)
        xv, _ = _prepare_x(va, features)
        vp = Pool(xv, cat_features=cats, feature_names=features)

        usable = aux_train[["reverse", "middle"]].notna().all(axis=1).to_numpy()
        gate = ((aux_train["reverse"] == 0) & (aux_train["middle"] == 0)).to_numpy()
        f_weight = float(cfg["recipe"]["gate_conditional"]["f_weight"])
        gate_weights = np.where(
            tr.loc[usable, "game_type"].astype(str).to_numpy() == "F", f_weight, 1.0
        ).astype(np.float32)
        pool = Pool(
            x.loc[usable],
            gate[usable].astype(np.float32),
            weight=gate_weights,
            cat_features=cats,
            feature_names=features,
        )
        model = CatBoostRegressor(**_params(cfg, "RMSE")).fit(pool)
        comp["gate600"] = np.clip(
            model.predict(
                vp, ntree_end=int(cfg["recipe"]["gate_conditional"]["gate_tree"])
            ),
            0,
            1,
        ).astype(np.float64)
        _save(model, out, "gate_brier", save_models)
        del model, pool, gate_weights
        gc.collect()

        cond = usable & gate
        cond_weights = np.where(
            tr.loc[cond, "game_type"].astype(str).to_numpy() == "F", f_weight, 1.0
        ).astype(np.float32)
        pool = Pool(
            x.loc[cond],
            tr.loc[cond, "control_success"].to_numpy(np.int8),
            weight=cond_weights,
            cat_features=cats,
            feature_names=features,
        )
        model = CatBoostClassifier(**_params(cfg, "Logloss")).fit(pool)
        comp["cond400"] = model.predict_proba(
            vp, ntree_end=int(cfg["recipe"]["gate_conditional"]["conditional_tree"])
        )[:, 1].astype(np.float64)
        _save(model, out, "conditional", save_models)
        del model, pool, cond_weights, x, xv, vp
        gc.collect()
    timing["gate_conditional_sec"] = time.perf_counter() - t0

    mixed, mixed_blend, _, _ = build_mixed(
        y,
        gt,
        comp["direct"],
        comp["reverse600"],
        comp["middle400"],
        comp["gate600"],
        comp["cond400"],
        cfg["recipe"]["mixed"],
    )
    comp["mixed"] = mixed
    mixed_metric = summary(y, mixed)
    log(
        f"[mixed] brier={mixed_metric['brier']:.12f} "
        f"R={mixed_blend['R']:.12f} F={mixed_blend['F']:.12f}"
    )

    t0 = time.perf_counter()
    with stage("offset cross1"):
        features = data.feature_sets["offset"]
        x, cats = _prepare_x(tr, features)
        xv, _ = _prepare_x(va, features)
        mean = float(tr.control_success.mean())
        prior_train = offset_core.prior(tr, "recent", mean)
        prior_valid = offset_core.prior(va, "recent", mean)
        residual = tr.control_success.to_numpy(np.float64) - prior_train
        pool = Pool(x, residual, cat_features=cats, feature_names=features)
        vp = Pool(xv, cat_features=cats, feature_names=features)
        model = CatBoostRegressor(**_params(cfg, "RMSE")).fit(pool)
        comp["offset"] = np.clip(
            prior_valid
            + model.predict(vp, ntree_end=int(cfg["recipe"]["offset"]["tree"])),
            0,
            1,
        ).astype(np.float64)
        _save(model, out, "offset_cross1", save_models)
        del model, pool, vp, x, xv, prior_train, prior_valid, residual
        gc.collect()
    timing["offset_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with stage("structured ids"):
        features = data.feature_sets["structured"]
        x, cats = _prepare_x(tr, features)
        xv, _ = _prepare_x(va, features)
        keep_struct = aux_train[["reverse", "middle"]].notna().all(axis=1).to_numpy()
        reverse = aux_train.reverse.to_numpy()
        middle = aux_train.middle.to_numpy()
        sy = tr.control_success.to_numpy()
        cls = np.where(
            sy == 1,
            0,
            np.where(
                (reverse == 0) & (middle == 0),
                1,
                np.where(
                    (reverse == 1) & (middle == 0),
                    2,
                    np.where((reverse == 0) & (middle == 1), 3, 4),
                ),
            ),
        ).astype(np.int8)
        pool = Pool(
            x.loc[keep_struct],
            cls[keep_struct],
            cat_features=cats,
            feature_names=features,
        )
        vp = Pool(xv, cat_features=cats, feature_names=features)
        model = CatBoostClassifier(**_params(cfg, "MultiClass")).fit(pool)
        comp["structured"] = model.predict_proba(
            vp, ntree_end=int(cfg["recipe"]["structured"]["tree"])
        )[:, 0].astype(np.float64)
        _save(model, out, "structured_ids", save_models)
        del model, pool, vp, x, xv, cls
        gc.collect()
    timing["structured_sec"] = time.perf_counter() - t0

    safe, safe_weights = build_safe_core(
        y, gt, comp["mixed"], comp["offset"], comp["joint"]
    )
    final, final_weights = build_final(y, gt, safe, comp["structured"])
    comp["safe"] = safe
    comp["pred"] = final

    metrics = {
        "mixed": mixed_metric,
        "safe": summary(y, safe),
        "final": summary(y, final),
        "components": {
            name: summary(y, comp[name])
            for name in ("direct", "reverse600", "middle400", "gate600", "cond400", "offset", "joint", "structured")
        },
        "mixed_blend": mixed_blend,
        "safe_weights": safe_weights,
        "final_weights": final_weights,
        "timing_sec": timing,
        "rows": {"train": len(tr), "valid": len(va)},
    }

    ref = cfg["reference"]
    metrics["reference_delta"] = {
        "mixed_brier": metrics["mixed"]["brier"] - float(ref["mixed_brier"]),
        "safe_core_brier": metrics["safe"]["brier"] - float(ref["safe_core_brier"]),
        "final_brier": metrics["final"]["brier"] - float(ref["final_brier"]),
    }
    metrics["reference_pass"] = (
        abs(metrics["final"]["brier"] - float(ref["final_brier"]))
        <= float(ref["brier_tolerance"])
    )

    # If the original Tablatent workspace is present next to Bitaboost, compare the
    # new vectors automatically.  This is read-only and never participates in fitting.
    metrics["prediction_audit"] = audit_if_available(cfg, y, comp)

    if cfg["output"].get("save_components", True):
        np.savez_compressed(
            out / "predictions.npz",
            y=y,
            gt=gt,
            **comp,
            safe_weights_R=np.asarray(safe_weights["R"]),
            safe_weights_F=np.asarray(safe_weights["F"]),
            final_weights_R=np.asarray(final_weights["R"]),
            final_weights_F=np.asarray(final_weights["F"]),
        )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    serial = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "resolved_config.json").write_text(
        json.dumps(serial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(
        f"[final] brier={metrics['final']['brier']:.12f} "
        f"score={metrics['final']['score']:.4f} reference_pass={metrics['reference_pass']}"
    )
    return metrics
