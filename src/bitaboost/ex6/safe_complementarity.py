from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitaboost.ex2.hypothesis_backward import _base_features
from bitaboost.night.common import (
    classification_metrics,
    load_yaml,
    logit,
    resolve_path,
    sigmoid,
)
from bitaboost.night.gpu2_structure import (
    _fit_predict,
    _materialize_trial_features,
    _prepare_frame,
    _temporal_weights,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if len(aa) < 2 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)
    return float(np.mean((yy - pp) ** 2))


def _load_safe(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"SAFE prediction vector not found: {path}. Run `python scripts/baseline_train.py --config configs/baseline_safe_981.yaml` first."
        )
    # predictions.npz also contains object-dtype gt; intentionally load only numeric arrays.
    with np.load(path, allow_pickle=False) as data:
        y = np.asarray(data["y"], dtype=np.float64)
        pred = np.asarray(data["pred"], dtype=np.float64)
    if len(y) != len(pred) or not np.isfinite(pred).all():
        raise RuntimeError("invalid SAFE prediction vector")
    return y, pred


def _best_gpu2_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = resolve_path(cfg["night"]["gpu2_summary"])
    if path.exists():
        payload = _load_json(path)
        best = payload.get("best")
        if isinstance(best, dict) and isinstance(best.get("config"), dict):
            return dict(best["config"]), str(best.get("trial_id", "night_best"))
    return dict(cfg["gpu2_fallback"]), "fallback"


def _fit_gpu2_2024(
    frame: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    seasons = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    train = frame.loc[seasons < 2024]
    valid = frame.loc[seasons == 2024]
    if train.empty or valid.empty:
        raise RuntimeError("EX6 GPU2 rolling split is empty")
    features = _materialize_trial_features(frame, config, _base_features())
    # _materialize_trial_features mutates the shared frame; re-slice after materialization.
    train = frame.loc[seasons < 2024]
    valid = frame.loc[seasons == 2024]
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8)
    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    weights = _temporal_weights(train, season_col, 2024, config)
    pred = _fit_predict(
        train,
        valid,
        features=features,
        y_train=y_train,
        weights=weights,
        cfg=config,
    )
    row_id = valid["row_id"].astype(str).to_numpy(dtype="U")
    return y_valid, pred.astype(np.float64), row_id, features


def _load_gpu3_artifact(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    summary_path = resolve_path(cfg["night"]["gpu3_summary"])
    if summary_path.exists():
        summary = _load_json(summary_path)
        best = summary.get("best")
        if isinstance(best, dict):
            base_id = str(best.get("base_id", ""))
            artifact = resolve_path(cfg["night"]["gpu3_artifact_dir"]) / f"{base_id}.npz"
            if base_id and artifact.exists():
                with np.load(artifact, allow_pickle=False) as data:
                    arrays = {name: np.asarray(data[name]) for name in data.files}
                return best, arrays, base_id
    raise FileNotFoundError(
        "GPU3 overnight artifact not found. Keep outputs/night_20260819/gpu3/base_predictions and final_summary.json."
    )


def _align_by_row_id(
    target_row_id: np.ndarray,
    source_row_id: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_row_id).astype(str)
    target = np.asarray(target_row_id).astype(str)
    if len(np.unique(source)) != len(source):
        raise RuntimeError("source row_id is not unique")
    lookup = {rid: i for i, rid in enumerate(source.tolist())}
    try:
        index = np.asarray([lookup[rid] for rid in target.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"row alignment failed, missing row_id={exc.args[0]}") from exc
    return np.asarray(source_values)[index]


def _fit_intercept_only(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    z = logit(pred)
    coarse = np.linspace(-0.35, 0.35, 71)
    scores = [(_brier(y, sigmoid(z + float(b))), float(b)) for b in coarse]
    _, b0 = min(scores, key=lambda x: x[0])
    fine = np.linspace(b0 - 0.02, b0 + 0.02, 81)
    scores = [(_brier(y, sigmoid(z + float(b))), float(b)) for b in fine]
    score, b = min(scores, key=lambda x: x[0])
    return b, score


def _fit_affine(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    z = logit(pred)
    best = (1.0, 0.0, float("inf"))
    for a in np.linspace(0.65, 1.25, 25):
        for b in np.linspace(-0.20, 0.20, 33):
            score = _brier(y, sigmoid(float(a) * z + float(b)))
            if score < best[2]:
                best = (float(a), float(b), score)
    a0, b0, _ = best
    for a in np.linspace(max(0.2, a0 - 0.05), a0 + 0.05, 21):
        for b in np.linspace(b0 - 0.03, b0 + 0.03, 25):
            score = _brier(y, sigmoid(float(a) * z + float(b)))
            if score < best[2]:
                best = (float(a), float(b), score)
    return best


def _gpu3_vectors(
    best: dict[str, Any],
    artifact: dict[str, np.ndarray],
    target_row_id: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    season = np.asarray(artifact["season"], dtype=np.int16)
    y = np.asarray(artifact["y"], dtype=np.float64)
    pred = np.asarray(artifact["pred"], dtype=np.float64)
    row_id = np.asarray(artifact["row_id"]).astype(str)
    hist = season < 2024
    test = season == 2024
    if not hist.any() or not test.any():
        raise RuntimeError("GPU3 artifact needs 2022/2023 history and 2024 test rows")

    base_2024 = _align_by_row_id(target_row_id, row_id[test], pred[test]).astype(np.float64)
    details = (best.get("fit_details") or {}).get("2024") or {}
    legacy_a = float(details.get("a", 1.0))
    legacy_b = float(details.get("b", 0.0))
    legacy = sigmoid(legacy_a * logit(base_2024) + legacy_b)

    b, hist_intercept_brier = _fit_intercept_only(y[hist], pred[hist])
    intercept = sigmoid(logit(base_2024) + b)
    a2, b2, hist_affine_brier = _fit_affine(y[hist], pred[hist])
    affine = sigmoid(a2 * logit(base_2024) + b2)

    vectors = {
        "gpu3_base": base_2024,
        "gpu3_night_calibration": legacy,
        "gpu3_pure_intercept": intercept,
        "gpu3_affine_corrected": affine,
    }
    calibration = {
        "night_labeled_method": best.get("method"),
        "night_parameters": {"a": legacy_a, "b": legacy_b},
        "pure_intercept": {"a": 1.0, "b": b, "history_brier": hist_intercept_brier},
        "corrected_affine": {"a": a2, "b": b2, "history_brier": hist_affine_brier},
        "note": "night 'intercept' used local slope refinement; pure_intercept fixes a=1.0",
    }
    return vectors, calibration


def _pair_diagnostics(y: np.ndarray, safe: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    se_safe = (y - safe) ** 2
    se_cand = (y - candidate) ** 2
    abs_safe = np.abs(y - safe)
    abs_cand = np.abs(y - candidate)
    threshold = float(np.quantile(se_safe, 0.90))
    worst = se_safe >= threshold
    cls_safe = (safe >= 0.5).astype(np.int8)
    cls_cand = (candidate >= 0.5).astype(np.int8)
    yy = y.astype(np.int8)
    safe_wrong = cls_safe != yy
    return {
        "prediction_corr": _corr(safe, candidate),
        "residual_corr": _corr(y - safe, y - candidate),
        "squared_error_corr": _corr(se_safe, se_cand),
        "candidate_lower_squared_error_rate": float(np.mean(se_cand < se_safe)),
        "candidate_lower_absolute_error_rate": float(np.mean(abs_cand < abs_safe)),
        "candidate_correct_when_safe_wrong_rate": float(np.mean(cls_cand[safe_wrong] == yy[safe_wrong])) if safe_wrong.any() else float("nan"),
        "safe_wrong_rows": int(safe_wrong.sum()),
        "safe_worst10_rows": int(worst.sum()),
        "mean_brier_gain_on_safe_worst10": float(np.mean(se_safe[worst] - se_cand[worst])),
    }


def _best_pair_blend(
    y: np.ndarray,
    safe: np.ndarray,
    candidate: np.ndarray,
    *,
    max_weight: float,
    step: float,
) -> dict[str, float]:
    count = int(math.floor(max_weight / step + 1e-9))
    best = {"candidate_weight": 0.0, "safe_weight": 1.0, "brier": _brier(y, safe)}
    for i in range(1, count + 1):
        w = i * step
        p = (1.0 - w) * safe + w * candidate
        score = _brier(y, p)
        if score < best["brier"]:
            best = {"candidate_weight": float(w), "safe_weight": float(1.0 - w), "brier": score}
    best["delta_vs_safe"] = float(best["brier"] - _brier(y, safe))
    return best


def _best_triple_blend(
    y: np.ndarray,
    safe: np.ndarray,
    gpu2: np.ndarray,
    gpu3: np.ndarray,
    *,
    max_total: float,
    step: float,
) -> dict[str, float]:
    count = int(math.floor(max_total / step + 1e-9))
    best = {"safe_weight": 1.0, "gpu2_weight": 0.0, "gpu3_weight": 0.0, "brier": _brier(y, safe)}
    for i in range(count + 1):
        w2 = i * step
        for j in range(count + 1 - i):
            w3 = j * step
            if w2 == 0.0 and w3 == 0.0:
                continue
            ws = 1.0 - w2 - w3
            p = ws * safe + w2 * gpu2 + w3 * gpu3
            score = _brier(y, p)
            if score < best["brier"]:
                best = {
                    "safe_weight": float(ws),
                    "gpu2_weight": float(w2),
                    "gpu3_weight": float(w3),
                    "brier": score,
                }
    best["delta_vs_safe"] = float(best["brier"] - _brier(y, safe))
    return best


def _experience_labels(n: np.ndarray) -> np.ndarray:
    nn = np.asarray(n, dtype=np.float64)
    return np.where(
        nn <= 0,
        "no_prior",
        np.where(nn < 50, "lt50", np.where(nn < 200, "50_199", np.where(nn < 500, "200_499", "ge500"))),
    ).astype(str)


def _group_complementarity(
    y: np.ndarray,
    safe: np.ndarray,
    candidates: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    max_weight: float,
    step: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted(set(str(x) for x in groups.tolist())):
        mask = groups == group
        if not mask.any():
            continue
        block: dict[str, Any] = {
            "rows": int(mask.sum()),
            "target_rate": float(np.mean(y[mask])),
            "safe_brier": _brier(y[mask], safe[mask]),
            "candidates": {},
        }
        for name, pred in candidates.items():
            block["candidates"][name] = {
                "brier": _brier(y[mask], pred[mask]),
                "delta_vs_safe": _brier(y[mask], pred[mask]) - block["safe_brier"],
                "prediction_corr": _corr(safe[mask], pred[mask]),
                "best_pair_blend": _best_pair_blend(
                    y[mask], safe[mask], pred[mask], max_weight=max_weight, step=step
                ),
            }
        out[group] = block
    return out


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# EX6 SAFE complementarity report",
        "",
        "> All blend weights in this report are **2024 diagnostic optima**. They measure complementarity and are not automatically promotable deployment weights.",
        "",
        "## Standalone 2024",
        "",
        "| vector | Brier | AUC | delta vs SAFE | pred corr vs SAFE | residual corr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    safe_brier = float(result["vectors"]["safe"]["brier"])
    for name, metric in result["vectors"].items():
        diag = result["pair_diagnostics"].get(name, {})
        delta = float(metric["brier"]) - safe_brier
        lines.append(
            f"| `{name}` | {float(metric['brier']):.9f} | {float(metric['auc']):.5f} | {delta:+.9f} | "
            f"{float(diag.get('prediction_corr', float('nan'))):.5f} | {float(diag.get('residual_corr', float('nan'))):.5f} |"
        )
    lines += ["", "## Pair blends with SAFE", "", "| candidate | candidate weight | Brier | delta vs SAFE |", "|---|---:|---:|---:|"]
    for name, blend in result["pair_blends"].items():
        lines.append(
            f"| `{name}` | {float(blend['candidate_weight']):.3f} | {float(blend['brier']):.9f} | {float(blend['delta_vs_safe']):+.9f} |"
        )
    t = result["triple_blend"]
    lines += [
        "",
        "## Triple blend diagnostic",
        "",
        f"SAFE={t['safe_weight']:.3f}, GPU2={t['gpu2_weight']:.3f}, GPU3={t['gpu3_weight']:.3f}",
        "",
        f"Brier `{t['brier']:.9f}` / delta vs SAFE `{t['delta_vs_safe']:+.9f}`",
        "",
        "## Calibration audit",
        "",
        "```json",
        json.dumps(result["gpu3_calibration"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Promotion rule",
        "",
        "Do not replace SAFE based on this run. A candidate advances only if its 2024 blend gain is non-trivial, its error correlation is meaningfully below 1, and the gain is not isolated to one tiny experience/domain bucket.",
    ]
    return "\n".join(lines) + "\n"


def run(config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    frame, season_col, target_col, _ = _prepare_frame(cfg)
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    valid = frame.loc[season == 2024].copy()
    if valid.empty:
        raise RuntimeError("no 2024 validation rows")
    y_frame = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    row_id = valid["row_id"].astype(str).to_numpy(dtype="U")

    safe_y, safe = _load_safe(resolve_path(cfg["safe"]["predictions"]))
    if len(safe_y) != len(y_frame) or not np.array_equal(safe_y.astype(np.int8), y_frame.astype(np.int8)):
        raise RuntimeError("SAFE predictions are not aligned with the 2024 canonical row order")
    safe_brier = _brier(y_frame, safe)
    expected = float(cfg["safe"].get("expected_brier", safe_brier))
    tolerance = float(cfg["safe"].get("tolerance", 2e-5))
    safe_reference_pass = abs(safe_brier - expected) <= tolerance

    gpu2_cfg, gpu2_trial = _best_gpu2_config(cfg)
    y2, gpu2, gpu2_row_id, gpu2_features = _fit_gpu2_2024(
        frame,
        season_col=season_col,
        target_col=target_col,
        config=gpu2_cfg,
    )
    if not np.array_equal(y2.astype(np.int8), y_frame.astype(np.int8)):
        raise RuntimeError("GPU2 target order differs from SAFE order")
    if not np.array_equal(gpu2_row_id.astype(str), row_id.astype(str)):
        gpu2 = _align_by_row_id(row_id, gpu2_row_id, gpu2)

    gpu3_best, gpu3_artifact, gpu3_base_id = _load_gpu3_artifact(cfg)
    gpu3_vectors, calibration = _gpu3_vectors(gpu3_best, gpu3_artifact, row_id)

    vectors: dict[str, np.ndarray] = {"safe": safe, "gpu2_night_best": gpu2, **gpu3_vectors}
    metrics = {name: classification_metrics(y_frame, pred) for name, pred in vectors.items()}
    pair_diag = {
        name: _pair_diagnostics(y_frame, safe, pred)
        for name, pred in vectors.items()
        if name != "safe"
    }

    blend_cfg = cfg.get("blend", {})
    pair_max = float(blend_cfg.get("pair_max_candidate_weight", 0.50))
    pair_step = float(blend_cfg.get("pair_step", 0.002))
    pair_blends = {
        name: _best_pair_blend(y_frame, safe, pred, max_weight=pair_max, step=pair_step)
        for name, pred in vectors.items()
        if name != "safe"
    }

    # Use the corrected affine GPU3 vector for the triple diagnostic because it is
    # explicitly defined and avoids the overnight label/method naming bug.
    triple = _best_triple_blend(
        y_frame,
        safe,
        gpu2,
        gpu3_vectors["gpu3_affine_corrected"],
        max_total=float(blend_cfg.get("triple_max_total_candidate_weight", 0.50)),
        step=float(blend_cfg.get("triple_step", 0.01)),
    )

    game_type = valid["game_type"].astype("string").fillna("<MISSING>").astype(str).to_numpy()
    prior_n = pd.to_numeric(valid["hist_prev_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    candidate_subset = {
        "gpu2_night_best": gpu2,
        "gpu3_night_calibration": gpu3_vectors["gpu3_night_calibration"],
        "gpu3_affine_corrected": gpu3_vectors["gpu3_affine_corrected"],
    }
    domains = _group_complementarity(
        y_frame,
        safe,
        candidate_subset,
        game_type,
        max_weight=pair_max,
        step=pair_step,
    )
    experience = _group_complementarity(
        y_frame,
        safe,
        candidate_subset,
        _experience_labels(prior_n),
        max_weight=pair_max,
        step=pair_step,
    )

    result = {
        "experiment": "ex6_safe_complementarity",
        "validation_season": 2024,
        "rows": int(len(y_frame)),
        "runtime_seconds": time.time() - started,
        "safe_reference": {
            "expected_brier": expected,
            "observed_brier": safe_brier,
            "tolerance": tolerance,
            "pass": safe_reference_pass,
        },
        "gpu2": {
            "source_trial": gpu2_trial,
            "config": gpu2_cfg,
            "feature_count": len(gpu2_features),
            "features": gpu2_features,
        },
        "gpu3": {
            "source_trial": gpu3_best.get("trial_id"),
            "base_id": gpu3_base_id,
            "base_config": gpu3_best.get("base_config"),
        },
        "gpu3_calibration": calibration,
        "vectors": metrics,
        "pair_diagnostics": pair_diag,
        "pair_blends": pair_blends,
        "triple_blend": triple,
        "game_type": domains,
        "experience": experience,
        "guardrails": [
            "all model features are current-row or frozen historical features",
            "GPU3 calibration parameters are fitted only from artifact rows before 2024",
            "blend search uses 2024 labels only as a complementarity diagnostic and is not a deployable weight selection proof",
            "no test/evaluation-row sorting, shift, rolling, aggregation, or cross-row reference is used",
        ],
    }

    np.savez_compressed(
        output_dir / "vectors_2024.npz",
        y=y_frame.astype(np.int8),
        row_id=row_id,
        **{name: np.asarray(pred, dtype=np.float32) for name, pred in vectors.items()},
    )
    _write_json(output_dir / "metrics.json", result)
    (output_dir / "report.md").write_text(_report_markdown(result), encoding="utf-8")
    return result
