from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.baseline import _params, _prepare_x
from bitaboost.config import load_config
from bitaboost.ex1.pitcher_season_backward import _season_profiles
from bitaboost.features import AUX_NAMES, auxiliary_targets, prepare
from bitaboost.metrics import summary
from bitaboost.night.common import attach_history_mode, shrunk_trait, weighted_profile_lookup


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else _root() / p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("EX7 config root must be a mapping")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, np.float64)
    pp = np.asarray(p, np.float64)
    return float(np.mean((yy - pp) ** 2))


def _domain_brier(y: np.ndarray, p: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for dom in ("R", "F"):
        mask = np.asarray(gt).astype(str) == dom
        if mask.any():
            out[dom] = _brier(y[mask], p[mask])
    return out


def _load_baseline_vectors(path: str | Path) -> dict[str, np.ndarray]:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"SAFE baseline vectors not found: {p}. Run `python scripts/baseline_train.py --config configs/baseline_safe_981.yaml` first."
        )
    wanted = (
        "y",
        "direct",
        "reverse600",
        "middle400",
        "gate600",
        "cond400",
        "mixed",
        "offset",
        "joint",
        "structured",
        "safe",
        "pred",
        "safe_weights_R",
        "safe_weights_F",
        "final_weights_R",
        "final_weights_F",
    )
    out: dict[str, np.ndarray] = {}
    with np.load(p, allow_pickle=False) as data:
        for name in wanted:
            if name not in data.files:
                raise KeyError(f"baseline predictions.npz missing {name}")
            out[name] = np.asarray(data[name], dtype=np.float64)
    return out


def _attach_career_traits(
    frame: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    target_col: str,
    traits: list[str],
    reliability_k: float,
) -> tuple[pd.DataFrame, list[str]]:
    profiles = _season_profiles(
        frame,
        aux,
        season_col=season_col,
        pitcher_col=pitcher_col,
        target_col=target_col,
    )
    current_seasons = sorted(
        int(x) for x in pd.to_numeric(frame[season_col], errors="raise").astype(int).unique().tolist()
    )
    lookup = weighted_profile_lookup(
        profiles,
        season_col=season_col,
        pitcher_col=pitcher_col,
        traits=traits,
        mode="career",
        current_seasons=current_seasons,
    )
    frame = attach_history_mode(
        frame,
        lookup,
        season_col=season_col,
        pitcher_col=pitcher_col,
        traits=traits,
        mode="career",
    )

    n = pd.to_numeric(frame["hist_career_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    rel = n / (n + float(reliability_k))
    frame["ex7_career_log_n"] = np.log1p(n).astype(np.float32)
    frame["ex7_career_reliability"] = rel.astype(np.float32)
    frame["ex7_career_has"] = (n > 0).astype(np.float32)
    available = ["career_log_n", "career_reliability", "career_has"]

    for trait in traits:
        values = shrunk_trait(frame, mode="career", trait=trait, k=float(reliability_k))
        frame[f"ex7_career_{trait}"] = np.asarray(values, np.float32)
        available.append(f"career_{trait}")

    current_sources = {
        "success": "asof_pitcher_success_rate",
        "strike": "asof_pitcher_strike_rate",
        "ball": "asof_pitcher_ball_rate",
        "reverse": "asof_pitcher_reverse_rate",
        "middle": "asof_pitcher_middle_rate",
    }
    for trait in traits:
        source = current_sources.get(trait)
        if source and source in frame.columns:
            current = pd.to_numeric(frame[source], errors="coerce").to_numpy(np.float64)
            stable = pd.to_numeric(frame[f"ex7_career_{trait}"], errors="coerce").to_numpy(np.float64)
            dev = current - stable
            # Preserve missingness instead of inventing a deviation where the current-row
            # as-of statistic is unavailable. CatBoost handles numeric NaN directly.
            frame[f"ex7_dev_{trait}"] = dev.astype(np.float32)
            available.append(f"dev_{trait}")
    return frame, available


def _variant_columns(requested: list[str], available: set[str]) -> tuple[list[str], list[str]]:
    mapping = {
        "career_success": "ex7_career_success",
        "career_strike": "ex7_career_strike",
        "career_log_n": "ex7_career_log_n",
        "career_reliability": "ex7_career_reliability",
        "career_has": "ex7_career_has",
        "dev_success": "ex7_dev_success",
        "dev_strike": "ex7_dev_strike",
    }
    columns: list[str] = []
    skipped: list[str] = []
    for key in requested:
        col = mapping.get(str(key))
        if col is None:
            raise KeyError(f"unknown EX7 requested feature {key!r}")
        if str(key) not in available or col not in mapping.values():
            skipped.append(str(key))
            continue
        columns.append(col)
    return columns, skipped


def _fit_direct(
    cfg: dict[str, Any],
    tr: pd.DataFrame,
    va: pd.DataFrame,
    aux_train: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    from catboost import CatBoostRegressor, Pool

    x, cats = _prepare_x(tr, features)
    xv, _ = _prepare_x(va, features)
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
        x.loc[keep],
        labels,
        weight=weights,
        cat_features=cats,
        feature_names=features,
    )
    vp = Pool(xv, cat_features=cats, feature_names=features)
    model = CatBoostRegressor(**_params(cfg, "MultiRMSE"))
    model.fit(pool)
    pred = np.clip(
        model.predict(vp, ntree_end=int(cfg["recipe"]["direct"]["tree"])), 0.0, 1.0
    )[:, 0].astype(np.float64)
    return pred


def _logic_from_baseline(base: dict[str, np.ndarray], recipe: dict[str, Any]) -> np.ndarray:
    c = float(recipe["interaction_c"])
    wi = float(recipe["independent_gate_weight"])
    wg = float(recipe["learned_gate_weight"])
    independent = np.clip(
        1.0 - base["reverse600"] - base["middle400"] + c * base["reverse600"] * base["middle400"],
        0.0,
        1.0,
    )
    return (wi * independent + wg * base["gate600"]) * base["cond400"]


def _frozen_recombine(
    direct: np.ndarray,
    *,
    base: dict[str, np.ndarray],
    gt: np.ndarray,
    baseline_metrics: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, np.ndarray]:
    gt_s = np.asarray(gt).astype(str)
    logic = _logic_from_baseline(base, recipe)
    mixed = np.asarray(direct, np.float64).copy()
    mixed_blend = baseline_metrics.get("mixed_blend") or {}
    for dom in ("R", "F"):
        mask = gt_s == dom
        w = float(mixed_blend[dom])
        mixed[mask] = direct[mask] + w * (logic[mask] - direct[mask])

    safe = np.empty(len(direct), dtype=np.float64)
    final = np.empty(len(direct), dtype=np.float64)
    for dom in ("R", "F"):
        mask = gt_s == dom
        sw = np.asarray(base[f"safe_weights_{dom}"], np.float64).reshape(-1)
        fw = np.asarray(base[f"final_weights_{dom}"], np.float64).reshape(-1)
        if len(sw) != 3 or len(fw) != 2:
            raise RuntimeError(f"unexpected frozen SAFE weights for domain {dom}")
        safe[mask] = np.column_stack(
            [mixed[mask], base["offset"][mask], base["joint"][mask]]
        ) @ sw
        final[mask] = np.column_stack([safe[mask], base["structured"][mask]]) @ fw
    return {"direct": direct, "mixed": mixed, "safe": safe, "final": final}


def _experience_metrics(
    y: np.ndarray, p: np.ndarray, va: pd.DataFrame
) -> dict[str, dict[str, float | int]]:
    n = pd.to_numeric(va["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    labels = np.where(
        n <= 0,
        "no_prior",
        np.where(n < 50, "lt50", np.where(n < 200, "50_199", np.where(n < 500, "200_499", "ge500"))),
    )
    out: dict[str, dict[str, float | int]] = {}
    for label in ("no_prior", "lt50", "50_199", "200_499", "ge500"):
        mask = labels == label
        if mask.any():
            out[label] = {"rows": int(mask.sum()), "brier": _brier(y[mask], p[mask])}
    return out


def _report(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    lines = [
        "# EX7 stable-trait injection report",
        "",
        "> Only the SAFE **direct MultiRMSE head** is retrained. Reverse/middle, hurdle, offset, joint, structured, mixed-domain weights, SAFE simplex weights, and final weights are frozen from SAFE982. This prevents a 2024 blend re-fit from hiding the real effect of the injected features.",
        "",
        "## Final 2024 Brier",
        "",
        "| variant | extra features | direct | mixed | final | delta final vs SAFE982 | R delta | F delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    base_final = float(baseline["final_brier"])
    base_domain = baseline["domain_final_brier"]
    for item in result["variants"]:
        dom = item["domain_final_brier"]
        lines.append(
            f"| `{item['name']}` | {', '.join(item['extra_features']) or '-'} | "
            f"{item['direct_brier']:.9f} | {item['mixed_brier']:.9f} | {item['final_brier']:.9f} | "
            f"{item['final_brier'] - base_final:+.9f} | "
            f"{dom.get('R', float('nan')) - base_domain.get('R', float('nan')):+.9f} | "
            f"{dom.get('F', float('nan')) - base_domain.get('F', float('nan')):+.9f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- SAFE982 final Brier: `{base_final:.12f}`",
        f"- best variant: `{result['best']['name']}`",
        f"- best final Brier: `{result['best']['final_brier']:.12f}`",
        f"- delta: `{result['best']['final_brier'] - base_final:+.12f}`",
        f"- promotion candidate: `{result['best']['promotion_candidate']}`",
        "",
        "The promotion flag is conservative: the final Brier must improve by the configured minimum and neither R nor F may regress beyond the configured domain tolerance. No ensemble weight is optimized on 2024 in EX7.",
    ]
    return "\n".join(lines) + "\n"


def run(experiment_config: str | Path) -> dict[str, Any]:
    exp = _load_yaml(experiment_config)
    cfg = load_config(_resolve(exp["baseline_config"]))
    baseline_metrics = _load_json(exp["baseline_metrics"])
    base = _load_baseline_vectors(exp["baseline_predictions"])

    output = _resolve(exp["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = prepare(cfg)
    frame = data.frame.copy()
    season_col = cfg["data"]["season_col"]
    target_col = cfg["data"]["target_col"]
    pitcher_col = "pitcher_id"

    profile_cfg = exp.get("profile") or {}
    traits = [str(x) for x in profile_cfg.get("traits", ["success", "strike"])]
    k = float(profile_cfg.get("reliability_k", 200.0))
    if str(profile_cfg.get("mode", "career")) != "career":
        raise ValueError("EX7 is intentionally restricted to career frozen profiles")

    frame, available_names = _attach_career_traits(
        frame,
        data.aux,
        season_col=season_col,
        pitcher_col=pitcher_col,
        target_col=target_col,
        traits=traits,
        reliability_k=k,
    )
    available = set(available_names)
    train_mask = pd.to_numeric(frame[season_col], errors="raise").astype(int).lt(2024).to_numpy()
    valid_mask = pd.to_numeric(frame[season_col], errors="raise").astype(int).eq(2024).to_numpy()
    tr = frame.loc[train_mask].reset_index(drop=True)
    va = frame.loc[valid_mask].reset_index(drop=True)
    aux_train = auxiliary_targets(tr)
    y = pd.to_numeric(va[target_col], errors="raise").to_numpy(np.float64)
    gt = va["game_type"].astype(str).to_numpy()

    if len(y) != len(base["pred"]) or not np.allclose(y, base["y"], atol=0.0, rtol=0.0):
        raise RuntimeError("EX7 row alignment disagrees with SAFE baseline vectors")

    baseline = {
        "direct_brier": _brier(y, base["direct"]),
        "mixed_brier": _brier(y, base["mixed"]),
        "safe_brier": _brier(y, base["safe"]),
        "final_brier": _brier(y, base["pred"]),
        "domain_final_brier": _domain_brier(y, base["pred"], gt),
    }

    variants: list[dict[str, Any]] = []
    vector_payload: dict[str, np.ndarray] = {"y": y, "safe982": base["pred"]}
    rich_base = list(data.feature_sets["rich"])
    promotion = exp.get("promotion") or {}
    min_gain = float(promotion.get("min_final_brier_gain", 0.00002))
    max_domain_regression = float(promotion.get("max_domain_regression", 0.00020))

    for spec in exp.get("variants", []):
        name = str(spec["name"])
        requested = [str(x) for x in spec.get("features", [])]
        extra, skipped = _variant_columns(requested, available)
        features = list(dict.fromkeys([*rich_base, *extra]))
        t0 = time.perf_counter()
        direct = _fit_direct(cfg, tr, va, aux_train, features)
        comp = _frozen_recombine(
            direct,
            base=base,
            gt=gt,
            baseline_metrics=baseline_metrics,
            recipe=cfg["recipe"]["mixed"],
        )
        domain = _domain_brier(y, comp["final"], gt)
        final_brier = _brier(y, comp["final"])
        domain_regressions = {
            dom: domain[dom] - baseline["domain_final_brier"][dom]
            for dom in domain
        }
        gain = baseline["final_brier"] - final_brier
        promote = bool(
            gain >= min_gain
            and all(v <= max_domain_regression for v in domain_regressions.values())
        )
        item = {
            "name": name,
            "requested_features": requested,
            "extra_features": extra,
            "skipped_features": skipped,
            "feature_count": len(features),
            "runtime_seconds": time.perf_counter() - t0,
            "direct_brier": _brier(y, comp["direct"]),
            "mixed_brier": _brier(y, comp["mixed"]),
            "safe_brier": _brier(y, comp["safe"]),
            "final_brier": final_brier,
            "delta_final_vs_safe982": final_brier - baseline["final_brier"],
            "domain_final_brier": domain,
            "domain_regression": domain_regressions,
            "experience_final_brier": _experience_metrics(y, comp["final"], va),
            "promotion_candidate": promote,
        }
        variants.append(item)
        vector_payload[f"direct__{name}"] = comp["direct"]
        vector_payload[f"final__{name}"] = comp["final"]
        print(
            f"[EX7:{name}] direct={item['direct_brier']:.9f} mixed={item['mixed_brier']:.9f} "
            f"final={item['final_brier']:.9f} delta={item['delta_final_vs_safe982']:+.9f} "
            f"features=+{len(extra)} skipped={skipped}",
            flush=True,
        )

    if not variants:
        raise RuntimeError("EX7 variants list is empty")
    variants.sort(key=lambda x: float(x["final_brier"]))
    best = dict(variants[0])
    result = {
        "experiment": "EX7 stable trait injection",
        "method": "retrain SAFE direct head only; frozen downstream SAFE982 components and weights",
        "profile": {"mode": "career", "traits": traits, "reliability_k": k},
        "available_synthetic_features": sorted(available),
        "baseline": baseline,
        "variants": variants,
        "best": best,
        "promotion_thresholds": {
            "min_final_brier_gain": min_gain,
            "max_domain_regression": max_domain_regression,
        },
        "runtime_seconds": time.perf_counter() - started,
        "safety": exp.get("safety") or {},
    }
    _write_json(output / "metrics.json", result)
    np.savez_compressed(output / "vectors_2024.npz", **vector_payload)
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
