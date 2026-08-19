from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.config import load_config
from bitaboost.ensemble import fit_simplex
from bitaboost.features import prepare


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else _root() / p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("EX8 config root must be a mapping")
    return value


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, np.float64)
    pp = np.asarray(p, np.float64)
    return float(np.mean((yy - pp) ** 2))


def _exp_label(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, np.float64)
    return np.where(
        x <= 0,
        "0",
        np.where(x < 50, "1_49", np.where(x < 200, "50_199", np.where(x < 500, "200_499", "ge500"))),
    ).astype(str)


def _load_vectors(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    wanted = ["y", *names]
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for name in wanted:
            if name not in data.files:
                raise KeyError(f"baseline predictions missing {name}")
            out[name] = np.asarray(data[name], np.float64)
    return out


def _group_block(
    y: np.ndarray,
    vectors: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    oracle_components: list[str],
    min_rows: int,
) -> dict[str, Any]:
    rows = int(mask.sum())
    pred = vectors["pred"]
    block: dict[str, Any] = {
        "rows": rows,
        "target_rate": float(np.mean(y[mask])) if rows else float("nan"),
        "final_brier": _brier(y[mask], pred[mask]) if rows else float("nan"),
        "component_brier": {},
    }
    if rows == 0:
        return block
    for name, values in vectors.items():
        block["component_brier"][name] = _brier(y[mask], values[mask])
    ranked = sorted(block["component_brier"].items(), key=lambda kv: kv[1])
    block["best_component"] = ranked[0][0]
    block["best_component_brier"] = float(ranked[0][1])
    block["best_component_gain_vs_final"] = float(block["final_brier"] - ranked[0][1])

    if rows >= int(min_rows):
        matrix = np.column_stack([vectors[name][mask] for name in oracle_components])
        w = fit_simplex(y[mask], matrix)
        oracle = matrix @ w
        oracle_brier = _brier(y[mask], oracle)
        block["oracle"] = {
            "components": oracle_components,
            "weights": [float(x) for x in w],
            "brier": oracle_brier,
            "gain_vs_final": float(block["final_brier"] - oracle_brier),
        }
    return block


def _routing_summary(groups: dict[str, Any], total_rows: int, overall_final: float) -> dict[str, Any]:
    """Summarize cohort-oracle room without discarding sparse cohorts.

    covered_oracle_brier is computed only over cohorts that meet min_rows.
    fallback_routed_brier uses each eligible cohort's oracle and leaves the existing
    SAFE final prediction unchanged for sparse cohorts.  The latter always covers all
    rows and is the conservative upper-bound diagnostic we actually need.
    """
    oracle_numer = 0.0
    final_numer_covered = 0.0
    fallback_numer = 0.0
    covered = 0
    nonempty = 0
    eligible = 0
    for item in groups.values():
        rows = int(item.get("rows", 0))
        if rows <= 0:
            continue
        nonempty += 1
        final_brier = float(item["final_brier"])
        oracle = item.get("oracle")
        if oracle:
            eligible += 1
            covered += rows
            oracle_numer += rows * float(oracle["brier"])
            final_numer_covered += rows * final_brier
            fallback_numer += rows * float(oracle["brier"])
        else:
            fallback_numer += rows * final_brier
    covered_oracle = oracle_numer / covered if covered else None
    covered_final = final_numer_covered / covered if covered else None
    fallback = fallback_numer / total_rows if total_rows else None
    return {
        "total_rows": int(total_rows),
        "covered_rows": int(covered),
        "coverage": float(covered / total_rows) if total_rows else 0.0,
        "nonempty_groups": int(nonempty),
        "eligible_groups": int(eligible),
        "covered_oracle_brier": covered_oracle,
        "covered_final_brier": covered_final,
        "covered_gain": (float(covered_final - covered_oracle) if covered_oracle is not None else None),
        "fallback_routed_brier": fallback,
        "fallback_gain_vs_safe": (float(overall_final - fallback) if fallback is not None else None),
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# EX8 cohort/component diagnostic",
        "",
        "> Oracle weights are 2024-only upper-bound diagnostics. They are not deployment weights.",
        "> Sparse cohorts below `min_rows_for_oracle` keep the existing SAFE final prediction in the fallback-routed upper bound.",
        "",
        f"SAFE982 final Brier: `{result['overall']['final_brier']:.12f}`",
        "",
        "## Pitcher experience",
        "",
        "| cohort | rows | target | final | best component | best component Brier | oracle Brier | oracle gain |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, item in result["pitcher_groups"].items():
        oracle = item.get("oracle") or {}
        lines.append(
            f"| `{name}` | {item['rows']:,} | {item['target_rate']:.5f} | {item['final_brier']:.9f} | "
            f"`{item.get('best_component','-')}` | {item.get('best_component_brier', float('nan')):.9f} | "
            f"{float(oracle.get('brier', float('nan'))):.9f} | {float(oracle.get('gain_vs_final', float('nan'))):+.9f} |"
        )
    lines += [
        "",
        "## Batter experience",
        "",
        "| cohort | rows | target | final | best component | best component Brier | oracle Brier | oracle gain |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, item in result["batter_groups"].items():
        oracle = item.get("oracle") or {}
        lines.append(
            f"| `{name}` | {item['rows']:,} | {item['target_rate']:.5f} | {item['final_brier']:.9f} | "
            f"`{item.get('best_component','-')}` | {item.get('best_component_brier', float('nan')):.9f} | "
            f"{float(oracle.get('brier', float('nan'))):.9f} | {float(oracle.get('gain_vs_final', float('nan'))):+.9f} |"
        )
    lines += [
        "",
        "## Pitcher x batter cells with >= min rows",
        "",
        "| cell | rows | final | best component | oracle Brier | oracle gain |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for name, item in result["cross_groups"].items():
        if "oracle" not in item:
            continue
        oracle = item["oracle"]
        lines.append(
            f"| `{name}` | {item['rows']:,} | {item['final_brier']:.9f} | `{item.get('best_component','-')}` | "
            f"{oracle['brier']:.9f} | {oracle['gain_vs_final']:+.9f} |"
        )
    lines += ["", "## Routing upper bounds", ""]
    for name in ("pitcher", "batter", "cross"):
        s = result["oracle_summary"][name]
        lines += [
            f"- **{name}**: coverage `{s['coverage']:.3%}` ({s['covered_rows']:,}/{s['total_rows']:,}), "
            f"eligible groups `{s['eligible_groups']}/{s['nonempty_groups']}`, "
            f"covered oracle `{s['covered_oracle_brier']}`, covered gain `{s['covered_gain']}`, "
            f"fallback-routed Brier `{s['fallback_routed_brier']}`, "
            f"gain vs SAFE `{s['fallback_gain_vs_safe']}`",
        ]
    lines += [
        "",
        "Interpretation: `fallback-routed Brier` is the useful upper-bound diagnostic: eligible cohorts receive their 2024 oracle mixture, while sparse cohorts retain SAFE. Only if that gap is material should we build rolling 2022/2023 OOF cohort routers.",
    ]
    return "\n".join(lines) + "\n"


def run(experiment_config: str | Path) -> dict[str, Any]:
    exp = _load_yaml(experiment_config)
    cfg = load_config(_resolve(exp["baseline_config"]))
    out = _resolve(exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    components = [str(x) for x in exp["components"]]
    oracle_components = [str(x) for x in exp["core_oracle_components"]]
    vectors = _load_vectors(_resolve(exp["baseline_predictions"]), components)
    y = vectors.pop("y")

    data = prepare(cfg)
    va = data.frame.loc[data.valid_mask].reset_index(drop=True)
    y_frame = pd.to_numeric(va[cfg["data"]["target_col"]], errors="raise").to_numpy(np.float64)
    if len(y_frame) != len(y) or not np.allclose(y_frame, y, atol=0.0, rtol=0.0):
        raise RuntimeError("EX8 baseline/frame alignment failed")

    pitcher = _exp_label(pd.to_numeric(va["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy())
    batter = _exp_label(pd.to_numeric(va["asof_batter_n"], errors="coerce").fillna(0).to_numpy())
    min_rows = int(exp.get("min_rows_for_oracle", 5000))

    labels = ["0", "1_49", "50_199", "200_499", "ge500"]
    pitcher_groups: dict[str, Any] = {}
    batter_groups: dict[str, Any] = {}
    cross_groups: dict[str, Any] = {}
    for label in labels:
        pitcher_groups[label] = _group_block(y, vectors, pitcher == label, oracle_components=oracle_components, min_rows=min_rows)
        batter_groups[label] = _group_block(y, vectors, batter == label, oracle_components=oracle_components, min_rows=min_rows)
    for p in labels:
        for b in labels:
            mask = (pitcher == p) & (batter == b)
            cross_groups[f"P:{p}|B:{b}"] = _group_block(
                y, vectors, mask, oracle_components=oracle_components, min_rows=min_rows
            )

    overall = _group_block(y, vectors, np.ones(len(y), dtype=bool), oracle_components=oracle_components, min_rows=min_rows)
    overall_final = float(overall["final_brier"])
    oracle_summary = {
        "pitcher": _routing_summary(pitcher_groups, len(y), overall_final),
        "batter": _routing_summary(batter_groups, len(y), overall_final),
        "cross": _routing_summary(cross_groups, len(y), overall_final),
    }
    result = {
        "overall": overall,
        "pitcher_groups": pitcher_groups,
        "batter_groups": batter_groups,
        "cross_groups": cross_groups,
        "oracle_summary": oracle_summary,
        "min_rows_for_oracle": min_rows,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    return result
