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


def _weighted_routed_brier(groups: dict[str, Any], total_rows: int) -> float | None:
    numer = 0.0
    covered = 0
    for item in groups.values():
        oracle = item.get("oracle")
        if not oracle:
            continue
        rows = int(item["rows"])
        numer += rows * float(oracle["brier"])
        covered += rows
    if covered != total_rows:
        return None
    return numer / covered


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# EX8 cohort/component diagnostic",
        "",
        "> Oracle weights are 2024-only upper-bound diagnostics. They are not deployment weights.",
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
    lines += [
        "",
        "## Global oracle upper bounds",
        "",
        f"- pitcher-routed oracle Brier: `{result['oracle_summary']['pitcher_routed_brier']}`",
        f"- batter-routed oracle Brier: `{result['oracle_summary']['batter_routed_brier']}`",
        f"- cross-routed oracle Brier: `{result['oracle_summary']['cross_routed_brier']}`",
        "",
        "Interpretation: only if a cohort routing scheme shows a material 2024 upper-bound gap should we spend GPU time building rolling 2022/2023 OOF cohort weights or specialized experts.",
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
    oracle_summary = {
        "pitcher_routed_brier": _weighted_routed_brier(pitcher_groups, len(y)),
        "batter_routed_brier": _weighted_routed_brier(batter_groups, len(y)),
        "cross_routed_brier": _weighted_routed_brier(cross_groups, len(y)),
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
