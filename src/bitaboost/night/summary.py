from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from bitaboost.night.common import atomic_write_json, atomic_write_text, read_jsonl, resolve_path, utc_timestamp


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _worker_state(path: Path) -> dict[str, Any]:
    trials = read_jsonl(path / "trials.jsonl")
    ok = [x for x in trials if x.get("status") == "ok"]
    ok.sort(key=lambda x: float(x.get("objective", float("inf"))))
    errors = [x for x in trials if x.get("status") == "error"]
    heartbeat = _load_json(path / "heartbeat.json")
    final = _load_json(path / "final_summary.json")
    return {
        "trials_total": len(trials),
        "trials_ok": len(ok),
        "errors": len(errors),
        "best": ok[0] if ok else None,
        "top10": ok[:10],
        "heartbeat": heartbeat,
        "final": final,
    }


def _gpu2_patterns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = rows[:20]
    traits = Counter()
    modes = Counter()
    feature_kinds = Counter()
    temporal = Counter()
    losses = Counter()
    for row in top:
        cfg = row.get("config") or {}
        traits["+".join(cfg.get("traits", []))] += 1
        modes[str(cfg.get("history_mode"))] += 1
        feature_kinds[str(cfg.get("feature_kind"))] += 1
        temporal[str(cfg.get("temporal_weight"))] += 1
        losses[str(cfg.get("loss"))] += 1
    return {
        "top20_trait_sets": dict(traits.most_common()),
        "top20_history_modes": dict(modes.most_common()),
        "top20_feature_kinds": dict(feature_kinds.most_common()),
        "top20_temporal_weights": dict(temporal.most_common()),
        "top20_losses": dict(losses.most_common()),
    }


def _gpu3_patterns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = rows[:20]
    methods = Counter()
    groups = Counter()
    base_kinds = Counter()
    for row in top:
        method = row.get("method") or {}
        methods[str(method.get("kind"))] += 1
        if method.get("group_kind") is not None:
            groups[str(method.get("group_kind"))] += 1
        base = row.get("base_config") or {}
        base_kinds[str(base.get("feature_kind"))] += 1
    return {
        "top20_methods": dict(methods.most_common()),
        "top20_domain_groups": dict(groups.most_common()),
        "top20_base_feature_kinds": dict(base_kinds.most_common()),
    }


def _fmt_fold(row: dict[str, Any], season: int) -> str:
    fold = (row.get("folds") or {}).get(str(season)) or {}
    value = fold.get("brier")
    return "-" if value is None else f"{float(value):.8f}"


def refresh(root: str | Path) -> dict[str, Any]:
    root = resolve_path(root)
    root.mkdir(parents=True, exist_ok=True)
    g2 = _worker_state(root / "gpu2")
    g3 = _worker_state(root / "gpu3")
    state = {
        "updated_utc": utc_timestamp(),
        "gpu2": g2,
        "gpu3": g3,
        "complete": bool(g2.get("final") and g3.get("final")),
        "patterns": {
            "gpu2": _gpu2_patterns(g2["top10"]),
            "gpu3": _gpu3_patterns(g3["top10"]),
        },
    }
    atomic_write_json(root / "campaign_state.json", state)

    lines = [
        "# Overnight campaign live report",
        "",
        f"Updated: `{state['updated_utc']}`",
        "",
        "## Worker status",
        "",
        "| worker | trials | ok | errors | phase | seconds left |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, worker in (("GPU2 structure", g2), ("GPU3 calibration", g3)):
        hb = worker.get("heartbeat") or {}
        lines.append(
            f"| {name} | {worker['trials_total']} | {worker['trials_ok']} | {worker['errors']} | "
            f"{hb.get('phase', '-')} | {float(hb.get('seconds_left', 0.0)):.0f} |"
        )

    lines += [
        "",
        "## Current best",
        "",
        "| worker | trial | objective | weighted Brier | 2022 | 2023 | 2024 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, worker in (("GPU2", g2), ("GPU3", g3)):
        row = worker.get("best")
        if not row:
            continue
        lines.append(
            f"| {name} | `{row.get('trial_id')}` | {float(row.get('objective', float('nan'))):.9f} | "
            f"{float(row.get('weighted_brier', float('nan'))):.9f} | {_fmt_fold(row, 2022)} | {_fmt_fold(row, 2023)} | {_fmt_fold(row, 2024)} |"
        )

    lines += ["", "## GPU2 top trials", ""]
    for rank, row in enumerate(g2["top10"], start=1):
        cfg = row.get("config") or {}
        lines.append(
            f"{rank}. `{row.get('trial_id')}` obj={float(row.get('objective', float('nan'))):.9f} "
            f"traits=`{'+'.join(cfg.get('traits', []))}` mode=`{cfg.get('history_mode')}` "
            f"features=`{cfg.get('feature_kind')}` temporal=`{cfg.get('temporal_weight')}` loss=`{cfg.get('loss')}`"
        )
    lines += ["", "## GPU3 top trials", ""]
    for rank, row in enumerate(g3["top10"], start=1):
        lines.append(
            f"{rank}. `{row.get('trial_id')}` obj={float(row.get('objective', float('nan'))):.9f} "
            f"base=`{row.get('base_id')}` method=`{json.dumps(row.get('method', {}), sort_keys=True)}`"
        )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Ranking objective weights 2022/2023/2024 as 0.20/0.30/0.50 and adds a cross-fold spread penalty.",
        "- Fold-s frozen profiles use only seasons before s.",
        "- GPU3 calibration/stacking for fold s is fitted only on earlier OOF folds.",
        "- SAFE982 is not re-calibrated against 2024 labels because equivalent earlier SAFE OOF vectors are unavailable.",
        "",
    ]
    atomic_write_text(root / "overnight_report.md", "\n".join(lines))
    return state


def watch(root: str | Path, *, interval_seconds: float = 60.0, hours: float = 8.5) -> None:
    deadline = time.time() + float(hours) * 3600.0
    while time.time() < deadline:
        state = refresh(root)
        if state.get("complete"):
            return
        time.sleep(max(5.0, float(interval_seconds)))
    refresh(root)
