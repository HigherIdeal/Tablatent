from __future__ import annotations

import argparse
import gc
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_game_type_temporal_regime_ablation as regime_core
import run_pitcher_trajectory_probe as trajectory_core
from src.utils import load_config, save_json, seed_everything


MODEL_NAMES = (
    "T0_REGIME",
    "T1_SUCCESS_CURVATURE",
    "T3_MIDDLE_TRAJECTORY",
)


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate integers: {values}")
    return sorted(values)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return regime_core.binary_metrics(
        np.asarray(y, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
    )


def gate_catalog(thresholds: list[int]) -> list[dict[str, int | str | None]]:
    """Build a deliberately small, interpretable gate family.

    G0: T0 everywhere.
    G1: low n -> T3, otherwise T0.
    G2: one n band -> T1, otherwise T0.
    G3: low n -> T3; a disjoint higher n band -> T1; otherwise T0.

    G3 uses low_end < t1_start < t1_end so the two specialists never overlap and
    at least one baseline region exists between them. This prevents arbitrary
    per-bin routing and keeps the strict temporal search low-capacity.
    """
    gates: list[dict[str, int | str | None]] = [
        {
            "gate": "G0_BASE",
            "family": "G0",
            "low_end": None,
            "t1_start": None,
            "t1_end": None,
        }
    ]

    for t in thresholds:
        gates.append(
            {
                "gate": f"G1_LOW_T3_LT{t}",
                "family": "G1",
                "low_end": int(t),
                "t1_start": None,
                "t1_end": None,
            }
        )

    for lo, hi in itertools.combinations(thresholds, 2):
        gates.append(
            {
                "gate": f"G2_T1_{lo}_{hi}",
                "family": "G2",
                "low_end": None,
                "t1_start": int(lo),
                "t1_end": int(hi),
            }
        )

    for low_end, t1_start, t1_end in itertools.combinations(thresholds, 3):
        gates.append(
            {
                "gate": f"G3_T3_LT{low_end}_T1_{t1_start}_{t1_end}",
                "family": "G3",
                "low_end": int(low_end),
                "t1_start": int(t1_start),
                "t1_end": int(t1_end),
            }
        )
    return gates


def routed_prediction(
    n: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p3: np.ndarray,
    gate: dict[str, int | str | None],
) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=np.float64)
    pred = np.asarray(p0, dtype=np.float64).copy()
    route = np.full(len(pred), "T0", dtype="<U2")
    finite = np.isfinite(n)

    low_end = gate.get("low_end")
    t1_start = gate.get("t1_start")
    t1_end = gate.get("t1_end")

    if low_end is not None:
        mask = finite & (n < float(low_end))
        pred[mask] = np.asarray(p3, dtype=np.float64)[mask]
        route[mask] = "T3"

    if t1_start is not None and t1_end is not None:
        mask = finite & (n >= float(t1_start)) & (n < float(t1_end))
        pred[mask] = np.asarray(p1, dtype=np.float64)[mask]
        route[mask] = "T1"

    return pred, route


def pooled_gate_score(
    fold_payload: dict[int, dict[str, np.ndarray]],
    years: list[int],
    gate: dict[str, int | str | None],
) -> tuple[float, dict[int, float], dict[int, float]]:
    se_sum = 0.0
    n_sum = 0
    fold_brier: dict[int, float] = {}
    fold_db: dict[int, float] = {}

    for year in years:
        d = fold_payload[year]
        pred, _ = routed_prediction(
            d["n"],
            d["T0_REGIME"],
            d["T1_SUCCESS_CURVATURE"],
            d["T3_MIDDLE_TRAJECTORY"],
            gate,
        )
        y = d["y"]
        brier = float(np.mean((y - pred) ** 2))
        base = float(np.mean((y - d["T0_REGIME"]) ** 2))
        fold_brier[year] = brier
        fold_db[year] = brier - base
        se_sum += float(np.sum((y - pred) ** 2))
        n_sum += int(len(y))

    return se_sum / n_sum, fold_brier, fold_db


def choose_gate(
    sweep: pd.DataFrame,
    earlier: list[int],
    *,
    robust: bool,
) -> pd.Series:
    work = sweep.copy()
    if robust:
        db_cols = [f"dB_{year}" for year in earlier]
        eligible = np.ones(len(work), dtype=bool)
        for col in db_cols:
            eligible &= work[col].to_numpy(np.float64) <= 0.0
        work = work.loc[eligible].copy()
        if work.empty:
            work = sweep.loc[sweep["family"].eq("G0")].copy()

    # Prefer lower pooled Brier; on exact ties prefer simpler gate family then name.
    complexity = {"G0": 0, "G1": 1, "G2": 1, "G3": 2}
    work["_complexity"] = work["family"].map(complexity).fillna(99).astype(int)
    work = work.sort_values(
        ["pooled_brier", "_complexity", "gate"],
        ascending=[True, True, True],
        kind="stable",
    )
    return work.iloc[0]


def evaluate_latest(
    d: dict[str, np.ndarray],
    gate_row: pd.Series,
    *,
    selection_name: str,
    selected_from: list[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    gate = {
        "gate": str(gate_row["gate"]),
        "family": str(gate_row["family"]),
        "low_end": None if pd.isna(gate_row["low_end"]) else int(gate_row["low_end"]),
        "t1_start": None if pd.isna(gate_row["t1_start"]) else int(gate_row["t1_start"]),
        "t1_end": None if pd.isna(gate_row["t1_end"]) else int(gate_row["t1_end"]),
    }
    pred, route = routed_prediction(
        d["n"],
        d["T0_REGIME"],
        d["T1_SUCCESS_CURVATURE"],
        d["T3_MIDDLE_TRAJECTORY"],
        gate,
    )

    y = d["y"]
    gt = d["gt"].astype(str)
    masks: dict[str, np.ndarray] = {
        "ALL": np.ones(len(y), dtype=bool),
        "R": gt == "R",
        "F": gt == "F",
        "ROUTE_T0": route == "T0",
        "ROUTE_T1": route == "T1",
        "ROUTE_T3": route == "T3",
    }

    rows: list[dict] = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        m = metrics(y[mask], pred[mask])
        b = metrics(y[mask], d["T0_REGIME"][mask])
        n_group = d["n"][mask]
        rows.append(
            {
                "selection": selection_name,
                "gate": gate["gate"],
                "family": gate["family"],
                "selected_from_folds": ",".join(map(str, selected_from)),
                "group": group,
                "rows": int(mask.sum()),
                "n_median": float(np.nanmedian(n_group)) if np.isfinite(n_group).any() else np.nan,
                **m,
                "delta_brier_vs_T0": float(m["brier"] - b["brier"]),
            }
        )
    return pd.DataFrame(rows), pred, route


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict temporal prediction-level gate for pitcher trajectory experts. "
            "All three CatBoost experts are trained on every prior-season row. Gate thresholds "
            "use only asof_pitcher_n from the current row. The latest-fold gate is selected "
            "only from earlier validation folds."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument(
        "--thresholds",
        default="250,500,1000,2000,3000,4000,6000",
        help="Candidate asof_pitcher_n thresholds used by the low-capacity gate search.",
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--trend-eps", type=float, default=0.005)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/trajectory_experience_gate")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.trend_eps < 0.0:
        raise ValueError("--trend-eps must be >= 0")
    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

    folds = parse_ints(args.folds)
    thresholds = parse_ints(args.thresholds)
    if len(folds) < 2:
        raise ValueError("At least two folds are required for a strict temporal selection test")
    if any(t <= 0 for t in thresholds):
        raise ValueError("All thresholds must be positive")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)

    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    unexpected = sorted(set(frame["game_type"].dropna().unique()) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    trajectory_core.add_pitcher_trajectory_features(frame, trend_eps=args.trend_eps)
    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    base_features = recent_core.feature_set("recent_raw_game_type")
    all_sets = trajectory_core.feature_sets(base_features)
    feature_sets = {name: all_sets[name] for name in MODEL_NAMES}

    params = regime_core.build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gates = gate_catalog(thresholds)
    tqdm.write(
        f"Trajectory experience gate | folds={folds} | thresholds={thresholds} | gates={len(gates)} "
        f"| rows={len(frame):,} | GPU={args.devices if args.task_type == 'GPU' else 'CPU'} "
        f"| iterations={args.iterations} | catboost={catboost.__version__}"
    )
    tqdm.write(
        "STRICT: T0/T1/T3 are global models trained on all prior rows; no experience specialist is trained. "
        "Latest-fold n-gate selection uses earlier validation folds only."
    )

    fold_payload: dict[int, dict[str, np.ndarray]] = {}
    progress = tqdm(
        total=len(folds) * len(MODEL_NAMES),
        desc="fit global trajectory experts",
        unit="model",
        dynamic_ncols=True,
    )

    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col].eq(val_year)].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid")

        payload: dict[str, np.ndarray] = {
            "y": pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64),
            "gt": valid["game_type"].astype(str).to_numpy(),
            "n": pd.to_numeric(valid["asof_pitcher_n"], errors="coerce").to_numpy(np.float64),
        }
        if row_id_col in valid.columns:
            payload["row_id"] = valid[row_id_col].to_numpy()

        for name in MODEL_NAMES:
            features, extra_cat = feature_sets[name]
            seed_everything(seed)
            payload[name] = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target_col,
                features=features,
                extra_categorical=extra_cat,
                params=params,
            )
            progress.update(1)

        fold_payload[int(val_year)] = payload
        del train, valid
        gc.collect()

    progress.close()

    # Core-model metrics, useful for reproducing the previous trajectory experiment.
    core_rows: list[dict] = []
    for year in folds:
        d = fold_payload[year]
        for name in MODEL_NAMES:
            for group, mask in (
                ("ALL", np.ones(len(d["y"]), dtype=bool)),
                ("R", d["gt"] == "R"),
                ("F", d["gt"] == "F"),
            ):
                if not mask.any():
                    continue
                m = metrics(d["y"][mask], d[name][mask])
                b = metrics(d["y"][mask], d["T0_REGIME"][mask])
                core_rows.append(
                    {
                        "fold": int(year),
                        "model": name,
                        "group": group,
                        "rows": int(mask.sum()),
                        **m,
                        "delta_brier_vs_T0": float(m["brier"] - b["brier"]),
                    }
                )
    core_df = pd.DataFrame(core_rows)
    core_df.to_csv(output_dir / "core_model_metrics.csv", index=False)

    latest = max(folds)
    earlier = [year for year in folds if year < latest]

    # Gate search uses ONLY earlier validation folds for strict latest-fold selection.
    sweep_rows: list[dict] = []
    for gate in gates:
        pooled_brier, fold_brier, fold_db = pooled_gate_score(fold_payload, earlier, gate)
        row: dict[str, object] = {
            **gate,
            "pooled_brier": float(pooled_brier),
        }
        for year in earlier:
            row[f"brier_{year}"] = float(fold_brier[year])
            row[f"dB_{year}"] = float(fold_db[year])
        sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df = sweep_df.sort_values(["pooled_brier", "gate"], kind="stable").reset_index(drop=True)
    sweep_df.to_csv(output_dir / "strict_gate_search_earlier_folds.csv", index=False)

    selected_pooled = choose_gate(sweep_df, earlier, robust=False)
    selected_robust = choose_gate(sweep_df, earlier, robust=True)

    latest_rows: list[pd.DataFrame] = []
    latest_predictions: dict[str, np.ndarray] = {}
    latest_routes: dict[str, np.ndarray] = {}
    for selection_name, selected in (
        ("STRICT_POOLED", selected_pooled),
        ("STRICT_ROBUST", selected_robust),
    ):
        df, pred, route = evaluate_latest(
            fold_payload[latest],
            selected,
            selection_name=selection_name,
            selected_from=earlier,
        )
        latest_rows.append(df)
        latest_predictions[selection_name] = pred
        latest_routes[selection_name] = route

    latest_df = pd.concat(latest_rows, ignore_index=True)
    latest_df.to_csv(output_dir / "strict_latest_fold_metrics.csv", index=False)

    # Oracle latest-fold gate is diagnostic headroom only; it is never used for selection.
    oracle_rows: list[dict] = []
    d_latest = fold_payload[latest]
    for gate in gates:
        pred, route = routed_prediction(
            d_latest["n"],
            d_latest["T0_REGIME"],
            d_latest["T1_SUCCESS_CURVATURE"],
            d_latest["T3_MIDDLE_TRAJECTORY"],
            gate,
        )
        m = metrics(d_latest["y"], pred)
        b = metrics(d_latest["y"], d_latest["T0_REGIME"])
        oracle_rows.append(
            {
                **gate,
                "brier": m["brier"],
                "score": m["score"],
                "delta_brier_vs_T0": float(m["brier"] - b["brier"]),
                "route_T0_share": float(np.mean(route == "T0")),
                "route_T1_share": float(np.mean(route == "T1")),
                "route_T3_share": float(np.mean(route == "T3")),
            }
        )
    oracle_df = pd.DataFrame(oracle_rows).sort_values("brier", kind="stable").reset_index(drop=True)
    oracle_df.to_csv(output_dir / "oracle_latest_gate_diagnostic_DO_NOT_SELECT.csv", index=False)

    np.savez_compressed(
        output_dir / f"validation_{latest}_predictions.npz",
        y=d_latest["y"],
        gt=d_latest["gt"],
        asof_pitcher_n=d_latest["n"],
        t0=d_latest["T0_REGIME"],
        t1=d_latest["T1_SUCCESS_CURVATURE"],
        t3=d_latest["T3_MIDDLE_TRAJECTORY"],
        strict_pooled=latest_predictions["STRICT_POOLED"],
        strict_robust=latest_predictions["STRICT_ROBUST"],
        strict_pooled_route=latest_routes["STRICT_POOLED"],
        strict_robust_route=latest_routes["STRICT_ROBUST"],
    )

    manifest = {
        "experiment": "strict temporal trajectory x experience prediction gate",
        "folds": folds,
        "latest_holdout_fold": int(latest),
        "selection_folds": earlier,
        "thresholds": thresholds,
        "gate_count": len(gates),
        "models": list(MODEL_NAMES),
        "strict_pooled_gate": {
            "gate": str(selected_pooled["gate"]),
            "family": str(selected_pooled["family"]),
            "low_end": None if pd.isna(selected_pooled["low_end"]) else int(selected_pooled["low_end"]),
            "t1_start": None if pd.isna(selected_pooled["t1_start"]) else int(selected_pooled["t1_start"]),
            "t1_end": None if pd.isna(selected_pooled["t1_end"]) else int(selected_pooled["t1_end"]),
            "pooled_brier": float(selected_pooled["pooled_brier"]),
        },
        "strict_robust_gate": {
            "gate": str(selected_robust["gate"]),
            "family": str(selected_robust["family"]),
            "low_end": None if pd.isna(selected_robust["low_end"]) else int(selected_robust["low_end"]),
            "t1_start": None if pd.isna(selected_robust["t1_start"]) else int(selected_robust["t1_start"]),
            "t1_end": None if pd.isna(selected_robust["t1_end"]) else int(selected_robust["t1_end"]),
            "pooled_brier": float(selected_robust["pooled_brier"]),
        },
        "oracle_latest_gate_diagnostic_only": str(oracle_df.iloc[0]["gate"]),
        "inference_contract": (
            "All experts are global full-prior-data CatBoost models. The gate uses only current-row "
            "asof_pitcher_n. No player lookup, validation peer row, rolling reconstruction, or target-derived "
            "inference feature is used. Latest-fold gate selection never sees latest-fold labels."
        ),
        "canonical_invariants": invariant_check,
    }
    save_json(manifest, output_dir / "manifest.json")

    tqdm.write("\n[Earlier-fold gate selection]")
    for label, row in (("STRICT_POOLED", selected_pooled), ("STRICT_ROBUST", selected_robust)):
        db_text = " ".join(f"dB{year}={row[f'dB_{year}']:+.8f}" for year in earlier)
        tqdm.write(
            f"{label:<14s} gate={row['gate']} pooled={row['pooled_brier']:.8f} {db_text}"
        )

    tqdm.write(f"\n[Strict temporal latest-fold test | validation={latest}]")
    for selection in ("STRICT_POOLED", "STRICT_ROBUST"):
        sub = latest_df.loc[latest_df["selection"].eq(selection)]
        gate_name = str(sub.iloc[0]["gate"])
        tqdm.write(f"{selection} gate={gate_name}")
        for group in ("ALL", "R", "F", "ROUTE_T0", "ROUTE_T1", "ROUTE_T3"):
            row = sub.loc[sub["group"].eq(group)]
            if row.empty:
                continue
            r = row.iloc[0]
            tqdm.write(
                f"  {group:<8s} rows={int(r['rows']):>7,d} brier={r['brier']:.8f} "
                f"score={r['score']:+9.2f} dB={r['delta_brier_vs_T0']:+.8f} n_med={r['n_median']:.1f}"
            )

    oracle = oracle_df.iloc[0]
    tqdm.write("\n[Oracle latest diagnostic | DO NOT SELECT FROM THIS]")
    tqdm.write(
        f"gate={oracle['gate']} brier={oracle['brier']:.8f} score={oracle['score']:+.2f} "
        f"dB={oracle['delta_brier_vs_T0']:+.8f}"
    )
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
