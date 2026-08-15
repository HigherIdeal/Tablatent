from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_frame
from src.utils import load_config, save_json, seed_everything

TARGET = "control_success"

BASE_790 = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before",
    "run_bot_before", "run_total_before", "score_diff_home",
    "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state", "home_win_expectancy", "away_win_expectancy",
    "li", "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id",
    "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

ENGINEERED_790 = [
    "pitcher_success_eb100", "pitcher_success_eb500", "pitcher_reliability_500",
    "batter_success_eb100", "batter_success_eb500", "batter_reliability_500",
    "pitcher_n_log", "batter_n_log", "pitchmix_n_log",
]
REFERENCE_790 = BASE_790 + ENGINEERED_790

CATEGORICAL = {
    "game_month", "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
}

GROUPS = {
    "batter_id": ["batter_id"],
    "pitcher_team_id": ["pitcher_team_id"],
    "batter_team_id": ["batter_team_id"],
    "team_ids": ["pitcher_team_id", "batter_team_id"],
    "retained_ids": ["batter_id", "pitcher_team_id", "batter_team_id"],
    "season": ["season"],
    "game_context": [
        "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
        "balls_before", "strikes_before", "outs_before", "run_top_before",
        "run_bot_before", "run_total_before", "score_diff_home",
        "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        "num_runners_on", "base_state",
    ],
    "leverage": ["home_win_expectancy", "away_win_expectancy", "li"],
    "handedness": ["pitcher_hand", "batter_hand"],
    "pitcher_profile": [
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    ],
    "pitcher_recent": [
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    ],
    "batter_profile": ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"],
    "pitchmix": [
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ],
    "empirical_bayes": [
        "pitcher_success_eb100", "pitcher_success_eb500",
        "batter_success_eb100", "batter_success_eb500",
    ],
    "reliability_logs": [
        "pitcher_reliability_500", "batter_reliability_500",
        "pitcher_n_log", "batter_n_log", "pitchmix_n_log",
    ],
    "all_engineered": ENGINEERED_790,
}

DEFAULT_VARIANTS = [
    "reference_790", "add_pitcher_id", "drop_batter_id", "drop_pitcher_team_id",
    "drop_batter_team_id", "drop_team_ids", "drop_retained_ids", "drop_season",
    "drop_game_context", "drop_leverage", "drop_handedness", "drop_pitcher_profile",
    "drop_pitcher_recent", "drop_batter_profile", "drop_pitchmix",
    "drop_empirical_bayes", "drop_reliability_logs", "drop_all_engineered",
]


def add_790_features(frame: pd.DataFrame, prior: float) -> None:
    pn = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
    pr = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").fillna(prior).clip(0, 1)
    bn = pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)
    br = pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").fillna(prior).clip(0, 1)
    for alpha in (100, 500):
        frame[f"pitcher_success_eb{alpha}"] = (pr * pn + alpha * prior) / (pn + alpha)
        frame[f"batter_success_eb{alpha}"] = (br * bn + alpha * prior) / (bn + alpha)
    frame["pitcher_reliability_500"] = pn / (pn + 500)
    frame["batter_reliability_500"] = bn / (bn + 500)
    frame["pitcher_n_log"] = np.log1p(pn)
    frame["batter_n_log"] = np.log1p(bn)
    pmn = pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0).clip(lower=0)
    frame["pitchmix_n_log"] = np.log1p(pmn)


def variant_features(name: str) -> list[str]:
    if name == "reference_790":
        return list(REFERENCE_790)
    if name == "add_pitcher_id":
        return [*REFERENCE_790, "pitcher_id"]
    if not name.startswith("drop_"):
        raise ValueError(f"Unknown variant: {name}")
    group = name[5:]
    if group not in GROUPS:
        raise ValueError(f"Unknown ablation group: {group}")
    drop = set(GROUPS[group])
    return [f for f in REFERENCE_790 if f not in drop]


def prepare_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = frame[features].copy()
    cats = [f for f in features if f in CATEGORICAL]
    cat_set = set(cats)
    for col in features:
        if col in cat_set:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(np.float32)
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    return x, cats


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    brier = float(np.mean((p - y) ** 2))
    ref = float(y.mean() * (1 - y.mean()))
    return {
        "brier": brier,
        "brier_skill": float(1 - brier / ref),
        "competition_score": float(max(0.0, 100000 * (1 - brier / ref))),
        "auc": float(roc_auc_score(y, p)),
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "target_mean": float(y.mean()),
        "reference_brier": ref,
    }


def run_ablation(config: dict, folds: list[int], variants: list[str], iterations: int,
                 task_type: str, devices: str, verbose: int) -> dict:
    try:
        import catboost
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError("catboost가 없습니다.") from exc

    seed_everything(int(config["seed"]))
    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    # Only raw source columns are required here. Engineered columns are generated
    # fold-by-fold using a prior computed from training seasons only.
    required_raw = set(BASE_790 + [target, season, row_id, "pitcher_id"])
    missing = sorted(required_raw - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)
    out = Path(config["paths"]["output_dir"]) / "catboost_ablation"
    out.mkdir(parents=True, exist_ok=True)
    feature_sets = {v: variant_features(v) for v in variants}
    (out / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict] = []
    print(f"[Ablation] folds={folds}, variants={len(variants)}, iterations={iterations}, "
          f"task_type={task_type}, catboost={catboost.__version__}")
    print("[Ablation] fixed H2/J0-style hyperparameters; no early stopping")

    for val_year in folds:
        use = frame[season] <= val_year
        fold = frame.loc[use].copy()
        tr_mask = fold[season] < val_year
        va_mask = fold[season] == val_year
        if not tr_mask.any() or not va_mask.any():
            raise ValueError(f"Fold {val_year}: empty train or validation split")

        prior = float(pd.to_numeric(fold.loc[tr_mask, target], errors="raise").mean())
        add_790_features(fold, prior)
        train = fold.loc[tr_mask]
        valid = fold.loc[va_mask]
        y_tr = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
        y_va = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float32)
        print(f"\n[Fold {val_year}] train={len(train):,}, val={len(valid):,}, prior={prior:.6f}")

        for i, variant in enumerate(variants, 1):
            feats = feature_sets[variant]
            x_tr, cats = prepare_x(train, feats)
            x_va, _ = prepare_x(valid, feats)
            tr_pool = Pool(x_tr, label=y_tr, cat_features=cats, feature_names=feats)
            va_pool = Pool(x_va, label=y_va, cat_features=cats, feature_names=feats)
            params = dict(
                iterations=int(iterations), learning_rate=0.03, depth=8,
                l2_leaf_reg=10.0, random_strength=0.5, bootstrap_type="Bayesian",
                bagging_temperature=0.5, border_count=128, random_seed=int(config["seed"]),
                loss_function="Logloss", has_time=True, one_hot_max_size=10,
                allow_writing_files=False, task_type=task_type, verbose=verbose,
            )
            if task_type == "GPU":
                params["devices"] = devices
            model = CatBoostClassifier(**params)
            print(f"  [{i:02d}/{len(variants):02d}] {variant:<24s} features={len(feats):2d}", flush=True)
            model.fit(tr_pool, verbose=verbose)
            pred = model.predict_proba(va_pool)[:, 1]
            m = metrics(y_va, pred)
            rows.append({
                "variant": variant, "validation_year": val_year,
                "train_start_year": int(train[season].min()),
                "train_end_year": int(train[season].max()),
                "train_rows": len(train), "val_rows": len(valid),
                "feature_count": len(feats), "categorical_count": len(cats),
                "prior": prior, **m,
            })
            print(f"       brier={m['brier']:.8f} skill={m['brier_skill']:+.3e} "
                  f"auc={m['auc']:.5f} p_std={m['prediction_std']:.5f}")
            del model, tr_pool, va_pool, x_tr, x_va, pred
            gc.collect()
        del fold, train, valid, y_tr, y_va
        gc.collect()

    fold_results = pd.DataFrame(rows)
    ref = fold_results.loc[fold_results.variant == "reference_790", ["validation_year", "brier"]]
    ref = ref.rename(columns={"brier": "reference_variant_brier"})
    fold_results = fold_results.merge(ref, on="validation_year", how="left")
    fold_results["delta_brier_vs_reference"] = (
        fold_results["brier"] - fold_results["reference_variant_brier"]
    )
    fold_results.to_csv(out / "fold_results.csv", index=False)

    summary = (fold_results.groupby("variant", as_index=False).agg(
        folds=("validation_year", "count"), feature_count=("feature_count", "first"),
        mean_brier=("brier", "mean"), worst_brier=("brier", "max"),
        mean_delta_brier=("delta_brier_vs_reference", "mean"),
        worst_delta_brier=("delta_brier_vs_reference", "max"),
        mean_skill=("brier_skill", "mean"), mean_auc=("auc", "mean"),
        mean_prediction_std=("prediction_std", "mean"),
    ).sort_values(["mean_brier", "worst_brier"]).reset_index(drop=True))
    summary.to_csv(out / "summary.csv", index=False)

    result = {
        "reference": "uploaded ~790-point H2/J0 feature set",
        "folds": folds, "variants": variants, "iterations": iterations,
        "output_dir": str(out),
    }
    save_json(result, out / "run_config.json")
    print("\n[Ablation summary] lower mean_delta_brier is better")
    cols = ["variant", "feature_count", "mean_brier", "mean_delta_brier", "worst_delta_brier", "mean_auc"]
    print(summary[cols].to_string(index=False))
    print(f"\nSaved: {out / 'summary.csv'}")
    return result


def parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_strings(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal CatBoost feature ablation around the H2/J0 reference.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--folds", default="2023,2024")
    p.add_argument("--variants", default="all")
    p.add_argument("--iterations", type=int, default=520)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    config = load_config(ROOT / args.config)
    folds = parse_ints(args.folds)
    variants = list(DEFAULT_VARIANTS) if args.variants == "all" else parse_strings(args.variants)
    unknown = [v for v in variants if v not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    run_ablation(config, folds, variants, args.iterations, args.task_type, args.devices, args.verbose)


if __name__ == "__main__":
    main()
