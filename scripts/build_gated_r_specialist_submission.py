from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_gated_r_specialist_suite as gated_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config, save_json, seed_everything


FULL_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
RECENT_TRAIN_SEASONS = [2023, 2024]
REQUIREMENTS = "catboost==1.2.10\n"


def gated_prediction(
    p_full: np.ndarray,
    p_recent: np.ndarray,
    p_r_fast: np.ndarray,
    is_r: np.ndarray,
    alpha_recent: float,
    beta_r: float,
) -> np.ndarray:
    p_full = np.asarray(p_full, dtype=np.float64)
    p_recent = np.asarray(p_recent, dtype=np.float64)
    p_r_fast = np.asarray(p_r_fast, dtype=np.float64)
    is_r = np.asarray(is_r, dtype=bool)
    if not (p_full.shape == p_recent.shape == p_r_fast.shape == is_r.shape):
        raise ValueError("prediction/mask shape mismatch")
    base = (1.0 - alpha_recent) * p_full + alpha_recent * p_recent
    out = base.copy()
    out[is_r] = (1.0 - beta_r) * base[is_r] + beta_r * p_r_fast[is_r]
    return out


def _inference_script(
    full_features: list[str],
    recent_features: list[str],
    r_fast_features: list[str],
    full_categorical: list[str],
    recent_categorical: list[str],
    r_fast_categorical: list[str],
    alpha_recent: float,
    beta_r: float,
) -> str:
    return f'''from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
FULL_FEATURES = {full_features!r}
RECENT_FEATURES = {recent_features!r}
R_FAST_FEATURES = {r_fast_features!r}
FULL_CATEGORICAL = {full_categorical!r}
RECENT_CATEGORICAL = {recent_categorical!r}
R_FAST_CATEGORICAL = {r_fast_categorical!r}
ALPHA_RECENT = {float(alpha_recent)!r}
BETA_R = {float(beta_r)!r}


def find_data_dir() -> Path:
    for name in ("data", "open"):
        candidate = ROOT / name
        if (candidate / "test.csv").is_file():
            return candidate
    raise FileNotFoundError("test.csv not found under ./data or ./open")


def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype(np.float32)


def add_features(frame: pd.DataFrame) -> None:
    top_bottom = frame["top_bottom"].astype(str)
    unknown = sorted(set(top_bottom.unique()) - {{"T", "B"}})
    if unknown:
        raise ValueError(f"Unexpected top_bottom values: {{unknown}}")
    home = pd.to_numeric(frame["home_win_expectancy"], errors="coerce")
    away = pd.to_numeric(frame["away_win_expectancy"], errors="coerce")
    frame["pitcher_team_win_expectancy"] = np.where(top_bottom.eq("T"), home, away)

    ps_long = numeric(frame, "asof_pitcher_success_rate")
    ps1 = numeric(frame, "asof_pitcher_prev1_game_success_rate")
    ps3 = numeric(frame, "asof_pitcher_prev3_game_success_rate")
    ps5 = numeric(frame, "asof_pitcher_prev5_game_success_rate")
    frame["eng_ps_prev1_minus_long"] = ps1 - ps_long
    frame["eng_ps_prev3_minus_long"] = ps3 - ps_long
    frame["eng_ps_prev5_minus_long"] = ps5 - ps_long
    frame["eng_ps_prev1_minus_prev3"] = ps1 - ps3
    frame["eng_ps_prev3_minus_prev5"] = ps3 - ps5
    frame["eng_ps_prev1_minus_prev5"] = ps1 - ps5
    stack = pd.concat([ps1, ps3, ps5], axis=1)
    frame["eng_ps_recent_mean_135"] = stack.mean(axis=1, skipna=False)
    frame["eng_ps_recent_mean_minus_long"] = frame["eng_ps_recent_mean_135"] - ps_long
    frame["eng_ps_recent_range_135"] = stack.max(axis=1, skipna=False) - stack.min(axis=1, skipna=False)


def prepare_x(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inference features: {{missing}}")
    x = frame.loc[:, features].copy()
    cat = set(categorical)
    for column in features:
        if column in cat:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x


def predict(model_path: Path, x: pd.DataFrame, features: list[str], categorical: list[str]) -> np.ndarray:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    pool = Pool(x, cat_features=categorical, feature_names=features)
    return np.asarray(model.predict_proba(pool)[:, 1], dtype=np.float64)


def main() -> None:
    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    if "row_id" not in test.columns:
        raise ValueError("test.csv missing row_id")
    if "game_type" not in test.columns:
        raise ValueError("test.csv missing game_type")
    add_features(test)

    x_full = prepare_x(test, FULL_FEATURES, FULL_CATEGORICAL)
    x_recent = prepare_x(test, RECENT_FEATURES, RECENT_CATEGORICAL)
    x_r_fast = prepare_x(test, R_FAST_FEATURES, R_FAST_CATEGORICAL)

    p_full = predict(MODEL_DIR / "full_raw.cbm", x_full, FULL_FEATURES, FULL_CATEGORICAL)
    p_recent = predict(MODEL_DIR / "recent_raw.cbm", x_recent, RECENT_FEATURES, RECENT_CATEGORICAL)
    p_r_fast = predict(MODEL_DIR / "r_fast.cbm", x_r_fast, R_FAST_FEATURES, R_FAST_CATEGORICAL)

    base = (1.0 - ALPHA_RECENT) * p_full + ALPHA_RECENT * p_recent
    is_r = test["game_type"].astype("string").fillna("<MISSING>").astype(str).eq("R").to_numpy()
    pred = base.copy()
    pred[is_r] = (1.0 - BETA_R) * base[is_r] + BETA_R * p_r_fast[is_r]

    if len(pred) != len(test):
        raise RuntimeError("prediction row count mismatch")
    if not np.isfinite(pred).all() or np.any((pred < 0.0) | (pred > 1.0)):
        raise RuntimeError("invalid probability output")

    sample_path = data_dir / "sample_submission.csv"
    if sample_path.is_file():
        sample = pd.read_csv(sample_path)
        if len(sample) != len(test):
            raise RuntimeError("sample_submission/test row count mismatch")
        if "row_id" not in sample.columns:
            raise ValueError("sample_submission.csv missing row_id")
        if not sample["row_id"].astype(str).equals(test["row_id"].astype(str)):
            raise RuntimeError("sample_submission row_id order differs from test.csv")
        submission = sample[["row_id"]].copy()
    else:
        submission = pd.DataFrame({{"row_id": test["row_id"].to_numpy()}})
    submission["control_success"] = pred
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    print(
        f"gated-r-fast rows={{len(submission):,}} R={{int(is_r.sum()):,}} "
        f"alpha_recent={{ALPHA_RECENT:.3f}} beta_r={{BETA_R:.3f}} "
        f"full_mean={{p_full.mean():.6f}} recent_mean={{p_recent.mean():.6f}} "
        f"r_fast_mean={{p_r_fast[is_r].mean() if is_r.any() else float('nan'):.6f}} "
        f"final_mean={{pred.mean():.6f}} final_std={{pred.std():.6f}}"
    )


if __name__ == "__main__":
    main()
'''


def _write_zip(
    *,
    output_zip: Path,
    full_model: Path,
    recent_model: Path,
    r_fast_model: Path,
    metadata: dict,
    full_features: list[str],
    recent_features: list[str],
    r_fast_features: list[str],
    full_categorical: list[str],
    recent_categorical: list[str],
    r_fast_categorical: list[str],
    alpha_recent: float,
    beta_r: float,
    smoke_data_dir: Path | None,
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="gated_r_fast_submit_") as temp_dir:
        package = Path(temp_dir)
        model_dir = package / "model"
        model_dir.mkdir(parents=True)
        shutil.copy2(full_model, model_dir / "full_raw.cbm")
        shutil.copy2(recent_model, model_dir / "recent_raw.cbm")
        shutil.copy2(r_fast_model, model_dir / "r_fast.cbm")
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (package / "script.py").write_text(
            _inference_script(
                full_features,
                recent_features,
                r_fast_features,
                full_categorical,
                recent_categorical,
                r_fast_categorical,
                alpha_recent,
                beta_r,
            ),
            encoding="utf-8",
        )
        (package / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

        if smoke_data_dir is not None:
            local_data = package / "data"
            local_data.mkdir()
            shutil.copy2(smoke_data_dir / "test.csv", local_data / "test.csv")
            sample = smoke_data_dir / "sample_submission.csv"
            if sample.is_file():
                shutil.copy2(sample, local_data / "sample_submission.csv")
            subprocess.run([sys.executable, "script.py"], cwd=package, check=True)
            out = pd.read_csv(package / "output" / "submission.csv")
            if list(out.columns) != ["row_id", "control_success"] or len(out) == 0:
                raise RuntimeError("gated R-fast smoke test failed")
            if not np.isfinite(out["control_success"].to_numpy(np.float64)).all():
                raise RuntimeError("smoke submission contains non-finite predictions")
            print(f"[smoke] OK rows={len(out):,}")

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(package.rglob("*")):
                if path.is_file() and "output" not in path.parts and "data" not in path.parts:
                    zf.write(path, path.relative_to(package).as_posix())

    with zipfile.ZipFile(output_zip) as zf:
        names = set(zf.namelist())
    required = {
        "script.py",
        "requirements.txt",
        "model/full_raw.cbm",
        "model/recent_raw.cbm",
        "model/r_fast.cbm",
        "model/metadata.json",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"ZIP missing entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train/package the current primary 2025 submission: full-history raw-game_type + "
            "2023-2024 recent raw-game_type, then apply a small recent-R fastball/hand specialist gate."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--full-iterations", type=int, default=500)
    parser.add_argument("--recent-iterations", type=int, default=500)
    parser.add_argument("--r-fast-iterations", type=int, default=500)
    parser.add_argument("--alpha-recent", type=float, default=0.2)
    parser.add_argument("--beta-r", type=float, default=0.1)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--models-dir", default="outputs/gated_r_fast_final")
    parser.add_argument(
        "--output",
        default="dist/gated_r/gated_r_fast_full80_recent20_beta10.zip",
    )
    parser.add_argument("--smoke-data-dir", default=None)
    args = parser.parse_args()

    if min(args.full_iterations, args.recent_iterations, args.r_fast_iterations) <= 0:
        raise ValueError("all iteration counts must be positive")
    if not (0.0 <= args.alpha_recent <= 1.0):
        raise ValueError("alpha_recent must be in [0, 1]")
    if not (0.0 <= args.beta_r <= 1.0):
        raise ValueError("beta_r must be in [0, 1]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    sort_columns = [season_col]
    if "game_month" in frame.columns:
        sort_columns.append("game_month")
    if row_id_col in frame.columns:
        sort_columns.append(row_id_col)
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    full_train = frame.loc[frame[season_col].isin(FULL_TRAIN_SEASONS)].copy()
    recent_train = frame.loc[frame[season_col].isin(RECENT_TRAIN_SEASONS)].copy()
    r_fast_train = recent_train.loc[
        recent_train["game_type"].astype("string").fillna("<MISSING>").astype(str).eq("R")
    ].copy()
    if sorted(full_train[season_col].unique().tolist()) != FULL_TRAIN_SEASONS:
        raise RuntimeError("full expert did not receive exactly 2019-2024")
    if sorted(recent_train[season_col].unique().tolist()) != RECENT_TRAIN_SEASONS:
        raise RuntimeError("recent expert did not receive exactly 2023-2024")
    if r_fast_train.empty or not r_fast_train["game_type"].astype(str).eq("R").all():
        raise RuntimeError("R-fast expert training split is invalid")

    base_features = recent_core.feature_set("recent_raw_game_type")
    feature_sets = gated_core._feature_sets(base_features)
    full_features = feature_sets["full_raw"]
    recent_features = feature_sets["recent_raw"]
    r_fast_features = feature_sets["r_fast"]
    categorical_set = set(CANONICAL_CATEGORICAL)
    full_categorical = [f for f in full_features if f in categorical_set]
    recent_categorical = [f for f in recent_features if f in categorical_set]
    r_fast_categorical = [f for f in r_fast_features if f in categorical_set]

    models_dir = (ROOT / args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    full_model = models_dir / "full_raw.cbm"
    recent_model = models_dir / "recent_raw.cbm"
    r_fast_model = models_dir / "r_fast.cbm"

    print("[Gated R-Fast Final Submission]")
    print(
        f"  full_raw    : seasons={FULL_TRAIN_SEASONS}, rows={len(full_train):,}, "
        f"features={len(full_features)}, trees={args.full_iterations}"
    )
    print(
        f"  recent_raw  : seasons={RECENT_TRAIN_SEASONS}, rows={len(recent_train):,}, "
        f"features={len(recent_features)}, trees={args.recent_iterations}"
    )
    print(
        f"  r_fast      : recent R only, rows={len(r_fast_train):,}, "
        f"features={len(r_fast_features)}, trees={args.r_fast_iterations}"
    )
    print(
        f"  blend       : base={(1.0 - args.alpha_recent):.3f}*full + {args.alpha_recent:.3f}*recent; "
        f"R only -> {(1.0 - args.beta_r):.3f}*base + {args.beta_r:.3f}*r_fast"
    )

    print("\n[1/3] Train full-history raw-game_type expert")
    seed_everything(seed)
    full_stats = recent_core.train_variant(
        train=full_train,
        target=target_col,
        features=full_features,
        config=config,
        iterations=args.full_iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
        model_path=full_model,
    )

    print("\n[2/3] Train recent 2023-2024 raw-game_type expert")
    seed_everything(seed)
    recent_stats = recent_core.train_variant(
        train=recent_train,
        target=target_col,
        features=recent_features,
        config=config,
        iterations=args.recent_iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
        model_path=recent_model,
    )

    print("\n[3/3] Train recent-R fastball/hand specialist")
    seed_everything(seed)
    r_fast_stats = recent_core.train_variant(
        train=r_fast_train,
        target=target_col,
        features=r_fast_features,
        config=config,
        iterations=args.r_fast_iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
        model_path=r_fast_model,
    )

    metadata = {
        "purpose": "primary gated R-fast 2025 hidden submission",
        "selection_source": "outputs/gated_r_specialist_suite internal temporal proxy",
        "formula": {
            "base": "(1-alpha_recent)*full_raw + alpha_recent*recent_raw",
            "R": "(1-beta_r)*base + beta_r*r_fast",
            "non_R": "base",
            "alpha_recent": float(args.alpha_recent),
            "beta_r": float(args.beta_r),
        },
        "full_raw": {
            "train_seasons": FULL_TRAIN_SEASONS,
            "iterations": int(args.full_iterations),
            "features": full_features,
            "categorical": full_categorical,
            "training_stats": full_stats,
        },
        "recent_raw": {
            "train_seasons": RECENT_TRAIN_SEASONS,
            "iterations": int(args.recent_iterations),
            "features": recent_features,
            "categorical": recent_categorical,
            "training_stats": recent_stats,
        },
        "r_fast": {
            "train_seasons": RECENT_TRAIN_SEASONS,
            "train_filter": "game_type == R",
            "iterations": int(args.r_fast_iterations),
            "features": r_fast_features,
            "categorical": r_fast_categorical,
            "training_stats": r_fast_stats,
        },
        "canonical_invariants": invariant_check,
        "training_order": sort_columns,
    }
    save_json(metadata, models_dir / "metadata.json")

    smoke_data_dir = Path(args.smoke_data_dir).resolve() if args.smoke_data_dir else None
    if smoke_data_dir is not None and not (smoke_data_dir / "test.csv").is_file():
        raise FileNotFoundError(f"smoke test.csv not found: {smoke_data_dir / 'test.csv'}")

    output_zip = (ROOT / args.output).resolve()
    _write_zip(
        output_zip=output_zip,
        full_model=full_model,
        recent_model=recent_model,
        r_fast_model=r_fast_model,
        metadata=metadata,
        full_features=full_features,
        recent_features=recent_features,
        r_fast_features=r_fast_features,
        full_categorical=full_categorical,
        recent_categorical=recent_categorical,
        r_fast_categorical=r_fast_categorical,
        alpha_recent=float(args.alpha_recent),
        beta_r=float(args.beta_r),
        smoke_data_dir=smoke_data_dir,
    )
    print(f"\n[Gated R-Fast Final] ZIP ready: {output_zip}")


if __name__ == "__main__":
    main()
