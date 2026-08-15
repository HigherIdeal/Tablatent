from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config, save_json, seed_everything


RECENT_VARIANT = "recent_raw_game_type"
STABLE_VARIANT = "recent_drop_game_type"
RECENT_TRAIN_SEASONS = [2023, 2024]
STABLE_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
REQUIREMENTS = "catboost==1.2.10\n"


def _load_recommendation(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"recommendation not found: {path}. Run scripts/run_dual_track_blend_screen.py first "
            "or pass --recent-iterations, --stable-iterations, and --alpha-recent explicitly."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("recent_iterations", "stable_iterations", "alpha_recent"):
        if key not in payload:
            raise RuntimeError(f"recommendation missing {key}: {path}")
    return payload


def _resolve_hyperparameters(args: argparse.Namespace) -> tuple[int, int, float, dict | None]:
    explicit = [args.recent_iterations, args.stable_iterations, args.alpha_recent]
    if all(value is not None for value in explicit):
        recent_iterations = int(args.recent_iterations)
        stable_iterations = int(args.stable_iterations)
        alpha_recent = float(args.alpha_recent)
        recommendation = None
    elif any(value is not None for value in explicit):
        raise ValueError(
            "Either pass all of --recent-iterations/--stable-iterations/--alpha-recent, or none of them."
        )
    else:
        recommendation_path = (ROOT / args.recommendation).resolve()
        recommendation = _load_recommendation(recommendation_path)
        recent_iterations = int(recommendation["recent_iterations"])
        stable_iterations = int(recommendation["stable_iterations"])
        alpha_recent = float(recommendation["alpha_recent"])

    if recent_iterations <= 0 or stable_iterations <= 0:
        raise ValueError("iterations must be positive")
    if not (0.0 <= alpha_recent <= 1.0):
        raise ValueError("alpha_recent must be in [0, 1]")
    return recent_iterations, stable_iterations, alpha_recent, recommendation


def _inference_script(
    recent_features: list[str],
    stable_features: list[str],
    recent_categorical: list[str],
    stable_categorical: list[str],
    alpha_recent: float,
) -> str:
    return f'''from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
RECENT_FEATURES = {recent_features!r}
STABLE_FEATURES = {stable_features!r}
RECENT_CATEGORICAL = {recent_categorical!r}
STABLE_CATEGORICAL = {stable_categorical!r}
ALPHA_RECENT = {float(alpha_recent)!r}


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
    add_features(test)

    x_recent = prepare_x(test, RECENT_FEATURES, RECENT_CATEGORICAL)
    x_stable = prepare_x(test, STABLE_FEATURES, STABLE_CATEGORICAL)
    p_recent = predict(MODEL_DIR / "recent.cbm", x_recent, RECENT_FEATURES, RECENT_CATEGORICAL)
    p_stable = predict(MODEL_DIR / "stable.cbm", x_stable, STABLE_FEATURES, STABLE_CATEGORICAL)
    pred = ALPHA_RECENT * p_recent + (1.0 - ALPHA_RECENT) * p_stable

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
        f"dual-track rows={{len(submission):,}} alpha_recent={{ALPHA_RECENT:.4f}} "
        f"recent_mean={{p_recent.mean():.6f}} stable_mean={{p_stable.mean():.6f}} "
        f"blend_mean={{pred.mean():.6f}} blend_std={{pred.std():.6f}}"
    )


if __name__ == "__main__":
    main()
'''


def _write_zip(
    output_zip: Path,
    recent_model: Path,
    stable_model: Path,
    metadata: dict,
    recent_features: list[str],
    stable_features: list[str],
    recent_categorical: list[str],
    stable_categorical: list[str],
    alpha_recent: float,
    smoke_data_dir: Path | None,
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="dual_track_submit_") as temp_dir:
        package = Path(temp_dir)
        model_dir = package / "model"
        model_dir.mkdir(parents=True)
        shutil.copy2(recent_model, model_dir / "recent.cbm")
        shutil.copy2(stable_model, model_dir / "stable.cbm")
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (package / "script.py").write_text(
            _inference_script(
                recent_features,
                stable_features,
                recent_categorical,
                stable_categorical,
                alpha_recent,
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
                raise RuntimeError("dual-track smoke test failed")
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
        "model/recent.cbm",
        "model/stable.cbm",
        "model/metadata.json",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"ZIP missing entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the final dual-track experts and package one blended submission. "
            "Recent expert: 2023+2024 with raw game_type. Stable expert: 2019-2024 without game_type."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--recommendation",
        default="outputs/dual_track_blend_screen/recommended_config.json",
        help="Recommendation generated by run_dual_track_blend_screen.py.",
    )
    parser.add_argument("--recent-iterations", type=int, default=None)
    parser.add_argument("--stable-iterations", type=int, default=None)
    parser.add_argument("--alpha-recent", type=float, default=None)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--models-dir", default="outputs/dual_track_final")
    parser.add_argument("--output", default="dist/dual_track/dual_track_blend.zip")
    parser.add_argument("--smoke-data-dir", default=None)
    args = parser.parse_args()

    recent_iterations, stable_iterations, alpha_recent, recommendation = _resolve_hyperparameters(args)
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

    recent_train = frame.loc[frame[season_col].isin(RECENT_TRAIN_SEASONS)].copy()
    stable_train = frame.loc[frame[season_col].isin(STABLE_TRAIN_SEASONS)].copy()
    if sorted(recent_train[season_col].unique().tolist()) != RECENT_TRAIN_SEASONS:
        raise RuntimeError("recent expert did not receive exactly 2023+2024")
    if sorted(stable_train[season_col].unique().tolist()) != STABLE_TRAIN_SEASONS:
        raise RuntimeError("stable expert did not receive exactly 2019-2024")

    recent_features = recent_core.feature_set(RECENT_VARIANT)
    stable_features = recent_core.feature_set(STABLE_VARIANT)
    if set(recent_features) - {"game_type"} != set(stable_features):
        raise RuntimeError("final expert feature sets must differ only by game_type")
    recent_categorical = [f for f in recent_features if f in set(CANONICAL_CATEGORICAL)]
    stable_categorical = [f for f in stable_features if f in set(CANONICAL_CATEGORICAL)]

    models_dir = (ROOT / args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    recent_model = models_dir / "recent.cbm"
    stable_model = models_dir / "stable.cbm"

    print(
        f"[Dual-Track Final] recent={RECENT_TRAIN_SEASONS} rows={len(recent_train):,} "
        f"trees={recent_iterations}; stable={STABLE_TRAIN_SEASONS} rows={len(stable_train):,} "
        f"trees={stable_iterations}; alpha_recent={alpha_recent:.4f}"
    )

    print("\n[1/2] Train recent expert: 2023+2024 + raw game_type")
    seed_everything(seed)
    recent_stats = recent_core.train_variant(
        train=recent_train,
        target=target_col,
        features=recent_features,
        config=config,
        iterations=recent_iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
        model_path=recent_model,
    )

    print("\n[2/2] Train stable expert: 2019-2024 - game_type")
    seed_everything(seed)
    stable_stats = recent_core.train_variant(
        train=stable_train,
        target=target_col,
        features=stable_features,
        config=config,
        iterations=stable_iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
        model_path=stable_model,
    )

    metadata = {
        "purpose": "dual-track 2025 hidden submission",
        "blend": "p = alpha_recent * recent + (1 - alpha_recent) * stable",
        "alpha_recent": alpha_recent,
        "recent": {
            "train_seasons": RECENT_TRAIN_SEASONS,
            "variant": RECENT_VARIANT,
            "iterations": recent_iterations,
            "features": recent_features,
            "categorical": recent_categorical,
            "training_stats": recent_stats,
        },
        "stable": {
            "train_seasons": STABLE_TRAIN_SEASONS,
            "variant": STABLE_VARIANT,
            "iterations": stable_iterations,
            "features": stable_features,
            "categorical": stable_categorical,
            "training_stats": stable_stats,
        },
        "recommendation": recommendation,
        "canonical_invariants": invariant_check,
        "training_order": sort_columns,
    }
    save_json(metadata, models_dir / "metadata.json")

    smoke_data_dir = Path(args.smoke_data_dir).resolve() if args.smoke_data_dir else None
    if smoke_data_dir is not None and not (smoke_data_dir / "test.csv").is_file():
        raise FileNotFoundError(f"smoke test.csv not found: {smoke_data_dir / 'test.csv'}")

    output_zip = (ROOT / args.output).resolve()
    _write_zip(
        output_zip,
        recent_model,
        stable_model,
        metadata,
        recent_features,
        stable_features,
        recent_categorical,
        stable_categorical,
        alpha_recent,
        smoke_data_dir,
    )
    print(f"\n[Dual-Track Final] ZIP ready: {output_zip}")


if __name__ == "__main__":
    main()
