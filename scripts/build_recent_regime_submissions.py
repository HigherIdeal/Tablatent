from __future__ import annotations

import argparse
import gc
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

import run_asof_state_engineering as asof_core
import run_context_interaction_screen as context_core
from src.canonical_features import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.utils import load_config, seed_everything


TRAIN_SEASONS = [2023, 2024]
VARIANTS = ["recent_raw_game_type", "recent_drop_game_type"]
REQUIREMENTS = "catboost==1.2.10\n"


def feature_set(variant: str) -> list[str]:
    base = list(CANONICAL_FEATURES) + list(asof_core.SUCCESS_STATE)
    if variant == "recent_raw_game_type":
        features = base
    elif variant == "recent_drop_game_type":
        features = [feature for feature in base if feature != "game_type"]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate features in {variant}")
    return features


def prepare_frame(config: dict) -> tuple[pd.DataFrame, dict]:
    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    raw_canonical = [feature for feature in CANONICAL_FEATURES if feature != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(
        raw_canonical
        + CANONICAL_SOURCE_COLUMNS
        + [
            target,
            season_col,
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
        ]
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    asof_core.add_asof_state_features(frame)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    return frame, invariant_check


def train_variant(
    train: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
    model_path: Path,
) -> dict[str, float | int]:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = context_core.prepare_x(train, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    params = context_core.catboost_params(config, iterations, task_type, devices, verbose)
    pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(pool, verbose=verbose)
    model.save_model(str(model_path))

    stats = {
        "train_rows": int(len(train)),
        "target_rate": float(y_train.mean()),
        "iterations": int(iterations),
        "feature_count": int(len(features)),
        "categorical_count": int(len(categorical)),
        "model_bytes": int(model_path.stat().st_size),
    }
    del model, pool, x_train, y_train
    gc.collect()
    return stats


def _inference_script(features: list[str], categorical: list[str]) -> str:
    return f'''from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
FEATURES = {features!r}
CATEGORICAL = {categorical!r}


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


def prepare_x(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inference features: {{missing}}")
    x = frame.loc[:, FEATURES].copy()
    cat = set(CATEGORICAL)
    for column in FEATURES:
        if column in cat:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x


def main() -> None:
    data_dir = find_data_dir()
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    if "row_id" not in test.columns:
        raise ValueError("test.csv missing row_id")
    add_features(test)
    x = prepare_x(test)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "model.cbm"))
    pool = Pool(x, cat_features=CATEGORICAL, feature_names=FEATURES)
    pred = np.asarray(model.predict_proba(pool)[:, 1], dtype=np.float64)
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
        f"submission rows={{len(submission):,}} mean={{pred.mean():.6f}} "
        f"std={{pred.std():.6f}} min={{pred.min():.6f}} max={{pred.max():.6f}}"
    )


if __name__ == "__main__":
    main()
'''


def write_zip(
    output_zip: Path,
    model_path: Path,
    features: list[str],
    categorical: list[str],
    metadata: dict,
    smoke_data_dir: Path | None,
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="recent_regime_submit_") as temp_dir:
        package = Path(temp_dir)
        model_dir = package / "model"
        model_dir.mkdir(parents=True)
        shutil.copy2(model_path, model_dir / "model.cbm")
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (package / "script.py").write_text(
            _inference_script(features, categorical), encoding="utf-8"
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
            values = out["control_success"].to_numpy(np.float64)
            if len(out) == 0 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise RuntimeError("submission smoke test failed")
            print(f"[smoke] {{output_zip.name}} OK rows={{len(out):,}}")

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(package.rglob("*")):
                if path.is_file() and "output" not in path.parts and "data" not in path.parts:
                    zf.write(path, path.relative_to(package).as_posix())

    with zipfile.ZipFile(output_zip) as zf:
        names = set(zf.namelist())
    required = {"script.py", "requirements.txt", "model/model.cbm", "model/metadata.json"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"ZIP missing entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train two no-validation 2023+2024 CatBoost probes and package evaluator-compatible ZIPs: "
            "raw game_type vs dropping game_type."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--output-dir", default="dist/recent_regime")
    parser.add_argument(
        "--smoke-data-dir",
        default=None,
        help="Optional directory containing test.csv and sample_submission.csv for package smoke testing.",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    frame, invariant_check = prepare_frame(config)
    train = frame.loc[frame[season_col].isin(TRAIN_SEASONS)].copy()
    if train.empty:
        raise RuntimeError("No rows found for seasons 2023 and 2024")
    observed = sorted(train[season_col].unique().tolist())
    if observed != TRAIN_SEASONS:
        raise RuntimeError(f"Expected train seasons {TRAIN_SEASONS}, got {observed}")

    f_mask = train["game_type"].astype(str).eq("F")
    print(
        f"[Recent Regime Submission] train seasons={TRAIN_SEASONS}, rows={len(train):,}, "
        f"target_rate={pd.to_numeric(train[target]).mean():.6f}, F_share={f_mask.mean():.6f}"
    )
    for year in TRAIN_SEASONS:
        group = train.loc[train[season_col].eq(year)]
        gf = group["game_type"].astype(str).eq("F")
        y = pd.to_numeric(group[target], errors="raise")
        f_rate = float(y.loc[gf].mean())
        r_rate = float(y.loc[~gf].mean())
        print(
            f"  {year}: rows={len(group):,} target={y.mean():.6f} F_share={gf.mean():.6f} "
            f"F_rate={f_rate:.6f} R_rate={r_rate:.6f} F-R={f_rate-r_rate:+.6f}"
        )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_data_dir = Path(args.smoke_data_dir).resolve() if args.smoke_data_dir else None
    if smoke_data_dir is not None and not (smoke_data_dir / "test.csv").is_file():
        raise FileNotFoundError(f"smoke test.csv not found: {smoke_data_dir / 'test.csv'}")

    for variant in VARIANTS:
        features = feature_set(variant)
        categorical = [feature for feature in features if feature in set(CANONICAL_CATEGORICAL)]
        model_path = output_dir / f"{variant}.cbm"
        print(
            f"\n[{variant}] training rows={len(train):,} features={len(features)} "
            f"categorical={len(categorical)} iterations={args.iterations}"
        )
        stats = train_variant(
            train=train,
            target=target,
            features=features,
            config=config,
            iterations=args.iterations,
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
            model_path=model_path,
        )
        metadata = {
            "variant": variant,
            "train_seasons": TRAIN_SEASONS,
            "no_validation": True,
            "features": features,
            "categorical": categorical,
            "training_stats": stats,
            "canonical_invariants": invariant_check,
            "purpose": "2025 leaderboard probe for post-2023 regime continuity",
        }
        zip_path = output_dir / f"{variant}.zip"
        write_zip(zip_path, model_path, features, categorical, metadata, smoke_data_dir)
        print(f"[{variant}] ZIP ready: {zip_path}")

    print("\nSubmit in this order:")
    print(f"  1) {output_dir / 'recent_raw_game_type.zip'}")
    print(f"  2) {output_dir / 'recent_drop_game_type.zip'}")
    print("Interpretation: raw > drop supports 2023-2024 game_type regime continuity into hidden 2025.")


if __name__ == "__main__":
    main()
