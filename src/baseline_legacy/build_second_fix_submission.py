from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


REQUIREMENTS = "catboost==1.2.10\n"


INFERENCE_SCRIPT = r'''from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ID_COL = "row_id"
TARGET_COL = "control_success"
AUX_NAMES = ["reverse", "middle", "ball", "strike"]

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
CODE_DIR = MODEL_DIR / "code"
for _p in (CODE_DIR, CODE_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_recent_regime_submissions as recent_core
import run_asof_state_engineering as asof_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_frozen_season_anchor_probe as anchor_core
import run_offset_residual_boosting as offset_core
from src.canonical_features import add_canonical_derived_features


# =======================
# 데이터 로드 유틸
# =======================

def find_data_dir() -> Path:
    # Official baseline uses ./data. Keep ./open only as a compatibility fallback.
    for name in ("data", "open"):
        p = ROOT / name
        if (p / "test.csv").is_file():
            return p
    raise FileNotFoundError("test.csv not found under ./data or ./open")


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if ID_COL not in df.columns:
        raise ValueError(f"test data missing {ID_COL}: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path: Path, test: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET_COL: np.nan})
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission columns are not ({ID_COL}, {TARGET_COL}): {list(df.columns)}"
        )
    return df


def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def add_regime_continuous(frame: pd.DataFrame, hand_levels: list[str]) -> None:
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    gt = frame["game_type"].astype("string").fillna("<MISSING>").astype(str).str.upper()
    hand = frame["batter_hand"].astype("string").fillna("<MISSING>").astype(str)
    recent = season.ge(2023).to_numpy()
    old = ~recent
    is_r = gt.eq("R").to_numpy()
    h1 = hand.eq(hand_levels[0]).to_numpy()
    h2 = hand.eq(hand_levels[1]).to_numpy()
    fast = pd.to_numeric(frame["asof_pitcher_fastball_rate"], errors="coerce").to_numpy(np.float32)
    rrng = pd.to_numeric(frame["eng_ps_recent_range_135"], errors="coerce").to_numpy(np.float32)

    frame["regime_recent"] = recent.astype(np.float32)
    for name, mask in {
        "rr_fastball_hand1": is_r & recent & h1,
        "rr_fastball_hand2": is_r & recent & h2,
        "ro_fastball_hand1": is_r & old & h1,
        "ro_fastball_hand2": is_r & old & h2,
    }.items():
        out = np.full(len(frame), np.nan, dtype=np.float32)
        out[mask] = fast[mask]
        frame[name] = out

    rr = np.full(len(frame), np.nan, dtype=np.float32)
    ro = np.full(len(frame), np.nan, dtype=np.float32)
    rr[is_r & recent] = rrng[is_r & recent]
    ro[is_r & old] = rrng[is_r & old]
    frame["rr_recent_range"] = rr
    frame["ro_recent_range"] = ro


def build_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Recreate the exact frozen-history features used by the trained models.

    Training history is stored as gzip CSV rather than pickle so the package is
    independent of the NumPy/pandas version that originally created the artifacts.
    Test targets and auxiliary outcomes remain NaN and are never reconstructed.
    """
    test = df.copy()
    add_canonical_derived_features(test)
    asof_core.add_asof_state_features(test)
    test["season"] = pd.to_numeric(test["season"], errors="raise").astype(int)
    test["game_type"] = test["game_type"].astype("string").str.strip().str.upper()
    test[TARGET_COL] = np.nan
    test["_submit_pos"] = np.arange(len(test), dtype=np.int64)

    history = pd.read_csv(MODEL_DIR / "history.csv.gz", low_memory=False)
    aux_hist = pd.read_csv(MODEL_DIR / "history_aux.csv.gz", low_memory=False).reset_index(drop=True)
    history = history.reset_index(drop=True)

    if len(history) != len(aux_hist):
        raise RuntimeError(f"history/aux row mismatch: {len(history)} != {len(aux_hist)}")

    history["_submit_pos"] = -1
    combo = pd.concat([history, test], ignore_index=True, sort=False)
    aux_test = pd.DataFrame(
        np.nan,
        index=np.arange(len(test)),
        columns=AUX_NAMES,
        dtype=np.float32,
    )
    aux = pd.concat([aux_hist[AUX_NAMES], aux_test], ignore_index=True)

    anchor_core.add_frozen_anchor_features(
        combo,
        season_col="season",
        pitcher_col="pitcher_id",
        n_col="asof_pitcher_n",
        count_tolerance=0.05,
    )
    offset_core.add_batter_anchor(combo)
    offset_core.add_anchor_cross(combo)
    offset_core.add_frozen_matchup(combo)
    offset_core.add_frozen_count_profiles(combo)
    offset_core.add_frozen_pressure_profiles(combo)
    offset_core.add_frozen_domain_profiles(combo)
    offset_core.add_frozen_aux_profiles(combo, aux, "pressure")
    offset_core.add_frozen_conditional_profiles(combo, aux, False)
    if bool(meta.get("use_paths", False)):
        path_core.add_paths(combo, "pitcher_id", "season", TARGET_COL)

    out = combo.loc[combo["_submit_pos"].ge(0)].sort_values("_submit_pos").copy()
    if len(out) != len(test):
        raise RuntimeError("engineered test row count mismatch")

    context_core.add_context_interactions(out)
    add_regime_continuous(out, meta["hand_levels"])
    return out


def make_pool(frame: pd.DataFrame, spec: dict) -> Pool:
    features = spec["features"]
    cats = set(spec["categorical"])
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing inference features: {missing[:20]}")

    x = frame.loc[:, features].copy()
    for col in features:
        if col in cats:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = (
                pd.to_numeric(x[col], errors="coerce")
                .astype(np.float32)
                .replace([np.inf, -np.inf], np.nan)
            )
    return Pool(x, cat_features=spec["categorical"], feature_names=features)


# =======================
# 모델 로드 / 추론 유틸
# =======================

def load_regressor(name: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(MODEL_DIR / f"{name}.cbm"))
    return model


def load_classifier(name: str) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / f"{name}.cbm"))
    return model


def predict_regression(model, pool: Pool, tree: int):
    try:
        return model.predict(pool, ntree_end=int(tree), task_type="GPU")
    except Exception:
        return model.predict(pool, ntree_end=int(tree), thread_count=-1)


def predict_probability(model, pool: Pool, tree: int):
    try:
        return model.predict_proba(pool, ntree_end=int(tree), task_type="GPU")
    except Exception:
        return model.predict_proba(pool, ntree_end=int(tree), thread_count=-1)


def offset_prior(frame: pd.DataFrame, mean: float) -> np.ndarray:
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").to_numpy(float)

    shr = (n * p + 200.0 * mean) / (n + 200.0)
    shr = np.nan_to_num(shr, nan=mean)
    b = np.nan_to_num(b, nan=mean)

    recent = frame[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ].mean(axis=1).to_numpy(float)
    recent = np.nan_to_num(recent, nan=mean)
    return 0.65 * shr + 0.25 * recent + 0.10 * b


def inference_model(features: pd.DataFrame, meta: dict) -> np.ndarray:
    rich_pool = make_pool(features, meta["specs"]["rich"])
    hurdle_pool = make_pool(features, meta["specs"]["hurdle"])
    offset_pool = make_pool(features, meta["specs"]["offset"])

    multi = load_regressor("multi")
    all_multi = np.clip(
        predict_regression(multi, rich_pool, meta["selection"]["multi_tree"]),
        0.0,
        1.0,
    )
    direct = all_multi[:, 0]

    reverse_model = load_classifier("aux_reverse")
    middle_model = load_classifier("aux_middle")
    gs = meta["selection"]["gate_cond"]
    reverse = predict_probability(reverse_model, rich_pool, gs["reverse_tree"])[:, 1]
    middle = predict_probability(middle_model, rich_pool, gs["middle_tree"])[:, 1]

    gate_model = load_classifier("hurdle_gate")
    cond_model = load_classifier("hurdle_cond")
    hurdle_tree = int(meta["selection"]["hurdle_tree"])
    gate_h = predict_probability(gate_model, hurdle_pool, hurdle_tree)[:, 1]
    cond_h = predict_probability(cond_model, hurdle_pool, hurdle_tree)[:, 1]
    hurdle = gate_h * cond_h

    cond_logic = predict_probability(cond_model, hurdle_pool, gs["cond_tree"])[:, 1]
    logic = np.clip(
        1.0 - reverse - middle + float(gs["c"]) * reverse * middle,
        0.0,
        1.0,
    ) * cond_logic

    gt = features["game_type"].astype(str).to_numpy()
    blend = np.where(
        gt == "F",
        float(gs["blend"]["F"]),
        float(gs["blend"]["R"]),
    )
    gate_cond = direct + blend * (logic - direct)

    offset_model = load_regressor("offset")
    prior = offset_prior(features, float(meta["offset_mean"]))
    offset = np.clip(
        prior
        + predict_regression(
            offset_model,
            offset_pool,
            meta["selection"]["offset_tree"],
        ),
        0.0,
        1.0,
    )

    joint_model = load_classifier("joint")
    joint_prob = predict_probability(
        joint_model,
        rich_pool,
        meta["selection"]["joint_tree"],
    )
    q = np.asarray(meta["joint_q"], dtype=np.float64)
    group_index = (gt == "F").astype(np.int8)
    joint = np.sum(joint_prob * q[group_index], axis=1)

    candidates = np.column_stack([gate_cond, direct, offset, hurdle, joint])
    expected = meta["candidate_names"]
    if expected != ["gate_cond", "multi", "offset", "hurdle", "joint"]:
        raise RuntimeError(f"unexpected candidate order: {expected}")

    weight_r = np.asarray(meta["ensemble_weights"]["R"], dtype=np.float64)
    weight_f = np.asarray(meta["ensemble_weights"]["F"], dtype=np.float64)
    pred = np.empty(len(features), dtype=np.float64)

    mask_r = gt == "R"
    mask_f = gt == "F"
    pred[mask_r] = candidates[mask_r] @ weight_r
    pred[mask_f] = candidates[mask_f] @ weight_f

    other = ~(mask_r | mask_f)
    if other.any():
        pred[other] = candidates[other] @ ((weight_r + weight_f) / 2.0)

    pred = np.clip(pred, 0.0, 1.0)
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite prediction")
    return pred


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub: pd.DataFrame, ids, preds) -> pd.DataFrame:
    if len(ids) != len(preds):
        raise RuntimeError(f"id/pred row mismatch: {len(ids)} != {len(preds)}")
    pred_map = dict(zip(ids, preds))
    values = []
    missing = []
    for rid in sub[ID_COL]:
        if rid not in pred_map:
            missing.append(rid)
        else:
            values.append(float(pred_map[rid]))
    if missing:
        raise RuntimeError(f"predictions missing for {len(missing)} row_id values")
    sub = sub[[ID_COL]].copy()
    sub[TARGET_COL] = values
    return sub


def save_submission(path: Path, sub: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main() -> None:
    t0 = time.time()

    data_dir = find_data_dir()
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    output_path = ROOT / "output" / "submission.csv"

    print("Load metadata...")
    meta = load_metadata()

    print("Load test data...")
    test = load_test(test_path)
    sub = load_sample_submission(sample_path, test)
    print(f" test={len(test)} submission={len(sub)}")

    print("Build features...")
    features = build_features(test, meta)
    print(f" features={len(features.columns)}")

    print("Inference model...")
    preds = inference_model(features, meta)
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, test[ID_COL].tolist(), preds)
    save_submission(output_path, sub)

    print(
        f"Saved: {output_path} rows={len(sub)} "
        f"mean={preds.mean():.6f} std={preds.std():.6f} sec={time.time()-t0:.1f}"
    )


if __name__ == "__main__":
    main()
'''


def extract_source(source_zip: Path, root: Path) -> None:
    with zipfile.ZipFile(source_zip) as zf:
        zf.extractall(root)


def validate_source(root: Path) -> tuple[Path, Path, Path]:
    script = root / "script.py"
    model = root / "model"
    requirements = root / "requirements.txt"
    required_models = [
        "multi.cbm",
        "aux_reverse.cbm",
        "aux_middle.cbm",
        "hurdle_gate.cbm",
        "hurdle_cond.cbm",
        "offset.cbm",
        "joint.cbm",
        "metadata.json",
        "history.pkl.gz",
        "history_aux.pkl.gz",
    ]
    if not script.is_file() or not model.is_dir() or not requirements.is_file():
        raise RuntimeError("source ZIP must contain model/ + script.py + requirements.txt")
    missing = [name for name in required_models if not (model / name).is_file()]
    if missing:
        raise RuntimeError(f"source ZIP missing model artifacts: {missing}")
    if not (model / "code").is_dir():
        raise RuntimeError("source ZIP missing model/code")
    return script, model, requirements


def convert_history_to_csv(model: Path) -> None:
    pairs = [
        ("history.pkl.gz", "history.csv.gz"),
        ("history_aux.pkl.gz", "history_aux.csv.gz"),
    ]
    for old_name, new_name in pairs:
        old_path = model / old_name
        new_path = model / new_name
        df = pd.read_pickle(old_path)
        print(f"[read] {old_name}: rows={len(df):,} cols={len(df.columns)}")
        df.to_csv(
            new_path,
            index=False,
            compression="gzip",
            float_format="%.17g",
        )
        old_path.unlink()
        print(f"[write] {new_name}: {new_path.stat().st_size / 1024**2:.1f} MiB")


def validate_package(root: Path) -> None:
    script = root / "script.py"
    model = root / "model"
    req = root / "requirements.txt"

    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    if req.read_text(encoding="utf-8") != REQUIREMENTS:
        raise RuntimeError("requirements.txt differs from expected minimal requirements")

    for name in ("history.csv.gz", "history_aux.csv.gz"):
        if not (model / name).is_file():
            raise RuntimeError(f"missing converted artifact: {name}")
    if list(model.glob("*.pkl")) or list(model.glob("*.pkl.gz")):
        raise RuntimeError("pickle artifacts remain in model directory")

    meta = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
    if int(meta.get("target_season", -1)) != 2025:
        raise RuntimeError(f"unexpected target_season: {meta.get('target_season')}")


def write_zip(root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())

    with zipfile.ZipFile(output_zip) as zf:
        names = zf.namelist()
        roots = {name.split("/", 1)[0] for name in names}
        expected = {"model", "script.py", "requirements.txt"}
        if roots != expected:
            raise RuntimeError(f"unexpected ZIP roots: {sorted(roots)}")
        if any(name.endswith((".pkl", ".pkl.gz")) for name in names):
            raise RuntimeError("pickle artifact found in output ZIP")
        extracted = sum(info.file_size for info in zf.infolist())

    print(
        f"[zip] {output_zip} compressed={output_zip.stat().st_size/1024**2:.1f} MiB "
        f"extracted={extracted/1024**2:.1f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repackage the already-trained current_best submission without retraining. "
            "The only artifact change is pickle -> gzip CSV for pandas/NumPy portability, "
            "plus a baseline-style inference script and minimal requirements.txt."
        )
    )
    parser.add_argument("--source", default="dist/current_best_submit.zip")
    parser.add_argument("--output", default="dist/build_second_fix.zip")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_zip = (repo_root / args.source).resolve()
    output_zip = (repo_root / args.output).resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)

    with tempfile.TemporaryDirectory(prefix="aimers_second_fix_") as tmp:
        root = Path(tmp)
        extract_source(source_zip, root)
        script, model, requirements = validate_source(root)

        convert_history_to_csv(model)
        script.write_text(INFERENCE_SCRIPT, encoding="utf-8")
        requirements.write_text(REQUIREMENTS, encoding="utf-8")

        validate_package(root)
        write_zip(root, output_zip)

    print("[done] no retraining performed")
    print("[done] requirements.txt: catboost==1.2.10 only")
    print(f"[done] {output_zip}")


if __name__ == "__main__":
    main()
