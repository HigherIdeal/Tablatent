from __future__ import annotations

import argparse, gc, json, shutil, tempfile, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import build_final_optimized_submission as zipper
import build_recent_regime_submissions as recent_core
import run_asof_prefix_inversion_probe as prefix_core
import run_game_type_temporal_regime_ablation as regime_core
from run_multitask_outcome_boosting import auxiliary_targets
from src.utils import load_config


DIRECT = {
    "prefix_a4_l6": ("A4_MULTI_SCALE_STATE", 6, 800, False),
    "prefix_a4_r6": ("A4_MULTI_SCALE_STATE", 6, 1000, True),
    "prefix_a4_r7": ("A4_MULTI_SCALE_STATE", 7, 1000, True),
    "prefix_a4_l7": ("A4_MULTI_SCALE_STATE", 7, 1000, False),
    "prefix_a3_l6": ("A3_MULTI_PREFIX_STATE", 6, 800, False),
}
JOINT = {"prefix_j6": (6, 1200), "prefix_j7": (7, 1000), "prefix_j8": (8, 800)}


def params(depth: int, iterations: int) -> dict:
    return dict(iterations=iterations, depth=depth, learning_rate=.03, l2_leaf_reg=20,
        random_strength=.5, bootstrap_type="Bayesian", bagging_temperature=.5,
        border_count=128, has_time=True, one_hot_max_size=10, allow_writing_files=False,
        task_type="GPU", devices="0", verbose=False, random_seed=42)


def patch_script(text: str) -> str:
    text = text.replace("import run_offset_residual_boosting as offset_core\n", "import run_offset_residual_boosting as offset_core\nimport run_asof_prefix_inversion_probe as prefix_core\n", 1)
    needle = '    combo = pd.concat([history, test], ignore_index=True, sort=False)\n'
    text = text.replace(needle, needle + '''    prefix_core.add_prefix_inversion_features(
        combo, pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05
    )
    combo["eng_recent_f"] = (
        pd.to_numeric(combo["season"], errors="raise").ge(2023)
        & combo["game_type"].astype("string").str.upper().eq("F")
    ).astype(np.float32)
''', 1)
    needle = "    pred = np.clip(pred, 0.0, 1.0)\n"
    replacement = '''    # Prefix-state SOTA ensemble; old gate_cond/offset remain proven components.
    pmeta = meta["prefix_sota"]
    pools = {name: make_pool(features, pmeta["specs"][name]) for name in ("a3", "a4")}
    pc = {"mixed": gate_cond, "offset": offset}
    for name, spec_name in pmeta["direct_specs"].items():
        spec = pmeta["models"][name]
        model = load_regressor(name) if spec["regression"] else load_classifier(name)
        pc[name] = np.clip(
            predict_regression(model, pools[spec_name], spec["trees"])
            if spec["regression"] else predict_probability(model, pools[spec_name], spec["trees"])[:, 1],
            0.0, 1.0,
        )
    jq = np.asarray(pmeta["joint_q"], dtype=np.float64)
    gi = (gt == "F").astype(np.int8)
    for name in pmeta["joint_names"]:
        jp = predict_probability(load_classifier(name), pools["a4"], pmeta["models"][name]["trees"])
        pc[name] = np.sum(jp * jq[gi], axis=1)
    names = pmeta["candidate_names"]
    matrix = np.column_stack([pc[name] for name in names])
    wr, wf = np.asarray(pmeta["weights_R"]), np.asarray(pmeta["weights_F"])
    pred[mask_r] = matrix[mask_r] @ wr
    pred[mask_f] = matrix[mask_f] @ wf
    pred[mask_r] = .5 + float(pmeta["calibration_R"]) * (pred[mask_r] - .5)
    pred[mask_f] = .5 + float(pmeta["calibration_F"]) * (pred[mask_f] - .5)
    pred = np.clip(pred, 0.0, 1.0)
'''
    if text.count(needle) != 1:
        raise RuntimeError("inference patch point mismatch")
    return text.replace(needle, replacement, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dist/sota_submit.zip")
    ap.add_argument("--output", default="dist/prefix_sota_submit.zip")
    a = ap.parse_args()
    cfg = load_config(ROOT / "configs/default.yaml")
    frame, _ = recent_core.prepare_frame(cfg)
    frame["season"] = pd.to_numeric(frame.season, errors="raise").astype(int)
    frame["game_type"] = frame.game_type.astype("string").str.upper()
    prefix_core.add_prefix_inversion_features(frame, pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05)
    regime_core.add_regime_features(frame, season_col="season", regime_start_year=2023)
    sets = prefix_core.feature_sets(recent_core.feature_set("recent_raw_game_type"))
    prepared = {k: regime_core.prepare_x(frame, sets[k]) for k in ("A3_MULTI_PREFIX_STATE", "A4_MULTI_SCALE_STATE")}
    y = frame.control_success.to_numpy(np.float32)

    with tempfile.TemporaryDirectory(prefix="prefix_sota_") as td:
        package = Path(td)
        with zipfile.ZipFile(ROOT / a.source) as zf: zf.extractall(package)
        model_dir = package / "model"
        for name, (variant, depth, trees, regression) in DIRECT.items():
            x, cats = prepared[variant]
            cls = CatBoostRegressor if regression else CatBoostClassifier
            kw = params(depth, trees)
            if regression: kw["loss_function"] = "RMSE"
            model = cls(**kw).fit(Pool(x, y, cat_features=cats, feature_names=sets[variant]))
            model.save_model(str(model_dir / f"{name}.cbm")); del model; gc.collect()
            print(f"fit {name}")

        aux = auxiliary_targets(frame)[["reverse", "middle", "ball", "strike"]]
        keep = aux.notna().all(axis=1).to_numpy(); bits=aux.loc[keep].to_numpy(np.int8)
        codes=bits @ (1 << np.arange(4,dtype=np.int16)); classes=np.unique(codes); labels=np.searchsorted(classes,codes)
        x4,c4=prepared["A4_MULTI_SCALE_STATE"]
        for name,(depth,trees) in JOINT.items():
            model=CatBoostClassifier(loss_function="MultiClass",**params(depth,trees)).fit(Pool(x4.loc[keep],labels,cat_features=c4,feature_names=sets["A4_MULTI_SCALE_STATE"]))
            model.save_model(str(model_dir/f"{name}.cbm"));del model;gc.collect();print(f"fit {name}")
        target=y[keep];gt=frame.loc[keep,"game_type"].astype(str).to_numpy();sy=frame.loc[keep,"season"].to_numpy();q=np.zeros((2,len(classes)))
        for gi,d in enumerate(("R","F")):
            dm=gt==d
            if d=="F": dm &= sy>=2023
            for ci in range(len(classes)):
                cm=dm&(labels==ci);q[gi,ci]=target[cm].mean() if cm.any() else target[dm].mean()

        meta=json.loads((model_dir/"metadata.json").read_text())
        names=["mixed","offset",*DIRECT,*JOINT]
        wr=[.319021936,0,.176211179,.126646379,.216197527,0,0,0,0,.161922979]
        wf=[.289773765,.117266991,0,0,0,.235485230,.128534016,.153961520,.005867201,.069111277]
        meta["prefix_sota"]={"specs":{"a3":{"features":sets["A3_MULTI_PREFIX_STATE"],"categorical":prepared["A3_MULTI_PREFIX_STATE"][1]},"a4":{"features":sets["A4_MULTI_SCALE_STATE"],"categorical":c4}},"direct_specs":{n:("a3" if v[0].startswith("A3") else "a4") for n,v in DIRECT.items()},"models":{**{n:{"trees":v[2],"regression":v[3]} for n,v in DIRECT.items()},**{n:{"trees":v[1],"regression":False} for n,v in JOINT.items()}},"joint_names":list(JOINT),"joint_q":q.tolist(),"joint_classes":classes.tolist(),"candidate_names":names,"weights_R":wr,"weights_F":wf,"calibration_R":1.15,"calibration_F":1.25,"validation_score":1222.8806105919587}
        (model_dir/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
        shutil.copy2(ROOT/"scripts/run_asof_prefix_inversion_probe.py",model_dir/"code/scripts/run_asof_prefix_inversion_probe.py")
        script=package/"script.py";script.write_text(patch_script(script.read_text()),encoding="utf-8");compile(script.read_bytes(),"script.py","exec")
        zipper.write_fast_zip(package,ROOT/a.output)
    print(f"ok {ROOT/a.output}")


if __name__ == "__main__": main()
