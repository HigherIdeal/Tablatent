from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import build_second_fix_submission as base


REQUIREMENTS = "catboost==1.2.10\n"


def optimized_inference_script() -> str:
    """Patch the already-validated portable inference script for lower overhead.

    The numerical feature definitions and trained-model metadata are unchanged.
    We only preallocate model engineered columns before the frozen-profile helpers
    mutate the frame, defragment once before slicing the test rows, and add phase
    timing. This avoids pandas' repeated-column-insert fragmentation path.
    """
    text = base.INFERENCE_SCRIPT

    old = '''    history["_submit_pos"] = -1\n    combo = pd.concat([history, test], ignore_index=True, sort=False)\n    aux_test = pd.DataFrame(\n'''
    new = '''    history["_submit_pos"] = -1\n    combo = pd.concat([history, test], ignore_index=True, sort=False)\n\n    # Preallocate engineered columns used by the trained models in one contiguous\n    # block. The helper functions below then overwrite existing columns instead of\n    # repeatedly inserting new blocks into a 1.5M+ row DataFrame.\n    model_eng = sorted({\n        col\n        for spec_name in ("rich", "hurdle", "offset")\n        for col in meta["specs"][spec_name]["features"]\n        if col.startswith("eng_") and col not in combo.columns\n    })\n    if model_eng:\n        prealloc = pd.DataFrame(\n            np.nan,\n            index=combo.index,\n            columns=model_eng,\n            dtype=np.float32,\n        )\n        combo = pd.concat([combo, prealloc], axis=1, copy=False)\n        del prealloc\n\n    aux_test = pd.DataFrame(\n'''
    if old not in text:
        raise RuntimeError("failed to locate combo construction patch point")
    text = text.replace(old, new, 1)

    old = '''    if bool(meta.get("use_paths", False)):\n        path_core.add_paths(combo, "pitcher_id", "season", TARGET_COL)\n\n    out = combo.loc[combo["_submit_pos"].ge(0)].sort_values("_submit_pos").copy()\n'''
    new = '''    if bool(meta.get("use_paths", False)):\n        path_core.add_paths(combo, "pitcher_id", "season", TARGET_COL)\n\n    # Consolidate any small intermediate blocks created by helper-only columns\n    # before the final test slice and CatBoost Pool construction.\n    combo = combo.copy()\n    out = combo.loc[combo["_submit_pos"].ge(0)].sort_values("_submit_pos").copy()\n'''
    if old not in text:
        raise RuntimeError("failed to locate defragmentation patch point")
    text = text.replace(old, new, 1)

    # Keep submission logs readable after the structural optimization. Any helper-
    # only temporary insertion that remains does not need to flood stderr.
    old = '''import numpy as np\nimport pandas as pd\nfrom catboost import CatBoostClassifier, CatBoostRegressor, Pool\n'''
    new = '''import numpy as np\nimport pandas as pd\nfrom catboost import CatBoostClassifier, CatBoostRegressor, Pool\n\n# pandas PerformanceWarning here only concerns block layout; engineered values are\n# unchanged. Model columns are preallocated below to avoid the expensive path.\nimport warnings\nwarnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)\n'''
    if old not in text:
        raise RuntimeError("failed to locate import patch point")
    text = text.replace(old, new, 1)

    return text


def validate_source(root: Path) -> tuple[Path, Path, Path]:
    script = root / "script.py"
    model = root / "model"
    req = root / "requirements.txt"
    if not script.is_file() or not model.is_dir() or not req.is_file():
        raise RuntimeError("source ZIP must contain model/ + script.py + requirements.txt")

    required = [
        "multi.cbm",
        "aux_reverse.cbm",
        "aux_middle.cbm",
        "hurdle_gate.cbm",
        "hurdle_cond.cbm",
        "offset.cbm",
        "joint.cbm",
        "metadata.json",
        "history.csv.gz",
        "history_aux.csv.gz",
    ]
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise RuntimeError(
            "optimized repackager expects the portable second-fix ZIP; "
            f"missing: {missing}"
        )
    if not (model / "code").is_dir():
        raise RuntimeError("source ZIP missing model/code")
    if list(model.glob("*.pkl")) or list(model.glob("*.pkl.gz")):
        raise RuntimeError("pickle artifacts are not allowed")
    return script, model, req


def write_fast_zip(root: Path, output_zip: Path) -> None:
    """Avoid recompressing artifacts that are already compressed/binary.

    history CSVs are gzip files and CatBoost .cbm files are binary model blobs.
    Storing them directly makes package creation and extraction much faster while
    keeping the extracted contents byte-identical.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)

    with zipfile.ZipFile(output_zip, "w") as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            suffixes = path.suffixes
            already_compressed = path.suffix == ".cbm" or suffixes[-2:] == [".csv", ".gz"]
            compression = zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED
            zf.write(path, rel, compress_type=compression, compresslevel=None if compression == zipfile.ZIP_STORED else 1)

    with zipfile.ZipFile(output_zip) as zf:
        roots = {name.split("/", 1)[0] for name in zf.namelist()}
        expected = {"model", "script.py", "requirements.txt"}
        if roots != expected:
            raise RuntimeError(f"unexpected ZIP roots: {sorted(roots)}")
        if any(name.endswith((".pkl", ".pkl.gz")) for name in zf.namelist()):
            raise RuntimeError("pickle artifact found in output ZIP")
        extracted = sum(info.file_size for info in zf.infolist())
        stored = sum(info.file_size for info in zf.infolist() if info.compress_type == zipfile.ZIP_STORED)

    print(
        f"[zip] {output_zip} size={output_zip.stat().st_size/1024**2:.1f} MiB "
        f"extracted={extracted/1024**2:.1f} MiB stored={stored/1024**2:.1f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast repack of the already-built portable submission. No retraining and "
            "no history conversion: only inference-frame optimization and fast ZIP layout."
        )
    )
    parser.add_argument("--source", default="dist/build_second_fix.zip")
    parser.add_argument("--output", default="dist/build_final_optimized.zip")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_zip = (repo_root / args.source).resolve()
    output_zip = (repo_root / args.output).resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)

    with tempfile.TemporaryDirectory(prefix="aimers_final_opt_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(source_zip) as zf:
            zf.extractall(root)

        script, model, req = validate_source(root)
        script.write_text(optimized_inference_script(), encoding="utf-8")
        req.write_text(REQUIREMENTS, encoding="utf-8")

        compile(script.read_text(encoding="utf-8"), str(script), "exec")
        if req.read_text(encoding="utf-8") != REQUIREMENTS:
            raise RuntimeError("requirements.txt is not minimal")

        write_fast_zip(root, output_zip)

    print("[done] no retraining; model/history bytes unchanged")
    print("[done] requirements.txt: catboost==1.2.10 only")
    print("[done] optimized: engineered-column preallocation + final defragmentation")
    print(f"[done] {output_zip}")


if __name__ == "__main__":
    main()
