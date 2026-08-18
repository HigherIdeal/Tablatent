from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# The frozen baseline core intentionally preserves the exact historical feature
# construction order for numerical reproducibility. That legacy code inserts many
# DataFrame columns one by one, which pandas reports as PerformanceWarning. The
# warning is noisy but does not indicate incorrect values, so keep baseline logs
# readable. New experimental code should avoid fragmentation rather than rely on
# this suppression.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "src" / "baseline_legacy"
REFERENCE_BRIER = 0.247355098
REFERENCE_SCORE = 981.5

RUNTIME_LEGACY_FILES = [
    "build_recent_regime_submissions.py",
    "run_catboost_ablation.py",
    "run_asof_state_engineering.py",
    "run_context_interaction_screen.py",
    "run_2025_proxy_validation.py",
    "run_game_type_temporal_regime_ablation.py",
    "run_frozen_season_anchor_probe.py",
    "run_asof_prefix_inversion_probe.py",
    "run_regime_feature_prediction_suite.py",
    "run_offset_residual_boosting.py",
    "run_multitask_outcome_boosting.py",
]


def _all_gpu_ids() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            return ids
    except Exception:
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [x.strip() for x in visible.split(",") if x.strip()]
    return ["0"]


def configure_parallelism(gpus: str) -> tuple[list[str], str]:
    physical = _all_gpu_ids() if gpus.strip().lower() == "all" else [
        x.strip() for x in gpus.replace(":", ",").split(",") if x.strip()
    ]
    if not physical:
        raise ValueError("no GPU selected")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)
    cpu_threads = str(os.cpu_count() or 1)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(key, cpu_threads)
    logical = ":".join(str(i) for i in range(len(physical)))
    print(f"[parallel] physical GPUs={physical} -> CatBoost devices={logical}; CPU threads={cpu_threads}")
    return physical, logical


def _exec_legacy(path: Path, fake_script_name: str, argv: list[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    old_argv = sys.argv[:]
    try:
        fake_file = ROOT / "scripts" / fake_script_name
        sys.argv = [str(fake_file), *argv]
        ns = {
            "__name__": "__main__",
            "__file__": str(fake_file),
            "__package__": None,
        }
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(fake_file), "exec"), ns, ns)
    finally:
        sys.argv = old_argv


def _rewrite_zip(root: Path, output_zip: Path) -> None:
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())


def patch_runtime_dependencies(raw_zip: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="bitaboost_patch_") as td:
        root = Path(td)
        with zipfile.ZipFile(raw_zip) as zf:
            zf.extractall(root)
        legacy_src = root / "model" / "code" / "src" / "baseline_legacy"
        runtime_dst = root / "model" / "code" / "scripts"
        runtime_dst.mkdir(parents=True, exist_ok=True)
        for name in RUNTIME_LEGACY_FILES:
            src = legacy_src / name
            if not src.is_file():
                raise FileNotFoundError(f"missing frozen runtime module: {src}")
            shutil.copy2(src, runtime_dst / name)
        _rewrite_zip(root, raw_zip)
    print(f"[package] injected {len(RUNTIME_LEGACY_FILES)} frozen runtime modules into ZIP only")


def smoke_test_final(package_zip: Path, data_dir: Path, timeout: int) -> None:
    test = data_dir / "test.csv"
    if not test.is_file():
        raise FileNotFoundError(f"smoke test requires {test}")
    with tempfile.TemporaryDirectory(prefix="bitaboost_smoke_") as td:
        root = Path(td)
        with zipfile.ZipFile(package_zip) as zf:
            zf.extractall(root)
        packaged_data = root / "data"
        packaged_data.mkdir(exist_ok=True)
        shutil.copy2(test, packaged_data / "test.csv")
        sample = data_dir / "sample_submission.csv"
        if sample.is_file():
            shutil.copy2(sample, packaged_data / "sample_submission.csv")
        proc = subprocess.run(
            [sys.executable, "script.py"],
            cwd=root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        print(proc.stdout.strip())
        if proc.returncode != 0:
            raise RuntimeError(f"final submission smoke test failed: code={proc.returncode}")
        out = pd.read_csv(root / "output" / "submission.csv")
        p = pd.to_numeric(out["control_success"], errors="raise").to_numpy(float)
        if len(out) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise RuntimeError("invalid smoke-test predictions")
        print(f"[smoke] final package OK rows={len(out):,} mean={p.mean():.6f} std={p.std():.6f}")


def read_metadata(package_zip: Path) -> dict:
    with zipfile.ZipFile(package_zip) as zf:
        return json.loads(zf.read("model/metadata.json").decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and package the rule-safe Bitaboost baseline/current-best model.")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--gpus", default="2", help="Physical GPU id. Default: 2 (single RTX 4090).")
    ap.add_argument("--output", default="dist/current_best_SAFE.zip")
    ap.add_argument("--smoke-data-dir", default="data")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-timeout", type=int, default=900)
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    _, logical_devices = configure_parallelism(args.gpus)
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bitaboost_build_") as td:
        td = Path(td)
        raw_zip = td / "raw.zip"
        portable_zip = td / "portable.zip"

        _exec_legacy(
            LEGACY / "build_current_best_submission_impl.py",
            "build_current_best_submission.py",
            [
                "--config", args.config,
                "--devices", logical_devices,
                "--output", str(raw_zip),
                "--skip-smoke",
                "--disable-paths",
            ],
        )
        patch_runtime_dependencies(raw_zip)

        _exec_legacy(
            LEGACY / "build_second_fix_submission.py",
            "build_second_fix_submission.py",
            ["--source", str(raw_zip), "--output", str(portable_zip)],
        )
        _exec_legacy(
            LEGACY / "build_final_optimized_submission.py",
            "build_final_optimized_submission.py",
            ["--source", str(portable_zip), "--output", str(output)],
        )

        if args.keep_intermediate:
            keep = output.parent / "intermediate"
            keep.mkdir(exist_ok=True)
            shutil.copy2(raw_zip, keep / "raw.zip")
            shutil.copy2(portable_zip, keep / "portable.zip")

    meta = read_metadata(output)
    b = float(meta["validation_2024_brier"])
    s = float(meta["validation_2024_score"])
    print(f"[reference] reproduced validation Brier={b:.9f} score={s:.1f}")
    print(f"[reference] Codex SAFE max      Brier={REFERENCE_BRIER:.9f} score={REFERENCE_SCORE:.1f}")
    print(f"[reference] delta_brier={b-REFERENCE_BRIER:+.3e}")

    if not args.skip_smoke:
        smoke_test_final(output, (ROOT / args.smoke_data_dir).resolve(), args.smoke_timeout)

    print(f"[done] {output}")


if __name__ == "__main__":
    main()
