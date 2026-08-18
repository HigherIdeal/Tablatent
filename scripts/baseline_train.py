from __future__ import annotations

import argparse
import importlib.metadata as metadata_lib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BRIER = 0.247355098
REFERENCE_SCORE = 981.5
REFERENCE_LB = 1098.86143

# Competition/test_trial environment fingerprint. Only CatBoost is installed by
# requirements.txt; these packages are expected to already exist on the server.
EXPECTED_RUNTIME = {
    "torch": "2.7.1+cu128",
    "pandas": "2.0.3",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "scikit-learn": "1.8.0",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "narwhals": "2.21.2",
    "transformers": "4.46.3",
    "accelerate": "1.9.0",
    "sentencepiece": "0.1.99",
    "regex": "2023.12.25",
    "tqdm": "4.66.4",
    "loguru": "0.7.2",
    "PyYAML": "6.0.1",
    "rich": "13.7.1",
    "catboost": "1.2.10",
}
CRITICAL_RUNTIME = {"torch", "pandas", "numpy", "scipy", "scikit-learn", "catboost"}


def package_version(name: str) -> str:
    try:
        return metadata_lib.version(name)
    except metadata_lib.PackageNotFoundError:
        return "MISSING"


def preflight_environment(strict: bool = True) -> None:
    print("=== ENVIRONMENT PREFLIGHT ===")
    mismatches = []
    for name, expected in EXPECTED_RUNTIME.items():
        actual = package_version(name)
        mark = "OK" if actual == expected else "DIFF"
        print(f"{name:<16s} {actual:<16s} expected={expected:<16s} [{mark}]")
        if name in CRITICAL_RUNTIME and actual != expected:
            mismatches.append((name, actual, expected))
    if mismatches and strict:
        detail = ", ".join(f"{n}={a} (expected {e})" for n, a, e in mismatches)
        raise SystemExit(
            "Critical runtime does not match the test_trial/competition baseline: " + detail
        )
    if mismatches:
        print("[env] WARNING: critical version mismatch accepted because --no-strict-env was used")
    print()


def read_package_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("model/metadata.json").decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the frozen rule-safe Codex baseline.")
    ap.add_argument(
        "--gpus",
        default="2",
        help="Physical GPU id. Default=2 (single RTX 4090 on iclab4GPU).",
    )
    ap.add_argument("--output", default="dist/baseline_SAFE.zip")
    ap.add_argument("--tolerance", type=float, default=1.5e-4)
    ap.add_argument("--smoke", action="store_true", help="Also run final package inference smoke test on data/test.csv.")
    ap.add_argument("--no-strict-env", action="store_true", help="Print environment differences but do not stop on critical version mismatch.")
    args = ap.parse_args()

    preflight_environment(strict=not args.no_strict_env)

    out = (ROOT / args.output).resolve()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_current_best_submission.py"),
        "--config", "configs/baseline.yaml",
        "--gpus", args.gpus,
        "--output", str(out),
    ]
    if not args.smoke:
        cmd.append("--skip-smoke")
    subprocess.run(cmd, cwd=ROOT, check=True)

    meta = read_package_metadata(out)
    brier = float(meta["validation_2024_brier"])
    score = float(meta["validation_2024_score"])
    delta = brier - REFERENCE_BRIER

    print("\n=== BASELINE CHECK ===")
    print(f"reproduced Brier : {brier:.9f}")
    print(f"reference Brier  : {REFERENCE_BRIER:.9f}")
    print(f"delta            : {delta:+.9f}")
    print(f"reproduced score : {score:.2f}")
    print(f"reference score  : {REFERENCE_SCORE:.1f}")
    print(f"external LB ref  : {REFERENCE_LB:.5f}")

    if abs(delta) > args.tolerance:
        raise SystemExit(
            f"baseline reproduction outside tolerance ({args.tolerance:g}); "
            "do not start new experiments until this is explained"
        )
    print("BASELINE_OK")


if __name__ == "__main__":
    main()
