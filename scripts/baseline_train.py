from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BRIER = 0.247355098
REFERENCE_SCORE = 981.5
REFERENCE_LB = 1098.86143


def metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("model/metadata.json").decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the frozen rule-safe Codex baseline.")
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--output", default="dist/baseline_SAFE.zip")
    ap.add_argument("--tolerance", type=float, default=1.5e-4)
    ap.add_argument("--smoke", action="store_true", help="Also run final package inference smoke test on data/test.csv.")
    args = ap.parse_args()

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

    meta = metadata(out)
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
