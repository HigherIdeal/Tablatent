from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.cycle3.sota_regime_bridge import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycle 3: transfer 2023 SAFE-family residual calibration to frozen SAFE982 2024")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--no-reuse-source", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = run(reuse_source=not args.no_reuse_source)
    print("\n[Cycle3 complete]")
    print(f"SAFE982={result['target_2024']['brier']:.12f}")
    p = result["predeclared"]
    print(f"predeclared game_type a=0.50: brier={p['brier']:.12f} delta={p['delta_vs_safe']:+.12f}")
    b = result["best"]
    print(f"best exploratory: {b['kind']} a={b['alpha']:.2f} brier={b['brier']:.12f} delta={b['delta_vs_safe']:+.12f}")
    print("report=outputs/experiments/cycle3_sota_regime_bridge/report.md")


if __name__ == "__main__":
    main()
