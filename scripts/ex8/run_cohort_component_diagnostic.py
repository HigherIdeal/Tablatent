from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.ex8.cohort_component_diagnostic import run


def main() -> None:
    parser = argparse.ArgumentParser(description="EX8: SAFE component performance by pitcher/batter experience cohort")
    parser.add_argument("--config", default="experiments/configs/ex8_cohort_component_diagnostic.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print("\n[EX8 complete]")
    print(f"SAFE982={result['overall']['final_brier']:.12f}")
    for name in ("pitcher", "batter", "cross"):
        item = result["oracle_summary"][name]
        print(
            f"{name}: coverage={item['coverage']:.3%} "
            f"fallback_routed={item['fallback_routed_brier']:.12f} "
            f"gain={item['fallback_gain_vs_safe']:+.12f}"
        )
    print("report=outputs/experiments/ex8_cohort_component_diagnostic/report.md")


if __name__ == "__main__":
    main()
