from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex3.strength_sweep import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX3 scale diagnostic: reuse the same backward model while sweeping "
            "counterfactual pseudo-count strength lambda."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex3_counterfactual_strength_sweep.yaml",
        help="EX3 strength-sweep config",
    )
    args = parser.parse_args()
    result = run(args.config)

    print("\n[EX3-SWEEP summary]")
    for variant, payload in result["variants"].items():
        print(f"  [{variant}]")
        for strength, summary in payload["strength_summary"].items():
            aucs = summary.get("fold_aucs", [])
            auc_text = ", ".join(f"{x:.5f}" for x in aucs)
            print(
                f"    lambda={strength:>4} "
                f"mean_auc={summary['mean_auc_margin']:.5f} "
                f"min_auc={summary['min_auc_margin']:.5f} "
                f"gap={summary['mean_hard_success_gap']:.5f} "
                f"folds=[{auc_text}]"
            )
        best = payload.get("best_by_mean_auc")
        if best:
            print(
                f"    best_mean_auc: lambda={best['strength']:g} "
                f"auc={best['mean_auc_margin']:.5f} "
                f"min_fold={best['min_auc_margin']:.5f}"
            )


if __name__ == "__main__":
    main()
