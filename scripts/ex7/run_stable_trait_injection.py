from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EX7: inject frozen career success/strike traits into the SAFE direct head"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex7_stable_trait_injection.yaml",
    )
    parser.add_argument("--gpu", default="2", help="physical GPU exposed as logical CatBoost device 0")
    args = parser.parse_args()

    # Set CUDA visibility before importing the experiment module / CatBoost path.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from bitaboost.ex7.stable_trait_injection import run

    result = run(args.config)
    base = result["baseline"]["final_brier"]
    print("\n[EX7 complete]", flush=True)
    print(f"SAFE982={base:.12f}", flush=True)
    for item in result["variants"]:
        print(
            f"{item['name']:<38s} final={item['final_brier']:.12f} "
            f"delta={item['delta_final_vs_safe982']:+.12f} "
            f"R={item['domain_regression'].get('R', float('nan')):+.9f} "
            f"F={item['domain_regression'].get('F', float('nan')):+.9f}",
            flush=True,
        )
    print("report=outputs/experiments/ex7_stable_trait_injection/report.md", flush=True)


if __name__ == "__main__":
    main()
