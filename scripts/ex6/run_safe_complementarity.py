from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.ex6.safe_complementarity import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EX6: measure complementarity of SAFE vs overnight GPU2/GPU3 candidates on 2024"
    )
    parser.add_argument("--config", default="experiments/configs/ex6_safe_complementarity.yaml")
    parser.add_argument("--gpu", default="2", help="physical GPU exposed to CatBoost; default 2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = run(args.config)

    print("\n[EX6 complete]", flush=True)
    print(
        f"safe={result['vectors']['safe']['brier']:.9f} "
        f"gpu2={result['vectors']['gpu2_night_best']['brier']:.9f} "
        f"gpu3={result['vectors']['gpu3_affine_corrected']['brier']:.9f}",
        flush=True,
    )
    print("\n[Pair blend diagnostic]", flush=True)
    for name, item in result["pair_blends"].items():
        print(
            f"{name:<26s} w={item['candidate_weight']:.3f} "
            f"brier={item['brier']:.9f} delta={item['delta_vs_safe']:+.9f}",
            flush=True,
        )
    t = result["triple_blend"]
    print(
        f"\n[Triple] SAFE={t['safe_weight']:.3f} GPU2={t['gpu2_weight']:.3f} "
        f"GPU3={t['gpu3_weight']:.3f} brier={t['brier']:.9f} delta={t['delta_vs_safe']:+.9f}",
        flush=True,
    )
    print("report=outputs/experiments/ex6_safe_complementarity/report.md", flush=True)


if __name__ == "__main__":
    main()
