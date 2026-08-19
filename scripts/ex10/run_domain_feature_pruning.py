from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.ex10.domain_feature_pruning import run


def main() -> None:
    parser = argparse.ArgumentParser(description="EX10: prune top 2023-domain-separating features from SAFE direct head")
    parser.add_argument("--config", default="experiments/configs/ex10_domain_feature_pruning.yaml")
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = run(args.config)

    print("\n[EX10 complete]")
    print(f"SAFE982={result['baseline']['final_brier']:.12f}")
    print(f"domain_holdout_auc={result['domain']['holdout_auc']}")
    print("top_shift_features=" + ", ".join(x["feature"] for x in result["domain_ranking"][:10]))
    best = result["best"]
    print(
        f"best drop_k={best['drop_k']} final={best['final_brier']:.12f} "
        f"delta={best['final_brier'] - result['baseline']['final_brier']:+.12f}"
    )
    print("report=outputs/experiments/ex10_domain_feature_pruning/report.md")


if __name__ == "__main__":
    main()
