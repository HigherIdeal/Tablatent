from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bitaboost.ex9.density_ratio_reweight as ex9_core
from bitaboost.ex9.controlfix import fit_direct_weighted_exact_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="EX9: reweight old seasons by 2023 input-density similarity")
    parser.add_argument("--config", default="experiments/configs/ex9_density_ratio_reweight.yaml")
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Patch the EX9 direct fit with the baseline-scale-preserving implementation.
    # This makes alpha=0 a true SAFE retrain control instead of accidentally
    # changing CatBoost's effective regularization through global weight rescaling.
    ex9_core._fit_direct_weighted = fit_direct_weighted_exact_scale

    result = ex9_core.run(args.config)
    print("\n[EX9 complete]")
    print(f"SAFE982={result['baseline']['final_brier']:.12f}")
    print(f"domain_holdout_auc={result['domain']['holdout_auc']}")
    best = result["best"]
    print(
        f"best alpha={best['alpha']:.2f} final={best['final_brier']:.12f} "
        f"delta={best['final_brier'] - result['baseline']['final_brier']:+.12f}"
    )
    print("report=outputs/experiments/ex9_density_ratio_reweight/report.md")


if __name__ == "__main__":
    main()
