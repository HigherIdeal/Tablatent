from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex2.hypothesis_backward import run


def _print_summary(result: dict) -> None:
    print("\n[EX2 summary]", flush=True)
    for variant, payload in result.get("variants", {}).items():
        for season, fold in payload.get("folds", {}).items():
            if fold.get("skipped"):
                continue
            m = fold["metrics"]
            r = fold["reconstruction"]
            true_rmse = r["true_hypothesis_rmse"]["macro"]
            false_rmse = r["false_hypothesis_rmse"]["macro"]
            gap = sum(r["counterfactual_gap_mean_abs"].values()) / max(
                1, len(r["counterfactual_gap_mean_abs"])
            )
            targets = ", ".join(
                f"{name}:{info['auc_margin']:.3f}"
                for name, info in fold["target_contribution"].items()
                if info["auc_margin"] is not None
            )
            groups = ", ".join(
                f"{name}:{info['sign_accuracy']:.3f}"
                for name, info in fold["tf_ft_groups"]["groups"].items()
            )
            print(
                f"  [{variant}:{season}] auc={m['auc_margin']:.5f} "
                f"brier={m['brier']:.8f} prior={m['prior_brier']:.8f} "
                f"true_rmse={true_rmse:.6f} false_rmse={false_rmse:.6f} "
                f"cf_gap={gap:.6f}",
                flush=True,
            )
            print(f"    target_auc: {targets}", flush=True)
            print(f"    TT/TF/FT/FF acc: {groups}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX2: candidate-label-conditioned backward reconstruction. "
            "Assume y=0 and y=1 separately, then score which hypothesis better reconstructs the known past state."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex2_hypothesis_backward.yaml",
        help="EX2 experiment config",
    )
    args = parser.parse_args()
    result = run(args.config)
    _print_summary(result)


if __name__ == "__main__":
    main()
