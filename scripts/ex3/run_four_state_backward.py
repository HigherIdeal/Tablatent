from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex3.four_state_backward import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX3: four-state counterfactual backward energy. "
            "Generate F_hard/F_soft/S_soft/S_hard post-pitch states, reconstruct the known past, "
            "and derive success probability only from backward consistency."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex3_four_state_backward_energy.yaml",
        help="EX3 experiment config",
    )
    args = parser.parse_args()
    results = run(args.config)

    print("\n[EX3 summary]")
    for variant, block in results["variants"].items():
        for season, fold in block["folds"].items():
            if fold.get("skipped"):
                continue
            m = fold["metrics"]
            d = fold["latent_state_diagnostics"]
            print(
                f"  [{variant}:{season}] auc={m['auc_margin']:.5f} "
                f"brier={m['brier']:.8f} prior={m['prior_brier']:.8f} "
                f"amb={d['ambiguity_mean']:.3f} conf={d['confidence_mean']:.3f}"
            )
            print(
                "    mean P: "
                + ", ".join(f"{k}={v:.3f}" for k, v in d["mean_probability"].items())
            )
            print(
                "    winners: "
                + ", ".join(f"{k}={v}" for k, v in d["winner_counts"].items())
            )
            if "failure_true_branch" in d:
                f = d["failure_true_branch"]
                s = d["success_true_branch"]
                print(
                    f"    true-branch hard share: failure={f['hard_share']:.3f}, "
                    f"success={s['hard_share']:.3f}"
                )


if __name__ == "__main__":
    main()
