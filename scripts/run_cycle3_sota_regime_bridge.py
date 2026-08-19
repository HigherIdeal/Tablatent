from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bitaboost.baseline as baseline_core
from bitaboost.features import prepare as prepare_features
from bitaboost.cycle3.sota_regime_bridge import run


def _install_early_fold_prepare_fix() -> None:
    """Make the recovered SAFE trainer safe for the intentional 2023 source fold.

    SAFE982 normally validates on 2024, so the prepared frame and the full auxiliary
    reconstruction both contain all 2019-2024 rows. Cycle3 intentionally moves the
    validation season back to 2023. `features.prepare()` then keeps only <=2023 rows
    after regime-feature construction, while its auxiliary table still contains the
    original full frame. The historical baseline trainer indexes that table with the
    shortened train mask, which causes a boolean-length mismatch.

    Keep the core SAFE implementation untouched and patch only this diagnostic runner:
    align auxiliary rows to the exact prepared-frame index before baseline training.
    The ordinary 2024 SAFE run is unchanged because its lengths already match.

    The baseline's forensic reference audit is also 2024-specific. It must be disabled
    for the 2023 source fold; otherwise a correctly trained 2023 vector would be
    rejected only because it cannot match the 2024 reference target/vector lengths.
    """

    def prepare_aligned(cfg):
        data = prepare_features(cfg)
        if len(data.aux) != len(data.frame):
            missing = data.frame.index.difference(data.aux.index)
            if len(missing):
                raise RuntimeError(
                    f"Cycle3 auxiliary alignment lost {len(missing)} prepared rows"
                )
            data.aux = data.aux.loc[data.frame.index].copy()
        if len(data.aux) != len(data.frame):
            raise RuntimeError(
                f"Cycle3 auxiliary alignment failed: aux={len(data.aux)} frame={len(data.frame)}"
            )
        return data

    baseline_core.prepare = prepare_aligned
    baseline_core.audit_if_available = lambda cfg, y, components: {
        "available": False,
        "reason": "cycle3_2023_source_fold",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycle 3: transfer 2023 SAFE-family residual calibration to frozen SAFE982 2024")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--no-reuse-source", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _install_early_fold_prepare_fix()
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
