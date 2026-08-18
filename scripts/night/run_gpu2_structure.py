from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.night.gpu2_structure import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight GPU2 structural / stable-trait search")
    parser.add_argument("--config", default="experiments/configs/night_campaign_20260819.yaml")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    result = run(args.config, hours=args.hours, expected_gpu=args.gpu)
    best = result.get("best")
    print("\n[GPU2 complete]", flush=True)
    if best:
        print(
            f"best={best.get('trial_id')} objective={best.get('objective'):.9f} "
            f"weighted_brier={best.get('weighted_brier'):.9f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
