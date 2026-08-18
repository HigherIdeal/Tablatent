from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.night.summary import watch


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh overnight_report.md while workers run")
    parser.add_argument("--root", default="outputs/night_20260819")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--hours", type=float, default=8.5)
    args = parser.parse_args()
    watch(args.root, interval_seconds=args.interval, hours=args.hours)


if __name__ == "__main__":
    main()
