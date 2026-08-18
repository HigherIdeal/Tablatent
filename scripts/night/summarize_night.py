from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.night.summary import refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the combined overnight report")
    parser.add_argument("--root", default="outputs/night_20260819")
    args = parser.parse_args()
    state = refresh(args.root)
    print(f"updated={state['updated_utc']} complete={state['complete']}")
    for name in ("gpu2", "gpu3"):
        worker = state[name]
        best = worker.get("best")
        hb = worker.get("heartbeat") or {}
        print(
            f"{name}: phase={hb.get('phase')} trials={worker['trials_total']} ok={worker['trials_ok']} "
            f"errors={worker['errors']} best={best.get('trial_id') if best else None}"
        )


if __name__ == "__main__":
    main()
