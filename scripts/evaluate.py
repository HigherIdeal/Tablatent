from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import evaluate
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="2024 holdout Brier/BSS 평가")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate(load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

