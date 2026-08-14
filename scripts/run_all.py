from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import evaluate, train_stage1, train_stage2
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 -> Stage2 -> 평가 전체 실행")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train_stage1(config)
    train_stage2(config)
    print(json.dumps(evaluate(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

