from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from src.data import prepare_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--target", default="control_success")
    parser.add_argument("--season-col", default="season")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = prepare_dataset(
        root=args.root,
        target_col=args.target,
        season_col=args.season_col,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
