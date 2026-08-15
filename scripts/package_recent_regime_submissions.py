from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core


VARIANTS = ["recent_raw_game_type", "recent_drop_game_type"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package already-trained recent-regime CatBoost models without retraining."
    )
    parser.add_argument("--models-dir", default="outputs/recent_regime_models")
    parser.add_argument("--output-dir", default="dist/recent_regime")
    parser.add_argument(
        "--smoke-data-dir",
        default=None,
        help="Optional directory containing test.csv and sample_submission.csv.",
    )
    args = parser.parse_args()

    models_dir = (ROOT / args.models_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_data_dir = Path(args.smoke_data_dir).resolve() if args.smoke_data_dir else None

    if smoke_data_dir is not None and not (smoke_data_dir / "test.csv").is_file():
        raise FileNotFoundError(f"smoke test.csv not found: {smoke_data_dir / 'test.csv'}")

    for variant in VARIANTS:
        model_path = models_dir / f"{variant}.cbm"
        metadata_path = models_dir / f"{variant}.json"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"trained model not found: {model_path}. "
                "Run scripts/train_recent_regime_models.py first."
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"training metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("variant") != variant:
            raise RuntimeError(
                f"metadata/model variant mismatch for {variant}: {metadata.get('variant')}"
            )
        if metadata.get("train_seasons") != [2023, 2024]:
            raise RuntimeError(
                f"unexpected train_seasons for {variant}: {metadata.get('train_seasons')}"
            )
        features = metadata.get("features")
        categorical = metadata.get("categorical")
        if not isinstance(features, list) or not isinstance(categorical, list):
            raise RuntimeError(f"invalid feature metadata for {variant}")

        zip_path = output_dir / f"{variant}.zip"
        recent_core.write_zip(
            output_zip=zip_path,
            model_path=model_path,
            features=features,
            categorical=categorical,
            metadata=metadata,
            smoke_data_dir=smoke_data_dir,
        )
        print(f"[{variant}] ZIP ready: {zip_path}")

    print("\nSubmit in this order:")
    print(f"  1) {output_dir / 'recent_raw_game_type.zip'}")
    print(f"  2) {output_dir / 'recent_drop_game_type.zip'}")
    print("No model retraining was performed during packaging.")


if __name__ == "__main__":
    main()
