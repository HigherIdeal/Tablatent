from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_gated_r_specialist_submission as builder


def _auto_smoke_dir(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).resolve()
        if not (path / "test.csv").is_file():
            raise FileNotFoundError(f"smoke test.csv not found: {path / 'test.csv'}")
        return path

    candidates = [ROOT / "data", ROOT / "data" / "raw", ROOT / "open"]
    for path in candidates:
        if (path / "test.csv").is_file():
            return path.resolve()

    # Last-resort local discovery. Keep it scoped to the repository.
    matches = sorted(ROOT.rglob("test.csv"))
    for match in matches:
        if match.is_file():
            return match.parent.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package already-trained gated R-fast models without retraining."
    )
    parser.add_argument("--models-dir", default="outputs/gated_r_fast_final")
    parser.add_argument(
        "--output",
        default="dist/gated_r/gated_r_fast_full80_recent20_beta10.zip",
    )
    parser.add_argument("--smoke-data-dir", default=None)
    args = parser.parse_args()

    models_dir = (ROOT / args.models_dir).resolve()
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    full_model = models_dir / "full_raw.cbm"
    recent_model = models_dir / "recent_raw.cbm"
    r_fast_model = models_dir / "r_fast.cbm"
    missing = [str(p) for p in (full_model, recent_model, r_fast_model) if not p.is_file()]
    if missing:
        raise FileNotFoundError("trained model(s) missing: " + ", ".join(missing))

    formula = metadata["formula"]
    alpha_recent = float(formula["alpha_recent"])
    beta_r = float(formula["beta_r"])
    full_features = list(metadata["full_raw"]["features"])
    recent_features = list(metadata["recent_raw"]["features"])
    r_fast_features = list(metadata["r_fast"]["features"])
    full_categorical = list(metadata["full_raw"]["categorical"])
    recent_categorical = list(metadata["recent_raw"]["categorical"])
    r_fast_categorical = list(metadata["r_fast"]["categorical"])

    smoke_dir = _auto_smoke_dir(args.smoke_data_dir)
    if smoke_dir is None:
        print("[package] smoke test skipped: no local test.csv found")
    else:
        print(f"[package] smoke data: {smoke_dir}")

    output_zip = (ROOT / args.output).resolve()
    builder._write_zip(
        output_zip=output_zip,
        full_model=full_model,
        recent_model=recent_model,
        r_fast_model=r_fast_model,
        metadata=metadata,
        full_features=full_features,
        recent_features=recent_features,
        r_fast_features=r_fast_features,
        full_categorical=full_categorical,
        recent_categorical=recent_categorical,
        r_fast_categorical=r_fast_categorical,
        alpha_recent=alpha_recent,
        beta_r=beta_r,
        smoke_data_dir=smoke_dir,
    )
    print(f"[package] ZIP ready: {output_zip}")


if __name__ == "__main__":
    main()
