from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_zip")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    src = Path(args.input_zip).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = Path(args.output).resolve() if args.output else src.with_name(src.stem + "_pandas_pinned.zip")

    with tempfile.TemporaryDirectory(prefix="aimers_pin_pandas_") as td:
        root = Path(td)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(root)

        req = root / "requirements.txt"
        script = root / "script.py"
        model = root / "model"
        if not req.is_file() or not script.is_file() or not model.is_dir():
            raise RuntimeError("invalid submission ZIP structure")

        lines = [x.strip() for x in req.read_text(encoding="utf-8").splitlines() if x.strip()]
        lines = [x for x in lines if not x.lower().startswith("pandas==")]
        lines.append(f"pandas=={pd.__version__}")
        req.write_text("\n".join(lines) + "\n", encoding="utf-8")

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())

    print(f"pandas={pd.__version__}")
    print(f"repaired={dst}")


if __name__ == "__main__":
    main()
