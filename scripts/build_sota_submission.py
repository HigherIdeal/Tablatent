from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath


ROOTS = {"model", "script.py", "requirements.txt"}
FORBIDDEN = (".pkl", ".pkl.gz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submission_build.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    source, output = root / cfg["source_zip"], root / cfg["output_zip"]
    overrides = {k: root / v for k, v in cfg.get("overrides", {}).items()}

    with zipfile.ZipFile(source) as src:
        names = [n for n in src.namelist() if not n.endswith("/")]
        selected = [n for n in names if any(n == x or n.startswith(x) for x in cfg["include"])]
        missing = [x for x in cfg["include"] if not any(n == x or n.startswith(x) for n in names)]
        if missing:
            raise RuntimeError(f"missing source members: {missing}")
        members = sorted(set(selected) | set(overrides))
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in members):
            raise RuntimeError("unsafe ZIP member")
        if any(n.endswith(FORBIDDEN) for n in members):
            raise RuntimeError("pickle artifact is forbidden")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        with zipfile.ZipFile(output, "w") as dst:
            for name in members:
                data = overrides[name].read_bytes() if name in overrides else src.read(name)
                stored = name.endswith((".cbm", ".csv.gz"))
                dst.writestr(name, data, zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED, compresslevel=None if stored else 1)

    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        roots = {n.split("/", 1)[0] for n in names}
        if roots != ROOTS:
            raise RuntimeError(f"bad roots: {sorted(roots)}")
        compile(zf.read("script.py"), "script.py", "exec")
        if not zf.read("requirements.txt").strip():
            raise RuntimeError("empty requirements.txt")
    if output.stat().st_size > int(cfg["max_bytes"]):
        raise RuntimeError(f"ZIP too large: {output.stat().st_size}")
    print(f"ok {output} {output.stat().st_size / 2**20:.1f}MiB {len(names)}files")


if __name__ == "__main__":
    main()
