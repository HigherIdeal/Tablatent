from __future__ import annotations

import sys
from pathlib import Path


def legacy_root() -> Path:
    return Path(__file__).resolve().parent / "_legacy"


def activate() -> Path:
    root = legacy_root(); scripts = root / "scripts"
    for path in (root, scripts):
        text = str(path)
        if text not in sys.path: sys.path.insert(0, text)
    return root
