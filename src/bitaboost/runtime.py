from __future__ import annotations

import os
import time
import warnings
from contextlib import contextmanager
from typing import Iterator


def configure_cuda(cfg: dict) -> None:
    physical = str(cfg["runtime"]["physical_gpu"])
    logical = str(cfg["runtime"]["catboost_device"])
    if physical != "2" or logical != "0":
        raise RuntimeError("expected physical GPU 2 -> logical CatBoost device 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical


def configure_warnings(cfg: dict) -> None:
    if not bool(cfg["runtime"].get("suppress_performance_warnings", True)):
        return
    try:
        import pandas as pd
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    except Exception:
        pass


def log(message: str) -> None:
    print(message, flush=True)


@contextmanager
def stage(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    log(f"[stage] {name} ...")
    try:
        yield
    finally:
        log(f"[stage] {name} done in {time.perf_counter() - t0:.1f}s")
