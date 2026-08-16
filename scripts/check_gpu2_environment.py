from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the isolated RTX 4090 GPU2 training environment.")
    parser.add_argument("--smoke-catboost", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    import catboost

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("[Environment check]")
    print(f"  python={sys.version.split()[0]}")
    print(f"  CUDA_VISIBLE_DEVICES={visible!r}")
    print(f"  torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"  catboost={catboost.__version__}")
    print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count={torch.cuda.device_count()}")

    if visible != "2":
        raise RuntimeError("source scripts/activate_gpu2.sh first; expected CUDA_VISIBLE_DEVICES=2")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("GPU isolation failed: exactly one logical CUDA device must be visible")

    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024**3)
    print(f"  logical_cuda0={props.name} memory={total_gib:.2f} GiB")

    # Small CUDA operation catches broken driver/runtime combinations without
    # spending meaningful time or memory.
    x = torch.randn((2048, 2048), device="cuda:0")
    y = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(y).all().item():
        raise RuntimeError("CUDA matmul produced non-finite values")
    del x, y
    torch.cuda.empty_cache()
    print("  torch CUDA smoke=OK")

    if args.smoke_catboost:
        from catboost import CatBoostClassifier

        rng = np.random.default_rng(42)
        x_np = rng.normal(size=(4096, 16)).astype(np.float32)
        y_np = (x_np[:, 0] + 0.25 * x_np[:, 1] > 0).astype(np.int32)
        model = CatBoostClassifier(
            iterations=8,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
            random_seed=42,
        )
        model.fit(x_np, y_np)
        p = model.predict_proba(x_np[:32])[:, 1]
        if not np.isfinite(p).all():
            raise RuntimeError("CatBoost GPU smoke produced non-finite predictions")
        print("  CatBoost GPU smoke=OK")

    print("[Environment check] PASS — physical GPU 2 is isolated as logical cuda:0")


if __name__ == "__main__":
    main()
