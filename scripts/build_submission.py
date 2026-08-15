from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_LOCAL_FILES = {
    ROOT / "outputs" / "checkpoints" / "stage1_context.pt": "model/stage1_context.pt",
    ROOT / "outputs" / "checkpoints" / "stage1_history.pt": "model/stage1_history.pt",
    ROOT / "outputs" / "checkpoints" / "preprocessors.joblib": "model/preprocessors.joblib",
    ROOT / "outputs" / "stage2_catboost" / "stage2_catboost.cbm": "model/stage2_catboost.cbm",
    ROOT / "src" / "data.py": "model/src/data.py",
    ROOT / "src" / "models.py": "model/src/models.py",
}

# The evaluation server already provides torch/numpy/pandas/sklearn/joblib.
# CatBoost is the only additional runtime dependency for this submission.
REQUIREMENTS = "catboost==1.2.10\n"

SCRIPT_PY = r'''from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"

# preprocessors.joblib was created from src.data classes.  Keep the exact
# training-time class definitions under model/src and expose them only for
# deserialization/inference.
sys.path.insert(0, str(MODEL_DIR))
from src.models import ContextVAE, HistoryVAE  # noqa: E402


def _find_data_dir() -> Path:
    # The competition page describes server-added data/, while one current
    # caution line mentions open/.  Supporting both is path compatibility only;
    # inference never aggregates across test rows.
    for name in ("data", "open"):
        candidate = ROOT / name
        if (candidate / "test.csv").is_file():
            return candidate
    raise FileNotFoundError("test.csv not found under ./data or ./open")


def _assert_rowwise_context_contract(test: pd.DataFrame) -> None:
    # Current official data uses T/B.  The training preprocessor also supports
    # textual top/bottom variants.  Reject unknown encodings rather than using
    # its training-time fallback that can infer a mapping from multiple rows.
    if "top_bottom" not in test.columns:
        raise ValueError("missing top_bottom")
    tb = (
        test["top_bottom"]
        .astype("string")
        .fillna("<MISSING>")
        .str.strip()
        .str.lower()
    )
    allowed = {"t", "b", "top", "bottom", "초", "말", "top_inning", "bottom_inning"}
    unknown = sorted(set(tb.unique().tolist()) - allowed)
    if unknown:
        raise ValueError(f"unsupported top_bottom values for row-wise inference: {unknown}")


def _load_stage1():
    prep_bundle = joblib.load(MODEL_DIR / "preprocessors.joblib")
    context_prep = prep_bundle["context"]
    history_prep = prep_bundle["history"]

    c = torch.load(MODEL_DIR / "stage1_context.pt", map_location="cpu", weights_only=True)
    h = torch.load(MODEL_DIR / "stage1_history.pt", map_location="cpu", weights_only=True)
    if c.get("model_type") != "vae" or h.get("model_type") != "vae":
        raise RuntimeError("packaged Stage1 checkpoints are not VAE checkpoints")

    context_model = ContextVAE(
        cardinalities=c["cardinalities"],
        numeric_dim=c["numeric_dim"],
        hidden_dims=c["hidden_dims"],
        latent_dim=c["latent_dim"],
        embedding_dim_max=c["embedding_dim_max"],
        dropout=c["dropout"],
    )
    context_model.load_state_dict(c["state_dict"])

    history_model = HistoryVAE(
        input_dim=h["input_dim"],
        hidden_dims=h["hidden_dims"],
        latent_dim=h["latent_dim"],
        dropout=h["dropout"],
    )
    history_model.load_state_dict(h["state_dict"])
    return prep_bundle, context_model, history_model


def _encode_latent(
    test: pd.DataFrame,
    prep_bundle,
    context_model: ContextVAE,
    history_model: HistoryVAE,
    batch_size: int = 16384,
) -> np.ndarray:
    context_arrays = prep_bundle["context"].transform(test)
    history_x = prep_bundle["history"].transform(test)
    if len(context_arrays.categorical) != len(test) or len(history_x) != len(test):
        raise RuntimeError("preprocessor row count mismatch")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_model.to(dev).eval()
    history_model.to(dev).eval()
    chunks = []

    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            stop = min(start + batch_size, len(test))
            cat = torch.from_numpy(
                np.asarray(context_arrays.categorical[start:stop], dtype=np.int64)
            ).to(dev, non_blocking=True)
            num = torch.from_numpy(
                np.asarray(context_arrays.numeric[start:stop], dtype=np.float32)
            ).to(dev, non_blocking=True)
            hist = torch.from_numpy(
                np.asarray(history_x[start:stop], dtype=np.float32)
            ).to(dev, non_blocking=True)

            c_mu, _ = context_model.encode_distribution(cat, num)
            h_mu, _ = history_model.encode_distribution(hist)
            chunks.append(torch.cat([c_mu, h_mu], dim=1).cpu().numpy())

    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def main() -> None:
    data_dir = _find_data_dir()
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    if "row_id" not in test.columns:
        raise ValueError("test.csv missing row_id")
    _assert_rowwise_context_contract(test)

    prep_bundle, context_model, history_model = _load_stage1()
    z = _encode_latent(test, prep_bundle, context_model, history_model)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "stage2_catboost.cbm"))
    pred = np.asarray(model.predict_proba(z)[:, 1], dtype=np.float64)

    if len(pred) != len(test):
        raise RuntimeError("prediction row count mismatch")
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite prediction detected")
    if np.any((pred < 0.0) | (pred > 1.0)):
        raise RuntimeError("prediction outside [0, 1]")

    sample_path = data_dir / "sample_submission.csv"
    if sample_path.is_file():
        submission = pd.read_csv(sample_path)
        if len(submission) != len(test):
            raise RuntimeError("sample_submission/test row count mismatch")
        if "row_id" not in submission.columns:
            raise ValueError("sample_submission.csv missing row_id")
        if not submission["row_id"].astype(str).equals(test["row_id"].astype(str)):
            raise RuntimeError("sample_submission row_id order differs from test.csv")
        submission = submission[["row_id"]].copy()
    else:
        submission = pd.DataFrame({"row_id": test["row_id"].to_numpy()})

    submission["control_success"] = pred
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    print(
        f"submission rows={len(submission):,} "
        f"mean={pred.mean():.6f} min={pred.min():.6f} max={pred.max():.6f}"
    )


if __name__ == "__main__":
    main()
'''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_local_artifacts() -> None:
    missing = [path for path in REQUIRED_LOCAL_FILES if not path.is_file()]
    if missing:
        listing = "\n".join(f"  - {p.relative_to(ROOT)}" for p in missing)
        raise FileNotFoundError(
            "Submission artifact가 부족합니다. Stage1 및 CatBoost Stage2 학습을 먼저 완료하세요:\n"
            + listing
        )


def _write_package_tree(package_root: Path) -> dict:
    model_dir = package_root / "model"
    (model_dir / "src").mkdir(parents=True, exist_ok=True)

    copied = []
    for src, archive_path in REQUIRED_LOCAL_FILES.items():
        dst = package_root / archive_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(
            {
                "path": archive_path,
                "bytes": dst.stat().st_size,
                "sha256": _sha256(dst),
            }
        )

    # Ensure model/src is a normal package for preprocessors.joblib imports.
    (model_dir / "src" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "script.py").write_text(SCRIPT_PY, encoding="utf-8")
    (package_root / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    provenance = {
        "purpose": "2026-08-15 CatBoost-on-frozen-latent leaderboard probe",
        "stage1": "two-branch VAE posterior means, context 16D + history 16D",
        "stage2": "CatBoostClassifier on frozen 32D latent",
        "warning": "development probe artifacts; not the final 2019-2024 retrained submission",
        "artifacts": copied,
    }
    (model_dir / "submission_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance


def _zip_package(package_root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_root).as_posix())


def _validate_zip(output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip) as zf:
        names = zf.namelist()
        top = {name.split("/", 1)[0] for name in names}
        expected_top = {"model", "script.py", "requirements.txt"}
        if top != expected_top:
            raise RuntimeError(f"invalid ZIP top-level entries: {sorted(top)}")
        required = {
            "script.py",
            "requirements.txt",
            "model/stage1_context.pt",
            "model/stage1_history.pt",
            "model/preprocessors.joblib",
            "model/stage2_catboost.cbm",
            "model/src/data.py",
            "model/src/models.py",
        }
        missing = sorted(required - set(names))
        if missing:
            raise RuntimeError(f"ZIP missing required entries: {missing}")


def _smoke_test(package_root: Path, data_dir: Path) -> None:
    test_path = data_dir / "test.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"smoke-test test.csv 없음: {test_path}")

    local_data = package_root / "data"
    local_data.mkdir(exist_ok=True)
    shutil.copy2(test_path, local_data / "test.csv")
    sample = data_dir / "sample_submission.csv"
    if sample.is_file():
        shutil.copy2(sample, local_data / "sample_submission.csv")

    subprocess.run([sys.executable, "script.py"], cwd=package_root, check=True)
    pred_path = package_root / "output" / "submission.csv"
    if not pred_path.is_file():
        raise RuntimeError("smoke test did not create output/submission.csv")

    import numpy as np
    import pandas as pd

    test = pd.read_csv(local_data / "test.csv")
    pred = pd.read_csv(pred_path)
    if list(pred.columns) != ["row_id", "control_success"]:
        raise RuntimeError(f"unexpected submission columns: {pred.columns.tolist()}")
    if len(pred) != len(test):
        raise RuntimeError("smoke-test row count mismatch")
    values = pred["control_success"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise RuntimeError("smoke-test invalid probabilities")
    print(f"[smoke test] OK: rows={len(pred):,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evaluator-compatible submit.zip for the current CatBoost latent probe."
    )
    parser.add_argument(
        "--output",
        default="dist/submit.zip",
        help="Output ZIP path relative to repository root (default: dist/submit.zip)",
    )
    parser.add_argument(
        "--smoke-data-dir",
        default=None,
        help="Optional directory containing official-format test.csv and sample_submission.csv.",
    )
    args = parser.parse_args()

    _check_local_artifacts()
    output_zip = (ROOT / args.output).resolve()

    with tempfile.TemporaryDirectory(prefix="tablatent_submit_") as tmp:
        package_root = Path(tmp)
        provenance = _write_package_tree(package_root)
        if args.smoke_data_dir:
            _smoke_test(package_root, Path(args.smoke_data_dir).expanduser().resolve())
        _zip_package(package_root, output_zip)

    _validate_zip(output_zip)
    size_mib = output_zip.stat().st_size / (1024**2)
    print(f"[submission] built: {output_zip}")
    print(f"[submission] zip size: {size_mib:.2f} MiB")
    print("[submission] top-level: model/, script.py, requirements.txt")
    print(f"[submission] artifacts: {len(provenance['artifacts'])}")
    print("[submission] NOTE: this is the current development probe, not a final 2019-2024 retrain.")


if __name__ == "__main__":
    main()
