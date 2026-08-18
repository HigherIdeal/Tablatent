# Bitaboost

Clean restart of the LG Aimers 9 baseball `control_success` project.

The active workflow exposes only five scripts. Historical experiment code needed for exact baseline reproduction is frozen under `src/baseline_legacy/` and should not be executed directly.

## Reference baseline

Authoritative **rule-safe** reference:

| Metric | Reference |
|---|---:|
| 2024 validation Brier | **0.247355098** |
| internal score | **981.5** |
| external leaderboard | **1098.86143** |

The historical prefix-state branch reached `1222.8806` internally, but it is **competition-invalid** because hidden-test rows influenced peer-row features. It remains documented only in `docs/EXPERIMENT_HISTORY_NUMERICAL.md`.

## Repository layout

```text
Bitaboost/
├── configs/
│   └── baseline.yaml
├── docs/
│   └── EXPERIMENT_HISTORY_NUMERICAL.md
├── scripts/
│   ├── prepare_data.py
│   ├── baseline_train.py
│   ├── train.py
│   ├── eval.py
│   └── build_current_best_submission.py
├── src/
│   ├── canonical_features.py
│   ├── data.py
│   ├── evaluation_metrics.py
│   ├── utils.py
│   └── baseline_legacy/   # frozen internal reproduction core
└── requirements.txt
```

## 1. Environment

Reference data/runtime versions are pinned:

```text
Python      3.11
NumPy       1.26.4
pandas      2.2.3
PyArrow     17.0.0
CatBoost    1.2.10
```

Existing environment:

```bash
cd ~/Aimers/Bitaboost
conda activate bitaboost
python -m pip install --upgrade -r requirements.txt
```

Verify:

```bash
python - <<'PY'
import numpy, pandas, pyarrow, catboost
print('numpy   ', numpy.__version__)
print('pandas  ', pandas.__version__)
print('pyarrow ', pyarrow.__version__)
print('catboost', catboost.__version__)
PY
```

Expected NumPy version is exactly `1.26.4`.

### GPU policy

The project uses **one RTX 4090 only: physical GPU 2**. The point is to use the large VRAM of one 4090, not multi-GPU training.

Recommended shell setting:

```bash
echo 'export CUDA_VISIBLE_DEVICES=2' >> ~/.bashrc
source ~/.bashrc
```

After that physical GPU 2 appears to CUDA programs as logical GPU `0`.

The Bitaboost wrappers also default to physical GPU 2, so the normal baseline command needs no GPU option.

## 2. Data cache: Parquet only

Bitaboost does **not** read `train.pkl`. Pickle caches are intentionally ignored because NumPy/Pandas module paths can change across versions (`numpy._core.numeric` vs `numpy.core.numeric`).

Processed training data is:

```text
data/processed/train.parquet
```

Format:

```text
Parquet + PyArrow + Zstandard
```

If reusing the old data directory:

```bash
cd ~/Aimers/Bitaboost
ln -s /home/kjw/Aimers/Tablatent_backup/data data
```

If `data` already exists, do not recreate the symlink.

### Build the Parquet cache

```bash
python scripts/prepare_data.py
```

The script first reuses an already extracted `train.csv` when possible. If unavailable, it downloads/extracts the official dataset. An old `train.pkl` may remain on disk but is ignored.

Force a full rebuild only when necessary:

```bash
python scripts/prepare_data.py --force
```

Sanity check:

```bash
python - <<'PY'
import pandas as pd
x = pd.read_parquet('data/processed/train.parquet')
print(x.shape)
print(x.groupby('season')['control_success'].agg(['size','mean']))
PY
```

Expected total rows are approximately `1,475,092`, covering seasons `2019..2024`.

## 3. Reproduce the safe Codex maximum

After `train.parquet` exists:

```bash
python scripts/baseline_train.py
```

The command:

1. loads the official 2019-2024 Parquet cache;
2. builds the frozen rule-safe feature/profile set;
3. trains the decomposed CatBoost ensemble on the single RTX 4090;
4. evaluates the 2024 validation fold;
5. refits on 2019-2024;
6. creates `dist/baseline_SAFE.zip`;
7. compares the reproduced metric with `0.247355098`.

Expected check:

```text
reproduced Brier ≈ 0.247355098
reproduced score ≈ 981.5
BASELINE_OK
```

Tiny GPU CatBoost numerical variation is allowed by the checker tolerance.

Inspect the package without retraining:

```bash
python scripts/eval.py --zip dist/baseline_SAFE.zip
```

If `data/test.csv` and `data/sample_submission.csv` are present, run the final packaged inference smoke test:

```bash
python scripts/baseline_train.py --smoke
```

## 4. Build the current-best submission directly

```bash
python scripts/build_current_best_submission.py \
  --output dist/current_best_SAFE.zip
```

Skip the final test-file smoke check when only retraining/package generation is required:

```bash
python scripts/build_current_best_submission.py \
  --output dist/current_best_SAFE.zip \
  --skip-smoke
```

The frozen safe architecture is:

```text
canonical/as-of/context features
        + frozen historical profiles
        ↓
MultiRMSE current-success head
reverse + middle auxiliary heads
success hurdle
residual offset
joint auxiliary outcome
        ↓
R/F-specific convex ensemble
```

The final ZIP uses portable `csv.gz` history artifacts; no pickle history remains in the final submission package.

## 5. New experiments

`baseline_train.py` is the immutable reference checkpoint. Do not modify it when trying new ideas.

Use:

```bash
python scripts/train.py --output dist/train_latest.zip
```

At this clean-restart checkpoint, `train.py` aliases the frozen baseline. New work should be implemented in `src/` and exposed through `train.py` while keeping `baseline_train.py` unchanged.

Evaluate an experiment that stores `y` and `pred` in an NPZ:

```bash
python scripts/eval.py --npz outputs/my_experiment/validation_predictions.npz
```

## 6. Fresh-pull recovery path

```bash
cd ~/Aimers/Bitaboost
git pull
conda activate bitaboost
python -m pip install --upgrade -r requirements.txt
python scripts/prepare_data.py
python scripts/baseline_train.py
```

If baseline reproduction is outside tolerance, do not start a new experiment until the environment/data difference is explained.

## 7. Experiment history

All previous numerical attempts, rejected directions, and rejection reasons are recorded in:

```text
docs/EXPERIMENT_HISTORY_NUMERICAL.md
```

Old experiment runners should not be restored into `scripts/` just to preserve history.
