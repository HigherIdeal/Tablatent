# Bitaboost

Clean restart of the LG Aimers 9 baseball `control_success` project.

The active workflow exposes only five scripts. Historical experiment code required for exact baseline reproduction is frozen under `src/baseline_legacy/` and should not be executed directly.

## Reference baseline

Authoritative **rule-safe** reference:

| Metric | Reference |
|---|---:|
| 2024 validation Brier | **0.247355098** |
| internal score | **981.5** |
| external leaderboard | **1098.86143** |

The historical prefix-state branch reached `1222.8806` internally, but it is competition-invalid because hidden-test rows influenced peer-row features. It remains documentation-only in `docs/EXPERIMENT_HISTORY_NUMERICAL.md`.

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
│   └── baseline_legacy/
└── requirements.txt
```

## 1. Runtime environment

`bitaboost` is cloned from `test_trial` and is expected to match the competition/runtime environment:

```text
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1
```

Only the library missing from the competition environment is listed in this repository:

```text
catboost==1.2.10
```

Install only that extra dependency:

```bash
cd ~/Aimers/Bitaboost
conda activate bitaboost
pip install -r requirements.txt
```

Do **not** run a broad `pip install --upgrade ...` over the competition packages. The baseline preflight deliberately checks the important versions before training.

## 2. GPU policy

Use **one RTX 4090: physical GPU 2**. The goal is to exploit the VRAM of one 4090, not multi-GPU training.

Recommended shell setting:

```bash
export CUDA_VISIBLE_DEVICES=2
```

The wrappers also default to physical GPU 2, so the normal training command needs no GPU argument:

```bash
python scripts/baseline_train.py
```

Expected mapping:

```text
[parallel] physical GPUs=['2'] -> CatBoost devices=0
```

## 3. Data cache

The processed cache is intentionally **not pickle and not Parquet**.

```text
data/processed/train.csv.gz
```

Why `csv.gz`:

- no NumPy pickle module-path compatibility problem;
- no `pyarrow`/`fastparquet` dependency;
- works with the existing pandas 2.0.3 runtime;
- keeps `requirements.txt` limited to CatBoost.

If the existing official data directory is reused:

```bash
cd ~/Aimers/Bitaboost
ln -s /home/kjw/Aimers/Tablatent_backup/data data
```

`prepare_data.py` searches `data/` for the official training CSV containing `control_success`, preferring `data/train.csv` and `data/raw/extracted/train.csv`.

Build or refresh the portable cache:

```bash
python scripts/prepare_data.py --force
```

Check it:

```bash
python - <<'PY'
import pandas as pd
x = pd.read_csv('data/processed/train.csv.gz', compression='gzip', low_memory=False)
print('shape:', x.shape)
print(x.groupby('season')['control_success'].agg(['size', 'mean']))
PY
```

Expected total rows: about `1,475,092`, seasons `2019..2024`.

## 4. Baseline validation — first command before experiments

### Step A: confirm environment + install CatBoost

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
pip install -r requirements.txt
```

Optional manual version check:

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, torch, catboost
print('numpy   ', numpy.__version__)
print('pandas  ', pandas.__version__)
print('scipy   ', scipy.__version__)
print('sklearn ', sklearn.__version__)
print('torch   ', torch.__version__)
print('catboost', catboost.__version__)
PY
```

### Step B: build the cache

```bash
python scripts/prepare_data.py --force
```

### Step C: reproduce the Codex SAFE maximum

```bash
python scripts/baseline_train.py
```

At startup the script prints the runtime fingerprint. Critical mismatches in NumPy, pandas, SciPy, scikit-learn, Torch, or CatBoost stop the run before expensive training.

Expected end-of-run check:

```text
=== BASELINE CHECK ===
reproduced Brier : ~0.247355098
reference Brier  : 0.247355098
reproduced score : ~981.5
external LB ref  : 1098.86143
BASELINE_OK
```

GPU CatBoost can show tiny numerical variation, so the checker uses a small tolerance.

If you intentionally want to inspect a slightly different runtime without hard failure:

```bash
python scripts/baseline_train.py --no-strict-env
```

Do not treat such a run as the canonical reproduction until the difference is explained.

## 5. Inspect a reproduced package

Without retraining:

```bash
python scripts/eval.py --zip dist/baseline_SAFE.zip
```

If `data/test.csv` and `data/sample_submission.csv` are present, validate the actual packaged inference path too:

```bash
python scripts/baseline_train.py --smoke
```

## 6. Build current-best submission directly

```bash
python scripts/build_current_best_submission.py \
  --output dist/current_best_SAFE.zip
```

Skip final test-file smoke checking:

```bash
python scripts/build_current_best_submission.py \
  --output dist/current_best_SAFE.zip \
  --skip-smoke
```

The safe architecture is the frozen decomposed CatBoost ensemble:

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

## 7. New experiments

`baseline_train.py` is the immutable reference checkpoint. New ideas go through:

```bash
python scripts/train.py --output dist/train_latest.zip
```

At the clean-restart checkpoint `train.py` aliases the frozen baseline. Future modeling code should be added under `src/` and exposed through `train.py`; do not modify the baseline checkpoint merely to test an idea.

For NPZ validation predictions containing `y` and `pred`:

```bash
python scripts/eval.py --npz outputs/my_experiment/validation_predictions.npz
```

## 8. Experiment history

All previous numerical attempts, rejected directions, and rejection reasons are recorded in:

```text
docs/EXPERIMENT_HISTORY_NUMERICAL.md
```

Old experiment runners should not be restored into `scripts/` just to preserve history.
