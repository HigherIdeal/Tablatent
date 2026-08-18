# Bitaboost

Clean restart of the LG Aimers 9 baseball `control_success` project.

The active repository intentionally exposes only five executable scripts. Historical experiment runners are not part of the public workflow; the frozen implementation required to reproduce the last rule-safe Codex baseline lives under `src/baseline_legacy/` and is invoked only through the five entry points below.

## Reference baseline

Authoritative **rule-safe** reference:

| Metric | Reference |
|---|---:|
| 2024 validation Brier | **0.247355098** |
| internal score | **981.5** |
| external leaderboard | **1098.86143** |

The historical prefix-state branch reached `1222.8806` internally, but it is **competition-invalid** because hidden-test rows influenced peer-row features. It is documented in `docs/EXPERIMENT_HISTORY_NUMERICAL.md` and intentionally has no runnable path in this clean repository.

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
│   └── baseline_legacy/   # frozen internal implementation; do not run directly
└── requirements.txt
```

## 1. Fresh environment

Use the actual Conda base path instead of assuming where Conda was installed:

```bash
cd ~/Aimers/Bitaboost

CONDA_BASE="$(conda info --base)"
conda create -p "$CONDA_BASE/envs/bitaboost" python=3.11 -y
conda activate "$CONDA_BASE/envs/bitaboost"

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the GPUs:

```bash
nvidia-smi -L
```

## 2. Data

### Option A — reuse the existing local dataset

If the old prepared dataset is still available:

```bash
cd ~/Aimers/Bitaboost
ln -s /home/kjw/Aimers/Tablatent_backup/data data
```

Expected training cache:

```text
data/processed/train.pkl
```

### Option B — rebuild the training cache

```bash
python scripts/prepare_data.py
```

Sanity check:

```bash
python - <<'PY'
import pandas as pd
x = pd.read_pickle('data/processed/train.pkl')
print(x.shape)
print(x.groupby('season')['control_success'].agg(['size','mean']))
PY
```

Expected total rows: approximately `1,475,092`, seasons `2019..2024`.

## 3. Reproduce the safe Codex baseline

This is the first command to run in a clean environment:

```bash
python scripts/baseline_train.py --gpus all
```

Default behavior:

- detects every NVIDIA GPU with `nvidia-smi`;
- exposes all detected GPUs to CatBoost;
- passes all logical devices to every CatBoost fit (`0:1:2:...`);
- allows CPU libraries to use all available CPU threads;
- validates on 2024;
- refits the selected model on all 2019-2024 rows;
- builds the portable optimized package at `dist/baseline_SAFE.zip`;
- verifies that the reproduced 2024 Brier is close to `0.247355098`.

Expected final line:

```text
BASELINE_OK
```

Inspect the result without retraining:

```bash
python scripts/eval.py --zip dist/baseline_SAFE.zip
```

### Optional final inference smoke test

If `data/test.csv` and `data/sample_submission.csv` are present:

```bash
python scripts/baseline_train.py --gpus all --smoke
```

The smoke test runs the final packaged `script.py`, checks finite `[0,1]` probabilities, and therefore validates the actual submission artifact rather than only the training code.

## 4. Build the current-best submission directly

`baseline_train.py` is the reference checker. The underlying build command is:

```bash
python scripts/build_current_best_submission.py \
  --gpus all \
  --output dist/current_best_SAFE.zip
```

For training/package generation without test-file smoke testing:

```bash
python scripts/build_current_best_submission.py \
  --gpus all \
  --output dist/current_best_SAFE.zip \
  --skip-smoke
```

The builder reproduces the frozen safe architecture:

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

The final ZIP is automatically converted away from pandas pickle history and then repacked using the previously validated portable/optimized submission path.

## 5. New experiments

Use one training entry point:

```bash
python scripts/train.py --gpus all --output dist/train_latest.zip
```

At the clean-restart checkpoint `train.py` intentionally aliases the frozen baseline. New modeling work should extend `src/` and then change `train.py`; **do not modify `baseline_train.py`**. This guarantees that the reference can always be rerun after experimental changes.

For a candidate that saves `y` and `pred` in NPZ:

```bash
python scripts/eval.py --npz outputs/my_experiment/validation_predictions.npz
```

The evaluator reports Brier, raw competition-style score, target mean, delta versus the safe baseline, and R/F Brier when `game_type` is present.

## 6. GPU policy

Training defaults to:

```bash
--gpus all
```

Do not manually restrict to a single GPU for normal experimentation. The wrapper detects all physical GPUs and maps them to CatBoost logical devices. Example on a 4-GPU server:

```text
physical GPUs=['0','1','2','3'] -> CatBoost devices=0:1:2:3
```

If a machine has a broken/unavailable GPU, explicitly select the healthy devices:

```bash
python scripts/baseline_train.py --gpus 0,1,3
```

This is a recovery option, not the normal path.

## 7. Experiment history

Numerical history, rejected ideas, and rejection reasons are maintained in:

```text
docs/EXPERIMENT_HISTORY_NUMERICAL.md
```

That document is the historical record. Old experiment runners should not be restored into `scripts/` just to preserve history.
