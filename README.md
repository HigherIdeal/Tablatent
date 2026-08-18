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

The current server already has Conda environments under `~/.conda/envs/`. A convenient clean environment is a clone of `test_trial`:

```bash
cd ~/Aimers/Bitaboost
conda create -n bitaboost --clone test_trial
conda activate bitaboost
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the selected GPU:

```bash
nvidia-smi -i 2
```

### GPU policy

The project uses **one RTX 4090 only: physical GPU 2**.

`GPU가 빵빵하다` means the selected 4090 has enough VRAM, not that training should span multiple GPUs. Keep the model on one GPU and use its VRAM aggressively. The wrappers therefore default to physical GPU `2`.

Normal command:

```bash
python scripts/baseline_train.py --gpus 2
```

Equivalent explicit isolation:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/baseline_train.py --gpus 0
```

Do **not** use `--gpus all` for the normal Bitaboost workflow.

## 2. Data

Reuse the existing prepared dataset:

```bash
cd ~/Aimers/Bitaboost
ln -s /home/kjw/Aimers/Tablatent_backup/data data
```

Expected cache:

```text
data/processed/train.pkl
```

If the cache does not exist:

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

Expected total rows are approximately `1,475,092`, covering seasons `2019..2024`.

## 3. Reproduce the safe Codex maximum

This is the first command to run after a clean setup:

```bash
python scripts/baseline_train.py --gpus 2
```

The command:

1. loads the official 2019-2024 training data;
2. builds the frozen rule-safe feature/profile set;
3. trains the decomposed CatBoost ensemble on GPU 2;
4. evaluates the 2024 validation fold;
5. refits on 2019-2024;
6. creates `dist/baseline_SAFE.zip`;
7. compares the reproduced metric with the reference `0.247355098`.

Expected check:

```text
reproduced Brier ≈ 0.247355098
reproduced score ≈ 981.5
BASELINE_OK
```

The exact reference is allowed a small tolerance because GPU CatBoost can exhibit tiny numerical variation.

Inspect the package without retraining:

```bash
python scripts/eval.py --zip dist/baseline_SAFE.zip
```

If `data/test.csv` and `data/sample_submission.csv` are present, also run the final packaged inference smoke test:

```bash
python scripts/baseline_train.py --gpus 2 --smoke
```

## 4. Build the current-best submission directly

```bash
python scripts/build_current_best_submission.py \
  --gpus 2 \
  --output dist/current_best_SAFE.zip
```

Skip the final test-file smoke check when only retraining/package generation is required:

```bash
python scripts/build_current_best_submission.py \
  --gpus 2 \
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

The final ZIP is converted to the portable `csv.gz` history format and uses the optimized inference package.

## 5. New experiments

`baseline_train.py` is the immutable reference checkpoint. Do not modify it when trying new ideas.

Use:

```bash
python scripts/train.py --gpus 2 --output dist/train_latest.zip
```

At this clean-restart checkpoint, `train.py` aliases the frozen baseline. New work should be implemented in `src/` and exposed through `train.py` while keeping `baseline_train.py` unchanged.

Evaluate an experiment that stores `y` and `pred` in an NPZ:

```bash
python scripts/eval.py --npz outputs/my_experiment/validation_predictions.npz
```

## 6. Experiment history

All previous numerical attempts, rejected directions, and rejection reasons are recorded in:

```text
docs/EXPERIMENT_HISTORY_NUMERICAL.md
```

Old experiment runners should not be restored into `scripts/` just to preserve history.
