# Physics-Arsenal MoE V2

V2 predicts control success directly and lets historical physics make only a
bounded residual correction:

```text
logit(P(control success))
  = prior + direct R/F head + 0.25 * tanh(arsenal residual)
```

Pitch selection and physical prediction remain low-weight auxiliary tasks;
they no longer form a mandatory path to the final probability.

The four arsenal tokens are `fastball`, `breaking`, `offspeed`, and `other`.
The deployable token for season `S` uses only Trackman seasons `< S`.

Exact current-pitch Trackman matches are never model inputs. They supervise
only the pitch-selection and physical-prediction auxiliary heads during
training. In 2025 inference, every prediction uses one test row and frozen
pre-2025 arsenal profiles; no other test row is read to construct a feature.

## 1. Build the processed dataset

This scans the large train and Trackman CSV files:

```bash
python src/build_pitch_arsenal_moe_data.py --force
```

Main outputs under `data/processed/pitch_arsenal_moe/`:

- `pitcher_arsenal_by_season.parquet`: four long-form arsenal tokens per mapped pitcher and feature season
- `team_hand_arsenal_by_season.parquet`: first fallback
- `league_hand_arsenal_by_season.parquet`: second fallback
- `league_arsenal_by_season.parquet`: final fallback
- `pitcher_arsenal_2025.parquet`: frozen player profiles for deployment
- `aligned_pitch_auxiliary.parquet`: current-pitch training targets only
- `build_summary.json`: mapping, alignment, and row-count diagnostics

## 2. Temporal cross-validation

Use physical GPU 2 while exposing it as the process-local CUDA device:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_pitch_arsenal_moe.py \
  --mode cv \
  --device cuda \
  --epochs 24 \
  --batch-size 4096 \
  --eval-batch-size 8192
```

The batch tqdm reports `loss`, `train_brier`, and `train_score`. Each epoch
reports those values plus `valid_brier` and `valid_score`. Loss values use
scientific notation such as `1.23e-02`.
BFloat16 autocast is enabled by default on CUDA; `--no-amp` uses Float32.
Brier, R/F metrics, auxiliary losses, pitch-selection accuracy, and selected
epochs remain available in the final CSV/JSON report artifacts.

CV outputs under `outputs/pitch_arsenal_moe_v2/`:

- `fold_metrics.csv`
- `cv_summary.json`
- `validation_predictions.parquet`
- `checkpoints/*.pt`

The primary comparison value is `weighted_brier` in `cv_summary.json`.

## 3. Final 2019-2024 fit

After accepting CV results, use the fold-weighted best epoch:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_pitch_arsenal_moe.py \
  --mode final \
  --device cuda \
  --use-cv-epoch
```

The deployable checkpoint is `outputs/pitch_arsenal_moe_v2/final/model.pt`.

## 4. 2025 row-independent inference

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_pitch_arsenal_moe.py \
  --test data/raw/test.csv \
  --device cuda \
  --force
```

The output is `outputs/pitch_arsenal_moe_v2/submission.csv`. Multiple independently
trained checkpoints can be supplied through `--checkpoints` and fixed weights
through `--blend-weights`; weights must be chosen before hidden-test inference.

PyTorch is intentionally not pinned in `requirements.txt` because the correct
wheel depends on the server CUDA version. Use the CUDA-compatible PyTorch build
already installed in the training environment.
