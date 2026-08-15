# Experiment status — 2026-08-15

## Current representation

Stage 1 is a two-branch VAE with the original dimensionality preserved.

- Context VAE: 16D posterior mean `mu_context`
- History VAE: 16D posterior mean `mu_history`
- Stage 2 base input: concatenated 32D posterior mean
- Stage 2 never uses random posterior samples for inference/probing.
- Current VAE KL beta: `5e-4` for both branches, with 10-epoch KL warm-up.
- Train seasons: 2019-2022
- Validation season: 2023
- 2024 remains untouched during Stage 2 design.

The current goal is to determine whether the learned latent itself contains stable predictive signal before adding raw features, local-probability correction, or calibration.

## Stage 2 results

### Linear baseline

Architecture: `32 -> Linear(1) -> sigmoid`

- validation BCE: `0.69387267`
- validation Brier: `0.25032231`
- validation accuracy: `0.520273`
- validation AUC: `0.528750`
- prediction mean/std: `0.506097 / 0.051291`

### Nonlinear MLP probe

Architecture: `32 -> 64 -> GELU -> 32 -> GELU -> 1 -> sigmoid`

- parameters: `4,225`
- best epoch: `2`
- validation BCE: `0.69618374`
- validation Brier: `0.25137261`
- validation accuracy: `0.523556`
- validation AUC: `0.529444`
- prediction mean/std: `0.513825 / 0.058535`
- official-style 2023 score: `0`

### Bilinear context-history probe

Architecture:

- `c = mu_context` (16D)
- `h = mu_history` (16D)
- `g = Bilinear(c, h)` (16D)
- `[c, h, g]` (48D) -> `Linear(48,1)` -> sigmoid

Result:

- parameters: `4,161`
- best epoch: `1`
- validation BCE: `0.69469438`
- validation Brier: `0.25070280`
- validation accuracy: `0.520525`
- validation AUC: `0.527509`
- prediction mean/std: `0.512296 / 0.053069`
- official-style 2023 score: `0`

The bilinear probe did not beat the linear baseline. Increasing neural Stage 2 capacity is therefore not the current priority.

## Current experiment: CatBoost on frozen latent

The next probe keeps Stage 1 completely frozen and changes only the probability estimator.

Input:

`[mu_context(16D); mu_history(16D)] -> 32 numeric features`

Model:

`CatBoostClassifier`

Constraints:

- uses only the frozen 32D latent
- no raw context/history columns
- no pitcher/batter/team IDs
- no local probability features
- no calibration layer
- no 2024 holdout during model selection
- no latent standardization

Training:

- objective: `Logloss`
- validation/model-selection metric: `BrierScore`
- iterations: `2000`
- learning rate: `0.03`
- depth: `6`
- L2 leaf regularization: `10`
- random strength: `1`
- early stopping: `100`
- default backend: GPU device 0

Run:

```bash
python scripts/train_stage2.py --config configs/default.yaml --head catboost
```

Existing probes remain available:

```bash
python scripts/train_stage2.py --config configs/default.yaml --head linear
python scripts/train_stage2.py --config configs/default.yaml --head mlp
python scripts/train_stage2.py --config configs/default.yaml --head bilinear
```

Primary comparison target is the linear latent baseline Brier `0.25032231`. If CatBoost clearly improves it, the latent contains nonlinear tree-extractable signal that the previous heads did not use effectively. If it remains around `0.25`, the next investigation should shift toward Stage 1 representation quality or raw-feature controls rather than adding a larger Stage 2 head.

## Stage 1 cache

Stage 1 artifacts and `data/processed/train.pkl` can be persisted to Google Drive and restored before Stage 2.

```bash
python scripts/stage1_cache.py push
python scripts/stage1_cache.py pull
```

The cache uses a SHA256 manifest. `--drive-dir` can point to a Colab-mounted Drive path or a locally synchronized Google Drive folder.
