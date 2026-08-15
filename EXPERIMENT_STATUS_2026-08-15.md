# Experiment status — 2026-08-15

## Current representation

Stage 1 is a two-branch VAE with the original dimensionality preserved.

- Context VAE: 16D posterior mean `mu_context`
- History VAE: 16D posterior mean `mu_history`
- Stage 2 input: concatenated 32D posterior mean
- Stage 2 never uses random posterior samples for inference/probing.
- Current VAE KL beta: `5e-4` for both branches, with 10-epoch KL warm-up.

The current goal is to test whether the learned latent contains a useful conditional control-success signal before adding any calibration or local-probability correction.

## Stage 2 results

### Linear baseline

Architecture:

`32 -> Linear(1) -> sigmoid`

Training:

- raw `control_success` labels (0/1)
- `BCEWithLogitsLoss`
- train seasons: 2019-2022
- validation season: 2023

At VAE beta `5e-4`:

- validation BCE: `0.69387267`
- validation Brier: `0.25032231`
- validation accuracy: `0.520273`
- validation AUC: `0.528750`
- prediction mean/std: `0.506097 / 0.051291`

### Nonlinear MLP probe

Architecture:

`32 -> 64 -> GELU -> 32 -> GELU -> 1 -> sigmoid`

Parameters: `4,225`

Best epoch: 2

- validation BCE: `0.69618374`
- validation Brier: `0.25137261`
- validation accuracy: `0.523556`
- validation AUC: `0.529444`
- prediction mean/std: `0.513825 / 0.058535`
- official-style 2023 score: `0`

Interpretation: the MLP lowers training BCE more than the linear head and slightly improves validation AUC, but worsens validation Brier. This is consistent with nonlinear signal extraction plus temporal overfitting / over-confident probability spread. Therefore simply enlarging Stage 2 is not the next priority.

## Next experiment

Test an explicitly constrained context-history interaction rather than a generic MLP.

Preferred formulation:

- `c = mu_context` (16D)
- `h = mu_history` (16D)
- learned bilinear interaction `g = Bilinear(c, h)` (candidate output 16D)
- concatenate `[c, h, g]` -> 48D
- `Linear(48, 1) -> sigmoid`
- train with raw 0/1 labels and `BCEWithLogitsLoss`

This is preferred over a simple Hadamard product because the same latent coordinates of the independently trained context/history VAEs are not guaranteed to be semantically aligned. The bilinear layer can learn which context and history dimensions should interact.

Do not delete the existing linear baseline. Compare all Stage 2 probes under the same frozen Stage-1 latent and the same temporal split.

## Implementation status

Bilinear Stage 2 is implemented and is now the default `train_stage2.py` head. Existing probes remain selectable.

```bash
python scripts/train_stage2.py --config configs/default.yaml --head bilinear
python scripts/train_stage2.py --config configs/default.yaml --head linear
python scripts/train_stage2.py --config configs/default.yaml --head mlp
```

For Colab session loss, Stage 1 artifacts and `data/processed/train.pkl` can be persisted to Google Drive and restored before Stage 2.

```bash
# immediately after Stage 1
python scripts/stage1_cache.py push

# in a new Colab session
python scripts/stage1_cache.py pull
python scripts/train_stage2.py --config configs/default.yaml --head bilinear
```

The cache uses a SHA256 manifest and defaults to `/content/drive/MyDrive/Tablatent/stage1_cache`.
