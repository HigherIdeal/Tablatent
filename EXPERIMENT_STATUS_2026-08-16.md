# Experiment status — 2026-08-16

## Current deployable backbone

Keep the fixed CatBoost ensemble as the reference system:

- full-history raw-`game_type` expert
- recent raw-`game_type` expert trained on the recent regime (2023+2024 for final 2025 training)
- R-fast specialist applied only to `game_type=R`
- fixed blend: `alpha_recent=0.20`, `beta_r=0.10`

Hidden-test rows must remain independent. Inference may use only the current row and fixed train-time model parameters. No hidden-test aggregation, sequential state update, target-prior estimation, or test-time adaptation is allowed.

## Regime-adaptation experiment suite

### 1. Row-wise dynamic gate — not promoted

Strict forward OOF experts were used to train a per-row gate. The richer `row_context` gate selected strength 0. The `outputs_only` gate produced only a noise-scale weighted improvement and improved one of three proxy folds.

Proxy summary:

- fixed weighted Brier: `0.24792295`
- outputs-only weighted Brier: `0.24792259`
- delta: `-0.00000036`
- improved folds: `1/3`
- row-context: identical to fixed (`strength=0`)

Conclusion: keep the fixed `alpha_recent=0.20`. The dynamic-gating idea is not promoted in its current form.

### 2. Season-aware latent HMM robustness — diagnostic only

The first monthly HMM had linked off-season boundaries as ordinary transitions. The robustness run fixed this by using canonical/de-duplicated signals and independent season sequences (`lengths=[8,6,7,7,7,8]`).

Across the best seed for every valid PCA/state setting:

- 2023 overlap with the 2024 dominant state: `1.000`
- 2024 dominant-state purity: `0.875` or `1.000`
- global partition ARI: mean `0.457`, median `0.572`, minimum `-0.034`

Conclusion: the earlier apparent 2024-only state was not robust. The stronger recurring pattern is that 2023 and 2024 belong to the same broad recent environment. Exact HMM state partitions are not stable enough to use as prediction features.

### 3. Smooth recency weighting — not promoted

Only the full-history expert was exponentially sample-weighted; recent/R-fast experts and ensemble weights were fixed.

Weighted proxy Brier:

| half-life (months) | weighted Brier | delta vs unweighted | improved folds |
|---:|---:|---:|---:|
| 0 (unweighted) | **0.24791945** | **0** | - |
| 60 | 0.24794424 | +0.00002479 | 1/3 |
| 36 | 0.24794656 | +0.00002711 | 1/3 |
| 24 | 0.24797332 | +0.00005387 | 1/3 |
| 12 | 0.24801205 | +0.00009260 | 1/3 |

Conclusion: old seasons remain useful. Smoothly suppressing them hurts overall generalization, despite a mid-2024 local gain.

## Closed / low-priority directions

- GRU temporal state: no robust gain; sequential hidden-test use would violate row independence.
- Tiny Transformer temporal state: no robust gain.
- HMM/BOCPD test-time filtering: forbidden by hidden-row independence.
- test-time adaptation / transductive hidden-test statistics: forbidden.
- recency-weighted full expert: rejected by proxy suite.
- current row-wise dynamic gate: rejected by promotion rule.

## Next modeling focus

Stop spending the main search budget on generic regime adaptation. Preserve the fixed full+recent+R-fast backbone and search for genuinely new predictive structure: conditional target relationships, interaction/specialist structure, calibration, and information not already encoded by the existing as-of statistics.

## Server profile

The current compute server has four RTX 4090 GPUs. Project runs must use **physical GPU 2 only**. On the server, source `scripts/activate_gpu2.sh`; this sets `CUDA_VISIBLE_DEVICES=2`, so physical GPU 2 becomes process-local `cuda:0` / CatBoost `devices=0` and the other three GPUs are inaccessible to the process.
