# Next Regime-Adaptation Experiments

## Fixed constraints

- Keep the current deployable backbone frozen unless an experiment beats it robustly:
  - full-history raw-`game_type` CatBoost
  - recent raw-`game_type` CatBoost
  - R-fast specialist
  - fixed `alpha_recent=0.20`, `beta_r=0.10`
- Hidden-test rows are independent. Prediction for row `i` must not use any other hidden-test row.
- Therefore no test-batch normalization, test-set clustering, TENT/SAR-style batch adaptation, sequential hidden-test HMM filtering, or hidden-test aggregate statistics are allowed.
- HMM is a train-only discovery/diagnostic tool, not a hidden-test state tracker.
- 2024 has already been used repeatedly in proxy design, so promotion decisions rely on consistency across established temporal folds, not a single 2024 optimum.

## Experiment A — Season-aware latent-regime robustness

Script: `scripts/run_latent_regime_robustness.py`

Purpose: verify whether the apparent 2024 distribution state survives the weaknesses of the first HMM probe.

Corrections relative to the first HMM experiment:

1. Use canonical/de-duplicated signals instead of all redundant official encodings.
2. Exclude direct time labels, team IDs, target, and `game_type` in the primary blind run.
3. Split 2019–2024 into independent season sequences with `hmmlearn` `lengths`; off-season gaps are not ordinary HMM transitions.
4. Test `K={2,3}` and PCA dimensions `{3,5,8}` across multiple seeds.
5. Do not compare BIC numerically across different PCA dimensions. Compare BIC only within a fixed transformed dimension.
6. Measure label-invariant partition stability with Adjusted Rand Index (ARI).
7. Treat state/feature ranking as distribution diagnostics only. `control_success` is merged after fitting only for post-hoc interpretation.

Primary decision criterion: a distinct 2024 state is credible only if the partition is stable across state count/PCA/seeds and 2024 has high state purity with low overlap with 2023.

Run:

```powershell
python scripts/run_latent_regime_robustness.py --config configs/default.yaml
```

Optional sensitivity with `game_type` restored:

```powershell
python scripts/run_latent_regime_robustness.py --config configs/default.yaml --include-game-type --output-dir outputs/latent_regime_robustness_with_game_type
```

## Experiment B — Row-wise dynamic full/recent gate

Script: `scripts/run_rowwise_dynamic_gate.py`

This is the main model experiment.

### Gate target

Generate strict historical OOF predictions:

- validation 2021: full=`2019–2020`, recent=`2020`
- validation 2022: full=`2019–2021`, recent=`2021`
- validation 2023: full=`2019–2022`, recent=`2022`

For each OOF row define:

`advantage_recent = (y - p_full)^2 - (y - p_recent)^2`

Positive values mean the recent expert had lower Brier loss on that row.

A CatBoost regressor learns the conditional expected advantage from current-row information. 2023 OOF is used only to calibrate gate strength; the final gate is then refit on all historical OOF rows before the 2024 proxy evaluation.

### Gate variants

1. `outputs_only`
   - `p_full(x_i)`
   - `p_recent(x_i)`
   - signed/absolute disagreement and mean
2. `row_context`
   - the above plus current-row canonical context/history features
   - direct `season`, team IDs, target, and row ID are excluded from the gate

The learned score changes the recent weight around the existing fixed prior `alpha=0.20`:

`alpha_i = sigmoid(logit(0.20) + strength * normalized_gate_score_i)`

with conservative clipping (`0.02 <= alpha_i <= 0.60` by default).

The R-fast specialist is applied after the dynamic full/recent blend with the existing fixed `beta_r=0.10`.

### Leakage rule

At deployment the gate may use only:

- current row `x_i`
- `p_full(x_i)`
- `p_recent(x_i)`

It must not use any statistic, prediction, state, or normalization derived from other hidden-test rows.

Promotion rule: dynamic gating must improve weighted Brier versus fixed `alpha=0.20`, should improve more than one established proxy fold, and must not depend on an extreme alpha distribution.

Run:

```powershell
python scripts/run_rowwise_dynamic_gate.py --config configs/default.yaml --iterations 500 --gate-iterations 300 --alpha-base 0.20 --beta-r 0.10 --task-type GPU --devices 0
```

## Experiment C — Smooth recency weighting of the full expert

Script: `scripts/run_recency_weighted_full_expert.py`

Purpose: test whether hard old/recent separation can be improved by reducing the influence of old rows smoothly while leaving the recent and R-fast branches unchanged.

For the full expert only:

`w_i = 0.5 ** (age_months_i / half_life_months)`

Weights are normalized to mean one inside each training fold. `half_life=0` is the exact unweighted baseline.

Default screen: `0, 12, 24, 36, 60` months.

Run:

```powershell
python scripts/run_recency_weighted_full_expert.py --config configs/default.yaml --iterations 500 --half-life-months 0,12,24,36,60 --alpha-recent 0.20 --beta-r 0.10 --task-type GPU --devices 0
```

Promotion rule: require weighted Brier improvement without a material regression in an established proxy fold. If weighting merely reproduces the recent expert, keep the simpler current ensemble.

## Execution order

1. Run **B (row-wise dynamic gate)** first: highest direct modeling value and fully compatible with hidden-row independence.
2. Run **A (HMM robustness)** in parallel/afterward: analysis only; use it to interpret gate feature importance, not as test-time state.
3. Run **C (recency weighting)** if B is weak or as an orthogonal low-risk improvement.

Do not reopen GRU/Transformer temporal-state branches unless new evidence shows a large missing trajectory signal. Their previous gains were unstable/negative and sequence-based hidden-test adaptation is incompatible with the row-independence constraint.
