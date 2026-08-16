# Stable Player Dynamics Experiment

## 1. Hypothesis and Scope

The current primary system is the gated dual-track CatBoost ensemble with an R-fast specialist. Do **not** replace it yet. This experiment tests only whether a compact temporal representation of pitcher-specific, regime-stable signals adds information that the current row-wise CatBoost features do not already contain.

Observed regime diagnostics motivating the experiment:

- `game_type` has a dominant persistent breakpoint at **2023-04** (`score≈0.1446`), reproduced for 5/8/12 bins.
- Several long-run success/reverse signals also show historical breakpoints and therefore should not be treated as stable player identity/state.
- A smaller set of pitcher-state deltas and pitch-mix/control signals show low regime drift and are better candidates for temporal modeling.

The intended decomposition is:

`prediction = current regime/context model + stable player dynamics`

The GRU branch must model **player trajectory only**. Regime-sensitive context (`game_type`, season-level shift, current count/base/game context) remains with CatBoost.

## 2. Experimental Design

### Stable temporal inputs

Initial fixed candidate set, chosen from the low-drift side of the monthly atlas and restricted to pitcher-specific pre-pitch state:

- `asof_pitcher_middle_rate`
- `asof_pitcher_ball_rate`
- `asof_pitcher_strike_rate`
- `asof_pitcher_fastball_rate`
- `asof_pitcher_breaking_rate`
- `asof_pitcher_offspeed_rate`
- `eng_ps_prev1_minus_long`
- `eng_ps_prev3_minus_long`
- `eng_ps_prev5_minus_long`
- `eng_ps_prev1_minus_prev3`
- `eng_ps_prev3_minus_prev5`
- `eng_ps_prev1_minus_prev5`
- `eng_ps_recent_mean_minus_long`
- `eng_ps_recent_range_135`

`asof_pitcher_n` and monthly pitch count are used only as reliability/exposure inputs. `game_type`, team IDs, batter features, and regime-sensitive long-run success/reverse rates are deliberately excluded from the temporal encoder.

### Temporal unit

Aggregate to `pitcher_id × season × game_month`.

- Rates/deltas: monthly median.
- `asof_pitcher_n`: monthly max then `log1p`.
- Exposure: `log1p(rows in pitcher-month)`.

The embedding attached to month `t` is generated from months **strictly before `t`**. Current-month rows never contribute to their own embedding.

### GRU objective

Use a small GRU (`hidden≈24`, 1 layer by default). It is **not** trained on `control_success`.

Self-supervised objective:

`history of stable monthly state -> next-month stable state`

This is intentional: the temporal encoder must not learn validation labels or become another opaque target model. CatBoost remains the supervised predictor.

### Outer folds

Use strict season-forward folds:

- train `<=2021`, validate `2022`
- train `<=2022`, validate `2023`  ← primary regime-crossing diagnostic
- train `<=2023`, validate `2024`

The GRU scaler and GRU weights are fit separately inside each outer fold using only months from the outer training period.

### Ablations

For each fold train the same CatBoost configuration with:

1. `base`: current row-wise recent-raw feature set only.
2. `lag1`: base + previous observed pitcher-month stable state.
3. `gru`: base + causal GRU hidden state.
4. `gru_lag1`: base + both.

`lag1` is essential. If GRU only beats `base` but not `lag1`, the gain is likely from adding a missing lagged statistic rather than learned temporal dynamics.

## 3. Decision Rules and Guardrails

Primary metric is Brier score / raw Brier skill score. Do not select the method by AUC.

Promote the temporal branch to the gated dual-track + R-fast system only if:

- `gru` or `gru_lag1` improves the **2023** fold versus both `base` and `lag1`;
- the same variant is non-degrading or positive on **2024**;
- the weighted multi-fold result is positive without a large worst-fold regression.

Hard guardrails:

- No `control_success` is used to train the GRU.
- No current/future month contributes to the embedding used for that month.
- GRU normalization statistics are fit on outer-train months only.
- Do not interpret a detected breakpoint as causal.
- Do not add `game_type` to the temporal encoder; the whole point is to isolate stable player dynamics from the 2023 regime break.
- Do not tune GRU depth/hidden size aggressively before the fixed small model demonstrates signal.
- Do not package a submission from this experiment until the ablation passes; phase 1 is representation validation only.

Run:

```bash
python scripts/run_stable_player_dynamics.py \
  --config configs/default.yaml \
  --folds 2022,2023,2024 \
  --iterations 300 \
  --torch-device auto \
  --task-type GPU \
  --devices 0
```

Outputs are written to `outputs/stable_player_dynamics/`.