# Stable Player Dynamics Experiment

## 1. Hypothesis and Scope

The current primary system is the gated dual-track CatBoost ensemble with an R-fast specialist. Do **not** replace it. The temporal experiment asks whether a compact representation of pitcher-specific, low-regime-drift signals contains information not already available to the row-wise CatBoost system.

Observed regime diagnostics motivating the experiment:

- `game_type` has a dominant persistent breakpoint at **2023-04** (`score≈0.1446`), reproduced for 5/8/12 bins.
- Several long-run success/reverse signals also show historical breakpoints and therefore should not be treated as stable player identity/state.
- A smaller set of pitcher-state deltas and pitch-mix/control signals have no comparably strong breakpoint and are better candidates for temporal modeling.

The intended decomposition is:

`prediction = current regime/context model + stable player dynamics`

The GRU branch models **player trajectory only**. Regime-sensitive context (`game_type`, current count/base/game context, and the full-vs-recent regime mixture) remains with CatBoost.

## 2. Phase 1: Representation Probe

### Stable temporal inputs

Initial fixed candidate set, restricted to pitcher-specific pre-pitch state:

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

`asof_pitcher_n` and monthly pitch count are reliability/exposure inputs only. `game_type`, team IDs, batter features, and the strongly regime-sensitive long-run success/reverse rates are excluded from the temporal encoder.

Aggregate to `pitcher_id × season × game_month` using monthly medians for rates/deltas, monthly maximum for `asof_pitcher_n`, and row count for exposure. The embedding attached to month `t` is generated only from months **strictly before `t`**.

The small GRU (`hidden=24` by default) is self-supervised:

`history of stable monthly state -> next-month stable state`

It never uses `control_success`.

Initial season-forward probe:

- train `<=2021`, validate `2022`
- train `<=2022`, validate `2023`
- train `<=2023`, validate `2024`

Ablations: `base`, `lag1`, `gru`, `gru_lag1`.

Observed result from the first run:

- 2022: all temporal variants slightly degraded the simple CatBoost baseline.
- 2023: `lag1`, `gru`, and especially `gru_lag1` improved the baseline; `gru_lag1` changed Brier by about `-0.000200`.
- 2024: temporal variants again slightly degraded the simple CatBoost baseline by roughly `+0.00002~0.00003` Brier.

This does **not** justify rejecting the temporal hypothesis because the absolute 2023 collapse is already known to affect ordinary CatBoost across the 2023 regime boundary. Phase 1 used a simple CatBoost baseline, not the current regime-aware production architecture.

## 3. Phase 2: Regime-Aware Integration

Phase 2 tests the temporal feature in the model family that actually handles the observed regime change:

`full_raw expert + recent_raw expert + R-fast specialist -> fixed gated prediction`

Only the **recent_raw expert** is modified. The full-history expert and R-fast specialist remain identical across variants. This isolates whether temporal player state improves the post-break/recent branch without confounding the already validated gate architecture.

Variants:

1. `base`: current recent_raw expert.
2. `lag1`: recent_raw + previous pitcher-month state.
3. `gru`: recent_raw + causal GRU hidden state.
4. `gru_lag1`: recent_raw + both.

The ensemble weights are fixed during this test (`alpha_recent=0.20`, `beta_r=0.10` by default); they are **not retuned** on the temporal experiment.

Use the established 2025 proxy folds:

- `season_forward_2024`: recent expert trains on 2023, full expert through 2023, validates all 2024.
- `mid_2024`: expanding history through 2024-05, validates 2024-06~07.
- `late_2024`: expanding history through 2024-07, validates 2024-08 onward.

The latter two folds are especially important because both training and validation are inside the post-2023 regime. GRU/scaler fitting stops at each fold cutoff. A validation-month embedding may use earlier validation months only when they are chronologically prior; it never uses its current or future month.

Promotion criterion:

- `gru` or `gru_lag1` should improve weighted proxy Brier versus `base`;
- improvement should not come only from the season-forward fold;
- at least one of `mid_2024` / `late_2024` should improve, preferably both;
- `gru` must be compared with `lag1` so a simple missing-lag effect is not mislabeled as learned temporal dynamics.

Run:

```bash
python scripts/run_regime_aware_stable_dynamics.py \
  --config configs/default.yaml \
  --iterations 500 \
  --alpha-recent 0.20 \
  --beta-r 0.10 \
  --torch-device auto \
  --task-type GPU \
  --devices 0
```

Outputs are written to `outputs/regime_aware_stable_dynamics/`.

Hard guardrails:

- No `control_success` is used to train the GRU.
- No current/future month contributes to the embedding for a row.
- GRU normalization and weights are fit only through each proxy training cutoff.
- `game_type` never enters the GRU; it remains in the CatBoost regime/context path.
- Full-history and R-fast branches are held fixed across temporal variants.
- No alpha/beta tuning is allowed in this phase.
- Do not package a submission until the regime-aware proxy test is positive.
