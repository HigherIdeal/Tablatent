# Warp Marker — R-only Experience Clustering

Date: 2026-08-17

## Why this marker exists

This is the restore point for the current experimental direction. If later experiments become noisy or overcomplicated, return to this point and resume from the design below.

## Evidence established before this point

- `game_type` is not a fixed player attribute. The same pitcher appears in both R and F, including within the same month, and month-level movement is bidirectional.
- R and F behave like separate operational baseball domains. R has 10 team IDs each season, while F has additional team IDs; same pitchers usually retain the same team ID across R/F.
- R is comparatively stable across 2019–2024. Its target rate drifts gradually rather than showing a sharp structural break.
- F is structurally unstable. The strongest break is 2022 -> 2023:
  - F overall control_success: about 0.709 -> 0.473.
  - Same-pitcher F shift: about -0.22.
  - All shared F teams fall together.
  - Historical-feature response curves also shift by roughly -0.23.
- Therefore the 2023 F change is not explained by player composition alone and is consistent with a domain/label/measurement regime change.
- ABS is a plausible external mechanism, but causality is not assumed or required for the model design.

## Experimental decision at this marker

Do **not** start by solving R and F together.

First isolate the cleaner domain and test the original hypothesis directly:

> Does hard experience-based specialization improve prediction when the underlying domain is relatively stable?

### Primary experiment

- Domain: `game_type == R` only
- Train: 2019–2023
- Validation / pseudo-test: 2024
- Baseline: one R-only CatBoost model trained on all R rows
- Treatment: completely independent CatBoost models for hard row-level pitcher-experience bands
- Initial pitcher experience bands:
  - P0: `asof_pitcher_n == 0`
  - P1: `1–10`
  - P2: `11–50`
  - P3: `51–200`
  - P4: `201–1000`
  - P5: `1001–4000`
  - P6: `4001+`
- Each 2024 R row is routed deterministically to the model matching its own `asof_pitcher_n`.
- No shared model, no soft gating, and no cross-cluster training in this first test.

### Required evaluation

For every experience band report:

- train rows
- validation rows
- target rate
- baseline R-only model Brier
- specialist model Brier
- `specialist - baseline` delta

Also report the row-weighted overall 2024 R Brier for the complete specialist system.

## What comes after this marker

Only after the R-only pitcher-experience experiment is understood:

1. refine/merge/split pitcher experience boundaries if evidence supports it;
2. test pitcher × batter experience specialization;
3. then return to F and handle its 2023+ regime separately.

The point of this marker is to preserve a clean causal experimental question. Do not mix F-regime handling, complex architectures, or shared gating into the first R-only test.
