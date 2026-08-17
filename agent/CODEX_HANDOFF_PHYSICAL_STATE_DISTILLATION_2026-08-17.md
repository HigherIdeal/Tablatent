# Codex Handoff — Physical State Distillation + Brier-Optimal Ensembling
**Project:** LG Aimers 9 Baseball Hackathon / `HigherIdeal/Tablatent`  
**Date:** 2026-08-17  
**Purpose:** Turn the newly recovered Trackman linkage into a conservative, temporally valid feature pipeline that can improve the current CatBoost backbone without repeating the failure mode of the standalone Pitch Arsenal MoE.

---

# 1. Objective, Current State, and Non-Negotiable Constraints

## 1.1 Competition objective

Predict `control_success` probability for each hidden 2025 test pitch.

Primary metric is Brier-based. Treat the task as calibrated conditional probability estimation, not hard classification.

The final inference function must be row-wise and deployable:

```text
test row
  -> fixed preprocessing
  -> fixed historical profile lookup / fallback
  -> fixed model(s)
  -> probability
```

Do not use hidden-test batch statistics, transductive adaptation, clustering of test rows, test-time normalization dependent on the hidden batch, HMM state updates across hidden rows, or any operation that makes one hidden row depend on another hidden row.

Only information that would have been available before the pitch may be used.

---

## 1.2 Current strongest backbone

The current deployable backbone is a gated CatBoost ensemble.

Base:

```text
p_base = 0.8 * p_full_raw + 0.2 * p_recent_raw
```

R-side specialist:

```text
p_final = 0.9 * p_base + 0.1 * p_Rfast
```

Current best weighted validation Brier:

```text
0.24789305
```

Current weighted subgroup metrics:

```text
R Brier = 0.24798863
F Brier = 0.24716205
```

This model is the reference and must not be replaced unless a candidate is clearly and consistently better.

---

## 1.3 Important known modeling facts

Keep these fixed unless an explicit ablation proves otherwise.

```text
- raw game_type is useful and should remain.
- 2023-2024 recency matters.
- pitcher_id and batter_id were not promoted as raw model features after temporal ablation.
- team IDs remain useful.
- generic neural tabular models / latent models have not beaten the CatBoost backbone.
- dynamic gates and smooth recency weighting produced little or negative gain.
- hidden 2025 rows must remain independent.
```

We are not trying to design another large end-to-end neural classifier.

The new research direction is:

```text
Trackman
  -> stable low-dimensional physical state
  -> explicit temporal state features
  -> CatBoost as the main predictor
  -> mathematically optimized Brier ensemble
```

---

## 1.4 Newly recovered Trackman linkage

A high-confidence deterministic record-linkage procedure has aligned train pitches with Trackman pitches through exact game-state sequences.

Matching state:

```text
inning
top_bottom
balls_before
strikes_before
outs_before
```

Results:

```text
exactly matched games             2,693
aligned pitches                 808,856
state-sequence mismatches             0

pitchers with mapping evidence      731
high-purity accepted pitchers       584
review candidates                    47
insufficient-support pitchers       100

team conflicts                        0
Trackman-ID duplicate conflicts       0
```

The 584 accepted pitchers explain:

```text
800,401 / 808,856 = ~98.95%
```

of all exactly aligned pitches.

Representative accepted mappings have:

```text
forward purity = 1.0
reverse purity = 1.0
multiple seasons
many games
thousands of aligned pitches
zero team conflicts
```

Hand mapping is verified as:

```text
Right -> 2
Left  -> 1
```

For the first physical-feature experiments, use only the accepted high-purity pitcher mapping.

Do not spend time trying to force all long-tail pitchers into the mapping.

---

# 2. Research Thesis: Physical State Distillation

The main hypothesis is **not** that a neural Trackman model should directly replace CatBoost.

The hypothesis is:

> Trackman contains pitcher-specific physical information that is partly absent from the official tabular rows. If we compress this information into a small, temporally valid, stable pitcher-state representation and feed it to the existing CatBoost backbone, the additional state may improve probability estimation.

This should be tested incrementally.

The proposed pipeline:

```text
Trackman 2019-2024
        |
        v
season-safe pitcher physical profiles
        |
        +---- raw physical summary features
        |
        +---- PCA physical state
        |
        +---- explicit temporal level / slope / drift / volatility
        |
        v
pitcher_id + feature_season keyed profile table
        |
        +----------------------------+
        |                            |
 known / mapped pitcher      unseen / sparse pitcher
        |                            |
 observed profile            inferred/fallback profile
        |                            |
        +-------------+--------------+
                      |
                      v
             canonical train features
               + physical state
                      |
                      v
                   CatBoost
                      |
                      v
              OOF probabilities
                      |
                      v
     Brier-optimal convex ensemble
```

The full system should be built only if the earlier cheap ablations justify it.

---

# 3. Work Package A — Brier-Optimal Convex Ensemble

## 3.1 Motivation

We already have several predictors with different inductive biases:

```text
full_raw
recent_raw
stable variants
R-fast / R specialists
Pitch Arsenal MoE
future Trackman physical CatBoost
```

Manual blend grids such as:

```text
0.8 * full + 0.2 * recent
```

are unnecessarily coarse.

Because Brier loss is squared error, fixed-weight probability blending with simplex constraints is a convex quadratic optimization problem.

Let:

```text
P: [N rows, M models] OOF prediction matrix
y: [N] binary target
w: [M] blend weights
```

Optimize:

```text
min_w  mean((P @ w - y)^2) + lambda * ||w - w_anchor||^2
```

subject to:

```text
w_m >= 0
sum(w_m) = 1
```

Start with `w_anchor` equal to the current backbone blend or uniform weights depending on the experiment.

The regularization term is important. Do not allow a tiny validation subset to drive extreme weights.

---

## 3.2 Separate R/F ensembles

The Pitch Arsenal MoE result strongly suggests that model usefulness differs by `game_type`.

Current MoE standalone result:

```text
weighted Brier = 0.24834853
weighted R Brier = 0.24807175
weighted F Brier = 0.25036706
```

Current strong baseline:

```text
weighted Brier = 0.24789305
weighted R Brier = 0.24798863
weighted F Brier = 0.24716205
```

The neural model is nearly competitive on R but catastrophically worse on F.

Therefore implement both:

```text
GLOBAL simplex blend
```

and:

```text
R-specific simplex blend
F-specific simplex blend
```

For R/F optimization, each row uses the weight vector corresponding to its game type.

Do not assume the MoE belongs in the final blend. Let the optimization determine that.

---

## 3.3 Avoid ensemble-weight overfitting

Do not optimize weights on a fold and report performance on the same fold as the only result.

Preferred evaluation:

### Option A: Leave-one-fold-out ensemble fitting

For each evaluation fold:

```text
1. fit blend weights on OOF predictions from all other available folds
2. apply frozen weights to the held-out fold
3. report held-out Brier
```

Then aggregate the held-out predictions.

### Option B: Strict time-forward fitting

If the repository has naturally ordered folds, prefer:

```text
earlier fold OOF -> choose weights -> later fold evaluation
```

Use whichever protocol best matches the existing canonical validation system.

For final 2025 deployment, after the blend design and regularization are frozen, weights may be fit on all accepted historical OOF predictions.

---

## 3.4 First cheap test using existing predictions

Before any new model training, test:

```text
baseline + Pitch Arsenal MoE
```

with blend weights:

```text
0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20
```

and especially:

```text
R only:
p_R = (1-gamma) * baseline + gamma * arsenal

F:
p_F = baseline
```

This should require no retraining if OOF prediction files already exist.

Also report prediction residual correlation:

```text
corr(p_model_a - y, p_model_b - y)
```

for R and F separately.

If Arsenal MoE has no useful diversity even on R, close that path.

---

# 4. Work Package B — Season-Safe Trackman Physical Profiles

## 4.1 Temporal rule

For a target feature season `S`, only Trackman data from seasons `< S` may be used.

Required chronology:

```text
train season 2020 -> Trackman 2019
train season 2021 -> Trackman 2019-2020
train season 2022 -> Trackman 2019-2021
train season 2023 -> Trackman 2019-2022
train season 2024 -> Trackman 2019-2023
test  season 2025 -> Trackman 2019-2024
```

Key:

```text
pitcher_id + feature_season
```

The original train row count, order, and `row_id` must remain unchanged after left join.

---

## 4.2 First physical feature set

Do not start with dozens of engineered features.

Use a compact physical summary first:

```text
tm_n

velo_mean
velo_std

spin_mean
spin_std

ivb_mean
ivb_std

hb_mean
hb_std

extension_mean
extension_std

release_height_mean
release_height_std

release_side_mean
release_side_std

plate_speed_mean
plate_speed_std

fastball_rate
breaking_rate
offspeed_rate
```

Use actual repository column names from `trackman_history.csv`.

Do not silently invent a field if it does not exist. Map every engineered feature to the source columns in code comments / report.

---

## 4.3 Recent-state delta features

Add a small set of recent-vs-long-term deltas.

For each feature where this is meaningful:

```text
recent300_mean - historical_mean
```

Initial set:

```text
recent300_velo_delta
recent300_spin_delta
recent300_ivb_delta
recent300_hb_delta
recent300_extension_delta
recent300_release_height_delta
recent300_release_side_delta
```

The "last 300" selection must be based only on pitches before `feature_season`.

If the source has a reliable chronological pitch timestamp/date, use it.

If chronological ordering is ambiguous, do not use recent300 until order is proven.

---

## 4.4 Season-aware fallback hierarchy

For missing or unmapped players:

```text
player profile
  -> team + hand + feature_season profile
  -> hand + feature_season league profile
  -> feature_season global league profile
```

This fallback itself must obey the temporal rule.

Example:

```text
feature_season = 2022
```

means the team/hand and league fallback tables are computed only from Trackman 2019-2021.

Do not use one static 2019-2024 fallback table for every train season.

---

## 4.5 Shrinkage

For low-support mapped pitchers, shrink player statistics toward the team+hand reference profile.

For a scalar feature:

```text
w = n / (n + lambda)

x_shrunk =
    w * x_player
  + (1-w) * x_team_hand
```

Test a small fixed lambda grid only:

```text
50
100
300
500
1000
```

Do not jointly tune dozens of lambdas.

Initially use one shared lambda across the physical summary features.

The main downstream metric is Brier, not physical reconstruction error.

---

# 5. Work Package C — PCA as a Stable Physical State

## 5.1 Motivation

Physical Trackman variables are highly correlated.

Examples:

```text
release velocity
plate velocity
fastball velocity

spin / movement

release height / release side / extension
```

Instead of giving every correlated aggregate directly to the model, test a low-dimensional physical state.

Target dimensionality:

```text
4, 6, 8 PCs
```

Do not use PCA as a replacement for raw canonical features.

Use:

```text
canonical CatBoost features
+ PCA state
```

Optionally compare against:

```text
canonical features
+ compact raw Trackman profile
```

and:

```text
canonical features
+ raw Trackman profile
+ PCA state
```

---

## 5.2 Leakage-safe PCA fitting

PCA/scaler must be fitted in a temporally valid way.

For each validation configuration:

```text
1. build only profiles allowed by the fold's training horizon
2. fit imputer/scaler/PCA using training-side profile rows only
3. freeze transform
4. transform validation profiles
```

Do not fit PCA globally on 2019-2024 physical profiles and then evaluate 2022/2023/2024.

For final 2025 deployment, fit the frozen final transform using historical profiles available through 2024 after the design is finalized.

---

## 5.3 PCA outputs

Store:

```text
phys_pc1
phys_pc2
...
phys_pcK
```

plus:

```text
phys_pc*_support
```

if useful.

Also store the PCA loadings and explained variance in artifacts for interpretation.

Do not assume semantic names such as "power axis" unless the loadings actually support that interpretation.

---

# 6. Work Package D — Explicit Time-Series Pitcher State

The prior Pitch Arsenal MoE overfit quickly:

```text
best epoch ~2-4
training Brier kept improving
validation Brier worsened
```

This suggests we should first try explicit low-capacity temporal state features instead of another large sequence model.

---

## 6.1 Basic state decomposition

For each physical variable or PC:

```text
level
recent level
slope
recent - long-term delta
volatility
```

Candidate definitions:

### Long-term level

Historical mean using all Trackman pitches before `feature_season`.

### Recent level

Mean over:

```text
last 100
last 300
last 500
```

Start with only `last 300`.

### Linear trend

For season-level or game-level physical summaries, fit:

```text
x_t = a + b * t
```

and store:

```text
slope = b
```

Do not fit a trend on unordered pitches.

### Volatility

Examples:

```text
recent std
season-to-season std
absolute recent-minus-longterm deviation
```

---

## 6.2 EMA state

Test a simple exponentially weighted state before Kalman/RNN models.

For ordered summaries:

```text
s_t = alpha * x_t + (1-alpha) * s_{t-1}
```

Small alpha grid:

```text
0.1
0.25
0.5
```

Potential outputs:

```text
phys_pc1_ema
phys_pc1_minus_ema
...
```

Only continue if this produces consistent fold improvement.

---

## 6.3 State-space / Kalman extension

Do NOT implement a Kalman model immediately.

Only consider it if:

```text
raw profile improves baseline
AND
PCA state improves or matches raw profile
AND
simple trend / EMA features show incremental gain
```

The first objective is to establish that temporal physical state is useful at all.

---

# 7. Work Package E — Unseen / Sparse Pitcher State Inference

## 7.1 Problem

Hidden 2025 may contain:

```text
- pitcher IDs observed in training but without accepted Trackman mapping
- sparse pitchers
- entirely unseen pitcher IDs
```

We need a fixed, row-wise way to produce compatible physical-state features without hidden-test aggregation.

---

## 7.2 Do not map an unseen pitcher to one existing player

Avoid:

```text
new pitcher -> nearest known pitcher identity
```

Instead infer a plausible physical profile/state:

```text
observable official features
   -> estimated physical profile / physical PCs
```

The identity of the nearest historical player is irrelevant.

---

## 7.3 Allowed inference inputs

The imputation model may only use columns available for the hidden row at inference time.

Candidate inputs:

```text
pitcher_hand
pitcher_team_id

asof_pitcher_n
asof_pitcher_success_rate
asof_pitcher_reverse_rate
asof_pitcher_middle_rate
asof_pitcher_ball_rate
asof_pitcher_strike_rate

asof_pitcher_prev1_game_*
asof_pitcher_prev3_game_*
asof_pitcher_prev5_game_*

asof_pitcher_fastball_rate
asof_pitcher_breaking_rate
asof_pitcher_offspeed_rate

season / game context only if justified
```

Use the exact dataset columns available in the repository.

Do not use target information or any Trackman measurement from the target season.

---

## 7.4 Cold-start validation

Create a dedicated cold-start benchmark.

Do not evaluate an imputer on the same pitcher identities it was trained to memorize.

Recommended unit:

```text
pitcher x feature_season
```

For each cold-start split:

```text
1. hold out complete pitcher identities or pitcher-season units
2. hide their real Trackman physical state
3. provide only official observable features
4. estimate physical state
5. feed estimated state into downstream control_success model
6. score final Brier
```

Strong preference: group holdout by pitcher identity for the hardest test.

---

## 7.5 Methods to compare

Start from low complexity:

```text
A. global season-aware mean
B. hand-specific mean
C. team+hand mean
D. KNN in official-feature space
E. CatBoost regression
F. TabPFN regression, only if the installed/compatible environment supports it cleanly
```

Do not jump to VAE/cVAE first.

If TabPFN is tested, use it on the **small pitcher-season state inference table**, not on 1.47M pitch rows.

The goal is to test whether small-N tabular foundation-model behavior is useful for cold-start state inference.

---

## 7.6 Prediction target for the state imputer

Prefer predicting:

```text
physical PCs
```

rather than dozens of raw physical features.

Example:

```text
official pitcher-season features
    -> phys_pc1 ... phys_pc6
```

This reduces the dimensionality of the cold-start regression target and may make the task more stable.

For known-but-sparse players, blend observed and inferred state:

```text
w = n / (n + lambda)

state_used =
    w * state_observed
  + (1-w) * state_inferred
```

For `n = 0`:

```text
state_used = state_inferred
```

---

# 8. Core Ablation Matrix

Do not build the entire final system in one shot.

Run the following sequence.

## Experiment 0 — Existing backbone

```text
canonical baseline
```

Record the exact reference metrics.

---

## Experiment 1 — Compact Trackman profile only

```text
canonical
+ compact season-safe physical profile
```

No PCA.
No temporal trend.
No learned cold-start inference beyond simple fallback.

Question:

> Does Trackman physical history contain incremental predictive signal at all?

If no, stop the Trackman branch.

---

## Experiment 2 — Shrinkage + hierarchical fallback

```text
canonical
+ physical profile
+ season-aware fallback
+ shrinkage
```

Question:

> Does support-aware stabilization improve transfer?

---

## Experiment 3 — PCA state

Compare:

```text
3A canonical + raw compact physical
3B canonical + physical PCs
3C canonical + raw compact physical + physical PCs
```

Question:

> Is a low-dimensional physical state more stable than raw correlated aggregates?

---

## Experiment 4 — Explicit temporal state

Add only a small set:

```text
recent300 - longterm
slope
EMA / current-minus-EMA
volatility
```

Question:

> Is physical change more useful than static identity/profile?

---

## Experiment 5 — Cold-start inference

Use forced unknown pitchers.

Compare:

```text
fallback mean
KNN
CatBoost state predictor
TabPFN state predictor (optional)
```

Select by **downstream Brier**, not PC reconstruction MSE alone.

---

## Experiment 6 — Final convex ensemble

Candidate OOF models:

```text
current gated baseline
physical CatBoost
R-fast
Pitch Arsenal MoE only if residual diversity is positive
other already-proven experts
```

Optimize regularized simplex weights, globally and R/F-specific.

Use fold-excluded evaluation.

---

# 9. Validation and Reporting Requirements

Every experiment must report at least:

```text
overall Brier
R Brier
F Brier

fold-wise Brier
fold-wise delta vs baseline

mapped/high-confidence rows
fallback rows
cold-start rows

high-support pitcher rows
low-support pitcher rows
```

For Trackman features additionally report:

```text
profile coverage
support distribution
fallback-tier distribution
missing-feature rate
PCA explained variance
```

For learned state inference additionally report:

```text
cold-start physical-state MSE/MAE
cold-start downstream Brier
known-pitcher downstream Brier
```

---

## 9.1 Promotion rule

Do not promote a feature/model because one fold improved by a tiny amount.

Desired evidence:

```text
- directionally consistent improvement across temporal folds
- no major F degradation
- no hidden test dependency
- improvement is at least plausibly above OOF noise
```

Given the current level of the project, improvements in the `1e-5 to 1e-4` Brier range can matter, but they must be stable.

---

# 10. Interpretation of the Failed Pitch Arsenal MoE

Do not treat the 3-hour experiment as useless.

Observed:

```text
model parameters ~607K
auxiliary pitch-type task learned
auxiliary physics loss decreased
standalone weighted Brier = 0.24834853
baseline = 0.24789305
```

Its major failure is F:

```text
Arsenal:
R = 0.24807175
F = 0.25036706

baseline:
R = 0.24798863
F = 0.24716205
```

Interpretation:

```text
- the network can learn Trackman-related representation
- that representation is not sufficiently season-stable as a standalone control predictor
- it may retain residual diversity on R
- do not spend more compute on large HPO until the cheap ensemble test is done
```

The Trackman branch should now move from:

```text
end-to-end neural prediction
```

to:

```text
low-dimensional state distillation
+ classical tabular prediction
```

---

# 11. Suggested Repository Structure

Prefer a small number of scripts and reuse existing project utilities.

Suggested files:

```text
scripts/
  build_pitcher_physical_profiles.py
  build_physical_pca_state.py
  build_physical_temporal_state.py
  train_physical_catboost_cv.py
  run_cold_start_state_inference.py
  optimize_brier_ensemble.py
  audit_physical_pipeline.py
```

Outputs:

```text
outputs/
  physical_state/
    profiles/
    pca/
    temporal/
    cold_start/
    cv/
    ensemble/
    report.md
```

Do not duplicate canonical preprocessing logic already present in `src/`.

---

# 12. Implementation Guardrails

1. **Never mutate train row order or row_id.**
2. **Every physical feature must be reproducible from information strictly before `feature_season`.**
3. **Every fallback table must be season-aware.**
4. **Every scaler/PCA transform must be fit only on the appropriate training horizon.**
5. **Cold-start models must not memorize held-out pitcher identity.**
6. **Do not use hidden-test row aggregates.**
7. **Do not replace the current backbone until temporal CV proves improvement.**
8. **Cache expensive profile tables and OOF predictions.**
9. **Keep exact feature provenance in the report.**
10. **Prefer one clean ablation at a time over broad architecture search.**

---

# 13. Recommended Execution Order

Execute in this exact order unless a repository dependency forces a small change:

```text
STEP 1
Re-use saved OOF predictions.
Test baseline + Arsenal R-only blend and residual correlations.
No retraining.

STEP 2
Build compact season-safe Trackman physical profiles for accepted 584 pitchers.
Add season-aware fallback.
Audit coverage and temporal correctness.

STEP 3
Train canonical CatBoost + compact physical profile.
Compare against exact current baseline.

STEP 4
Add shrinkage.
Small lambda sweep only.

STEP 5
Fit fold-safe PCA.
Test 4 / 6 / 8 PCs.

STEP 6
Add recent300 deltas, slope, and simple EMA features.
No Kalman/RNN yet.

STEP 7
Build forced cold-start benchmark.
Compare mean/KNN/CatBoost/optional TabPFN state inference.

STEP 8
Choose the best physical CatBoost design.

STEP 9
Regularized convex Brier ensemble using fold-excluded weight fitting.

STEP 10
Only if all above show positive evidence:
consider more sophisticated state-space or probabilistic latent models.
```

---

# 14. Required Final Deliverable from Codex

At the end of the run, produce one compact Markdown report with:

```text
1. What was implemented
2. Exact leakage controls
3. Feature definitions and source columns
4. Mapping/profile coverage
5. Ablation table
6. Fold-wise Brier / R / F
7. Cold-start benchmark
8. Ensemble weights
9. Best candidate vs current reference
10. Recommendation:
   PROMOTE / KEEP AS EXPERT / REJECT
```

The report must include exact commands needed to reproduce the best result.

Do not summarize a failed experiment as "promising" unless the downstream Brier actually supports that claim.
