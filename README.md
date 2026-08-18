# Bitaboost

Compact, reproducible research repository for the **LG Aimers 9 baseball next-pitch control-probability task**.

The project predicts `control_success` for each pitch row. The current stable checkpoint is the recovered **SAFE982** baseline: a rule-safe ensemble reconstructed from the historical Tablatent research lineage and then re-run end-to-end from source.

## Current checkpoint

**Stable branch:** `bitaboost-stable`  
**Frozen checkpoint:** `checkpoint/2026-08-18-safe982`  
**Checkpoint date:** 2026-08-18

Observed 2024 validation result on the RTX 4090 setup:

```text
Final Brier = 0.247352360403
Score       = 982.5854
reference_pass = True
```

Historical SAFE reference:

```text
Brier = 0.247355098397
Score = 981.4893
```

The recovered run slightly exceeds the historical reference while remaining within the configured reproduction tolerance. This is the baseline to return to before future ablations.

> 2024 is **not a pristine holdout**. It has already been used extensively for validation, model selection, and recovery. Treat the score above as an internal research reference, not as an unbiased estimate of hidden-test performance.

## What the model is

SAFE982 is **not one CatBoost model**. It is a small ensemble of complementary CatBoost models built around different views of the same row and frozen historical state.

At a high level:

```text
                         ┌─────────────────────────────┐
                         │  row + frozen history      │
                         │  regime/context features   │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
                 ▼                      ▼                      ▼
        direct multitask        auxiliary heads        hurdle decomposition
          MultiRMSE              reverse/middle        gate × conditional
                 │                      │                      │
                 └───────────────┬──────┴──────────────┬───────┘
                                 │                     │
                                 ▼                     │
                           MIXED predictor             │
                                 │                     │
                    ┌────────────┼────────────┐        │
                    │            │            │        │
                    ▼            ▼            ▼        │
                  mixed       offset400    joint600    │
                    └────────────┬────────────┘        │
                                 ▼                     │
                             SAFE core                 │
                                 │                     │
                                 └──────────┬──────────┘
                                            ▼
                              + structured-id600
                                            │
                                            ▼
                                          FINAL
```

The ensemble is fitted **separately for game-type domains `R` and `F`**, because the data exhibits a substantial regime/domain shift and the best constituent mix is not the same in both domains.

## Model components

### 1. Direct multitask model

A depth-8 CatBoost `MultiRMSE` model jointly predicts success and reconstructed pitch-state targets.

```text
outputs = [success × 8, reverse, middle, ball, strike]
trees   = 600
F row weight = 2.0
```

Repeating the success target gives the primary task more influence while retaining shared structure from the auxiliary outcomes.

### 2. Reverse / middle auxiliary heads

Two standalone CatBoost binary classifiers estimate:

```text
reverse600 = P(reverse)
middle400  = P(middle)
```

Their targets are reconstructed from cumulative pre-pitch `asof_*` rates. Historical parity matters here: the standalone heads use the full-frame reconstruction scope that existed in the original experiment.

### 3. Hurdle decomposition

The hurdle branch separates two questions:

```text
gate600 = P(no reverse and no middle)
cond400 = P(control_success | gate state)
```

The gate is trained with an RMSE/Brier-style regression objective; the conditional head is a Logloss classifier.

### 4. Mixed predictor

The direct and hurdle views are combined through the recovered logical decomposition:

```text
independent_gate = clip(
    1 - reverse600 - middle400 - 1.2 * reverse600 * middle400,
    0, 1
)

hybrid_gate = 0.4 * independent_gate + 0.6 * gate600
logic       = hybrid_gate * cond400

mixed = domain-wise blend(direct, logic)
```

The blend coefficient is fitted independently for `R` and `F`.

Recovered 2026-08-18 run:

```text
mixed Brier = 0.247377078372
R blend     = 0.374113214243
F blend     = 0.697950494554
```

### 5. Old-cross1 residual model

A separate 400-tree CatBoost residual model predicts correction around a recent prior.

It deliberately uses the preserved **old `cross1` feature definition**, not the later expanded cross implementation:

```text
eng_anchor_cross_success
eng_anchor_cross_middle
eng_anchor_pitch_success_shrunk
eng_anchor_batter_success_shrunk
eng_anchor_gap_logratio
```

This component reproduced the historical prediction vector **bit-for-bit**.

### 6. Joint auxiliary-state model

A 600-tree CatBoost `MultiClass` model predicts the joint combination of:

```text
reverse / middle / ball / strike
```

The predicted class distribution is converted back to a success probability using the historical domain-conditioned success mapping. This component also reproduced the historical prediction vector **bit-for-bit**.

### 7. Structured outcome model

A pre-rich 600-tree `MultiClass` model with `pitcher_id` and `batter_id` predicts five outcome states:

```text
0: success
1: failure with neither reverse nor middle
2: failure with reverse only
3: failure with middle only
4: failure with both
```

Its class-0 probability is used as the success prediction. Historically it contributed only a small `F`-domain correction, but that small correction improved the final Brier. The recovered vector matches the historical artifact to numerical precision.

## Final ensemble

The final combination has two stages.

```text
SAFE  = R/F simplex(mixed, offset400, joint600)
FINAL = R/F simplex(SAFE, structured-id600)
```

Historical SAFE-core weights were approximately:

```text
R:
  mixed   0.78368392
  offset  0.20305241
  joint   0.01326367

F:
  mixed   0.51146784
  offset  0.37011603
  joint   0.11841613
```

Historical final correction:

```text
R: SAFE only
F: 0.958476824 * SAFE + 0.041523176 * structured-id600
```

The current training script refits these simplex weights from the generated 2024 validation predictions instead of hard-coding rounded constants.

## Feature families

The model uses supplied row features plus inference-safe historical state. Main families are:

- raw game context, count, base state, score, hands, teams, IDs where explicitly enabled;
- supplied pitcher/batter `asof_*` history statistics;
- 2023+ regime indicator and R-domain regime interactions;
- frozen pitcher and batter anchor state;
- batter-anchor and anchor-cross state;
- frozen pitcher-batter matchup profiles;
- frozen count, pressure, and game-type domain profiles;
- frozen auxiliary and conditional profiles;
- target-free context interactions;
- prior-season entity path state.

All hidden-test rows must remain independent prediction targets. The SAFE contract forbids hidden-test peer aggregation, rolling updates from other hidden rows, or adaptation to the hidden-test distribution.

Current recovered feature counts:

```text
rich       185
hurdle     185
offset      90
structured  74
```

## Reproduction audit

The 2026-08-18 recovery was checked directly against preserved historical prediction artifacts:

| Component | MAE | RMSE | MAX |
|---|---:|---:|---:|
| mixed | 9.756e-04 | 1.328e-03 | 1.113e-02 |
| offset | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| joint | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| structured | 1.600e-08 | 1.954e-08 | 7.180e-08 |
| safe | 7.132e-04 | 9.359e-04 | 7.216e-03 |
| final | 7.085e-04 | 9.283e-04 | 7.216e-03 |

The residual difference is concentrated in the mixed lineage. Offset and joint are exact, structured is numerically identical, and the full pipeline passes the configured historical reference tolerance.

## Repository structure

```text
configs/
  baseline_safe_981.yaml       # frozen baseline recipe + provenance + runtime policy

scripts/
  prepare_data.py              # CSV -> fast local cache
  baseline_train.py            # one-command training + 2024 validation + artifact audit
  eval.py                      # evaluate saved predictions

src/bitaboost/
  baseline.py                  # training orchestration
  features.py                  # recovered SAFE feature composition
  ensemble.py                  # R/F mixed + simplex ensemble logic
  references.py                # read-only historical prediction audit
  runtime.py                   # one-GPU and quiet logging policy
  _legacy/                     # frozen historical helper/source definitions

experiments/
  configs/                     # future experiment configs

checkpoints/
  2026-08-18_SAFE982.md        # frozen checkpoint record

outputs/
  baseline/                    # baseline artifacts
  experiments/                 # isolated experiment outputs
```

`src/bitaboost/_legacy/` is **not** the research surface. It exists to preserve historical semantics required for reproducibility. New work should be implemented through clean reusable code/configs outside that directory.

## Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost

python scripts/prepare_data.py --config configs/baseline_safe_981.yaml
python scripts/baseline_train.py --config configs/baseline_safe_981.yaml
python scripts/eval.py --config configs/baseline_safe_981.yaml
```

The baseline runner exposes only physical GPU 2:

```text
CUDA_VISIBLE_DEVICES=2
CatBoost devices="0"
```

This intentionally uses exactly one RTX 4090. Multi-GPU settings are rejected by config validation.

## Checkpoint / restore

Return to the immutable 2026-08-18 checkpoint:

```bash
git fetch origin
git switch -c restore-safe982 --track origin/checkpoint/2026-08-18-safe982
```

Return to the promoted stable line:

```bash
git fetch origin
git switch -C bitaboost-stable origin/bitaboost-stable
```

The stable reference command is always:

```bash
python scripts/baseline_train.py --config configs/baseline_safe_981.yaml
```

## Research policy from this checkpoint

SAFE982 is the **control** for future research. New ideas should be measured as deltas against this checkpoint rather than silently changing the baseline.

Preferred workflow:

```text
SAFE982 checkpoint
      ↓
new hypothesis / ablation
      ↓
separate experiment config + prediction artifact
      ↓
compare Brier overall + R/F + robustness
      ↓
promote only if improvement is reproducible
```

Do not create a large collection of `run_foo_v2_final2.py` scripts. Prefer configs under `experiments/configs/` and reusable modules under `src/bitaboost/`. Every meaningful experiment should preserve enough metadata to recover the exact source/config that produced its predictions.

Submission ZIP construction remains intentionally separate from this research baseline.
