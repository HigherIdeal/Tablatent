# EX3 — Four-state counterfactual backward energy

EX3 keeps the backward-only research line and removes the main weakness seen in EX2: a candidate label should not be an ignorable feature. Instead, each hypothesis physically changes a row-local post-pitch state before the backward model sees it.

## Four latent states

For every evaluation row, EX3 constructs four mutually exclusive counterfactual states:

- `F_hard`: failure (`y=0`) assimilated with one full pitch of evidence;
- `F_soft`: failure assimilated with a fractional pitch of evidence;
- `S_soft`: success (`y=1`) assimilated with a fractional pitch of evidence;
- `S_hard`: success assimilated with one full pitch of evidence.

The soft evidence weight is configurable (`soft_event_weight`, default 0.35). This does not relabel the binary target. It changes how strongly the hypothetical pitch moves the cumulative state.

## Backward connection

Training uses the realized outcome only to build the actual post-pitch state:

```text
real pre-pitch asof state
        + actual outcome
        -> realized post-pitch state
        -> backward CatBoost
        -> previous-season pitcher profile
```

The backward target remains the frozen previous-season profile:

```text
[success, reverse, middle, ball, strike]
```

At evaluation, the observed target is not given to the model. Four counterfactual post-states are generated and each is mapped backward. The reconstruction energy against the known previous-season profile gives:

```text
E(F_hard), E(F_soft), E(S_soft), E(S_hard)
```

A softmax over negative energy yields the four latent probabilities. The binary probability required by the competition is only:

```text
P(success) = P(S_soft) + P(S_hard)
```

There is no forward success classifier or SAFE982 blend in EX3.

## State assimilation

EX3 currently mutates only row-local sufficient statistics that are explicitly present before the pitch:

- `asof_pitcher_n`
- `asof_pitcher_success_rate`
- `asof_pitcher_middle_rate`

For an event with effective evidence weight `w`:

```text
post_success_rate = (n * success_rate + w * event_success) / (n + w)
post_middle_rate  = (n * middle_rate  + w * event_middle)  / (n + w)
```

Hard states use `w=1`; soft states use the configured fractional weight. Therefore the same hypothetical event naturally moves low-history pitchers more than veterans.

For failures, middle/non-middle failure type is unobserved at inference. EX3 evaluates both substates and marginalizes them with `P(middle | failure)` estimated from the training seasons only. Success candidates use `middle=0`.

## Variants and validation

- `state_only`: only the transformed counterfactual state is used by the backward model.
- `context_state`: adds inference-safe current game context while excluding IDs and `asof_*` history features, preventing the EX2 history shortcut from trivially ignoring the counterfactual state.

Rolling folds are 2022, 2023 and 2024. Temperature calibration for the four-state energy uses only earlier held-out folds. The runner reports AUC, Brier vs prior, latent class mass/winners, ambiguity/confidence mass and row-independence audit.

## Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
python scripts/ex3/run_four_state_backward.py \
  --config experiments/configs/ex3_four_state_backward_energy.yaml
```

Outputs:

```text
outputs/experiments/ex3/four_state_backward_energy/
models/ex3/
```
