# EX4 — Bidirectional Stable-Trait Discovery

EX4 changes the role of backward modeling. Backward is no longer treated as another predictor of the future. Instead, it is used as a temporal consistency probe: a pitcher-season state component is considered useful only if it remains learnable when time is traversed in both directions.

## Core question

For the same adjacent pitcher-season pair, train two models on the same chronological data:

```text
Forward : Z(p, s-1) -> Z(p, s)
Backward: Z(p, s)   -> Z(p, s-1)
```

The state is:

```text
[success, reverse, middle, ball, strike]
```

The goal is not to prove that backward prediction is intrinsically easier. The goal is to identify components that are robust to temporal direction and therefore look more like persistent pitcher traits than one-way regime artifacts.

## Four diagnostics

1. **Raw adjacent-season persistence**
   - Pearson/Spearman between past and current state
   - identity RMSE and mean absolute season change

2. **Bidirectional learned prediction**
   - CatBoost forward and backward models trained on exactly the same adjacent pairs
   - RMSE/MAE/Pearson/Spearman per target
   - comparison with identity and chronological train-mean baselines

3. **Cycle consistency**
   - past -> predicted future -> reconstructed past
   - future -> predicted past -> reconstructed future
   - low error in both cycles is additional evidence that the component lies on a stable temporal manifold

4. **Cross-fold stable subset selection**
   - each state is classified as `bidirectional_stable`, `forward_only`, `backward_only`, or `weak_or_regime_sensitive`
   - selection requires both directions to satisfy transparent correlation and chronological-fold criteria
   - a ranking score is reported, but the raw metrics remain the source of truth

## Reliability and identity diagnostics

The default run checks minimum pitcher-season sample counts of 50, 200 and 500 pitches. It also runs:

- `state_only`: preferred scientific diagnostic; no pitcher identity
- `state_plus_id`: measures how much apparent temporal stability can be recovered by persistent identity

The primary result is `state_only`, minimum 200 pitches.

## Validation

For each held-out current season:

```text
2022: train on earlier adjacent pairs, test 2022 pair
2023: train on earlier adjacent pairs, test 2023 pair
2024: train on earlier adjacent pairs, test 2024 pair
```

No 2024 target information is used to train the 2024 fold. EX4 is a historical state-discovery experiment and does not produce a competition submission.

## Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
python scripts/ex4/run_bidirectional_traits.py \
  --config experiments/configs/ex4_bidirectional_stable_traits.yaml
```

Main output:

```text
outputs/experiments/ex4/bidirectional_stable_traits/metrics_bidirectional_stable_traits.json
```

At the end, the runner prints the primary stable-trait ranking and the selected bidirectional subset.
