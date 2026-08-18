# EX2 — Hypothesis-conditioned backward consistency

EX2 tests a different question from the normal forward predictor:

> If the unknown current outcome were assumed to be 0 or 1, which assumption makes the current row more consistent with the pitcher's known previous-season state?

No SAFE982 prediction is loaded, blended, or retrained.

## Core idea

Training rows use the true current label only as a conditioning input:

```text
(current pre-pitch X, true current y) -> previous-season pitcher state
```

The reconstructed previous-season state is:

```text
[success, reverse, middle, ball, strike]
```

At evaluation, the same row is evaluated twice:

```text
(X, hypothesis y=0) -> reconstructed past state 0
(X, hypothesis y=1) -> reconstructed past state 1
```

Both reconstructions are compared with the previous-season pitcher state that is already known from earlier official training data. The lower standardized reconstruction energy is treated as the more compatible outcome hypothesis.

```text
E0 = distance(past_truth, reconstruct(X, y=0))
E1 = distance(past_truth, reconstruct(X, y=1))
margin = E0 - E1
```

Positive margin favors `y=1`; negative margin favors `y=0`.

## Why this is not a normal forward model

The model is never trained with `control_success` as its prediction target. Its supervised target is the previous-season pitcher state. The current label is only injected as a hypothetical condition. At inference/evaluation both possible labels are tried explicitly and judged by backward consistency.

## Variants

- `context_only`: removes player/team IDs and supplied/asof history features. Tests whether game context plus the candidate label alone creates backward consistency.
- `history_no_id`: retains supplied historical/asof state but removes raw identity IDs.
- `history_plus_id`: retains the complete canonical recent feature set. This measures the upper bound when persistent player identity is available.

## Rolling evaluation

The default run evaluates 2022, 2023, and 2024 chronologically. A row is eligible only if the same pitcher has at least 200 pitches in the immediately previous season.

For each fold EX2 reports:

- energy-margin AUC and sign accuracy;
- Brier/logloss derived only from the backward energy margin;
- constant-prior Brier as a non-model reference;
- true-label vs false-label reconstruction RMSE;
- per-state energy contribution (`success/reverse/middle/ball/strike`);
- counterfactual reconstruction gap between the y=0 and y=1 assumptions;
- TT / TF / FT / FF diagnostics, where T/F denotes whether the previous-season success trait is above/below the training median and the current pitch outcome is success/failure;
- batch-vs-single-row independence audit.

Probability temperature for 2023/2024 is calibrated only from earlier held-out folds. The current fold never chooses its own temperature.

## SAFE property

Each evaluated row uses only:

- that row's own pre-pitch inputs;
- a candidate value 0 or 1;
- a frozen previous-season profile built from earlier official training seasons.

No other evaluation row is read, aggregated, shifted, rolled, or used to adapt the model.

## Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
python scripts/ex2/run_hypothesis_backward.py \
  --config experiments/configs/ex2_hypothesis_backward.yaml
```

Outputs:

```text
outputs/experiments/ex2/hypothesis_backward/
  metrics_hypothesis_backward.json
  hypothesis_<variant>_<season>.npz

models/ex2/
  hypothesis_backward_<variant>.cbm
```
