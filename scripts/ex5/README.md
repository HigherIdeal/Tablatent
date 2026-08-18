# EX5 — Frozen stable-trait pitch-level probe

EX5 is the first bridge from the EX4 bidirectional state analysis back to the pitch-level competition target. It does **not** blend SAFE982 and does **not** use backward prediction at inference.

## Question

EX4 found four pitcher-season states that remained useful in both time directions:

```text
ball / reverse / success / strike
```

`middle` was classified as weak or regime-sensitive. EX5 asks a narrower question:

> If we freeze only the previous season's bidirectionally stable state, how much standalone information does it contain about individual `control_success` outcomes in the next season?

## Causal feature construction

For every pitch from season `s`, EX5 attaches only the same pitcher's profile from season `s-1`.

```text
2019 profile -> 2020 pitch rows
2020 profile -> 2021 pitch rows
2021 profile -> 2022 pitch rows
2022 profile -> 2023 pitch rows
2023 profile -> 2024 pitch rows
```

No target aggregate from the current season is used as a feature. A pitcher absent from `s-1` receives the frozen league mean from `s-1` and reliability zero. This makes the construction directly deployable to 2025 using only 2024 training history.

The previous-season sample size provides a reliability term:

```text
reliability = n / (n + k)
```

with `k=200` by default. Stable rates are also shrunk toward the frozen source-season league mean using this reliability.

## Ablations

- `prior_success_only`: previous-season success rate only.
- `stable4_raw`: raw `ball/reverse/success/strike` rates.
- `stable4_reliable`: shrunk stable four + previous-season log-count/reliability/coverage.
- `stable4_reliable_plus_middle`: same representation with `middle` added as a negative-control trait.

If the stable four improve over `prior_success_only`, the EX4 state discovery is providing information beyond persistence of success rate alone. If adding `middle` consistently hurts or adds nothing, that supports the bidirectional filter.

## Validation

Rolling pitch-level folds are 2022, 2023 and 2024. For each fold the classifier trains only on earlier seasons whose features were themselves built from their immediately previous season.

Reported diagnostics:

- Brier and AUC;
- constant train-prior Brier;
- direct previous-season success-rate Brier;
- previous-profile coverage;
- performance by previous-season pitch count (`no_prior`, `<50`, `50-199`, `200-499`, `500+`);
- full-batch vs single-row independence audit.

This is intentionally a standalone probe. A positive result is required before any SAFE integration.

## Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
python scripts/ex5/run_stable_trait_pitch_probe.py \
  --config experiments/configs/ex5_stable_trait_pitch_probe.yaml
```

Outputs:

```text
outputs/experiments/ex5/stable_trait_pitch_probe/
models/ex5/
```
