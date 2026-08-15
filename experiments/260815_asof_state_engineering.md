# 2026-08-15 ASOF State Engineering Screen

## Question

Return to the Road-to-1500 search with a clean feature-signal experiment.

Previous diagnostics established that:

- hard recent-season training windows were worse than full history;
- per-pitcher recent-row caps were also worse than full history;
- naturally unseen 2024 pitchers were not the main failure mode;
- the official `asof_*` features remain available at inference and contain long-run plus recent pitcher/batter information.

Therefore this experiment keeps **all eligible historical training rows** and asks whether the same official `asof_*` values become more useful when expressed as state changes and interactions rather than only raw levels.

## Protocol

Temporal folds:

```text
2019-2021 -> 2022
2019-2022 -> 2023
2019-2023 -> 2024
```

Default model budget is the same 200-tree GPU CatBoost screening configuration used in the recent diagnostics. There is **no row sampling**.

Reference is the canonical 36-feature set.

Variants:

```text
reference_canonical
add_success_state
add_middle_state
add_matchup_state
add_recent_state
add_recent_experience
add_all_state
```

## Engineered families

### Success state

Use long-run pitcher success plus official previous 1/3/5-game success rates to expose directly:

- recent minus long-run deltas;
- 1-vs-3, 3-vs-5, and 1-vs-5 changes;
- recent 1/3/5 mean;
- recent mean minus long-run;
- recent range.

These are useful candidates for trees because a cross-column difference otherwise requires multiple sequential splits.

### Middle-rate state

Apply the same construction to pitcher middle-rate history.

### Pitcher-batter matchup state

Create inference-safe differences between pitcher and batter historical rates, including long-run and recent pitcher success/middle rates against batter long-run rates. This does not use pitcher or batter identity lookup.

### Experience-modulated state

Use a bounded gate

```text
w = asof_pitcher_n / (asof_pitcher_n + 2000)
```

and multiply key recent-minus-long deltas by `w`. The gate alone is monotonic in `asof_pitcher_n`; the intended new information is the interaction between experience and state change.

The constant 2000 is a diagnostic scale, not a tuned optimum. Do not grid-search it on 2024 unless this family first shows clear value.

## Run

```powershell
python scripts/run_asof_state_engineering.py --config configs/default.yaml
```

The script prints explicitly that no row sampling is used and reports Brier/score/AUC for every variant on every fold.

Outputs:

```text
outputs/asof_state_engineering/
  feature_sets.json
  fold_results.csv
  summary.csv
  run_config.json
```

## Decision rule

Do not select a feature family because of a tiny isolated 2024 win.

Prioritize families that:

1. improve Brier on multiple temporal folds;
2. do not catastrophically worsen the 2023 change-point fold;
3. retain or improve AUC without causing obviously excessive prediction spread.

If no family survives, stop hand-crafting these deltas and move to a different source of signal rather than tuning constants.
