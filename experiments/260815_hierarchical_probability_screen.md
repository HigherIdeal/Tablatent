# Hierarchical probability screen

## Question

Can leakage-safe historical conditional probabilities add a materially stronger signal than raw context features alone?

This experiment keeps every eligible training row. It does not use `pitcher_id` or `batter_id` in the probability tables or in the model.

## Protocol

For a row in season `Y`, every target-derived probability feature is built using labels from seasons strictly earlier than `Y`.

Examples:

- 2020 rows use 2019 labels only.
- 2021 rows use 2019-2020 labels.
- 2022 rows use 2019-2021 labels.
- 2023 rows use 2019-2022 labels.
- 2024 rows use 2019-2023 labels.
- The earliest season has no earlier labels and receives a neutral 0.5 fallback.

Thus validation rows never contribute their own labels to the mapping, and training rows do not receive same-season target statistics.

Sparse child groups are Empirical-Bayes smoothed toward their parent probability. Default smoothing strength is `alpha=200`.

## Probability families

- `count`: `P(success | balls, strikes)`, shrunk toward the historical global prior.
- `count_game_type`: `P(success | balls, strikes, game_type)`, shrunk toward count probability.
- `count_handedness`: `P(success | balls, strikes, pitcher_hand, batter_hand)`, shrunk toward count probability.
- `count_base`: `P(success | balls, strikes, base_state)`, shrunk toward count probability.
- `experience_count`: `P(success | balls, strikes, pitcher experience bucket)`, shrunk toward count probability.

Pitcher experience buckets are based only on `asof_pitcher_n`: `0-100`, `101-500`, `501-1000`, `1001-2000`, `2001-5000`, `5000+`.

## Variants

- `reference_canonical`
- `add_success_state`
- each hierarchical family separately
- `add_hp_all`
- `add_success_plus_hp_all`

All models use the same canonical CatBoost screening settings and temporal folds `2022,2023,2024` by default. No row sampling is used.

## Run

```powershell
git pull
python scripts/run_hierarchical_probability_screen.py --config configs/default.yaml
```

Optional smoothing sensitivity check, only if the default screen is promising:

```powershell
python scripts/run_hierarchical_probability_screen.py --config configs/default.yaml --alpha 50
python scripts/run_hierarchical_probability_screen.py --config configs/default.yaml --alpha 1000
```

## Outputs

`outputs/hierarchical_probability_screen/`

- `fold_results.csv`
- `summary.csv`
- `feature_sets.json`
- `mapping_diagnostics.csv`
- `run_config.json`

The primary decision signal is the sign and magnitude of `delta_brier_vs_reference` across all three temporal folds, not a single-year win.
