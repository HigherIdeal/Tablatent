# 2026-08-15 Game-Phase Follow-up Ablation

The 200-tree canonical screening found `drop_game_phase` to be the only large average improvement, but it was not uniformly better across 2023 and 2024. The next experiment therefore decomposes the group instead of removing all three columns at once.

## Variants

Baseline:

- `reference_canonical`

Game-phase decomposition:

- `drop_inning`
- `drop_top_bottom`
- `drop_game_type`
- `drop_inning_top_bottom`
- `drop_inning_game_type`
- `drop_top_bottom_game_type`
- `drop_game_phase`

Near-zero candidates from the previous screening:

- `drop_calendar`
- `drop_base_state`
- `drop_win_expectancy`
- `drop_weak_context` = calendar + base state + pitcher-team win expectancy

The canonical de-duplication policy remains unchanged. Pitcher/batter IDs remain excluded, deterministic duplicate encodings remain removed, and team IDs/profile/recent/pitchmix/handedness are not re-tested in this focused pass.

## Run

```powershell
python scripts/run_game_phase_followup.py --config configs/default.yaml
```

Defaults:

- folds: 2023, 2024
- iterations: 200
- GPU

Outputs are isolated from the previous broad screening under:

```text
outputs/game_phase_followup/catboost_ablation_canonical/
```

Console output includes Brier, competition-style score, AUC, and prediction spread.
