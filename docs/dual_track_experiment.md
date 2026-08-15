# Dual-track regime experiment

## Goal

Use old seasons for stable baseball signal without letting the pre-2023 `game_type` relationship contaminate the recent regime.

- **Recent expert A**: 2023 only -> validate on 2024, keeps raw `game_type`.
- **Stable expert B**: 2019-2023 -> validate on 2024, removes `game_type` but otherwise uses the same canonical + `success_state` features.
- Blend: `p = alpha_recent * p_A + (1 - alpha_recent) * p_B`.

## Screen

```powershell
python scripts/run_dual_track_blend_screen.py --config configs/default.yaml
```

Default search:

- CatBoost trees: `100,150,200,250,300,400` independently for A and B.
- Blend alpha: 0.05 grid plus the exact Brier-optimal clipped analytic alpha.
- Validation: 2024 only, with strictly older training seasons.

Outputs are written to `outputs/dual_track_blend_screen/`:

- `expert_results.csv`
- `blend_results.csv`
- `validation_predictions.npz`
- `recommended_config.json`
- `run_config.json`

The recommended config uses the coarse alpha grid to avoid carrying an overly precise 2024-only optimum into 2025.

## Final hidden-2025 build

After reviewing the screen output:

```powershell
python scripts/build_dual_track_submission.py --config configs/default.yaml
```

This reads `recommended_config.json`, then trains:

- A: 2023+2024 with raw `game_type`.
- B: 2019-2024 without `game_type`.

It packages both CatBoost models and the fixed blend into:

`dist/dual_track/dual_track_blend.zip`

To override the recommendation explicitly:

```powershell
python scripts/build_dual_track_submission.py --config configs/default.yaml --recent-iterations 200 --stable-iterations 200 --alpha-recent 0.70
```

All three override arguments must be supplied together.
