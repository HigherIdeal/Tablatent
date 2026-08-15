# Tablatent — VAE latent control-probability experiments

LG Aimers 9기 야구 해커톤에서 pitch 직전 tabular state를 두 개의 latent branch로 압축하고, `control_success` 확률을 예측하는 실험 저장소입니다.

## Current split

- 2019~2022: train
- 2023: validation
- 2024: untouched holdout

## Stage 1 — two-branch VAE

- `mu_context`: 16D, 현재 경기 상황
- `mu_history`: 16D, 해당 시점까지의 `asof_*` history snapshot
- Stage 1은 `control_success`를 사용하지 않음
- `pitcher_id`, `batter_id`, team ID도 latent 입력에서 제외
- Stage 2에는 posterior sample이 아니라 deterministic posterior mean `mu`를 전달
- 현재 KL beta: context/history 모두 `5e-4`, warm-up 10 epochs

```bash
python scripts/run_stage1.py --config configs/default.yaml
```

고정 데이터 URL:

`https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link`

## Colab Stage 1 cache

Colab 세션 종료 후 Stage 1을 다시 학습하지 않도록 Google Drive에 Stage 1 결과를 보존합니다. 기본 cache 위치는 `/content/drive/MyDrive/Tablatent/stage1_cache`입니다.

Stage 1 학습 직후:

```bash
python scripts/stage1_cache.py push
```

새 Colab 세션에서 repository와 requirements를 준비한 뒤:

```bash
python scripts/stage1_cache.py pull
```

`pull`이 끝나면 바로 Stage 2를 실행할 수 있습니다. 기본 cache에는 다음을 포함합니다.

```text
outputs/latents/context.npy
outputs/latents/history.npy
outputs/latents/context_logvar.npy
outputs/latents/history_logvar.npy
outputs/checkpoints/stage1_context.pt
outputs/checkpoints/stage1_history.pt
outputs/checkpoints/preprocessors.joblib
outputs/logs/stage1_training.json
data/processed/train.pkl
```

각 파일은 SHA256 manifest로 검증합니다. processed dataset을 Drive에 저장하고 싶지 않으면 push/pull 모두 `--exclude-data`를 사용합니다.

```bash
python scripts/stage1_cache.py push --exclude-data
python scripts/stage1_cache.py pull --exclude-data
```

## Stage 2 — current experiment

2026-08-15 기본 실험은 generic MLP 대신 context-history interaction을 명시적으로 학습하는 bilinear probe입니다.

```text
c = mu_context  # 16D
h = mu_history  # 16D
g = Bilinear(c, h)  # 16D
[c, h, g]  # 48D
 -> Linear(48, 1)
 -> sigmoid
```

raw `control_success` 0/1 label과 `BCEWithLogitsLoss`를 사용하며, Stage 1은 frozen입니다.

```bash
python scripts/train_stage2.py --config configs/default.yaml --head bilinear
```

기존 baseline도 같은 latent와 temporal split에서 그대로 실행할 수 있습니다.

```bash
python scripts/train_stage2.py --config configs/default.yaml --head linear
python scripts/train_stage2.py --config configs/default.yaml --head mlp
```

주요 Bilinear 출력:

```text
outputs/stage2_bilinear/stage2_bilinear_best.pt
outputs/stage2_bilinear/metrics.json
outputs/stage2_bilinear/validation_predictions.csv
```

## Legacy / diagnostic experiments

`evaluate_knn.py`, `build_stage2_dataset.py`, `src/stage2.py`, `src/stage2_regularized.py`는 latent neighborhood와 local-probability 실험을 재현하기 위해 유지합니다. 현재 bilinear 실험에는 사용하지 않습니다.

```bash
python scripts/evaluate_knn.py --config configs/default.yaml
```
