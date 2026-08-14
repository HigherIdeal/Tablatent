# LG Aimers 9기 - 다음 투구 제구 성공 확률

현재 경기 상황과 과거 이력을 각각 독립적인 latent로 학습한 뒤, 두 표현을 고정해
`control_success` 확률을 예측하는 재현 가능한 PyTorch 베이스라인입니다.

## Colab에서 실행

```bash
pip install -r configs/requirements.txt
python scripts/prepare_data.py
python scripts/train_stage1.py --config configs/default.yaml
python scripts/train_stage2.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

한 번에 실행하려면 다음 명령을 사용합니다.

```bash
python scripts/run_all.py --config configs/default.yaml
```

첫 명령은 고정 Google Drive URL에서 원본을 받아 `data/raw/`에 보존하고,
CSV를 찾아 UTF-8로 정규화한 뒤 `data/processed/`에 시즌별 파일과 manifest를 만듭니다.
기본 split은 2019~2022(train), 2023(validation), 2024(test)입니다.

## 결과물

- `outputs/checkpoints/`: Stage1 encoder, Stage2 predictor, 전처리기
- `outputs/latents/`: 고정된 latent와 row id export
- `outputs/logs/`: 학습 이력, Brier/BSS 지표, 예측 CSV

`Brier Skill Score = 1 - model Brier / climatology Brier`이며, climatology 확률은
훈련 시즌의 target 평균만 사용합니다. 2024 test는 최종 평가 전까지 학습, 전처리,
조기 종료에 사용하지 않습니다.

## 컬럼 자동 분류

`asof_*` 및 이름에 pitcher/batter history 의미가 있는 컬럼은 과거 이력으로,
나머지 투구 직전 입력 컬럼은 현재 상황으로 분류합니다. target, 시즌, 행 식별자와
명백한 투구 결과/사후 컬럼은 입력에서 제외합니다. 실제 별첨 데이터의 컬럼명이
다르면 `configs/default.yaml`의 include/exclude 목록으로 명시할 수 있습니다.

