# Tablatent — Stage 1

LG Aimers 9기 야구 해커톤에서 **예측(Stage 2) 전에 tabular state representation 자체가 제대로 학습되는지 검증**하기 위한 Stage 1 코드입니다.

현재 단계에서는 `control_success` 예측기를 학습하지 않습니다.

## 목표

한 투구 직전의 정보를 두 상태로 분리합니다.

1. `z_context`: 현재 경기 상황
2. `z_history`: 그 시점까지의 과거 이력 snapshot

각 encoder는 자신의 입력을 latent로 압축한 뒤 다시 복원합니다. Stage 1의 성공 여부는 **복원 성능과 latent collapse 여부**로 먼저 확인합니다.

## 구조

```text
Tablatent/
├─ configs/
│  ├─ default.yaml
│  └─ requirements.txt
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train_stage1.py
│  ├─ evaluate_stage1.py
│  └─ run_stage1.py
├─ src/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ models.py
│  ├─ pipeline.py
│  └─ utils.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ latents/
│  └─ logs/
├─ .gitignore
└─ README.md
```

## 실행

```bash
pip install -r configs/requirements.txt
python scripts/prepare_data.py
python scripts/train_stage1.py --config configs/default.yaml
python scripts/evaluate_stage1.py --config configs/default.yaml
```

한 번에:

```bash
python scripts/run_stage1.py --config configs/default.yaml
```

고정 데이터 URL:

`https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link`

`prepare_data.py`는 원본을 `data/raw/`에 두고 학습용 `data/processed/train.pkl` 하나만 생성합니다.

## 현재 상황 표현

현재 상황은 raw column을 그대로 숫자로 넣지 않습니다. 먼저 정보가 중복되는 값들을 canonical state로 정리합니다.

- `balls_before + strikes_before` -> `count_state`
- `runner_on_1b/2b/3b` -> `base_state` 0~7
- `inning + top_bottom` -> `game_phase_state`
- `home_win_expectancy + away_win_expectancy` -> 투수 팀 관점 `pitcher_win_expectancy`
- `score_diff_pitcher_team`, `run_total_before`, `li` 유지
- `outs_before`, `game_type`, `game_month`, `game_dayofweek`, `pitcher_hand`, `batter_hand` 유지

`base_state=7`의 7은 크기를 의미하지 않습니다. 모든 categorical state는 **embedding table index**로만 사용됩니다.

`pitcher_id`, `batter_id`, team ID는 current latent에 넣지 않습니다.

### Context reconstruction

- categorical state -> 각 컬럼별 softmax classification head
- numeric state -> Smooth-L1 reconstruction
- 입력 categorical 일부는 mask, numeric에는 작은 noise를 넣는 denoising 학습

## 과거 이력 표현

기본적으로 `asof_*`를 그 시점의 history snapshot으로 사용합니다.

- `*_n` / count 성격 값: `log1p`
- 나머지 수치형: train split에서만 impute + robust scaling
- `pitcher_id`, `batter_id`, target 미사용
- 시간 순서 sequence를 선험적으로 강제하지 않음

## Split

- 2019~2022: train
- 2023: validation / Stage 1 early stopping
- 2024: holdout reconstruction check

2024는 preprocessing fit이나 early stopping에 사용하지 않습니다.

## Stage 1 평가

```bash
python scripts/evaluate_stage1.py --config configs/default.yaml
```

생성 파일:

- `outputs/logs/stage1_reconstruction_metrics.json`
- `outputs/logs/reconstruction_sample_train.csv`
- `outputs/logs/reconstruction_sample_val.csv`
- `outputs/logs/reconstruction_sample_test.csv`

평가 항목:

- context categorical feature별 reconstruction accuracy
- context numeric normalized MAE / RMSE
- history feature별 normalized MAE / RMSE
- latent 각 축의 표준편차 요약(collapse 확인)

**Stage 2는 이 결과가 충분히 안정적인지 확인한 뒤 추가합니다.**
