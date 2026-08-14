# Tablatent — Stage 1 + latent kNN baseline

LG Aimers 9기 야구 해커톤에서 tabular state를 latent로 압축하고, 먼저 reconstruction을 검증한 뒤 latent 이웃의 실제 성공률로 `control_success` 확률을 예측합니다.

## 현재 구조

한 투구 직전의 정보를 두 상태로 분리합니다.

1. `z_context`: 현재 경기 상황
2. `z_history`: 그 시점까지의 과거 이력 snapshot

Stage 1 encoder는 reconstruction으로 학습되고, `control_success`는 Stage 1 학습에 사용하지 않습니다.

```text
Tablatent/
├─ configs/
│  ├─ default.yaml
│  └─ requirements.txt
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train_stage1.py
│  ├─ evaluate_stage1.py
│  ├─ run_stage1.py
│  └─ evaluate_knn.py
├─ src/
│  ├─ data.py
│  ├─ models.py
│  ├─ pipeline.py
│  ├─ knn_probability.py
│  └─ utils.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ latents/
│  └─ logs/
└─ README.md
```

## 설치

```bash
pip install -r configs/requirements.txt
```

## Stage 1

```bash
python scripts/run_stage1.py --config configs/default.yaml
```

또는 단계별로:

```bash
python scripts/prepare_data.py
python scripts/train_stage1.py --config configs/default.yaml
python scripts/evaluate_stage1.py --config configs/default.yaml
```

고정 데이터 URL:

`https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link`

## 현재 상황 표현

현재 상황은 raw column을 그대로 숫자로 넣지 않고 canonical state로 정리합니다.

- `balls_before + strikes_before` -> `count_state`
- `runner_on_1b/2b/3b` -> `base_state`
- `inning + top_bottom` -> `game_phase_state`
- `home_win_expectancy + away_win_expectancy` -> 투수 팀 관점 `pitcher_win_expectancy`
- `score_diff_pitcher_team`, `run_total_before`, `li` 유지
- `outs_before`, `game_type`, `game_month`, `game_dayofweek`, `pitcher_hand`, `batter_hand` 유지
- `pitcher_id`, `batter_id`, team ID는 latent 입력에서 제외

Categorical state는 embedding lookup으로 처리합니다.

## 과거 이력 표현

기본적으로 `asof_*`를 그 시점의 history snapshot으로 사용합니다.

- `*_n` / count 성격 값: `log1p`
- 나머지 수치형: train split에서만 impute + robust scaling
- ID와 target 미사용

## Split

- 2019~2022: train
- 2023: validation / Stage 1 early stopping / kNN의 k 선택
- 2024: holdout

## Stage 1 평가

```bash
python scripts/evaluate_stage1.py --config configs/default.yaml
```

주요 출력:

- `outputs/logs/stage1_reconstruction_metrics.json`
- `outputs/logs/reconstruction_sample_train.csv`
- `outputs/logs/reconstruction_sample_val.csv`
- `outputs/logs/reconstruction_sample_test.csv`

## Latent kNN probability baseline

Stage 1에서 저장한 `context.npy`와 `history.npy`를 이어 붙여 32차원 latent를 만들고, train split에서 각 latent dimension을 표준화합니다. 이후 2019~2022 latent를 neighbor pool로 사용합니다.

2023 validation:

```bash
python scripts/evaluate_knn.py --config configs/default.yaml
```

기본 `k` 후보:

```text
20, 50, 100, 200, 500, 1000
```

각 query의 가장 가까운 `k`개 train row에서 `control_success` 평균을 확률로 사용합니다.

```text
p(control_success=1 | z) = mean(neighbor labels)
```

출력:

- k별 Brier score
- train-mean baseline 대비 skill
- best k
- best-k calibration
- `outputs/logs/knn_validation_predictions.csv`
- `outputs/logs/knn_neighbor_examples.csv`
- `outputs/logs/knn_probability_metrics.json`

2023에서 best k를 고른 뒤 2024 holdout까지 한 번 평가하려면:

```bash
python scripts/evaluate_knn.py --config configs/default.yaml --test
```

기본 neighbor search는 대용량 데이터 때문에 FAISS `IVF-Flat`을 사용합니다. GPU FAISS가 설치된 환경이면 자동으로 GPU를 시도하고, 그렇지 않으면 CPU를 사용합니다.
