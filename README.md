# Tablatent — Stage 1 + Stage 2 probability model

LG Aimers 9기 야구 해커톤에서 pitch 직전의 tabular state를 latent로 압축한 뒤, 비슷한 latent state의 과거 관측으로 local success probability를 만들고 최종 `control_success` 확률을 학습합니다.

## 구조

- `z_context` 16-D: 현재 경기 상황
- `z_history` 16-D: 그 시점까지의 이력 snapshot
- Stage 1: target 없이 AE reconstruction
- Stage 2: frozen 32-D latent + local probability statistics -> 최종 probability

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
│  ├─ evaluate_knn.py
│  ├─ build_stage2_dataset.py
│  └─ train_stage2.py
├─ src/
│  ├─ data.py
│  ├─ models.py
│  ├─ pipeline.py
│  ├─ knn_probability.py
│  ├─ stage2.py
│  └─ utils.py
└─ outputs/
```

## 설치

```bash
pip install -r configs/requirements.txt
```

## Split

- 2019~2022: train
- 2023: validation
- 2024: untouched holdout

## Stage 1

```bash
python scripts/run_stage1.py --config configs/default.yaml
```

Stage 1은 `control_success`를 사용하지 않습니다. `pitcher_id`, `batter_id`, team ID도 latent 입력에서 제외합니다.

고정 데이터 URL:

`https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link`

## Latent kNN sanity baseline

```bash
python scripts/evaluate_knn.py --config configs/default.yaml
```

2019~2022를 neighbor pool로 두고 2023에서 `k={20,50,100,200,500,1000}`의 raw neighbor success mean을 비교합니다. 이 실험은 최종 Stage 2가 아니라 latent neighborhood에 predictive signal이 있는지 보는 baseline입니다.

## Stage 2 dataset

먼저 Stage 1에서 이미 저장된 `outputs/latents/context.npy`, `history.npy`를 사용합니다.

```bash
python scripts/build_stage2_dataset.py --config configs/default.yaml
```

각 row의 Stage 2 feature는 기본적으로 다음입니다.

```text
32-D standardized latent
+ local_prob_k100, local_effective_n_k100, local_radius_k100
+ local_prob_k500, local_effective_n_k500, local_radius_k500
+ local_prob_k1000, local_effective_n_k1000, local_radius_k1000
```

`local_prob`은 단순 0/1 neighbor mean이 아니라 adaptive Gaussian distance weight를 적용한 뒤 train-pool global probability로 empirical-Bayes shrinkage한 값입니다.

```text
p_local = (sum(w * y) + alpha * p_global) / (sum(w) + alpha)
```

기본 `alpha=50`입니다. `effective_n`과 `radius`도 함께 저장하여 Stage 2가 local probability의 신뢰도를 판단할 수 있게 합니다.

### Train leakage 방지

2019~2022 Stage 2 train row는 5-fold cross-fitting으로 local feature를 만듭니다. 한 train row의 `local_prob`을 만들 때 그 row가 속한 fold의 `control_success`는 neighbor pool에서 전부 제외합니다.

2023 local feature는 2019~2022만 neighbor pool로 사용합니다. 2024는 기본 실행에서 Stage 2 dataset으로도 만들지 않습니다.

생성 파일:

```text
outputs/stage2/train_features.npy
outputs/stage2/train_target.npy
outputs/stage2/train_global_index.npy
outputs/stage2/val_features.npy
outputs/stage2/val_target.npy
outputs/stage2/val_global_index.npy
outputs/stage2/metadata.json
```

2024 feature까지 실제로 만들 준비가 되었을 때만:

```bash
python scripts/build_stage2_dataset.py --config configs/default.yaml --include-test
```

## Stage 2 training

```bash
python scripts/train_stage2.py --config configs/default.yaml
```

기본 모델은 `local_prob_k1000`을 empirical prior로 두는 residual probability MLP입니다.

```text
base_logit = logit(local_prob_k1000)
final_logit = base_logit + MLP(z, local statistics)
p = sigmoid(final_logit)
```

마지막 correction head를 0으로 초기화하므로 학습 시작 시점의 출력은 정확히 local empirical probability입니다. 이후 latent와 neighborhood reliability가 설명하는 만큼만 correction을 학습합니다.

기본 loss는 Brier loss이며 2023 validation Brier로 early stopping합니다. 비교값으로 train-mean baseline과 local-prior Brier를 같이 출력합니다.

주요 출력:

```text
outputs/stage2/stage2_best.pt
outputs/stage2/stage2_metrics.json
outputs/stage2/stage2_val_predictions.csv
```
