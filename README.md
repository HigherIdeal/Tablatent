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

## Stage 1 cache

Stage 1을 다시 학습하지 않도록 Google Drive 또는 로컬 동기화 Drive 경로에 Stage 1 결과를 보존할 수 있습니다.

```bash
python scripts/stage1_cache.py push
python scripts/stage1_cache.py pull
```

Windows Google Drive처럼 별도 경로를 쓰면:

```powershell
python scripts/stage1_cache.py push --drive-dir "G:\내 드라이브\학습\LG Aimers 9기\해커톤\오프라인\data\Tablatent\stage1_cache"
python scripts/stage1_cache.py pull --drive-dir "G:\내 드라이브\학습\LG Aimers 9기\해커톤\오프라인\data\Tablatent\stage1_cache"
```

기본 cache에는 다음을 포함합니다.

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

각 파일은 SHA256 manifest로 검증합니다. processed dataset을 Drive에 저장하고 싶지 않으면 `--exclude-data`를 사용합니다.

## Stage 2 — current experiment: CatBoost on frozen latent

현재 기본 Stage 2는 Stage 1에서 만든 posterior mean만 사용합니다.

```text
mu_context 16D
mu_history 16D
     ↓ concat
32 numeric latent features
     ↓
CatBoostClassifier
     ↓
control_success probability
```

제약:

- Stage 1 frozen
- raw feature 미사용
- player/team ID 미사용
- local probability 미사용
- 2024 holdout 미사용
- latent standardization 미사용

GPU 학습과 early stopping은 `Logloss`를 사용하고, 최종 선택된 모델의 `predict_proba`로 validation Brier를 별도로 계산해 주 평가값으로 기록합니다.

```bash
python scripts/train_stage2.py --config configs/default.yaml --head catboost
```

현재 default head도 `catboost`입니다.

주요 출력:

```text
outputs/stage2_catboost/stage2_catboost.cbm
outputs/stage2_catboost/metrics.json
outputs/stage2_catboost/validation_predictions.csv
outputs/stage2_catboost/feature_importance.csv
```

현재 2023 validation 결과:

```text
best iteration  23
BCE             0.69242962
Brier           0.24964003
AUC             0.530320
official-style  143.99
```

## Stage 2 comparison probes

기존 probe는 삭제하지 않고 같은 frozen latent와 temporal split에서 비교합니다.

```bash
python scripts/train_stage2.py --config configs/default.yaml --head linear
python scripts/train_stage2.py --config configs/default.yaml --head mlp
python scripts/train_stage2.py --config configs/default.yaml --head bilinear
```

현재 기록된 validation Brier:

```text
CatBoost   0.24964003
linear     0.25032231
bilinear   0.25070280
mlp        0.25137261
```

## Diagnostic leaderboard submission

현재 CatBoost-on-latent 모델을 실제 2025 evaluator에서 확인하기 위한 진단용 `submit.zip`을 만들 수 있습니다. 이 패키지는 현재 개발 artifact를 그대로 사용하며 **최종 2019~2024 재학습 모델이 아닙니다.**

```powershell
python scripts/build_submission.py
```

출력:

```text
dist/submit.zip
```

ZIP 최상위 구조는 DACON 코드 제출 형식에 맞게 고정됩니다.

```text
submit.zip
├─ model/
├─ script.py
└─ requirements.txt
```

`model/`에는 현재 Stage1 context/history VAE checkpoint, train-fit preprocessor, CatBoost Stage2 모델과 inference에 필요한 최소 `src` 정의만 포함합니다. `script.py`는 서버의 `test.csv` 각 행을 학습 당시 preprocessor로 변환해 32D posterior mean을 만들고 CatBoost 확률을 계산한 뒤 `output/submission.csv`를 생성합니다.

공식 5행 샘플을 가진 로컬 디렉터리가 있으면 ZIP 생성 전에 end-to-end smoke test도 할 수 있습니다.

```powershell
python scripts/build_submission.py --smoke-data-dir "C:\path\to\official\data"
```

추론 패키지의 `requirements.txt`에는 평가 서버 기본 설치 패키지를 중복 설치하지 않고 `catboost==1.2.10`만 넣습니다.

## Legacy / diagnostic experiments

`evaluate_knn.py`, `build_stage2_dataset.py`, `src/stage2.py`, `src/stage2_regularized.py`는 latent neighborhood와 local-probability 실험을 재현하기 위해 유지합니다.

```bash
python scripts/evaluate_knn.py --config configs/default.yaml
```
