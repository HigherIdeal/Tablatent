# CODEX HANDOFF — LG Aimers 9기 야구 해커톤 / Tablatent

> **Snapshot date:** 2026-08-16  
> **Purpose:** Codex가 이 문서 하나만 읽고도 현재 문제 정의, 데이터 제약, 검증 철학, 모델 구조, 실패한 실험, 현재 최고 기준점, 서버 환경, Git 상태, 앞으로의 공격적 실험 우선순위를 이해하고 바로 작업을 이어갈 수 있도록 만든 handoff 문서다.  
> **Current repository:** `HigherIdeal/Tablatent`  
> **Active branch:** `agent/stable-player-dynamics-gru`  
> **Draft PR:** `#39` — `Experiment: regime adaptation results and GPU2 server profile`  
> **PR head at handoff:** `b644b92c0a57a15f9471218ca97d6e343e5010f3`

---

# Page 1 / 7 — 문제 정의, 절대 제약, Codex 작업 계약

## 1. 문제의 본질

이 프로젝트는 LG Aimers 9기 × LG스포츠 야구 해커톤의 **투구 단위 확률 예측 문제**다. 한 행은 현재 투구가 실제로 던져지기 **직전**의 상태를 나타낸다. 모델의 출력은 이진 분류 결과가 아니라 다음 조건부 확률이다.

```text
p_i = P(control_success = 1 | 현재 투구 직전까지 알 수 있는 정보)
```

`control_success`는 train에서 0/1 target으로 제공되지만, 제출은 각 행마다 `[0, 1]` 실수 확률을 출력한다. 따라서 핵심은 Accuracy가 아니라 **확률 품질과 calibration**이다.

공식 강의가 강조하는 문제의 목적은 단순히 “스트라이크를 던졌는가”가 아니다. 좋은 제구는 경기 상황과 요구된 방향까지 포함한 개념이며, 현장에서는 “주자가 있을 때 제구가 흔들리는가”, “불리한 카운트에서도 유지되는가”, “경기 후반에 안정적인가”, “최근 제구 상태가 좋아지는가”처럼 맥락에 따른 제구력 정량화가 더 중요하다.

## 2. 가장 중요한 hard constraint — hidden test row independence

2025 hidden test의 **각 행은 독립 예측 대상**으로 취급해야 한다. 최종 inference는 사실상 아래 형태여야 한다.

```text
p_i = f(x_i ; theta_train)
```

여기서 `x_i`는 현재 test row 하나이고, `theta_train`은 오직 허용된 2019~2024 학습 데이터로 고정된 파라미터/lookup/preprocessor/artifact다.

### 절대 금지

- 다른 hidden-test row를 이용한 groupby/aggregation
- hidden-test 전체 분포를 보고 prior를 재추정
- test-time batch normalization/statistics
- TENT/SAR 등 test batch adaptation
- 이전 hidden-test 행을 이용한 rolling/expanding state
- hidden-test row끼리 kNN/cluster/context 구성
- HMM/BOCPD state를 2025 test sequence를 읽으며 업데이트
- test 내 선수 빈도/팀 빈도/월별 분포를 사용한 보정
- 현재 투구 이후에 확정되는 실제 구종, 실제 위치, 판정, Trackman 측정값, target 사용

훈련 과정에서 regime/HMM/OOF 통계를 발견하는 것은 가능하지만, 그 결과가 최종 inference에 들어가려면 **현재 row 하나로 계산 가능한 고정 함수**로 변환되어야 한다.

## 3. 외부 데이터 및 익명 ID 원칙

- 공식 제공 데이터 외 외부 야구 데이터는 사용하지 않는다.
- 익명 `pitcher_id`, `batter_id`, 팀 ID를 외부 DB로 역식별하지 않는다.
- `trackman_history.csv`의 `pitcher_trackman_id`와 main table `pitcher_id`가 숫자가 같다고 해서 같은 ID 체계라고 가정하지 않는다.
- `game_type=R/F`, hand 코드의 구체적 의미처럼 공식 문서가 직접 정의하지 않은 것은 임의로 확정하지 않는다.

## 4. Codex에게 기대하는 작업 방식

이 문서는 **2026-08-16 시점의 최신 의사결정**을 나타낸다. repo 안의 더 오래된 `Strategy.md`, Stage 계획, VAE 문서와 충돌한다면 이 handoff와 최신 `README.md`, `EXPERIMENT_STATUS_2026-08-16.md`, 현재 branch 코드를 우선한다.

Codex는 다음 원칙으로 작업한다.

1. 먼저 현재 branch 코드와 결과 파일을 확인하고, 이미 실패한 방향을 무의미하게 반복하지 않는다.
2. 실험은 반드시 temporal validation을 유지한다.
3. 2024는 이미 반복적으로 사용되었으므로 더 이상 pristine holdout으로 주장하지 않는다.
4. score가 조금 좋아졌다는 이유만으로 승격하지 않는다. fold consistency와 재현성을 본다.
5. 새 feature/model은 hidden-test row independence를 먼저 증명한 뒤 구현한다.
6. 공격적으로 탐색하되, baseline을 항상 보존하고 결과를 CSV/JSON으로 저장한다.
7. 장시간 GPU 작업은 **physical GPU 2만** 사용한다.
8. 실험 파일과 결과의 의미를 영어 filename/experiment name으로 명확히 남기고, 사용자에게 설명할 때는 한국어로 핵심부터 말한다.

---

# Page 2 / 7 — 데이터 구조, 지표, 시간적 변화

## 5. 공식 데이터

### Main train

```text
train.csv
1,475,092 rows x 49 columns
= 48 input columns + control_success
2019~2024 seasons
```

### Local test/sample

배포 `test.csv`는 실제 평가 데이터가 아니라 형식 확인용 5행이다. 실제 평가 서버는 2025 hidden test로 교체한다.

```text
actual hidden test: 245,789 rows
season: 2025
```

### Trackman history

```text
trackman_history.csv
1,793,078 rows x 30 columns
2019~2024
```

Trackman은 현재 main row와 1:1 정답 테이블이 아니다. 구종/구속/회전/무브먼트/릴리스 등의 **과거 물리 로그**이며, 사용할 경우 반드시 시점과 linkage를 검증해야 한다.

## 6. main row에서 사용 가능한 신호의 큰 축

공식 강의 관점에서 데이터는 대략 네 축이다.

1. **Game context** — inning, count, score, runners, top/bottom 등
2. **Importance / pressure** — win expectancy, leverage index `li`
3. **Historical / as-of state** — 투수/타자의 누적 성공률, 최근 1/3/5 경기 상태, pitch mix
4. **Historical Trackman physical information** — velocity, spin, movement, release, extension 등

현재 주력 모델은 1~3을 직접 사용하고 있으며, 4는 아직 제대로 된 큰 승리로 연결하지 못한 **고가치 미개척 축**이다.

## 7. 시즌별 train 규모와 target rate

| Season | Rows | `control_success` rate |
|---:|---:|---:|
| 2019 | 237,413 | 0.56467 |
| 2020 | 244,087 | 0.53271 |
| 2021 | 247,088 | 0.53276 |
| 2022 | 247,472 | 0.52892 |
| 2023 | 245,525 | 0.49996 |
| 2024 | 253,507 | 0.48610 |

전체 성공률은 약 0.5238이고 심한 class imbalance는 없다. 오히려 2019→2024의 base-rate 하락이 중요하다.

## 8. Brier / official-style score

기본 metric은 Brier Score다.

```text
Brier = mean((p_i - y_i)^2)
```

공식 leaderboard는 해당 hidden set target mean `r`을 사용한 Brier Skill Score를 100,000 scale로 표현한다.

```text
reference_brier = r * (1-r)
score = max(0, 100000 * (1 - brier / reference_brier))
```

따라서 local에서 가장 신뢰할 숫자는 **Brier 자체**다. 서로 다른 validation 기간은 target mean이 다르므로 raw score를 단순 절대 비교하지 않는다.

## 9. 데이터 감사에서 반드시 기억할 것

### CSV 행 순서는 완전한 실제 시간축이 아니다

자체 감사에서 동일 투수의 인접 row에서도 월이 뒤로 가는 사례가 확인되었다. 즉 `row_id`나 CSV order를 그대로 실제 chronological pitch sequence로 보면 안 된다.

따라서 임의의:

```text
row-order rolling
row-order GRU sequence
row-order expanding target state
```

같은 것은 위험하다.

### exact duplicate/redundant state가 많다

예:

```text
run_total_before = run_top_before + run_bot_before
score_diff_home = run_bot_before - run_top_before
num_runners_on = runner_on_1b + runner_on_2b + runner_on_3b
base_state = runner flags의 결정적 표현
```

그래서 현재 canonical feature policy는 exact deterministic duplicate를 제거한다.

## 10. `game_type` structural change

`game_type`의 공식 의미는 공개되지 않았다. 하지만 데이터에서는 강한 변화가 있다.

```text
F success rate
2022: ~0.7087
2023: ~0.4729
```

즉 2022→2023에서 약 -23.6%p 급락한다. 이 변화는 특정 팀 하나가 아니라 넓게 관찰된다.

중요한 결론:

- `game_type`은 **삭제하면 안 된다**.
- raw `game_type`을 유지한 모델이 실제 2025 leaderboard에서도 drop보다 더 좋았다.
- 과거 실험에서 recent 2023~2024 + raw `game_type` 제출이 drop-game-type보다 우세했다.
- 그러나 “F가 무엇이다” 같은 의미 해석은 금지한다.

---

# Page 3 / 7 — 현재 feature policy와 모델 backbone

## 11. Canonical feature policy

현재 main CatBoost는 `src/canonical_features.py`의 canonical feature set을 기준으로 한다.

### 유지하는 대표 feature

- `season`, `game_month`, `game_dayofweek`
- `inning`, `top_bottom`, `game_type`
- `balls_before`, `strikes_before`, `outs_before`
- `run_total_before`, `score_diff_home`
- `base_state`
- pitcher-team perspective win expectancy
- `li`
- pitcher/batter hand
- pitcher/batter team IDs
- `asof_pitcher_*` cumulative success/reverse/middle/ball/strike
- prev1/3/5 success and middle rates
- batter cumulative state
- pitcher pitch-mix rates
- 일부 leakage-safe derived as-of dynamics

### 현재 제외

- `row_id`
- `pitcher_id`
- `batter_id`
- exact deterministic duplicate state columns

투수/타자 ID는 temporal ablation에서 generalization을 악화시켜 현재 canonical predictor에서 제외되었다. **팀 ID는 현재 general experts에서 유지**된다.

## 12. 현재 deployable backbone

현재 가장 강하고 안정적인 구조는 CatBoost 기반 3-expert mixture다.

```text
                full-history raw-game_type CatBoost
                            |
                            | (1-alpha)
                            |
                            +------ base prediction ------+
                            |                              |
                            | alpha                        | if game_type == R
                            |                              v
                recent raw-game_type CatBoost      R-fast specialist
                                                    beta blend
```

수식으로는:

```text
p_base = (1-alpha_recent) * p_full + alpha_recent * p_recent

if game_type == R:
    p_final = (1-beta_r) * p_base + beta_r * p_r_fast
else:
    p_final = p_base
```

현재 기준:

```text
alpha_recent = 0.20
beta_r       = 0.10
```

### Final 2025 training interpretation

- `full` expert: 2019~2024 전체 사용
- `recent` expert: 2023+2024 recent regime 사용
- `R-fast`: recent R rows 중심 specialist

최근 regime weighting/HMM 결과까지 고려하면 **recent expert는 full을 대체하는 모델이 아니라 small complementary correction**으로 해석한다.

## 13. R-fast feature philosophy

현재 `run_gated_r_specialist_suite.py`의 `r_fast`는 recent R rows만 학습하며 약 17개 feature를 사용한다. 대표적으로:

```text
game_month
inning
top_bottom
balls_before
strikes_before
outs_before
base_state
li
pitcher_hand
batter_hand
asof_pitcher_n
asof_pitcher_success_rate
asof_pitcher_ball_rate
asof_pitcher_strike_rate
asof_pitcher_fastball_rate
asof_pitcher_breaking_rate
asof_pitcher_offspeed_rate
```

specialist 단독 성능은 general expert보다 약해도 된다. 핵심은 **error diversity가 있어 R subset에 10% 정도 섞었을 때 전체 Brier가 줄어드는가**다.

## 14. 현재 3개 proxy validation fold

주요 model selection은 2024를 세 가지 시간 관점으로 보는 established proxy suite를 쓴다.

```text
season_forward_2024   weight = 0.50
mid_2024              weight = 0.20
late_2024             weight = 0.30
```

`season_forward_2024`가 2025 deployment에 가장 가까운 proxy다. `mid/late`는 2024의 earlier observed rows를 training에 포함하는 expanding-window 성격이므로 hidden 2025 row independence의 정확한 복제는 아니며 robustness 보조용이다.

중요: 2024는 이미 수많은 실험에 사용했기 때문에 더 이상 “untouched holdout”이라고 부르면 안 된다.

---

# Page 4 / 7 — 지금까지의 핵심 실험 결과와 폐기된 방향

## 15. 최신 baseline 재현 결과 — RTX 4090 server

명령:

```bash
python scripts/run_gated_r_specialist_suite.py \
  --config configs/default.yaml \
  --iterations 500 \
  --task-type GPU \
  --devices 0
```

실제로 server에서 재현된 weighted expert summary:

| Expert | Weighted Brier |
|---|---:|
| `full_raw` | 0.24792363 |
| `stable_drop_gt` | 0.24799039 |
| `recent_raw` | 0.24814370 |
| `r_full` | 0.24847725 |
| `r_both` | 0.24852254 |
| `r_fast` | 0.24863254 |
| `r_range` | 0.24872297 |

현재 best gated configuration:

```text
base_mode             = full_recent
specialist            = r_fast
old_iterations        = 500
recent_iterations     = 500
specialist_iterations = 500
alpha_recent          = 0.2
beta_r                = 0.1
weighted_brier        = 0.24789305
weighted_raw_score    = +741.44
```

`full_raw` 대비 개선은 작지만 반복적으로 확인되었다. 이 값이 현재 **reference baseline**이다.

### Fold별 최적은 완전히 동일하지 않다

```text
season_forward_2024 : full_recent + r_fast, alpha=.2, beta=.1
mid_2024            : full_recent + r_fast, alpha=.2, beta=.2
late_2024           : stable_recent + r_range 쪽이 일부 우세
```

따라서 한 fold만 보고 specialist 구조를 바꾸면 안 된다.

## 16. Dynamic row-wise gate — 실패 / 승격 금지

full과 recent 중 어떤 expert를 더 믿을지 row-wise gate CatBoost로 예측했다.

OOF advantage target:

```text
A_i = (y_i - p_full_i)^2 - (y_i - p_recent_i)^2
```

결과:

```text
outputs_only weighted Brier = 0.24792259
fixed       weighted Brier = 0.24792295
delta                     = -0.00000036
improved folds            = 1/3
row_context strength      = 0
```

결론: 개선이 noise scale이다. **현재 dynamic gate는 사용하지 않는다.**

## 17. HMM latent regime — 분석용으로 종료

첫 HMM은 43개월을 하나의 연속 sequence로 취급해 2024-only state처럼 보이는 결과를 냈다. 이후 robustness 실험에서 시즌을 독립 HMM sequence로 분리했다.

```text
lengths = [8, 6, 7, 7, 7, 8]
```

robustness 결과:

```text
2023 overlap with 2024 dominant state = 1.000
2024 purity                            = 0.875 or 1.000
partition ARI mean                     = 0.457
median                                 = 0.572
min                                    = -0.034
```

결론:

- 2024-only regime은 robust하지 않다.
- 더 큰 반복 패턴은 **2023 ≈ 2024 recent environment**다.
- 정확한 HMM state partition은 불안정하므로 prediction feature로 사용하지 않는다.
- hidden test에서 sequential HMM adaptation은 금지다.

## 18. Smooth recency weighting — 실패

full-history expert의 오래된 시즌에 exponential decay weight를 줬다.

| Half-life | Weighted Brier | Delta vs unweighted |
|---:|---:|---:|
| 0 | **0.24791945** | 0 |
| 60 | 0.24794424 | +0.00002479 |
| 36 | 0.24794656 | +0.00002711 |
| 24 | 0.24797332 | +0.00005387 |
| 12 | 0.24801205 | +0.00009260 |

모든 weighting이 전체적으로 더 나빴다. 즉 old seasons는 여전히 variance reduction/generalization에 기여한다.

## 19. GRU / Tiny Transformer — 더 투자하지 않음

과거 temporal branch 실험:

- GRU는 2023에서만 약간 나아졌지만 2024 robustness가 약했다.
- regime-aware GRU는 weighted improvement가 약 `1e-5` 이하 수준이었다.
- Tiny Transformer는 weighted 기준 개선하지 못했다.
- 더 중요한 문제: 이전 hidden-test row를 이용해 state를 업데이트하면 대회 row-independence 위반이다.

따라서 **sequence model을 그대로 final inference에 넣는 방향은 종료**한다. 현재 row의 공식 as-of field만으로 state가 계산되는 독립 구조라면 별도 새로운 아이디어로 다시 검토할 수 있다.

## 20. 오래된 latent/VAE branch

VAE/latent posterior mean → CatBoost, neural head, kNN 등도 과거에 시도했으나 현재 strong raw CatBoost보다 약했다. repo에 재현 코드는 남아 있지만 **현재 reference path가 아니다.** 새 아이디어가 실제로 strong CatBoost를 이길 명확한 근거가 없다면 이 branch를 기본값으로 부활시키지 않는다.

---

# Page 5 / 7 — Git/repo 구조와 현재 서버 환경

## 21. Git 상태

```text
repository : HigherIdeal/Tablatent
branch     : agent/stable-player-dynamics-gru
PR         : #39 (draft, open, mergeable)
base       : main
head       : b644b92c0a57a15f9471218ca97d6e343e5010f3
```

이 branch에는 regime/HMM/dynamic-gate/recency-weighting 실험과 4090 server bootstrap 문서가 포함되어 있다.

핵심 파일:

```text
README.md
EXPERIMENT_STATUS_2026-08-16.md
configs/default.yaml
configs/requirements.txt
src/canonical_features.py
src/evaluation_metrics.py
src/data.py
scripts/run_2025_proxy_validation.py
scripts/run_gated_r_specialist_suite.py
scripts/build_gated_r_specialist_submission.py
scripts/run_rowwise_dynamic_gate.py
scripts/run_latent_regime_hmm.py
scripts/run_latent_regime_robustness.py
scripts/run_recency_weighted_full_expert.py
scripts/run_context_interaction_screen.py
scripts/activate_gpu2.sh
scripts/check_gpu2_environment.py
scripts/bootstrap_4090_gpu2.sh
docs/server_4090_gpu2.md
```

## 22. 새 compute server

현재 사용 서버:

```text
host context : iclab4GPU
user         : kjw
repo path    : ~/Aimers/Tablatent
GPU count    : 4 x NVIDIA GeForce RTX 4090
project GPU  : physical GPU 2 ONLY
```

Conda environment:

```text
env        : tablatent
Python     : 3.11.15
PyTorch    : 2.7.1+cu128
CatBoost   : 1.2.10
GPU memory : 24,564 MiB physical / ~23.65 GiB visible
```

환경 smoke test는 이미 통과했다.

```text
CUDA_VISIBLE_DEVICES='2'
torch.cuda.is_available=True
torch.cuda.device_count=1
logical_cuda0=NVIDIA GeForce RTX 4090
torch CUDA smoke=OK
CatBoost GPU smoke=OK
```

## 23. GPU 번호 규칙 — 매우 중요

매 shell에서:

```bash
conda activate tablatent
source scripts/activate_gpu2.sh
```

`activate_gpu2.sh`는 다음을 설정한다.

```bash
CUDA_VISIBLE_DEVICES=2
```

그러면 **physical GPU 2가 process-local GPU 0으로 재번호화**된다.

따라서 프로젝트 명령은:

```text
PyTorch   -> cuda:0
CatBoost  -> --devices 0
```

이어야 한다. `source scripts/activate_gpu2.sh` 이후 `--devices 2`를 넘기면 안 된다.

## 24. Data preparation

현재 `prepare_data.py`는 `--config`를 받지 않는다. 올바른 명령은:

```bash
python scripts/prepare_data.py
```

processed output:

```text
data/processed/train.pkl
```

고정 데이터 source URL:

```text
https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link
```

## 25. shared Anaconda 주의

서버의 base Anaconda는 `/home/iclab/anaconda3`에 있고 read-only다. bootstrap은 사용자 writable path를 사용하도록 수정되어 있다.

```text
$HOME/.conda/pkgs
$HOME/.conda/envs
```

Conda 환경이 이미 정상 생성되었으므로 이 이슈는 해결 상태다.

## 26. final submission runtime 환경은 개발 서버와 다르다

개발은 4090이지만 최종 평가 서버는 더 제한적이다. 최종 package는 대략 다음 제약을 만족해야 한다.

```text
Ubuntu 22.04.x
Python 3.11.15
6 vCPU
28 GB RAM
NVIDIA L4 ~22.4 GiB
CUDA 12.8
inference <= 10 min
package installation <= 10 min
```

따라서 4090에서 큰 실험을 해도 최종 inference model은 L4 환경에서 실행 가능한지 반드시 마지막에 검증해야 한다.

---

# Page 6 / 7 — 이제부터의 공격적 실험 계획

## 27. 목표

현재 기준 `weighted_brier = 0.24789305`를 단순 noise가 아니라 **재현 가능한 구조적 개선**으로 넘어서는 것이 목표다.

앞으로는 `1e-6` 규모의 우연한 improvement보다 다음을 선호한다.

```text
최소 기대 개선 규모: 1e-5 ~ 1e-4 Brier
+ temporal fold consistency
+ seed robustness
+ deployment legality
```

컴퓨팅 파워가 충분하므로 탐색은 넓게 하되, 검증은 더 엄격하게 한다.

## 28. Priority A — tree count / blend grid 확대

현재 suite는 한 번 최대 tree까지 fit한 뒤 `ntree_end` prefix를 재사용하므로 iteration grid 확장은 상대적으로 싸다.

첫 aggressive sweep 예시:

```bash
python scripts/run_gated_r_specialist_suite.py \
  --config configs/default.yaml \
  --iterations-grid 500,750,1000,1500 \
  --alpha-step 0.05 \
  --beta-step 0.05 \
  --beta-max 0.4 \
  --task-type GPU \
  --devices 0 \
  --thread-count 16 \
  --output-dir outputs/gated_r_specialist_aggressive
```

주의: prediction cache가 이전 grid와 호환되지 않으면 새 output dir을 사용한다.

## 29. Priority B — CatBoost HPO 대규모 탐색

지금까지 성능은 특정 CatBoost parameter family 주변에서 얻은 것이다. 4090을 이용해 다음 축을 temporal folds로 체계적으로 탐색할 가치가 높다.

```text
depth                 : 5,6,7,8,9,10
learning_rate         : ~0.01 ... 0.08
l2_leaf_reg           : broad log-scale
random_strength       : 0 ... 2
bagging_temperature   : 0 ... 2
border_count          : 64,128,254 등
one_hot_max_size      : categorical sensitivity
iterations            : learning_rate와 연동
```

단, GPU CatBoost에서 parameter 조합별 지원 제약을 확인한다. 모든 후보를 무식하게 Cartesian product로 돌리기보다 random/Optuna-like search 또는 coarse→refine 형태가 낫다.

### HPO objective

단일 fold best가 아니라:

```text
primary   = weighted temporal Brier
secondary = season_forward_2024 Brier
penalty   = worst-fold regression
```

으로 선택한다.

## 30. Priority C — seed ensemble

CatBoost GPU는 seed에 따라 작은 variance가 있다. 컴퓨팅이 충분할 때 가장 저위험 개선 중 하나다.

계획:

```text
1. HPO로 1~3개의 parameter family를 고정
2. seed 5~10개 학습
3. 각 expert(full/recent/R-fast) 내부에서 probability average
4. 그 후 alpha/beta 재탐색
5. single-seed 대비 fold별 Brier와 calibration 비교
```

Brier는 평균 예측의 variance reduction에서 직접 이득을 볼 가능성이 있다. 단, seed ensemble로 개선이 `1e-6` 정도라면 최종 package complexity를 늘릴 이유가 없다.

## 31. Priority D — residual-driven specialists

현재 R-fast가 성공한 핵심은 “강한 general expert + 특정 coarse segment specialist + 작은 gate” 구조다. 이를 일반화한다.

OOF residual을 이용해 다음 coarse segment에서 base model의 systematic error를 찾는다.

```text
count state
pitcher/batter hand matchup
pitcher experience / asof_pitcher_n bucket
base_state + outs
high-LI / low-LI
inning / late-game
team/context combinations
game_month
game_type
```

중요: row-wise noisy gate를 다시 만들지 말고, **충분히 큰 segment에서 반복되는 residual bias**를 먼저 확인한다.

예:

```text
if segment S에서
  specialist - base Brier improvement가
  여러 temporal fold에서 같은 방향이고
  sample size가 충분하면
small beta gated specialist로 승격
```

## 32. Priority E — Trackman physical profile

가장 큰 신규 정보 후보다.

Trackman에서 투수의 과거 물리 profile을 만들 수 있다.

```text
pitch type group proportions
rel_speed mean/std/quantiles
spin_rate mean/std
induced_vert_break
horz_break
extension
rel_height
rel_side
zone_speed
recent vs long-term physical change
```

하지만 linkage가 핵심 난점이다. main `pitcher_id`와 Trackman ID를 숫자 직접 join하면 안 된다.

Trackman 실험을 할 경우 Codex는 먼저:

1. 공식 데이터만으로 안전한 linkage 가능성을 감사한다.
2. linkage confidence를 수치화한다.
3. 2024 validation feature 생성 시 미래 2024 Trackman을 보지 않는 strict cutoff 버전을 만든다.
4. 2025 final에서는 2024까지의 고정 profile만 사용한다.
5. hidden-test row를 전혀 사용하지 않는다.

Trackman이 직접 join 불가라면 억지 mapping을 만들지 말고, teacher/representation 방식도 고려할 수 있다. 단 과거 latent branch는 실패했으므로 새로운 방식은 strong baseline 대비 정보 이득이 명확해야 한다.

---

# Page 7 / 7 — 실험 승격 규칙, 하지 말아야 할 것, Codex 즉시 시작 체크리스트

## 33. Promotion rule

새 실험이 baseline을 대체/추가하려면 최소한 다음을 만족해야 한다.

### 반드시 확인

```text
1. weighted temporal Brier 개선
2. season_forward_2024에서 큰 regression 없음
3. 3개 fold 중 다수에서 같은 방향
4. seed를 바꿔도 유지되거나 variance가 작음
5. current-row-only inference 보장
6. final L4 runtime/VRAM 현실성
7. 결과 CSV + metadata + command가 재현 가능
```

단일 fold에서만 좋은 모델, 특정 seed에서만 좋은 모델, hidden-test distribution을 필요로 하는 모델은 승격하지 않는다.

## 34. 현재 명시적으로 닫은 방향

아래는 새로운 증거가 없으면 다시 파지 않는다.

```text
- generic VAE latent -> CatBoost
- GRU sequential hidden-test state
- Tiny Transformer sequential state
- HMM state as direct prediction feature
- HMM/BOCPD test-time filtering
- TENT/SAR/test-time adaptation
- hidden-test prior estimation
- smooth recency-weighted full expert
- current row-wise dynamic gate implementation
- dropping raw game_type
- pitcher_id/batter_id를 그대로 main canonical predictor에 복귀
```

## 35. 해석상 중요한 결론

지금까지의 실험이 말하는 큰 그림:

```text
Old data is still useful.
2023~2024 form a useful broad recent environment.
Recent-only is weaker than full-history.
Recent expert works as a correction, not a replacement.
A small R-specific specialist can add error diversity.
Generic temporal adaptation has not paid off.
The next gains should come from better conditional structure,
better CatBoost optimization/ensembling, or genuinely new information.
```

## 36. 실험 로그 작성 규칙

가능하면 각 experiment마다 아래를 저장한다.

```text
experiment_id
commit SHA
config / CLI
seed
dataset fingerprint
feature list
fold definition
per-fold Brier
weighted Brier
raw score (diagnostic)
subset metrics (R/F etc.)
training time
peak GPU memory if relevant
prediction artifact/cache
promotion decision
```

결과를 말할 때 “좋아 보인다”가 아니라 baseline 대비 `delta_brier`와 fold별 방향을 먼저 제시한다.

## 37. Codex가 시작할 때 확인할 명령

```bash
cd ~/Aimers/Tablatent

git status
git branch --show-current
git rev-parse HEAD

git pull --ff-only

conda activate tablatent
source scripts/activate_gpu2.sh

python scripts/check_gpu2_environment.py --smoke-catboost
```

예상 branch:

```text
agent/stable-player-dynamics-gru
```

예상 GPU semantics:

```text
physical GPU 2 -> CUDA_VISIBLE_DEVICES=2 -> logical cuda:0
```

## 38. Codex의 첫 실제 modeling task 권장안

사용자가 별도 방향을 지시하지 않았다면 순서는 다음이 적절하다.

```text
A. aggressive tree-prefix / alpha-beta sweep 결과 확인
B. CatBoost HPO suite 구현
C. HPO 상위 family seed ensemble
D. OOF residual atlas 생성
E. coarse specialist 후보 자동 screening
F. Trackman linkage/profile feasibility audit
```

### 첫 단계에서 만들면 좋은 파일

```text
scripts/run_catboost_hpo_temporal.py
scripts/run_seed_ensemble_suite.py
scripts/analyze_oof_residual_segments.py
scripts/run_segment_specialist_suite.py
docs/aggressive_search_2026-08.md
```

각 script는 기존 `run_2025_proxy_validation.py`, `run_gated_r_specialist_suite.py`, `src/canonical_features.py`를 최대한 재사용한다. 동일한 data preparation을 새로 복제하지 않는다.

## 39. 마지막 경고

이 프로젝트에서 가장 위험한 실패는 모델이 약한 것이 아니라 **검증이 잘못되어 강해 보이는 것**이다. 특히 다음 세 가지를 항상 경계한다.

```text
(1) hidden-test rows 사이의 정보 공유
(2) 미래 시즌 정보를 과거 validation feature에 섞는 leakage
(3) 2024 반복 튜닝으로 인한 proxy overfitting
```

컴퓨팅 파워가 커졌다는 것은 더 많은 모델을 돌릴 수 있다는 뜻이지, 더 느슨한 validation이 허용된다는 뜻이 아니다. 공격적 탐색의 기준은 **많이 돌리되 더 엄격하게 검증하는 것**이다.

---

# Quick Reference

```text
CURRENT BEST
  full_recent + r_fast
  alpha_recent = 0.20
  beta_r       = 0.10
  500 trees each
  weighted Brier = 0.24789305

SERVER
  conda env     = tablatent
  Python        = 3.11.15
  torch         = 2.7.1+cu128
  catboost      = 1.2.10
  physical GPU  = 2 only
  CLI device    = --devices 0 after activate_gpu2.sh

DATA
  train         = 1,475,092 rows, 2019~2024
  hidden test   = 245,789 rows, 2025
  Trackman      = 1,793,078 rows, 2019~2024

NON-NEGOTIABLE
  hidden-test row independence
  no external baseball data
  no future/post-pitch information
  no raw row-order temporal assumption
  keep raw game_type
  do not revive failed sequence/TTA/HMM deployment paths

NEXT SEARCH
  CatBoost HPO -> seed ensemble -> residual specialists -> Trackman profile
```

