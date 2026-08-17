# Qwen3-1.7B Offline Tabular-RAG

## 구조

- learned model: `Qwen/Qwen3-1.7B` 하나만 사용
- retrieval: Python의 deterministic historical lookup
- temporal safety: query season `S`에서는 `season < S`만 조회
- inference network access: 금지
- output: `control_success` probability `[0, 1]`
- trace: 모델 raw 출력, tool call, tool result를 JSONL로 저장

## 1. 온라인 머신에서 모델 1회 준비

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-qwen.txt
python scripts/download_qwen3_1p7b.py
```

생성되는 `models/Qwen3-1.7B/` 디렉터리를 통째로 오프라인 실행 머신에 복사한다. Git에는 모델 weight를 올리지 않도록 `models/`가 `.gitignore`에 포함되어 있다.

## 2. 완전 오프라인 추론

`run_qwen3_rag.py`는 시작할 때 다음을 강제한다.

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
local_files_only=True
trust_remote_code=False
```

따라서 로컬 snapshot이 불완전하면 인터넷으로 받으려 하지 않고 즉시 실패한다.

2024 validation 100행 sanity check:

```bash
python scripts/run_qwen3_rag.py \
  --model-dir models/Qwen3-1.7B \
  --query-season 2024 \
  --limit 100
```

전체 2024 validation:

```bash
python scripts/run_qwen3_rag.py \
  --model-dir models/Qwen3-1.7B \
  --query-season 2024 \
  --limit 0 \
  --resume
```

외부 `test.csv`에 season이 없고 평가 season을 별도로 지정해야 할 경우:

```bash
python scripts/run_qwen3_rag.py \
  --model-dir models/Qwen3-1.7B \
  --query data/raw/test.csv \
  --query-season-override <SEASON> \
  --limit 0 \
  --resume
```

## 출력

기본 위치: `outputs/qwen3_1p7b_rag/`

- `predictions.csv`: row별 probability, status, tool-call count
- `traces.jsonl`: Qwen raw response와 retrieval trace
- `summary.json`: Brier(정답 존재 시), prediction 통계, fallback 비율

최초 실험에서는 반드시 작은 validation slice로 JSON 준수율, tool-call 횟수, probability 분산, Brier를 확인한 뒤 전체 행으로 확장한다.
