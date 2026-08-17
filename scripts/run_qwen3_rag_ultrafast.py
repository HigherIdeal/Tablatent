#!/usr/bin/env python3
"""Ultra-fast offline Qwen3-1.7B tabular-RAG scorer.

This path is designed for throughput, not agentic tool-use tracing.

Speed strategy:
- physical NVIDIA GPU 2 only (mapped to logical cuda:0);
- Qwen3-1.7B is the only learned model;
- deterministic historical RAG statistics are pre-indexed once with pandas groupby;
- no per-row function-calling loop;
- no autoregressive text generation;
- one batched Qwen forward pass per batch;
- probability = softmax(next-token logits for labels 0 vs 1);
- Qwen3 logits_to_keep=1 avoids materializing vocabulary logits for every prompt token;
- BF16 + PyTorch SDPA attention.

Example:
    python scripts/run_qwen3_rag_ultrafast.py \
      --model-dir models/Qwen3-1.7B \
      --query-season 2024 --limit 10000 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# MUST be set before importing torch/transformers.  Physical GPU 2 becomes cuda:0.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3-1.7B"
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen3_1p7b_rag_ultrafast"
TARGET = "control_success"
REFERENCE_BRIER = 0.25
SCORE_SCALE = 100_000.0

# Compact row features.  Missing columns are skipped.
ROW_FEATURES = [
    "season",
    "game_month",
    "game_dayofweek",
    "game_type",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "base_state_before",
    "runners",
    "pitcher_hand",
    "batter_hand",
    "score_diff",
    "score_diff_before",
    "win_expectancy",
    "win_expectancy_before",
    "li",
    "leverage_index_before",
]

# Explicitly meaningful leak-safe/as-of fields.  We select only a bounded number.
EXTRA_PREFIXES = ("asof_", "prev1_", "prev3_", "prev5_", "tm_")
MAX_EXTRA_FIELDS = 16

SYSTEM = (
    "You predict whether a baseball pitch has control_success. "
    "Use the current situation and leak-safe historical evidence. "
    "The next answer is a binary label: 1 means control_success, 0 means failure. "
    "Do not reason aloud."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--query", type=Path)
    parser.add_argument("--query-season", type=int, default=2024)
    parser.add_argument("--query-season-override", type=int)
    parser.add_argument("--limit", type=int, default=10000, help="0 means all rows")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-every", type=int, default=4096)
    return parser.parse_args()


def official_style_score(value: float) -> float:
    return float(SCORE_SCALE * (1.0 - value / REFERENCE_BRIER))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def canon(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            if number.is_integer():
                return str(int(number))
            return f"{number:.6g}"
        return "<NA>"
    return str(value).strip()


def compact_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, 5)
    return str(value)


def validate_model_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"invalid local model directory: {path}")
    if not any((path / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
        raise FileNotFoundError(f"missing model weights: {path}")
    return path


def load_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(args.train, low_memory=False)
    if TARGET not in train.columns or "season" not in train.columns:
        raise ValueError(f"{args.train} must contain season and {TARGET}")

    if args.query is None:
        seasons = pd.to_numeric(train["season"], errors="coerce")
        query = train.loc[seasons.eq(args.query_season)].copy()
    else:
        query = pd.read_csv(args.query, low_memory=False)
        if args.query_season_override is not None:
            query["season"] = args.query_season_override
        if "season" not in query.columns:
            raise ValueError("query needs season or --query-season-override")

    if args.offset:
        query = query.iloc[args.offset:]
    if args.limit > 0:
        query = query.iloc[: args.limit]
    query = query.reset_index(drop=False).rename(columns={"index": "source_index"})
    return train, query


def aggregate_rate(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[str, ...], tuple[int, float]]:
    if any(column not in frame.columns for column in columns):
        return {}
    work = frame[columns + [TARGET]].copy()
    for column in columns:
        work[column] = work[column].map(canon)
    grouped = work.groupby(columns, sort=False, dropna=False)[TARGET].agg(["size", "mean"])
    result: dict[tuple[str, ...], tuple[int, float]] = {}
    for key, row in grouped.iterrows():
        key_tuple = key if isinstance(key, tuple) else (key,)
        result[tuple(str(item) for item in key_tuple)] = (int(row["size"]), float(row["mean"]))
    return result


class SeasonEvidenceIndex:
    """Vectorized/groupby-built historical evidence for one query season."""

    def __init__(self, train: pd.DataFrame, query_season: int) -> None:
        self.season = int(query_season)
        seasons = pd.to_numeric(train["season"], errors="coerce")
        prior = train.loc[seasons < self.season].copy()
        if prior.empty:
            raise ValueError(f"no history before season {self.season}")
        y = pd.to_numeric(prior[TARGET], errors="coerce").dropna()
        self.global_rate = float(y.mean()) if len(y) else 0.5

        self.pitcher = aggregate_rate(prior, ["pitcher_id"])
        self.batter = aggregate_rate(prior, ["batter_id"])
        self.matchup = aggregate_rate(prior, ["pitcher_id", "batter_id"])

        base_col = "base_state" if "base_state" in prior.columns else "base_state_before"
        self.context_levels: list[tuple[list[str], dict[tuple[str, ...], tuple[int, float]]]] = []
        candidates = [
            ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand", base_col],
            ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"],
            ["balls_before", "strikes_before", "outs_before"],
            ["balls_before", "strikes_before"],
        ]
        for columns in candidates:
            if all(column in prior.columns for column in columns):
                self.context_levels.append((columns, aggregate_rate(prior, columns)))

        # Latest prior season pitcher rate is often more relevant than all-history rate.
        latest_season = int(np.nanmax(pd.to_numeric(prior["season"], errors="coerce")))
        latest = prior.loc[pd.to_numeric(prior["season"], errors="coerce").eq(latest_season)]
        self.latest_prior_season = latest_season
        self.pitcher_latest = aggregate_rate(latest, ["pitcher_id"])

    @staticmethod
    def _lookup(table: dict[tuple[str, ...], tuple[int, float]], row: pd.Series, columns: list[str]) -> tuple[int, float] | None:
        if not table or any(column not in row.index for column in columns):
            return None
        return table.get(tuple(canon(row[column]) for column in columns))

    def evidence(self, row: pd.Series) -> dict[str, Any]:
        result: dict[str, Any] = {"prior": round(self.global_rate, 5)}

        p = self._lookup(self.pitcher, row, ["pitcher_id"])
        if p:
            result["pitcher"] = {"n": p[0], "rate": round(p[1], 5)}
        pl = self._lookup(self.pitcher_latest, row, ["pitcher_id"])
        if pl:
            result["pitcher_latest"] = {
                "season": self.latest_prior_season,
                "n": pl[0],
                "rate": round(pl[1], 5),
            }
        b = self._lookup(self.batter, row, ["batter_id"])
        if b:
            result["batter"] = {"n": b[0], "rate": round(b[1], 5)}
        m = self._lookup(self.matchup, row, ["pitcher_id", "batter_id"])
        if m:
            result["matchup"] = {"n": m[0], "rate": round(m[1], 5)}

        best_small: tuple[list[str], tuple[int, float]] | None = None
        for columns, table in self.context_levels:
            item = self._lookup(table, row, columns)
            if item is None:
                continue
            if item[0] >= 100:
                result["context"] = {
                    "n": item[0],
                    "rate": round(item[1], 5),
                    "on": "+".join(columns),
                }
                break
            if best_small is None or item[0] > best_small[1][0]:
                best_small = (columns, item)
        else:
            if best_small is not None:
                columns, item = best_small
                result["context"] = {
                    "n": item[0],
                    "rate": round(item[1], 5),
                    "on": "+".join(columns),
                }
        return result


def row_snapshot(row: pd.Series, extra_columns: list[str]) -> dict[str, Any]:
    columns = [column for column in ROW_FEATURES if column in row.index]
    columns += [column for column in extra_columns if column in row.index]
    return {column: compact_value(row[column]) for column in columns if compact_value(row[column]) is not None}


def build_prompt(tokenizer: Any, row: pd.Series, evidence: dict[str, Any], extra_columns: list[str]) -> str:
    current = row_snapshot(row, extra_columns)
    user = (
        "CURRENT=" + json.dumps(current, ensure_ascii=False, separators=(",", ":"))
        + "\nHISTORY=" + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        + "\nReturn the binary label now. LABEL:"
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def resolve_label_ids(tokenizer: Any) -> tuple[int, int]:
    ids0 = tokenizer.encode("0", add_special_tokens=False)
    ids1 = tokenizer.encode("1", add_special_tokens=False)
    if len(ids0) != 1 or len(ids1) != 1:
        raise RuntimeError(f"Expected single-token labels; got 0={ids0}, 1={ids1}")
    print(f"[labels] token '0'={ids0[0]}, token '1'={ids1[0]}")
    return int(ids0[0]), int(ids1[0])


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "GPU isolation failed: expected exactly one visible GPU after CUDA_VISIBLE_DEVICES=2, "
            f"got {torch.cuda.device_count()}"
        )
    print("[gpu] mandatory physical GPU=2 -> logical cuda:0")
    print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[gpu] name={torch.cuda.get_device_name(0)}")
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[gpu] total_memory={total_gib:.2f} GiB")

    model_dir = validate_model_dir(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    print(f"[model] dtype={next(model.parameters()).dtype} device={next(model.parameters()).device}")
    print("[model] thinking=False generation=NONE logits_to_keep=1 attention=sdpa")
    return tokenizer, model


def infer_batch(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    label0: int,
    label1: int,
    max_length: int,
    temperature: float,
) -> tuple[np.ndarray, int]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    width = int(encoded["input_ids"].shape[1])
    encoded = {name: tensor.to("cuda:0", non_blocking=True) for name, tensor in encoded.items()}

    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1, :]
        binary_logits = torch.stack((logits[:, label0], logits[:, label1]), dim=-1).float()
        if temperature != 1.0:
            binary_logits = binary_logits / temperature
        probabilities = torch.softmax(binary_logits, dim=-1)[:, 1]
    return probabilities.cpu().numpy(), width


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0 or args.logit_temperature <= 0:
        raise ValueError("batch-size/max-length/logit-temperature must be positive")

    torch.set_float32_matmul_precision("high")
    train, query = load_frames(args)
    if query.empty:
        raise ValueError("query slice is empty")

    extra_columns = [
        column for column in train.columns
        if column.startswith(EXTRA_PREFIXES)
    ][:MAX_EXTRA_FIELDS]

    t0 = time.perf_counter()
    season_indices: dict[int, SeasonEvidenceIndex] = {}
    for season in sorted({int(x) for x in pd.to_numeric(query["season"], errors="raise") }):
        season_indices[season] = SeasonEvidenceIndex(train, season)
    evidence_build_sec = time.perf_counter() - t0
    print(f"[rag] built season indexes in {evidence_build_sec:.2f}s")

    tokenizer, model = load_model(args)
    label0, label1 = resolve_label_ids(tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    running_sse = 0.0
    running_n = 0
    total_prompt_tokens_padded = 0
    infer_start = time.perf_counter()

    progress = tqdm(
        range(0, len(query), args.batch_size),
        desc=f"Qwen3-1.7B logits GPU2 x{args.batch_size}",
    )
    for start in progress:
        end = min(start + args.batch_size, len(query))
        batch_rows = [query.iloc[i] for i in range(start, end)]
        prompts = []
        for row in batch_rows:
            season = int(row["season"])
            evidence = season_indices[season].evidence(row)
            prompts.append(build_prompt(tokenizer, row, evidence, extra_columns))

        probabilities, prompt_width = infer_batch(
            tokenizer,
            model,
            prompts,
            label0,
            label1,
            args.max_length,
            args.logit_temperature,
        )
        total_prompt_tokens_padded += prompt_width * len(batch_rows)

        for local, (row, probability) in enumerate(zip(batch_rows, probabilities)):
            index = start + local
            result: dict[str, Any] = {
                "query_index": index,
                "source_index": int(row["source_index"]),
                "row_id": row.get("row_id", index),
                "season": int(row["season"]),
                "probability": float(probability),
            }
            if TARGET in row.index and not pd.isna(row[TARGET]):
                target = float(row[TARGET])
                result[TARGET] = target
                error = float(probability) - target
                running_sse += error * error
                running_n += 1
            rows.append(result)

        elapsed = time.perf_counter() - infer_start
        rate = len(rows) / max(elapsed, 1e-9)
        postfix: dict[str, str] = {"rows/s": f"{rate:.1f}", "tok": str(prompt_width)}
        if running_n:
            current_brier = running_sse / running_n
            postfix["brier"] = f"{current_brier:.6f}"
            postfix["score"] = f"{official_style_score(current_brier):.1f}"
        progress.set_postfix(postfix)

        if len(rows) % args.save_every < len(batch_rows) or end == len(query):
            pd.DataFrame(rows).to_csv(prediction_path, index=False, encoding="utf-8-sig")

    inference_sec = time.perf_counter() - infer_start
    predictions = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "model": "Qwen/Qwen3-1.7B",
        "mode": "single_forward_binary_logit_rag",
        "physical_gpu": 2,
        "logical_cuda": 0,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "thinking": False,
        "autoregressive_generation": False,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "rows": len(predictions),
        "rag_index_build_sec": evidence_build_sec,
        "inference_sec": inference_sec,
        "rows_per_sec": len(predictions) / max(inference_sec, 1e-9),
        "padded_prompt_tokens_per_sec": total_prompt_tokens_padded / max(inference_sec, 1e-9),
        "mean_probability": float(predictions["probability"].mean()),
        "std_probability": float(predictions["probability"].std(ddof=0)),
        "logit_temperature": args.logit_temperature,
    }
    if TARGET in predictions.columns:
        y = predictions[TARGET].to_numpy(dtype=float)
        p = predictions["probability"].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(p)
        value = brier(y[valid], p[valid])
        summary["brier"] = value
        summary["official_style_score"] = official_style_score(value)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if "brier" in summary:
        print(f"[validation] Brier={summary['brier']:.8f} score={summary['official_style_score']:.2f}")
    print(f"predictions: {prediction_path}")


if __name__ == "__main__":
    main()
