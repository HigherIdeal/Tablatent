#!/usr/bin/env python3
"""Qwen3-1.7B-only probabilistic forecaster with temporal-safe tabular RAG.

The only learned model in this experiment is Qwen/Qwen3-1.7B.
Python provides deterministic retrieval functions over historical train rows.
Qwen decides which functions to call, reads their results, and emits a final
control_success probability.

Recommended first run:
    python scripts/run_qwen3_rag.py --query-season 2024 --limit 1000

The script reports Brier score when labels are available and writes every model
response/tool call for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_tabular_rag import TARGET, TemporalTabularRAG  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen3_1p7b_rag"

SYSTEM_PROMPT = """You are a calibrated probabilistic baseball forecaster.
Your target is control_success for one future pitch.

You have access to historical retrieval tools. All tool results are guaranteed
to contain only seasons strictly earlier than the query season. Use the tools
when useful. Do not invent unavailable data.

Available tools:
- pitcher_history: historical control rate of this pitcher.
- batter_history: historical control rate for pitches faced by this batter.
- matchup_history: historical control rate for this exact pitcher-batter pair.
- context_history: historical rate for a similar count/game-state context.
- similar_examples: retrieves labeled historical pitch examples. Argument: k,
  integer from 4 to 20.

At each turn output EXACTLY one JSON object and no markdown.
To call a tool:
{"type":"tool","name":"pitcher_history","arguments":{}}
or
{"type":"tool","name":"similar_examples","arguments":{"k":12}}

When ready to predict:
{"type":"final","probability":0.5372}

The final probability must be a calibrated real number in [0,1].
Do not output a hard class label. /no_think"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--query",
        type=Path,
        help="Optional query CSV. If omitted, rows from --query-season in train.csv are evaluated.",
    )
    parser.add_argument("--query-season", type=int, default=2024)
    parser.add_argument(
        "--query-season-override",
        type=int,
        help="Set/replace season on the query table, useful for external test.csv.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="0 means all query rows")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_json(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return extract_json(raw), raw


def query_message(snapshot: dict[str, Any]) -> str:
    return (
        "QUERY_ROW\n"
        + json.dumps(snapshot, ensure_ascii=False, default=json_default, separators=(",", ":"))
        + "\nUse retrieval tools as needed, then return a calibrated probability."
    )


def fallback_prior(train: pd.DataFrame, season: int) -> float:
    y = pd.to_numeric(
        train.loc[pd.to_numeric(train["season"], errors="coerce") < season, TARGET],
        errors="coerce",
    ).dropna()
    if y.empty:
        return 0.5
    return float(y.mean())


def run_agent(
    query: pd.Series,
    rag: TemporalTabularRAG,
    tokenizer: Any,
    model: Any,
    max_tool_calls: int,
    max_new_tokens: int,
    prior: float,
) -> tuple[float, list[dict[str, Any]], str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query_message(rag.query_snapshot(query))},
    ]
    trace: list[dict[str, Any]] = []
    last_raw = ""
    used_tools: set[str] = set()

    for step in range(max_tool_calls + 1):
        action, raw = generate_json(tokenizer, model, messages, max_new_tokens)
        last_raw = raw
        trace.append({"step": step, "model_raw": raw, "parsed": action})
        if not action:
            messages.append({
                "role": "assistant",
                "content": raw,
            })
            messages.append({
                "role": "user",
                "content": 'Invalid format. Output one JSON object only. Example: {"type":"final","probability":0.5}',
            })
            continue

        if action.get("type") == "final":
            try:
                probability = float(action["probability"])
            except (KeyError, TypeError, ValueError):
                probability = math.nan
            if math.isfinite(probability) and 0.0 <= probability <= 1.0:
                return float(np.clip(probability, 1e-5, 1 - 1e-5)), trace, "ok"
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({
                "role": "user",
                "content": 'Probability was invalid. Return {"type":"final","probability":<number from 0 to 1>}.',
            })
            continue

        if action.get("type") != "tool":
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": "type must be tool or final."})
            continue

        name = str(action.get("name", ""))
        arguments = action.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if name == "similar_examples":
            arguments["k"] = int(np.clip(int(arguments.get("k", 12)), 4, 20))
        if name in used_tools and name != "similar_examples":
            result_payload = {"warning": f"{name} was already called; use existing result or finish."}
        else:
            try:
                result = rag.call(name, query, **arguments)
                result_payload = result.payload
                used_tools.add(name)
            except Exception as error:
                result_payload = {"error": f"{type(error).__name__}: {error}"}

        trace[-1]["tool_name"] = name
        trace[-1]["tool_arguments"] = arguments
        trace[-1]["tool_result"] = result_payload
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        messages.append({
            "role": "user",
            "content": "TOOL_RESULT\n" + json.dumps(
                {"name": name, "result": result_payload},
                ensure_ascii=False,
                default=json_default,
                separators=(",", ":"),
            ),
        })

    # A malformed generation must not destroy an entire experiment. The fallback
    # is the historical pre-query-season prior and is explicitly flagged.
    return float(np.clip(prior, 1e-5, 1 - 1e-5)), trace, "fallback_prior"


def load_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(args.train, low_memory=False)
    if TARGET not in train.columns:
        raise ValueError(f"{args.train} has no {TARGET}")
    if "season" not in train.columns:
        raise ValueError(f"{args.train} has no season")

    if args.query:
        query = pd.read_csv(args.query, low_memory=False)
        if args.query_season_override is not None:
            query["season"] = args.query_season_override
        if "season" not in query.columns:
            raise ValueError("query table needs season or --query-season-override")
    else:
        season = pd.to_numeric(train["season"], errors="coerce")
        query = train.loc[season.eq(args.query_season)].copy()

    if args.offset:
        query = query.iloc[args.offset:]
    if args.limit > 0:
        query = query.iloc[: args.limit]
    return train, query.reset_index(drop=False).rename(columns={"index": "source_index"})


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, query = load_frames(args)
    rag = TemporalTabularRAG(train, seed=args.seed)
    tokenizer, model = load_model(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    trace_path = args.output_dir / "traces.jsonl"

    done = 0
    existing: pd.DataFrame | None = None
    if args.resume and prediction_path.exists():
        existing = pd.read_csv(prediction_path)
        done = len(existing)
        if done > len(query):
            raise ValueError("resume file has more rows than the current query slice")
        print(f"[resume] {done:,} predictions already exist")

    rows: list[dict[str, Any]] = []
    if existing is not None:
        rows.extend(existing.to_dict(orient="records"))

    trace_mode = "a" if done and args.resume else "w"
    with trace_path.open(trace_mode, encoding="utf-8") as trace_file:
        for local_idx in tqdm(range(done, len(query)), desc="Qwen3-1.7B RAG"):
            row = query.iloc[local_idx]
            season = int(row["season"])
            prior = fallback_prior(train, season)
            probability, trace, status = run_agent(
                row,
                rag,
                tokenizer,
                model,
                max_tool_calls=args.max_tool_calls,
                max_new_tokens=args.max_new_tokens,
                prior=prior,
            )
            result = {
                "query_index": local_idx,
                "source_index": int(row["source_index"]),
                "row_id": row.get("row_id", local_idx),
                "season": season,
                "probability": probability,
                "status": status,
                "tool_calls": sum(1 for item in trace if "tool_name" in item),
            }
            if TARGET in row.index and not pd.isna(row[TARGET]):
                result[TARGET] = float(row[TARGET])
            rows.append(result)
            trace_file.write(json.dumps(
                {"result": result, "trace": trace},
                ensure_ascii=False,
                default=json_default,
            ) + "\n")
            trace_file.flush()

            # Keep progress recoverable without waiting for the whole run.
            if (local_idx + 1) % 50 == 0 or local_idx + 1 == len(query):
                pd.DataFrame(rows).to_csv(prediction_path, index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "model": args.model,
        "rows": len(predictions),
        "mean_probability": float(predictions["probability"].mean()),
        "std_probability": float(predictions["probability"].std(ddof=0)),
        "fallback_rows": int(predictions["status"].ne("ok").sum()),
        "mean_tool_calls": float(predictions["tool_calls"].mean()),
    }
    if TARGET in predictions.columns:
        y = predictions[TARGET].to_numpy(dtype=float)
        p = predictions["probability"].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(p)
        summary["brier"] = brier(y[valid], p[valid])
        prior = float(np.mean(y[valid]))
        summary["slice_prior"] = prior
        summary["slice_prior_brier"] = brier(y[valid], np.full(valid.sum(), prior))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"predictions: {prediction_path}")
    print(f"traces: {trace_path}")


if __name__ == "__main__":
    main()
