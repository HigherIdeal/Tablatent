#!/usr/bin/env python3
"""Fully offline batched Qwen3-1.7B forecaster with temporal-safe tabular RAG.

Only Qwen/Qwen3-1.7B is a learned model. Python performs deterministic
historical lookup and tool execution. Inference never accesses the network.

The LLM is evaluated in batches. Every active query in a batch advances one
tool/final-action round together, which keeps GPU utilization much higher than
row-by-row generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

# Network access is disabled before transformers is imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3-1.7B"
DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen3_1p7b_rag"

SYSTEM_PROMPT = """You are a calibrated probabilistic baseball forecaster.
Predict control_success for one future pitch.

Historical retrieval tools are available. Tool results contain only seasons
strictly earlier than the query season. Use tools when useful; never invent
unavailable information.

Tools:
- pitcher_history: historical control rate of this pitcher
- batter_history: historical control rate for pitches faced by this batter
- matchup_history: exact pitcher-batter historical control rate
- context_history: historical rate for similar count/game context
- similar_examples: labeled historical examples; argument k is 4..20

At every turn output exactly ONE JSON object and no markdown.
Tool call examples:
{"type":"tool","name":"pitcher_history","arguments":{}}
{"type":"tool","name":"similar_examples","arguments":{"k":12}}
Final answer:
{"type":"final","probability":0.5372}

Probability must be a calibrated real number in [0,1], not a hard class. /no_think"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--query", type=Path)
    parser.add_argument("--query-season", type=int, default=2024)
    parser.add_argument("--query-season-override", type=int)
    parser.add_argument("--limit", type=int, default=1000, help="0 means all rows")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of independent query rows advanced together on the GPU.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=256,
        help="Rewrite predictions.csv after this many newly completed rows.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_tool_calls < 0:
        raise ValueError("--max-tool-calls must be >= 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
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


def validate_local_model_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local Qwen3-1.7B directory not found: {path}\n"
            "Prepare it beforehand with scripts/download_qwen3_1p7b.py."
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json: {path}")
    if not any(
        (path / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        raise FileNotFoundError(f"Missing safetensors weights: {path}")
    if not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        raise FileNotFoundError(f"Missing tokenizer files: {path}")
    return path


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Path]:
    model_dir = validate_local_model_dir(args.model_dir)
    print(f"[offline model] {model_dir}")
    print("[offline] HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 local_files_only=True")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    # Decoder-only batched generation must use left padding so the newest token
    # of every prompt is aligned at the right edge.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model, model_dir


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_actions(
    tokenizer: Any,
    model: Any,
    message_batches: list[list[dict[str, str]]],
    max_new_tokens: int,
) -> list[tuple[dict[str, Any] | None, str]]:
    """Generate one action for every active conversation in a single GPU call."""
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in message_batches
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    device = model_device(model)
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[:, prompt_width:]
    raw_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [(extract_json(raw.strip()), raw.strip()) for raw in raw_outputs]


def query_message(snapshot: dict[str, Any]) -> str:
    return (
        "QUERY_ROW\n"
        + json.dumps(
            snapshot,
            ensure_ascii=False,
            default=json_default,
            separators=(",", ":"),
        )
        + "\nUse retrieval tools as needed, then return the calibrated probability."
    )


def _tool_arguments(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(action.get("name", ""))
    arguments = action.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "similar_examples":
        try:
            arguments["k"] = int(np.clip(int(arguments.get("k", 12)), 4, 20))
        except (TypeError, ValueError):
            arguments["k"] = 12
    return name, arguments


def run_agents_batch(
    queries: list[pd.Series],
    priors: list[float],
    rag: TemporalTabularRAG,
    tokenizer: Any,
    model: Any,
    max_tool_calls: int,
    max_new_tokens: int,
) -> list[tuple[float, list[dict[str, Any]], str]]:
    """Advance multiple independent tool-using agents synchronously by round."""
    if len(queries) != len(priors):
        raise ValueError("queries/priors length mismatch")

    states: list[dict[str, Any]] = []
    for query, prior in zip(queries, priors):
        states.append(
            {
                "query": query,
                "prior": float(prior),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": query_message(rag.query_snapshot(query)),
                    },
                ],
                "trace": [],
                "used_tools": set(),
                "probability": None,
                "status": None,
            }
        )

    for step in range(max_tool_calls + 1):
        active = [
            index
            for index, state in enumerate(states)
            if state["probability"] is None
        ]
        if not active:
            break

        generated = generate_actions(
            tokenizer,
            model,
            [states[index]["messages"] for index in active],
            max_new_tokens,
        )

        for state_index, (action, raw) in zip(active, generated):
            state = states[state_index]
            query = state["query"]
            messages = state["messages"]
            trace = state["trace"]
            used_tools = state["used_tools"]

            trace_entry: dict[str, Any] = {
                "step": step,
                "model_raw": raw,
                "parsed": action,
            }
            trace.append(trace_entry)

            if not action:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                'Invalid format. Output one JSON object only, e.g. '
                                '{"type":"final","probability":0.5}'
                            ),
                        },
                    ]
                )
                continue

            if action.get("type") == "final":
                try:
                    probability = float(action["probability"])
                except (KeyError, TypeError, ValueError):
                    probability = math.nan
                if math.isfinite(probability) and 0.0 <= probability <= 1.0:
                    state["probability"] = float(
                        np.clip(probability, 1e-5, 1 - 1e-5)
                    )
                    state["status"] = "ok"
                    continue

                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": json.dumps(action, ensure_ascii=False),
                        },
                        {
                            "role": "user",
                            "content": (
                                'Invalid probability. Return '
                                '{"type":"final","probability":<0..1>}.'
                            ),
                        },
                    ]
                )
                continue

            if action.get("type") != "tool":
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": json.dumps(action, ensure_ascii=False),
                        },
                        {
                            "role": "user",
                            "content": "type must be tool or final.",
                        },
                    ]
                )
                continue

            name, arguments = _tool_arguments(action)
            if name in used_tools and name != "similar_examples":
                result_payload = {
                    "warning": (
                        f"{name} was already called; "
                        "use its previous result or finish."
                    )
                }
            else:
                try:
                    result_payload = rag.call(name, query, **arguments).payload
                    used_tools.add(name)
                except Exception as error:
                    result_payload = {
                        "error": f"{type(error).__name__}: {error}"
                    }

            trace_entry.update(
                tool_name=name,
                tool_arguments=arguments,
                tool_result=result_payload,
            )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": "TOOL_RESULT\n"
                        + json.dumps(
                            {"name": name, "result": result_payload},
                            ensure_ascii=False,
                            default=json_default,
                            separators=(",", ":"),
                        ),
                    },
                ]
            )

    outcomes: list[tuple[float, list[dict[str, Any]], str]] = []
    for state in states:
        if state["probability"] is None:
            probability = float(
                np.clip(state["prior"], 1e-5, 1 - 1e-5)
            )
            status = "fallback_prior"
        else:
            probability = float(state["probability"])
            status = str(state["status"])
        outcomes.append((probability, state["trace"], status))
    return outcomes


def load_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(args.train, low_memory=False)
    if TARGET not in train.columns or "season" not in train.columns:
        raise ValueError(f"{args.train} must contain season and {TARGET}")

    if args.query:
        query = pd.read_csv(args.query, low_memory=False)
        if args.query_season_override is not None:
            query["season"] = args.query_season_override
        if "season" not in query.columns:
            raise ValueError("query needs season or --query-season-override")
    else:
        seasons = pd.to_numeric(train["season"], errors="coerce")
        query = train.loc[seasons.eq(args.query_season)].copy()

    if args.offset:
        query = query.iloc[args.offset:]
    if args.limit > 0:
        query = query.iloc[: args.limit]
    query = query.reset_index(drop=False).rename(columns={"index": "source_index"})
    return train, query


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, query = load_frames(args)
    if query.empty:
        raise ValueError("query slice is empty")

    rag = TemporalTabularRAG(train, seed=args.seed)
    query_seasons = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                query["season"], errors="raise"
            ).tolist()
        }
    )
    prior_by_season = {
        season: rag.prior_rate(season) for season in query_seasons
    }
    tokenizer, model, model_dir = load_model(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    trace_path = args.output_dir / "traces.jsonl"

    done = 0
    rows: list[dict[str, Any]] = []
    if args.resume and prediction_path.exists():
        existing = pd.read_csv(prediction_path)
        done = len(existing)
        if done > len(query):
            raise ValueError(
                "resume file has more rows than current query slice"
            )
        rows.extend(existing.to_dict(orient="records"))
        print(f"[resume] {done:,} predictions")

    print(
        f"[batch] batch_size={args.batch_size}, "
        f"rows={len(query):,}, max_tool_calls={args.max_tool_calls}"
    )

    trace_mode = "a" if done and args.resume else "w"
    newly_saved = 0
    with trace_path.open(trace_mode, encoding="utf-8") as trace_file:
        batch_starts = range(done, len(query), args.batch_size)
        progress = tqdm(
            batch_starts,
            total=math.ceil(max(len(query) - done, 0) / args.batch_size),
            desc=f"Qwen3-1.7B offline RAG x{args.batch_size}",
        )

        for batch_start in progress:
            batch_end = min(batch_start + args.batch_size, len(query))
            local_indices = list(range(batch_start, batch_end))
            query_rows = [query.iloc[index] for index in local_indices]
            priors = [
                prior_by_season[int(row["season"])]
                for row in query_rows
            ]

            outcomes = run_agents_batch(
                query_rows,
                priors,
                rag,
                tokenizer,
                model,
                max_tool_calls=args.max_tool_calls,
                max_new_tokens=args.max_new_tokens,
            )

            for local_idx, row, (probability, trace, status) in zip(
                local_indices, query_rows, outcomes
            ):
                season = int(row["season"])
                result: dict[str, Any] = {
                    "query_index": local_idx,
                    "source_index": int(row["source_index"]),
                    "row_id": row.get("row_id", local_idx),
                    "season": season,
                    "probability": probability,
                    "status": status,
                    "tool_calls": sum(
                        "tool_name" in item for item in trace
                    ),
                }
                if TARGET in row.index and not pd.isna(row[TARGET]):
                    result[TARGET] = float(row[TARGET])
                rows.append(result)

                trace_file.write(
                    json.dumps(
                        {"result": result, "trace": trace},
                        ensure_ascii=False,
                        default=json_default,
                    )
                    + "\n"
                )
                newly_saved += 1

            trace_file.flush()
            if newly_saved >= args.save_every or batch_end == len(query):
                pd.DataFrame(rows).to_csv(
                    prediction_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                newly_saved = 0

    predictions = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "model": "Qwen/Qwen3-1.7B",
        "model_dir": str(model_dir),
        "offline_inference": True,
        "batched_generation": True,
        "batch_size": args.batch_size,
        "rows": len(predictions),
        "mean_probability": float(predictions["probability"].mean()),
        "std_probability": float(
            predictions["probability"].std(ddof=0)
        ),
        "fallback_rows": int(
            predictions["status"].ne("ok").sum()
        ),
        "mean_tool_calls": float(
            predictions["tool_calls"].mean()
        ),
        "prior_by_query_season": prior_by_season,
    }
    if TARGET in predictions.columns:
        y = predictions[TARGET].to_numpy(dtype=float)
        p = predictions["probability"].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(p)
        summary["brier"] = brier(y[valid], p[valid])
        slice_prior = float(np.mean(y[valid]))
        summary["slice_prior"] = slice_prior
        summary["slice_prior_brier"] = brier(
            y[valid],
            np.full(valid.sum(), slice_prior),
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"predictions: {prediction_path}")
    print(f"traces: {trace_path}")


if __name__ == "__main__":
    main()
