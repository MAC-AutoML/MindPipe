"""SFT data adapters for compression-aware LoRA."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import Dataset


LOGGER = logging.getLogger(__name__)


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return [dict(row) for row in payload["data"]]
    raise TypeError(f"Unsupported SFT data payload in {path}: {type(payload)!r}")


def _format_alpaca_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def _response_token_count_after_truncation(
    prompt: str,
    response: str,
    tokenizer: Any,
    max_length: int,
) -> tuple[str, int]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    response_ids = tokenizer(str(response) + eos, add_special_tokens=False, truncation=False)["input_ids"]
    if not response_ids:
        return "dropped_empty_response_tokens", 0
    if len(prompt_ids) + len(response_ids) <= max_length:
        return "full", len(response_ids)
    available_response_tokens = max(0, int(max_length) - len(prompt_ids))
    if available_response_tokens <= 0:
        return "dropped_no_response_after_truncation", 0
    return "truncated_answer", min(len(response_ids), available_response_tokens)


def build_alpaca_sft_dataset(
    train_file: str | Path,
    sample_count: int,
    seed: int,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    min_response_tokens: int = 8,
    sample_start: int = 0,
):
    rows = _load_json_or_jsonl(train_file)
    full_examples: list[dict[str, Any]] = []
    truncated_examples: list[dict[str, Any]] = []
    dropped_empty = 0
    dropped_short_response = 0
    dropped_no_response_after_truncation = 0
    dropped_empty_response_tokens = 0
    for row in rows:
        instruction = str(row.get("instruction") or "").strip()
        input_text = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            dropped_empty += 1
            continue
        prompt = _format_alpaca_prompt(instruction, input_text)
        bucket = "full"
        response_tokens = 0
        if tokenizer is not None and max_length is not None and int(max_length) > 0:
            bucket, response_tokens = _response_token_count_after_truncation(prompt, output, tokenizer, int(max_length))
            if bucket == "dropped_no_response_after_truncation":
                dropped_no_response_after_truncation += 1
                continue
            if bucket == "dropped_empty_response_tokens":
                dropped_empty_response_tokens += 1
                continue
            if response_tokens < int(min_response_tokens):
                dropped_short_response += 1
                continue
        example = {
            "prompt": prompt,
            "response": output,
            "messages": [
                {"role": "user", "content": f"{instruction}\n{input_text}".strip()},
                {"role": "assistant", "content": output},
            ],
            "images": [],
            "sft_length_bucket": bucket,
            "sft_response_tokens_after_truncation": response_tokens,
        }
        if bucket == "truncated_answer":
            truncated_examples.append(example)
        else:
            full_examples.append(example)

    rng = random.Random(seed)
    rng.shuffle(full_examples)
    rng.shuffle(truncated_examples)
    ordered_examples = full_examples + truncated_examples
    sample_start = max(0, int(sample_start))
    if sample_count > 0:
        examples = ordered_examples[sample_start : sample_start + sample_count]
    else:
        examples = ordered_examples[sample_start:]
    if not examples:
        raise ValueError(f"No Alpaca SFT examples were built from {train_file}.")
    selected_full = sum(1 for example in examples if example["sft_length_bucket"] == "full")
    selected_truncated = sum(1 for example in examples if example["sft_length_bucket"] == "truncated_answer")
    LOGGER.info(
        "Alpaca SFT length filter: selected=%s sample_start=%s full=%s truncated_answer=%s available_full=%s available_truncated_answer=%s "
        "dropped_empty=%s dropped_no_response_after_truncation=%s dropped_empty_response_tokens=%s dropped_short_response=%s "
        "min_response_tokens=%s max_length=%s",
        len(examples),
        sample_start,
        selected_full,
        selected_truncated,
        len(full_examples),
        len(truncated_examples),
        dropped_empty,
        dropped_no_response_after_truncation,
        dropped_empty_response_tokens,
        dropped_short_response,
        int(min_response_tokens),
        max_length,
    )
    return Dataset.from_list(examples)


def build_llava_sft_dataset(
    train_file: str | Path,
    sample_count: int,
    seed: int,
):
    rows = _load_json_or_jsonl(train_file)
    examples: list[dict[str, Any]] = []
    for row in rows:
        image = str(row.get("image") or "").strip()
        conversations = row.get("conversations")
        if not image or not isinstance(conversations, list):
            continue
        messages: list[dict[str, str]] = []
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = message.get("role", message.get("from"))
            content = message.get("content", message.get("value"))
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            if role not in {"user", "assistant"} or content is None:
                continue
            messages.append({"role": str(role), "content": str(content)})
        if len(messages) < 2 or messages[0]["role"] != "user":
            continue
        sample_id = row.get("id")
        examples.append(
            {
                "id": None if sample_id is None else str(sample_id),
                "image": image,
                "images": [image],
                "messages": messages,
                "conversations": messages,
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {"id", "image", "conversations"}
                },
            }
        )

    if sample_count > 0 and len(examples) > sample_count:
        rng = random.Random(seed)
        examples = rng.sample(examples, sample_count)
    if not examples:
        raise ValueError(f"No LLaVA SFT examples were built from {train_file}.")
    return Dataset.from_list(examples)
