"""SFT data adapters for compression-aware LoRA."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset


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


def build_alpaca_sft_dataset(
    train_file: str | Path,
    sample_count: int,
    seed: int,
):
    rows = _load_json_or_jsonl(train_file)
    examples: list[dict[str, Any]] = []
    for row in rows:
        instruction = str(row.get("instruction") or "").strip()
        input_text = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            continue
        examples.append(
            {
                "prompt": _format_alpaca_prompt(instruction, input_text),
                "response": output,
                "messages": [
                    {"role": "user", "content": f"{instruction}\n{input_text}".strip()},
                    {"role": "assistant", "content": output},
                ],
                "images": [],
            }
        )

    if sample_count > 0 and len(examples) > sample_count:
        rng = random.Random(seed)
        examples = rng.sample(examples, sample_count)
    if not examples:
        raise ValueError(f"No Alpaca SFT examples were built from {train_file}.")
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
