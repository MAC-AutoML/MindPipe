"""Text training data for compression-aware LoRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_raw_text_jsonl_dataset(
    tokenizer,
    train_file: str | Path,
    sequence_length: int,
    sample_count: int,
    seed: int,
):
    """Build CPT-style chunks from a local JSONL file with a `text` field."""

    rows = _read_jsonl(train_file)
    texts = [str(row.get("text") or "").strip() for row in rows]
    texts = [text for text in texts if text]
    if seed is not None:
        import random

        rng = random.Random(seed)
        rng.shuffle(texts)

    examples: list[dict[str, Any]] = []
    eos = tokenizer.eos_token or ""
    for text in texts:
        token_ids = tokenizer(
            text + eos,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        if len(token_ids) < 2:
            continue
        for start in range(0, len(token_ids), sequence_length):
            chunk = token_ids[start : start + sequence_length]
            if len(chunk) < 2:
                continue
            examples.append({"input_ids": chunk})
            if sample_count > 0 and len(examples) >= sample_count:
                return Dataset.from_list(examples)
    if not examples:
        raise ValueError(f"No CPT training examples were built from {train_file}.")
    return Dataset.from_list(examples)
