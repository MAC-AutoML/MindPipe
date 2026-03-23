"""Offline-first calibration and evaluation datasets."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


_DATA_ROOT_CANDIDATES = [
    Path(os.environ.get("ALGORITHM_V1_DATA_ROOT", "/home/ma-user/work/algorithm-v1/datasets")),
    Path("/mnt/42_store/lcw/data2/Huawei/datasets"),
]


def _resolve_data_root() -> Path:
    for candidate in _DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return _DATA_ROOT_CANDIDATES[0]


DATA_ROOT = _resolve_data_root()
WIKITEXT2_DIR = DATA_ROOT / "wikitext2"
C4_DIR = DATA_ROOT / "c4"
PILEVAL_DIR = DATA_ROOT / "pileval"


class EncodedText:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_local_wikitext(split_name: str):
    split_to_candidates = {
        "train": [
            ("text", WIKITEXT2_DIR / "wiki.train.raw"),
            ("parquet", WIKITEXT2_DIR / "train-00000-of-00001.parquet"),
        ],
        "test": [
            ("text", WIKITEXT2_DIR / "wiki.test.raw"),
            ("parquet", WIKITEXT2_DIR / "test-00000-of-00001.parquet"),
        ],
        "validation": [
            ("text", WIKITEXT2_DIR / "wiki.valid.raw"),
            ("parquet", WIKITEXT2_DIR / "validation-00000-of-00001.parquet"),
        ],
    }
    for dataset_type, file_path in split_to_candidates[split_name]:
        if file_path.exists():
            return load_dataset(dataset_type, data_files=str(file_path), split="train")
    return None


def _load_local_c4(split_name: str):
    split_to_candidates = {
        "train": [
            C4_DIR / "c4-train.00000-of-01024.json.gz",
            C4_DIR / "en" / "c4-train.00000-of-01024.json.gz",
        ],
        "validation": [
            C4_DIR / "c4-validation.00000-of-00008.json.gz",
            C4_DIR / "en" / "c4-validation.00000-of-00008.json.gz",
        ],
    }
    for candidate in split_to_candidates[split_name]:
        if candidate.exists():
            return load_dataset("json", data_files={split_name: str(candidate)}, split=split_name)
    return None


def _load_local_pileval():
    file_path = PILEVAL_DIR / "val.jsonl"
    if not file_path.exists():
        return None
    return file_path


def _sample_text_corpus_from_jsonl(
    file_path: Path,
    tokenizer,
    sequence_length: int,
    sample_count: int,
    seed: int,
    candidate_limit: int = 512,
):
    random.seed(seed)
    candidates: list[torch.Tensor] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(candidates) >= candidate_limit:
                break
            payload = json.loads(line)
            text = payload.get("text")
            if not text:
                continue
            encoded = tokenizer(text, return_tensors="pt")
            if encoded.input_ids.shape[1] >= sequence_length:
                candidates.append(encoded.input_ids)
    if not candidates:
        raise ValueError(f"No pileval samples longer than {sequence_length} tokens found in {file_path}")

    calibration_batches = []
    for _ in range(sample_count):
        encoded_ids = random.choice(candidates)
        start = random.randint(0, encoded_ids.shape[1] - sequence_length)
        end = start + sequence_length
        input_ids = encoded_ids[:, start:end]
        labels = input_ids.clone()
        labels[:, :-1] = -100
        calibration_batches.append((input_ids, labels))
    return calibration_batches


def _sample_train_chunks(encoded, sample_count: int, seed: int, sequence_length: int):
    random.seed(seed)
    calibration_batches = []
    for _ in range(sample_count):
        start = random.randint(0, encoded.input_ids.shape[1] - sequence_length - 1)
        end = start + sequence_length
        input_ids = encoded.input_ids[:, start:end]
        labels = input_ids.clone()
        labels[:, :-1] = -100
        calibration_batches.append((input_ids, labels))
    return calibration_batches


def _load_wikitext2(tokenizer, sequence_length: int, sample_count: int, seed: int):
    train_split = _load_local_wikitext("train")
    test_split = _load_local_wikitext("test")
    if train_split is None or test_split is None:
        train_split = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        test_split = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train_encoded = tokenizer("\n\n".join(train_split["text"]), return_tensors="pt")
    test_encoded = tokenizer("\n\n".join(test_split["text"]), return_tensors="pt")
    return _sample_train_chunks(train_encoded, sample_count, seed, sequence_length), EncodedText(test_encoded.input_ids)


def _load_c4(tokenizer, sequence_length: int, sample_count: int, seed: int):
    train_split = _load_local_c4("train")
    validation_split = _load_local_c4("validation")
    if train_split is None:
        train_split = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    if validation_split is None:
        validation_split = train_split

    random.seed(seed)
    calibration_batches = []
    for _ in range(sample_count):
        while True:
            sample_index = random.randint(0, len(train_split) - 1)
            encoded = tokenizer(train_split[sample_index]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= sequence_length:
                break
        start = random.randint(0, encoded.input_ids.shape[1] - sequence_length - 1)
        end = start + sequence_length
        input_ids = encoded.input_ids[:, start:end]
        labels = input_ids.clone()
        labels[:, :-1] = -100
        calibration_batches.append((input_ids, labels))

    evaluation_slices = []
    random.seed(0)
    for _ in range(256):
        while True:
            sample_index = random.randint(0, len(validation_split) - 1)
            encoded = tokenizer(validation_split[sample_index]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= sequence_length:
                break
        start = random.randint(0, encoded.input_ids.shape[1] - sequence_length - 1)
        end = start + sequence_length
        evaluation_slices.append(encoded.input_ids[:, start:end])
    return calibration_batches, EncodedText(torch.hstack(evaluation_slices))


def _load_pileval(tokenizer, sequence_length: int, sample_count: int, seed: int):
    pileval = _load_local_pileval()
    if pileval is None:
        raise FileNotFoundError(f"Pileval dataset not found under {PILEVAL_DIR}")
    if isinstance(pileval, Path):
        return _sample_text_corpus_from_jsonl(
            file_path=pileval,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            sample_count=sample_count,
            seed=seed,
        ), None

    random.seed(seed)
    calibration_batches = []
    for _ in range(sample_count):
        while True:
            sample_index = random.randint(0, len(pileval) - 1)
            encoded = tokenizer(pileval[sample_index]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= sequence_length:
                break
        start = random.randint(0, encoded.input_ids.shape[1] - sequence_length)
        end = start + sequence_length
        input_ids = encoded.input_ids[:, start:end]
        labels = input_ids.clone()
        labels[:, :-1] = -100
        calibration_batches.append((input_ids, labels))
    return calibration_batches, None


def get_calibration_and_evaluation_data(
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    sample_count: int,
    seed: int,
):
    normalized_name = dataset_name.lower()
    if normalized_name == "wikitext2":
        return _load_wikitext2(tokenizer, sequence_length, sample_count, seed)
    if normalized_name == "c4":
        return _load_c4(tokenizer, sequence_length, sample_count, seed)
    if normalized_name == "pileval":
        return _load_pileval(tokenizer, sequence_length, sample_count, seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_evaluation_tokens(tokenizer, dataset_name: str, sequence_length: int, seed: int):
    _, evaluation_tokens = get_calibration_and_evaluation_data(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sequence_length=sequence_length,
        sample_count=1,
        seed=seed,
    )
    if evaluation_tokens is None:
        raise ValueError(f"Dataset {dataset_name} does not provide evaluation tokens")
    return evaluation_tokens
