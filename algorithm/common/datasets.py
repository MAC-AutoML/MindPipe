"""Offline-first calibration and evaluation datasets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

class EncodedText:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_local_wikitext(split_name: str, data_path: Path):
    wikitext2_dir = data_path / "wikitext2"
    split_to_file = {
        "train": wikitext2_dir / "wiki.train.raw",
        "test": wikitext2_dir / "wiki.test.raw",
        "validation": wikitext2_dir / "wiki.valid.raw",
    }
    file_path = split_to_file[split_name]
    if not file_path.exists():
        return None
    return load_dataset("text", data_files=str(file_path), split="train")


def _load_local_c4(split_name: str, data_path: Path):
    c4_dir = data_path / "c4"
    split_to_candidates = {
        "train": [
            c4_dir / "c4-train.00000-of-01024.json.gz",
            c4_dir / "en" / "c4-train.00000-of-01024.json.gz",
        ],
        "validation": [
            c4_dir / "c4-validation.00000-of-00008.json.gz",
            c4_dir / "en" / "c4-validation.00000-of-00008.json.gz",
        ],
    }
    for candidate in split_to_candidates[split_name]:
        if candidate.exists():
            return load_dataset("json", data_files={split_name: str(candidate)}, split=split_name)
    return None


def _load_local_pileval(data_path: Path):
    file_path = data_path / "pileval" / "val.jsonl"
    if not file_path.exists():
        return None
    return load_dataset("json", data_files={"train": str(file_path)}, split="train")


def _load_local_bookcorpus(data_path: Path):
    bookcorpus_dir = data_path / "bookcorpus"
    if not bookcorpus_dir.exists():
        return None
    from datasets import load_from_disk

    return load_from_disk(str(bookcorpus_dir))


def _sample_train_chunks(encoded, sample_count: int, seed: int, sequence_length: int):
    random.seed(seed)
    calibration_batches = []
    for _ in range(sample_count):
        max_start = encoded.input_ids.shape[1] - sequence_length
        start = 0 if max_start <= 0 else random.randint(0, max_start)
        end = start + sequence_length
        input_ids = encoded.input_ids[:, start:end]
        labels = input_ids.clone()
        labels[:, :-1] = -100
        calibration_batches.append((input_ids, labels))
    return calibration_batches


def _load_wikitext2(tokenizer, sequence_length: int, sample_count: int, seed: int, data_path: Path):
    train_split = _load_local_wikitext("train", data_path)
    test_split = _load_local_wikitext("test", data_path)
    if train_split is None or test_split is None:
        train_split = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        test_split = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train_encoded = tokenizer("\n\n".join(train_split["text"]), return_tensors="pt")
    test_encoded = tokenizer("\n\n".join(test_split["text"]), return_tensors="pt")
    return _sample_train_chunks(train_encoded, sample_count, seed, sequence_length), EncodedText(test_encoded.input_ids)


def _load_c4(tokenizer, sequence_length: int, sample_count: int, seed: int, data_path: Path):
    train_split = _load_local_c4("train", data_path)
    validation_split = _load_local_c4("validation", data_path)
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
        max_start = encoded.input_ids.shape[1] - sequence_length
        start = 0 if max_start <= 0 else random.randint(0, max_start)
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
        max_start = encoded.input_ids.shape[1] - sequence_length
        start = 0 if max_start <= 0 else random.randint(0, max_start)
        end = start + sequence_length
        evaluation_slices.append(encoded.input_ids[:, start:end])
    return calibration_batches, EncodedText(torch.hstack(evaluation_slices))


def _load_pileval(tokenizer, sequence_length: int, sample_count: int, seed: int, data_path: Path):
    pileval = _load_local_pileval(data_path)
    if pileval is None:
        raise FileNotFoundError(f"Pileval dataset not found under {data_path / 'pileval'}")

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


def _load_pg19(tokenizer, sequence_length: int, sample_count: int, seed: int, data_path: Path):
    pg19_dir = data_path / "pg19"
    if not pg19_dir.exists():
        raise FileNotFoundError(f"PG19 dataset not found under {pg19_dir}")
    from datasets import load_from_disk
    pg19 = load_from_disk(str(pg19_dir))

    random.seed(seed)
    calibration_batches = []
    for _ in range(sample_count):
        while True:
            sample_index = random.randint(0, len(pg19) - 1)
            encoded = tokenizer(pg19[sample_index]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= sequence_length:
                break
        start = random.randint(0, encoded.input_ids.shape[1] - sequence_length)
        end = start + sequence_length
        input_ids = encoded.input_ids[:, start:end]
        labels = input_ids.clone()
        labels[:, :-1] = -100
        calibration_batches.append((input_ids, labels))
    return calibration_batches, None


def _load_bookcorpus(tokenizer, sequence_length: int, sample_count: int, seed: int, data_path: Path):
    bookcorpus = _load_local_bookcorpus(data_path)
    if bookcorpus is None:
        bookcorpus = load_dataset("bookcorpus", split="train", trust_remote_code=True)

    random.seed(seed)
    calibration_batches = []
    for _ in range(sample_count):
        while True:
            sample_index = random.randint(0, len(bookcorpus) - 1)
            encoded = tokenizer(bookcorpus[sample_index]["text"], return_tensors="pt")
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
    data_path: str | Path,
):
    data_path = Path(data_path)
    normalized_name = dataset_name.lower()
    if normalized_name == "wikitext2":
        return _load_wikitext2(tokenizer, sequence_length, sample_count, seed, data_path)
    if normalized_name == "c4":
        return _load_c4(tokenizer, sequence_length, sample_count, seed, data_path)
    if normalized_name == "pileval":
        return _load_pileval(tokenizer, sequence_length, sample_count, seed, data_path)
    if normalized_name == "pg19":
        return _load_pg19(tokenizer, sequence_length, sample_count, seed, data_path)
    if normalized_name == "bookcorpus":
        return _load_bookcorpus(tokenizer, sequence_length, sample_count, seed, data_path)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_evaluation_tokens(
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    seed: int,
    data_path: str | Path,
):
    _, evaluation_tokens = get_calibration_and_evaluation_data(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sequence_length=sequence_length,
        sample_count=1,
        seed=seed,
        data_path=data_path,
    )
    if evaluation_tokens is None:
        raise ValueError(f"Dataset {dataset_name} does not provide evaluation tokens")
    return evaluation_tokens
# Maintenance touch for repository metadata refresh.
