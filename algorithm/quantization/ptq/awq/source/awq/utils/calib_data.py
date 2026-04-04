from pathlib import Path
import random

import torch
from datasets import load_dataset


def _load_local_dataset(data: str, data_path):
    data_path = Path(data_path)
    if data == "pileval":
        file_path = data_path / "pileval" / "val.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Pileval not found: {file_path}")
        return load_dataset("json", data_files={"train": str(file_path)}, split="train")
    if data == "wikitext2":
        file_path = data_path / "wikitext2" / "wiki.train.raw"
        if file_path.exists():
            return load_dataset("text", data_files={"train": str(file_path)}, split="train")
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    if data == "c4":
        file_path = data_path / "c4" / "c4-train.00000-of-01024.json.gz"
        if file_path.exists():
            return load_dataset("json", data_files={"train": str(file_path)}, split="train")
        return load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    raise ValueError(f"Unsupported calibration dataset: {data}")


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512, data_path=None):
    dataset = _load_local_dataset(data, data_path)
    dataset = dataset.shuffle(seed=42)
    samples = []
    random.seed(42)
    sample_count = 0
    for record in dataset:
        line = record.get("text", "").strip()
        if not line:
            continue
        encoded = tokenizer.encode(line)
        if len(encoded) < block_size:
            continue
        if len(encoded) > block_size:
            start = random.randint(0, len(encoded) - block_size)
            encoded = encoded[start : start + block_size]
        sample = torch.tensor([encoded])
        samples.append(sample)
        sample_count += 1
        if sample_count == n_samples:
            break
    cat_samples = torch.cat(samples, dim=1)
    split_count = cat_samples.shape[1] // block_size
    print(f" * Split into {split_count} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size]
        for i in range(split_count)
    ]
