from pathlib import Path
import random

import torch
from datasets import load_dataset


DATA_ROOT = Path("/mnt/42_store/lcw/data2/Huawei/datasets")


def _load_local_dataset(data: str):
    if data == "pileval":
        file_path = DATA_ROOT / "pileval" / "val.jsonl"
        return load_dataset("json", data_files={"train": str(file_path)}, split="train")
    if data == "wikitext2":
        file_path = DATA_ROOT / "wikitext2" / "wiki.train.raw"
        return load_dataset("text", data_files={"train": str(file_path)}, split="train")
    if data == "c4":
        file_path = DATA_ROOT / "c4" / "c4-train.00000-of-01024.json.gz"
        return load_dataset("json", data_files={"train": str(file_path)}, split="train")
    raise ValueError(f"Unsupported calibration dataset: {data}")


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    dataset = _load_local_dataset(data)
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
