import torch
import json
import random
import os


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        # 使用本地pileval数据
        data_path = os.path.join(os.path.dirname(__file__), '../../../../datasets/pileval/val.jsonl')
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise NotImplementedError
    random.seed(42)
    random.shuffle(dataset)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
