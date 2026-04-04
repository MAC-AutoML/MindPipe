# Offline-first calibration data helpers for structured Wanda.

from pathlib import Path
import random

from datasets import load_dataset


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _load_local_c4(split_name, data_path):
    c4_dir = Path(data_path) / "c4"
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


def get_c4(nsamples, seed, seqlen, tokenizer, data_path):
    traindata = _load_local_c4("train", data_path)
    valdata = _load_local_c4("validation", data_path)
    local_train_available = traindata is not None
    if traindata is None:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    if valdata is None:
        if local_train_available:
            valdata = traindata
        else:
            valdata = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[idx]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    return trainloader, TokenizerWrapper(valenc)


def get_loaders(name="c4", nsamples=128, seed=0, seqlen=2048, tokenizer=None, data_path=None):
    if "c4" not in name:
        raise ValueError(f"Structured Wanda only supports c4 calibration, got: {name}")
    return get_c4(nsamples, seed, seqlen, tokenizer, data_path)
