# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

from pathlib import Path
import numpy as np
import random
import torch
from datasets import load_dataset


def _load_local_wikitext(split_name, data_path):
    wikitext2_dir = Path(data_path) / "wikitext2"
    split_to_file = {
        "train": wikitext2_dir / "wiki.train.raw",
        "test": wikitext2_dir / "wiki.test.raw",
        "validation": wikitext2_dir / "wiki.valid.raw",
    }
    file_path = split_to_file[split_name]
    if not file_path.exists():
        return None
    return load_dataset("text", data_files=str(file_path), split="train")


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

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process PTB (Penn Treebank) dataset
def get_ptb(nsamples, seed, seqlen, tokenizer, data_path=None):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer, data_path):
    traindata = _load_local_wikitext("train", data_path)
    testdata = _load_local_wikitext("test", data_path)
    if traindata is None or testdata is None:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process C4 (Common Crawl) dataset
def get_c4(nsamples, seed, seqlen, tokenizer, data_path):
    traindata = _load_local_c4("train", data_path)
    valdata = _load_local_c4("validation", data_path)
    if traindata is None:
        traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    if valdata is None:
        valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None, data_path=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, data_path)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, data_path)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer, data_path)
