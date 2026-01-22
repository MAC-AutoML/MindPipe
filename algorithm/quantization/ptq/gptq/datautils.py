# Code adapted from https://github.com/IST-DASLab/gptq
# Modified to use local datasets

import numpy as np
import random
import torch
import os
from datasets import load_dataset

# 本地数据集目录 (algorithm/datasets)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'datasets'
)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class TokenizerWrapper:
    """Wrapper for tokenized input IDs"""
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, model):
    """Load and process wikitext2 dataset from local files."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    wikitext_dir = os.path.join(DATA_DIR, 'wikitext2')
    train_file = os.path.join(wikitext_dir, 'wiki.train.raw')
    test_file = os.path.join(wikitext_dir, 'wiki.test.raw')

    print(f"Loading wikitext2 from local: {wikitext_dir}")
    traindata = load_dataset('text', data_files=train_file, split='train')
    testdata = load_dataset('text', data_files=test_file, split='train')

    # Encode datasets
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
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


def get_c4(nsamples, seed, seqlen, model):
    """Load and process C4 dataset from local files."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    c4_dir = os.path.join(DATA_DIR, 'c4')
    train_file = os.path.join(c4_dir, 'c4-train.00000-of-01024.json.gz')

    print(f"Loading C4 from local: {train_file}")
    traindata = load_dataset('json', data_files=train_file, split='train')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset (use part of train data since we don't have validation file)
    random.seed(0)
    valenc = tokenizer(' '.join(traindata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    """
    Get data loaders for calibration and evaluation.

    Args:
        name: Dataset name ('wikitext2' or 'c4')
        nsamples: Number of calibration samples
        seed: Random seed
        seqlen: Sequence length
        model: Model name/path for tokenizer

    Returns:
        trainloader: List of (input, target) tuples for calibration
        testloader: Tokenized test data for evaluation
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)

    raise ValueError(f"Unknown dataset: {name}. Supported: wikitext2, c4")
