# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import os
from datasets import load_dataset

# 本地数据集目录 (algorithm/datasets)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'datasets')

# Set random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch for reproducibility.

    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset from local directory.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    wikitext_dir = os.path.join(DATA_DIR, 'wikitext2')
    train_file = os.path.join(wikitext_dir, 'wiki.train.raw')
    test_file = os.path.join(wikitext_dir, 'wiki.test.raw')

    print(f"Loading wikitext2 from local: {wikitext_dir}")
    traindata = load_dataset('text', data_files=train_file, split='train')
    testdata = load_dataset('text', data_files=test_file, split='train')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
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
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset from local directory.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    c4_dir = os.path.join(DATA_DIR, 'c4')
    train_file = os.path.join(c4_dir, 'c4-train.00000-of-01024.json.gz')

    print(f"Loading C4 from local: {train_file}")
    traindata = load_dataset('json', data_files=train_file, split='train')

    # Generate samples from training set using random seed and specified sequence length
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

    # Prepare validation dataset (use part of train data)
    valenc = tokenizer(' '.join(traindata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2' or 'c4').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation/test set.
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
