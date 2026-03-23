# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

from pathlib import Path
import numpy as np
import random
import torch
from datasets import load_dataset

DATA_ROOT_CANDIDATES = (
    Path("/home/ma-user/work/algorithm-v1/datasets"),
    Path("/mnt/42_store/lcw/data2/Huawei/datasets"),
)


def _resolve_data_root() -> Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATA_ROOT_CANDIDATES[0]


DATA_ROOT = _resolve_data_root()
WIKITEXT2_DIR = DATA_ROOT / "wikitext2"
C4_DIR = DATA_ROOT / "c4"


def _load_local_wikitext(split_name):
    split_to_file = {
        "train": WIKITEXT2_DIR / "wiki.train.raw",
        "test": WIKITEXT2_DIR / "wiki.test.raw",
        "validation": WIKITEXT2_DIR / "wiki.valid.raw",
    }
    file_path = split_to_file[split_name]
    if not file_path.exists():
        parquet_path = WIKITEXT2_DIR / f"{split_name}-00000-of-00001.parquet"
        if parquet_path.exists():
            return load_dataset("parquet", data_files={split_name: str(parquet_path)}, split=split_name)
        return None
    return load_dataset("text", data_files=str(file_path), split="train")


def _load_local_c4(split_name):
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

# Load and process PTB (Penn Treebank) dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    """
    Load and process PTB (Penn Treebank) dataset.
    
    Args:
        nsamples (int): Number of samples to extract.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for each sample.
        tokenizer (Tokenizer): Tokenizer to use for text encoding.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test set.
    """
    # Load train and test datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    
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

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = _load_local_wikitext("train")
    testdata = _load_local_wikitext("test")
    if traindata is None or testdata is None:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
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
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    traindata = _load_local_c4("train")
    valdata = _load_local_c4("validation")
    if traindata is None:
        traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    if valdata is None:
        valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
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

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2', 'c4', or 'ptb').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation/test set.
    """
    # Determine which dataset to use based on 'name' parameter and return corresponding loader
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    
if __name__ == "__main__": 
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)

# Note:
# This script is designed to load and process various text datasets for training language models.
# It includes functions to load PTB (Penn Treebank), Wikitext-2, and C4 (Common Crawl) datasets.
# Each loading function returns a trainloader (list of input and target pairs) and encoded validation/test set.
