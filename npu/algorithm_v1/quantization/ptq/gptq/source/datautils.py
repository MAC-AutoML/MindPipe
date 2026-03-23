import random
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = Path(__file__).resolve().parent.parent / 'datasets'
WIKITEXT2_DIR = DATA_ROOT / 'wikitext2'
C4_DIR = DATA_ROOT / 'c4'


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def _load_local_wikitext2(split):
    from datasets import load_dataset

    split_to_file = {
        'train': WIKITEXT2_DIR / 'wiki.train.raw',
        'test': WIKITEXT2_DIR / 'wiki.test.raw',
        'validation': WIKITEXT2_DIR / 'wiki.valid.raw',
    }
    file_path = split_to_file[split]
    if not file_path.exists():
        return None
    return load_dataset('text', data_files=str(file_path), split='train')


def _sample_train_sequences(trainenc, nsamples, seed, seqlen):
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    traindata = _load_local_wikitext2('train')
    testdata = _load_local_wikitext2('test')
    if traindata is None or testdata is None:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return _sample_train_sequences(trainenc, nsamples, seed, seqlen), testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')
    return _sample_train_sequences(trainenc, nsamples, seed, seqlen), testenc


def _load_local_c4(split):
    from datasets import load_dataset

    split_to_file = {
        'train': C4_DIR / 'c4-train.00000-of-01024.json.gz',
        'validation': C4_DIR / 'c4-validation.00000-of-00008.json.gz',
    }
    file_path = split_to_file[split]
    if not file_path.exists():
        return None
    return load_dataset('json', data_files={split: str(file_path)}, split=split)


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    traindata = _load_local_c4('train')
    valdata = _load_local_c4('validation')
    if traindata is None:
        traindata = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train'
        )
    if valdata is None:
        valdata = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation'
        )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

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

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return trainloader, TokenizerWrapper(valenc)


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer(' '.join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(' '.join(testdata['sentence']), return_tensors='pt')
    return _sample_train_sequences(trainenc, nsamples, seed, seqlen), testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    traindata = _load_local_c4('train')
    valdata = _load_local_c4('validation')
    if traindata is None:
        traindata = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train'
        )
    if valdata is None:
        valdata = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation'
        )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

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

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    return trainloader, TokenizerWrapper(valenc)


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    raise ValueError(f'Unknown dataset: {name}')
