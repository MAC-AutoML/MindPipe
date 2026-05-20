import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random

c4_path = "/SliderQuant/datasets_local/c4"

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    print(f"get_wikitext2 from_start:False")
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    print(f"wikitext train total tokens:{trainenc.input_ids.shape[1]}")

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


def get_c4(nsamples, seed, seqlen, model,from_start=False):
    print(f"get_c4 from_start:{from_start}")
    # import ipdb;ipdb.set_trace()

    traindata = load_dataset(
            c4_path, data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )


    valdata = load_dataset(
       c4_path, data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        if from_start is True:
            i = 0
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
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
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc



def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='',args=None
):

    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)

    if 'c4' in name:
        if "start" in name:
            from_start = True
        else:
            from_start = False
        return get_c4(nsamples, seed, seqlen, model,from_start=from_start)
