import os
import pickle
import datasets
import random
import transformers

# 数据集根目录
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'datasets'
)

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    wikitext_dir = os.path.join(DATA_DIR, 'wikitext2')
    train_file = os.path.join(wikitext_dir, 'wiki.train.raw')
    test_file = os.path.join(wikitext_dir, 'wiki.test.raw')

    if eval_mode:
        testdata = datasets.load_dataset('text', data_files=test_file, split='train')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('text', data_files=train_file, split='train')
        traindata = traindata.filter(lambda x: len(x['text']) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    c4_dir = os.path.join(DATA_DIR, 'c4')
    train_file = os.path.join(c4_dir, 'c4-train.00000-of-01024.json.gz')

    if eval_mode:
        # 用训练集的一部分作为验证集
        traindata = datasets.load_dataset('json', data_files=train_file, split='train')
        valenc = tokenizer(' '.join(traindata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset('json', data_files=train_file, split='train')
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
        return trainloader


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seqlen, tokenizer):
    traindata = datasets.load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    args, name, tokenizer, nsamples=128, seqlen=2048, eval_mode=False
):
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seqlen, tokenizer)

    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset
