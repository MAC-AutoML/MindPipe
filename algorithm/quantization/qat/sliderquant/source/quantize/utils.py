from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus
import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from datautils import get_loaders
from accelerate.utils import get_balanced_memory
from accelerate import dispatch_model,infer_auto_device_map
import copy


def get_max_memory_map(ratio=0.95):
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory
        mem_in_gib = int(total_mem * ratio / (1024 ** 3))  # 转换为 GiB
        max_memory[i] = f"{mem_in_gib}GiB"
    return max_memory

@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    assert not (args.multigpu and  args.parallelize)
    if args.multigpu:

        if "llama" in args.net.lower() or "vicuna" in args.net.lower()  or "qwen" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        else:
            raise NotImplementedError("Only support for llama/vicuna/qwen multigpu now")

    elif args.parallelize:
        balanced_mem = get_balanced_memory(
            lm.model,
            max_memory=get_max_memory_map(0.95),
            no_split_module_classes=["LlamaDecoderLayer","QuantLlamaDecoderLayer","Qwen3MoeDecoderLayer"]
        )
        logger.info(f"mem is {balanced_mem}")
        device_map = infer_auto_device_map(
            lm.model,
            max_memory=balanced_mem,
            no_split_module_classes=["LlamaDecoderLayer","QuantLlamaDecoderLayer","Qwen3MoeDecoderLayer"]
        )
        lm.model = dispatch_model(lm.model,device_map=device_map)

    else:
        if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        else:
            raise NotImplementedError("Only support for llama/vicuna/qwen single gpu now")



    if args.eval_ppl:
        logger.info(f"model seqlen is {lm.seqlen}")
        datasets = args.test_datasets.split(",")
        for dataset in datasets:
        # for dataset in ["wikitext2"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.net}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                    args=args
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            all_hidden_states = []
            output_hidden_states  = False

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                outputs = lm.model.model(batch,output_hidden_states=output_hidden_states)
                hidden_states = outputs[0]

                if  hasattr(lm.model.lm_head,"bias") and lm.model.lm_head.bias is not None:
                    lm.model.lm_head.bias = torch.nn.Parameter(lm.model.lm_head.bias.to(lm.model.lm_head.weight.device))
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break


            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    if args.tasks != "":
        args.tasks = args.tasks.split(",")
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM
        print(f"use lm_eval in {lm_eval}")

        task_manager = lm_eval.tasks.TaskManager(include_path="./datasets_local/lm_eval_configs/tasks", include_defaults=True)
        hflm = HFLM(pretrained=lm.model,tokenizer=lm.tokenizer, batch_size=args.lm_eval_batch_size)
        t_results = lm_eval.simple_evaluate(hflm, tasks=args.tasks, batch_size=args.lm_eval_batch_size,task_manager=task_manager)['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in t_results.items()}


        logger.info(metric_vals)
        pprint(metric_vals)
        if args.eval_ppl is True:
            metric_vals.update(results)

        reported_metric_vals = {}
        for k,v in metric_vals.items():
            if "mmlu" in k:
                if k == "mmlu":
                    reported_metric_vals[k] = v
            else:
                reported_metric_vals[k] = v

        import pandas as pd
        if os.path.exists(f"{args.output_dir}/results.csv"):
            df = pd.read_csv(f"{args.output_dir}/results.csv")
            new_df = pd.DataFrame(reported_metric_vals,index=[0])
            df[new_df.columns] = new_df
        else:
            df = pd.DataFrame(reported_metric_vals,index=[0])
        if args.eval_ppl:
             new_columns = ['wikitext2','c4']
        else:
             new_columns = []
        new_columns += args.tasks

        if len(args.tasks) >= 5:
            df["avg-5"] = df[['piqa','arc_easy', 'arc_challenge','hellaswag', 'winogrande']].mean(axis=1)
        if len(args.tasks) >= 6:
            df["avg-6"] = df[['piqa','arc_easy', 'arc_challenge','hellaswag', 'winogrande', 'boolq']].mean(axis=1)
        if 'mmlu' in args.tasks and len(args.tasks) >= 7:
            df["avg-7"] = df[['piqa','arc_easy', 'arc_challenge','hellaswag', 'winogrande', 'boolq','mmlu']].mean(axis=1)

        logger.info(df)
        df.to_csv(f"{args.output_dir}/results.csv",index=False)

    model = lm.model
    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")

    return results




def get_slider_parameters(sub_layers, use_list=["scale","alpha","shift"]):
    params = []
    print(f"get {use_list} parameters!")
    for sub_layer_idx in range(len(sub_layers)):
        for n, m in sub_layers[sub_layer_idx].named_parameters():
            if any(n.find(t) > -1 for t in use_list) and not n.find('bound_factor') > -1:
                params.append(m)
                # print(n)
    # print(params)
    return iter(params)

def get_lwc_parameters(sub_layers):
    params = []

    print("get lwc parameters!")
    for sub_layer_idx in range(len(sub_layers)):
        for n, m in sub_layers[sub_layer_idx].named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
    return iter(params)

def try_delete_object(object,logger,name=None):
    try:
        del object
    except Exception as e:
        logger.info(f"del tensor occurs {e}, skip!")
    else:
        logger.info(f"del tensor {name} successfully!")


def cleanup_memory(verbos=True,logger=None) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos and logger:
            logger.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def slider_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find('lora_') > -1 or name.find('Q_') > -1 :
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()
