import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.hf_llama.modeling_llama import LlamaForCausalLM

from importlib.metadata import version

from lib.prune import prune_wanda_sp, prune_flap, prune_magnitude_sp, check_sparsity
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model, 
    #     torch_dtype=torch.float16, 
    #     cache_dir=cache_dir, 
    #     low_cpu_mem_usage=True, 
    #     device_map="auto"
    # )
    model = LlamaForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        # device_map="auto"
    )
    
    for i in range(32):
        model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device='cpu'))  # 或 'cuda'
        model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device='cpu'))  # 或 'cuda'
        torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        
    model.seqlen = 128
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=2048, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap", "wanda_sp", "mag_sp"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()
    
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # Prune the model
    print("pruning starts")
    if args.prune_method == "flap":
        if args.metrics == 'N/A':
            raise ValueError("For FLAP pruning, the metrics parameter must be chosen from ['IFV', 'WIFV', 'WIFN']. 'N/A' is not a valid choice.")  
        if args.structure == 'N/A':
            raise ValueError("For FLAP pruning, the compressed model structure parameter must be chosen from ['UL-UM', 'UL-MM', 'AL-MM', 'AL-AM']. 'N/A' is not a valid choice.")  
        prune_flap(args, model, tokenizer, device)
    elif args.prune_method == "wanda_sp":
        prune_wanda_sp(args, model, tokenizer, device)
    elif args.prune_method == "mag_sp":
        prune_magnitude_sp(args, model, tokenizer, device)

    # Check the sparsity of the model
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*"*30)
    # Evaluate the model
    if args.eval:
        ppl = eval_ppl(model, tokenizer, device)    
        print(f"ppl on wikitext {ppl}")
        
    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        # torch.save(model, f'{args.save_model}/pruned_model.pt')
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    

if __name__ == '__main__':
    main()