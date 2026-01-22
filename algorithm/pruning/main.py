"""
Unified Pruning Algorithm Entry Point
Supports both structured (FLAP) and unstructured (Wanda/SparseGPT) pruning methods.

Usage:
    # Unstructured pruning (wanda)
    python main.py --task pruning --algorithm wanda --prune_method wanda --model <model_path> --sparsity_ratio 0.5

    # Structured pruning (flap)
    python main.py --task pruning --algorithm flap --prune_method flap --model <model_path> --pruning_ratio 0.5

    # Generate text with pruned model
    python main.py --task pruning --algorithm wanda --prune_method wanda --model <model_path> --sparsity_ratio 0.5 \\
        --save_model ./pruned_model --generate --prompts "Hello, how are you?"
"""

import argparse
import os
import sys

# Add parent directory to path for imports
ALGORITHM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ALGORITHM_ROOT not in sys.path:
    sys.path.insert(0, ALGORITHM_ROOT)


def run_wanda(args):
    """Run wanda-based unstructured pruning algorithms."""
    import os
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from importlib.metadata import version

    from pruning.unstructured.wanda.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
    from pruning.unstructured.wanda.eval import eval_ppl, eval_zero_shot

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    def get_llm(model_name, cache_dir="llm_weights"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.seqlen = model.config.max_position_embeddings
        return model

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True
        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Model saved to {args.save_model}")

    # Generate text if requested
    if args.generate:
        generate_text(model, tokenizer, device, args)

    return model, tokenizer, device


def run_flap(args):
    """Run FLAP-based structured pruning algorithms."""
    import os
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from pruning.models.hf_llama.modeling_llama import LlamaForCausalLM
    from importlib.metadata import version

    from pruning.structured.flap.prune import prune_wanda_sp, prune_flap, prune_magnitude_sp, check_sparsity
    from pruning.structured.flap.eval import eval_ppl

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    def get_llm(model_path, cache_dir="llm_weights"):
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        num_layers = model.config.num_hidden_layers
        for i in range(num_layers):
            model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
                torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device='cpu')
            )
            model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device='cpu')
            )
            torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
            torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        model.seqlen = 128
        return model

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

    if "30b" in args.model or "65b" in args.model:
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
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*" * 30)

    # Evaluate the model
    if args.eval:
        ppl = eval_ppl(model, tokenizer, device)
        print(f"ppl on wikitext {ppl}")

    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Model saved to {args.save_model}")

    # Generate text if requested
    if args.generate:
        generate_text(model, tokenizer, device, args)

    return model, tokenizer, device


def generate_text(model, tokenizer, device, args):
    """Generate text using the pruned model."""
    import torch

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "top_k": args.top_k if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
    }
    # Remove None values
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    prompts = args.prompts if args.prompts else ["Hello, how are you?"]

    print("\n" + "=" * 50)
    print("Text Generation Results")
    print("=" * 50)

    results = []
    for prompt in prompts:
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if input_ids[0][-1] == tokenizer.eos_token_id:
                input_ids = input_ids[:, :-1]
            input_ids = input_ids.to(device)

            generated_ids = model.generate(input_ids, **generate_kwargs)
            result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
            results.append(result)

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {result}")
            print("-" * 50)

    # Save generation results if save path is provided
    if args.save_generation:
        save_path = args.save_generation
        if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, "w", encoding="utf-8") as f:
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                f.write(f"=== Sample {i+1} ===\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {result}\n\n")
        print(f"\nGeneration results saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Pruning Algorithm")

    # Common arguments
    parser.add_argument('--algorithm', type=str, required=True, choices=['wanda', 'flap'],
                        help='Algorithm family: wanda (unstructured) or flap (structured)')
    parser.add_argument('--model', type=str, required=True, help='Model path or HuggingFace model name')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument("--cache_dir", default="llm_weights", type=str, help='Cache directory for model weights')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model')

    # Wanda-specific arguments
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level (wanda)')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"],
                        help='Sparsity type (wanda)')
    parser.add_argument("--prune_method", type=str, default="wanda",
                        choices=["magnitude", "wanda", "sparsegpt", "ablate_mag_seq", "ablate_wanda_seq",
                                 "ablate_mag_iter", "ablate_wanda_iter", "search", "flap", "wanda_sp", "mag_sp"],
                        help='Pruning method')
    parser.add_argument('--use_variant', action="store_true", help='Use wanda variant (appendix)')
    parser.add_argument('--save', type=str, default=None, help='Path to save results (wanda)')
    parser.add_argument("--eval_zero_shot", action="store_true", help='Run zero-shot evaluation (wanda)')

    # FLAP-specific arguments
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio (flap)')
    parser.add_argument('--remove_heads', type=int, default=8, help='Number of heads to remove (flap)')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", "N/A"],
                        help='Metrics for FLAP pruning')
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", "N/A"],
                        help='Compressed model structure (flap)')
    parser.add_argument('--unstr', action="store_true", help='Unstructured pruning flag (flap)')
    parser.add_argument('--eval', action="store_true", help='Run evaluation (flap)')

    # Generation arguments
    parser.add_argument('--generate', action="store_true", help='Generate text after pruning')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Prompts for text generation (can specify multiple)')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (0 for greedy)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--save_generation', type=str, default=None, help='Path to save generation results')

    args = parser.parse_args()

    if args.algorithm == 'wanda':
        run_wanda(args)
    elif args.algorithm == 'flap':
        run_flap(args)


if __name__ == '__main__':
    main()
