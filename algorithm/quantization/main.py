"""
Quantization Algorithm Entry Point

Supports PTQ (GPTQ, AWQ, FlatQuant) and QAT algorithms.

Usage:
    # GPTQ quantization (same as original)
    python main.py --method gptq --model <model_path> --dataset wikitext2 --wbits 4

    # GPTQ with better accuracy (act-order + true-sequential)
    python main.py --method gptq --model <model_path> --dataset wikitext2 --wbits 4 --act-order --true-sequential

    # GPTQ and save quantized model
    python main.py --method gptq --model <model_path> --dataset wikitext2 --wbits 3 --save ./model-3bit.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# Add parent directory to path for imports
ALGORITHM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ALGORITHM_ROOT not in sys.path:
    sys.path.insert(0, ALGORITHM_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Unified Quantization Algorithm")

    # Common arguments
    parser.add_argument('--method', type=str, default='gptq',
                        choices=['gptq', 'awq', 'flatquant'],
                        help='Quantization method')
    parser.add_argument('--model', type=str, required=True,
                        help='Model path or HuggingFace model name')

    # GPTQ-specific arguments (same as original llama.py)
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4'],
                        help='Calibration dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples')
    parser.add_argument('--percdamp', type=float, default=0.01,
                        help='Percent of the average Hessian diagonal to use for dampening')
    parser.add_argument('--wbits', type=int, default=16,
                        choices=[2, 3, 4, 8, 16],
                        help='#bits to use for quantization; use 16 for evaluating base model')
    parser.add_argument('--groupsize', type=int, default=-1,
                        help='Groupsize to use for quantization; default uses full row')
    parser.add_argument('--sym', action='store_true',
                        help='Whether to perform symmetric quantization')
    parser.add_argument('--act-order', action='store_true',
                        help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true',
                        help='Whether to run in true sequential model')
    parser.add_argument('--static-groups', action='store_true',
                        help='Whether to use static groups; recommended when using --act-order')
    parser.add_argument('--nearest', action='store_true',
                        help='Whether to run the RTN baseline')
    parser.add_argument('--save', type=str, default='',
                        help='Save quantized checkpoint under this name')

    # AWQ-specific arguments
    parser.add_argument('--w_bit', type=int, default=4,
                        help='[AWQ] Weight quantization bits')
    parser.add_argument('--q_group_size', type=int, default=128,
                        help='[AWQ] Quantization group size')
    parser.add_argument('--no_zero_point', action='store_true',
                        help='[AWQ] Disable zero point')
    parser.add_argument('--q_backend', type=str, default='fake',
                        choices=['fake', 'real'],
                        help='[AWQ] Quantization backend: fake (pseudo) or real')
    parser.add_argument('--dump_awq', type=str, default=None,
                        help='[AWQ] Save AWQ search results to this path')
    parser.add_argument('--load_awq', type=str, default=None,
                        help='[AWQ] Load AWQ search results from this path')
    parser.add_argument('--dump_fake', type=str, default=None,
                        help='[AWQ] Save pseudo-quantized model to this path')
    parser.add_argument('--dump_quant', type=str, default=None,
                        help='[AWQ] Save real-quantized model to this path')

    # FlatQuant-specific arguments
    parser.add_argument('--abits', type=int, default=16,
                        help='[FlatQuant] Activation quantization bits')
    parser.add_argument('--kv_bits', type=int, default=16,
                        help='[FlatQuant] KV cache quantization bits (applies to both K and V)')
    parser.add_argument('--cali_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4', 'pile'],
                        help='[FlatQuant] Calibration dataset')
    parser.add_argument('--epochs', type=int, default=15,
                        help='[FlatQuant] Number of training epochs')
    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='[FlatQuant] Batch size for calibration')
    parser.add_argument('--flat_lr', type=float, default=1e-5,
                        help='[FlatQuant] Learning rate for learnable transformation')
    parser.add_argument('--cali_trans', action='store_true',
                        help='[FlatQuant] Enable calibration of transformations')
    parser.add_argument('--add_diag', action='store_true',
                        help='[FlatQuant] Add per-channel scaling')
    parser.add_argument('--lwc', action='store_true',
                        help='[FlatQuant] Use learnable weight clipping')
    parser.add_argument('--lac', action='store_true',
                        help='[FlatQuant] Use learnable activation clipping')
    parser.add_argument('--use_gptq', action='store_true',
                        help='[FlatQuant] Use GPTQ for weight quantization (default: RTN)')
    parser.add_argument('--save_matrix', action='store_true',
                        help='[FlatQuant] Save the transformation matrices')
    parser.add_argument('--load_matrix', type=str, default=None,
                        help='[FlatQuant] Load pre-trained transformation matrices from path')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='[FlatQuant] Output directory for saving results')

    args = parser.parse_args()

    if args.method == 'gptq':
        run_gptq(args)
    elif args.method == 'awq':
        run_awq(args)
    elif args.method == 'flatquant':
        run_flatquant(args)


# ============================================================================
# GPTQ Implementation (same logic as original llama.py)
# ============================================================================

def get_llama(model_path):
    """Load LLaMA model with skipped initialization for speed."""
    def skip(*args, **kwargs):
        pass
    # Skip weight initialization for faster loading
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def gptq_sequential(model, dataloader, dev, args):
    """
    Apply GPTQ quantization to model layer by layer.
    (Same as llama_sequential in original llama.py)
    """
    from quantization.ptq.gptq.gptq import GPTQ
    from quantization.ptq.gptq.quant import Quantizer
    from quantization.ptq.gptq.modelutils import find_layers

    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def gptq_eval(model, testenc, dev, args):
    """
    Evaluate model perplexity on test data.
    (Same as llama_eval in original llama.py)
    """
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            from quantization.ptq.gptq.quant import Quantizer, quantize
            from quantization.ptq.gptq.modelutils import find_layers
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


def gptq_pack3(model, quantizers):
    """
    Pack quantized weights into 3-bit format for efficient inference.
    (Same as llama_pack3 in original llama.py)
    """
    from quantization.ptq.gptq.modelutils import find_layers
    from quantization.ptq.gptq.quant import Quant3Linear, make_quant3

    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


def run_gptq(args):
    """
    Run GPTQ quantization.
    (Same logic as original llama.py main function)
    """
    from quantization.ptq.gptq.datautils import get_loaders, set_seed
    from quantization.ptq.gptq.modelutils import DEV

    # Load model
    model = get_llama(args.model)
    model.eval()

    # Load calibration data
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen
    )

    # Run GPTQ quantization
    quantizers = None
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = gptq_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    # Evaluate on all datasets (same as original: wikitext2, c4)
    # Note: ptb removed since we don't have local data
    datasets = ['wikitext2', 'c4']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        gptq_eval(model, testloader, DEV, args)

    # Save quantized model (pack to 3-bit if wbits==3)
    if args.save:
        if quantizers is not None:
            gptq_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
        print(f'Model saved to {args.save}')


# ============================================================================
# AWQ Implementation
# ============================================================================

def run_awq(args):
    """
    Run AWQ quantization.
    (Based on original entry.py logic)

    AWQ流程:
    1. 加载模型
    2. run_awq(): 搜索最优scale和clip (可选保存)
    3. apply_awq(): 应用搜索结果
    4. pseudo/real_quantize: 伪量化或真量化
    5. 评估PPL
    6. 保存模型
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from accelerate import infer_auto_device_map, dispatch_model
    from accelerate.utils.modeling import get_balanced_memory

    from quantization.ptq.awq.quantize.pre_quant import run_awq as awq_search, apply_awq
    from quantization.ptq.awq.quantize.quantizer import (
        pseudo_quantize_model_weight,
        real_quantize_model_weight,
    )

    print("=" * 50)
    print("AWQ Quantization")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Bits: {args.w_bit}")
    print(f"Group size: {args.q_group_size}")
    print(f"Zero point: {not args.no_zero_point}")
    print("=" * 50)

    # Quantization config
    q_config = {
        "zero_point": not args.no_zero_point,
        "q_group_size": args.q_group_size,
    }

    # Load model and tokenizer
    print(f"Loading model from {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model.seqlen = 2048

    # Step 1: Run AWQ search (or load from file)
    if args.load_awq:
        print(f"Loading AWQ results from {args.load_awq}...")
        awq_results = torch.load(args.load_awq, map_location="cpu")
    else:
        print("Running AWQ search...")
        awq_results = awq_search(
            model,
            tokenizer,
            w_bit=args.w_bit,
            q_config=q_config,
            n_samples=128,
            seqlen=512,
        )
        # Save AWQ results if requested
        if args.dump_awq:
            os.makedirs(os.path.dirname(args.dump_awq) or '.', exist_ok=True)
            torch.save(awq_results, args.dump_awq)
            print(f"AWQ results saved to {args.dump_awq}")

    # Step 2: Apply AWQ results
    print("Applying AWQ results...")
    apply_awq(model, awq_results)

    # Step 3: Quantize weights
    if args.q_backend == "fake":
        print("Applying pseudo (fake) quantization...")
        pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
        if args.dump_fake:
            model.save_pretrained(args.dump_fake)
            tokenizer.save_pretrained(args.dump_fake)
            print(f"Pseudo-quantized model saved to {args.dump_fake}")
    elif args.q_backend == "real":
        print("Applying real quantization...")
        real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
        if args.dump_quant:
            os.makedirs(os.path.dirname(args.dump_quant) or '.', exist_ok=True)
            torch.save(model.cpu().state_dict(), args.dump_quant)
            print(f"Real-quantized model saved to {args.dump_quant}")
            return  # Exit after saving real quantized model

    # Step 4: Move model to GPU for evaluation
    kwargs = {"max_memory": get_balanced_memory(model)}
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=["LlamaDecoderLayer", "Qwen2DecoderLayer"],
        **kwargs,
    )
    model = dispatch_model(model, device_map=device_map)

    # Step 5: Evaluate PPL on wikitext2
    print("\nEvaluating on wikitext2...")
    awq_eval_ppl(model, tokenizer, model.device)

    # Save fake-quantized model if requested
    if args.save and args.q_backend == "fake":
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        print(f"Model saved to {args.save}")


@torch.no_grad()
def awq_eval_ppl(model, tokenizer, dev):
    """
    Evaluate model perplexity on wikitext2.
    (Same logic as entry.py wikitext evaluation)
    """
    from quantization.ptq.gptq.datautils import get_loaders

    # Load wikitext2 test data
    _, testloader = get_loaders(
        'wikitext2',
        seed=0,
        model=tokenizer.name_or_path,
        seqlen=model.seqlen
    )

    testenc = testloader.input_ids
    nsamples = testenc.numel() // model.seqlen

    model.eval()
    nlls = []

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:].to(dev)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"wikitext2 PPL: {ppl.item():.2f}")


# ============================================================================
# FlatQuant Implementation
# ============================================================================

def run_flatquant(args):
    """Run FlatQuant quantization."""
    import transformers
    from quantization.qat.flatquant import utils as flat_utils_module
    from quantization.qat.flatquant import model_utils as flat_model_utils
    from quantization.qat.flatquant import data_utils as flat_data_utils
    from quantization.qat.flatquant import eval_utils as flat_eval_utils
    from quantization.qat.flatquant import train_utils as flat_train_utils
    from quantization.qat.flatquant import flat_utils
    from quantization.qat.flatquant import gptq_utils as flat_gptq_utils

    print("=" * 50)
    print("FlatQuant Quantization")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Weight bits: {args.wbits}")
    print(f"Activation bits: {args.abits}")
    print(f"KV cache bits: {args.kv_bits}")
    print(f"Calibration dataset: {args.cali_dataset}")
    print("=" * 50)

    # Set random seed
    flat_utils_module.seed_everything(seed=args.seed)

    # Build FlatQuant args object (convert from our args format)
    class FlatQuantArgs:
        pass

    fq_args = FlatQuantArgs()
    fq_args.model = args.model
    fq_args.seed = args.seed
    fq_args.hf_token = None

    # Weight quantization
    fq_args.w_bits = args.wbits
    fq_args.w_groupsize = args.groupsize
    fq_args.w_asym = not args.sym
    fq_args.gptq = args.use_gptq
    fq_args.gptq_mse = False
    fq_args.percdamp = args.percdamp
    fq_args.act_order = args.act_order if hasattr(args, 'act_order') else False

    # Activation quantization
    fq_args.a_bits = args.abits
    fq_args.a_groupsize = -1
    fq_args.a_asym = False

    # KV cache quantization
    fq_args.k_bits = args.kv_bits
    fq_args.k_groupsize = -1
    fq_args.k_asym = False
    fq_args.v_bits = args.kv_bits
    fq_args.v_groupsize = -1
    fq_args.v_asym = False
    fq_args.q_bits = 16
    fq_args.q_groupsize = -1
    fq_args.q_asym = False

    # FlatQuant calibration
    fq_args.cali_dataset = args.cali_dataset
    fq_args.nsamples = args.nsamples
    fq_args.cali_bsz = args.cali_bsz
    fq_args.epochs = args.epochs
    fq_args.flat_lr = args.flat_lr
    fq_args.cali_trans = args.cali_trans
    fq_args.add_diag = args.add_diag
    fq_args.lwc = args.lwc
    fq_args.lac = args.lac
    fq_args.diag_init = "sq_style"
    fq_args.diag_alpha = 0.3
    fq_args.warmup = False
    fq_args.deactive_amp = False
    fq_args.direct_inv = False
    fq_args.separate_vtrans = False

    # Save/load
    fq_args.save_matrix = args.save_matrix
    fq_args.reload_matrix = args.load_matrix is not None
    fq_args.matrix_path = args.load_matrix
    fq_args.resume = False
    fq_args.output_dir = args.output_dir
    fq_args.exp_name = "exp"
    fq_args.model_name = args.model.split("/")[-1]
    fq_args.exp_dir = os.path.join(args.output_dir, fq_args.model_name, f"w{fq_args.w_bits}a{fq_args.a_bits}", fq_args.exp_name)
    fq_args.quantized_save = False
    fq_args.distribute_model = False

    # 创建输出目录
    os.makedirs(fq_args.exp_dir, exist_ok=True)

    # Determine if quantization is needed
    fq_args.quantize = (fq_args.w_bits < 16) or (fq_args.a_bits < 16) or \
                       (fq_args.k_bits < 16) or (fq_args.v_bits < 16)

    # Load model
    print(f"Loading model from {args.model}...")
    model, apply_flatquant_to_model = flat_model_utils.get_model(args.model, fq_args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Get calibration data
    print("Loading calibration data...")
    trainloader = flat_data_utils.get_loaders(
        fq_args, fq_args.cali_dataset, tokenizer,
        nsamples=fq_args.nsamples, seqlen=model.seqlen, eval_mode=False,
    )
    print("Finished loading training data.")

    # Apply FlatQuant transformations
    if fq_args.quantize:
        print("Applying FlatQuant to model...")
        model = apply_flatquant_to_model(fq_args, model)
        print("Finished applying FlatQuant to model.")

        if fq_args.resume:
            flat_utils.load_flat_parameters(fq_args, model)
        elif fq_args.reload_matrix:
            flat_utils.load_flat_matrices(fq_args, model, path=fq_args.matrix_path)
        elif (fq_args.cali_trans or fq_args.add_diag or fq_args.lwc or fq_args.lac):
            print("Calibrating FlatQuant transformations...")
            # 创建简单的 logger
            import logging
            logger = logging.getLogger("flatquant")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(message)s'))
                logger.addHandler(handler)
            flat_train_utils.cali_flat_quant(fq_args, model, trainloader, flat_utils_module.DEV, logger=logger)

        if fq_args.save_matrix and not fq_args.reload_matrix:
            flat_utils.save_flat_matrices(fq_args, model)

        flat_utils.reparameterize_model(model)
        print("Finished reparameterizing model.")

    # Weight quantization (GPTQ or RTN)
    if fq_args.w_bits < 16:
        print(f"Quantizing weights to {fq_args.w_bits} bits using {'GPTQ' if fq_args.gptq else 'RTN'}...")
        if fq_args.gptq:
            quantizers = flat_gptq_utils.gptq_fwrd(model, trainloader, flat_utils_module.DEV, fq_args)
        else:
            quantizers = flat_gptq_utils.rtn_fwrd(model, flat_utils_module.DEV, fq_args)
        print("Finished weight quantization.")

    # Move model to GPU for evaluation
    if fq_args.distribute_model:
        flat_utils_module.distribute_model(model)
    else:
        model.to(flat_utils_module.DEV)

    # Evaluate PPL
    print("\nEvaluating perplexity...")
    for eval_dataset in ["wikitext2"]:
        print(f"Evaluating on {eval_dataset}...")
        testloader = flat_data_utils.get_loaders(
            fq_args,
            eval_dataset,
            tokenizer,
            seqlen=model.seqlen,
            eval_mode=True
        )
        dataset_ppl = flat_eval_utils.ppl_eval(model, testloader)
        print(f"{eval_dataset} PPL: {dataset_ppl:.2f}")


if __name__ == '__main__':
    main()
