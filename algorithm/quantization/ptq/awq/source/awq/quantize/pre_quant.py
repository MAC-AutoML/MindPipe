import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from algorithm.common.device import empty_cache
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
try:
    from tinychat.models import LlavaLlamaForCausalLM
except ImportError as e:
    pass

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip
from .forward_utils import forward_in_chunks
from ..utils.device import resolve_device

__all__ = ["run_awq"]


def _is_qwen3_5_model(model) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return str(model_type) in {"qwen3_5", "qwen3_5_text"}


def get_named_linears(module, model=None, qwen3_5_quantize_linear_attn=False):
    linears = {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
    if model is not None and _is_qwen3_5_model(model):
        layer_type = getattr(module, "layer_type", None)
        # Keep Qwen3.5 attention in higher precision by default. The linear_attn
        # path can be enabled explicitly for targeted AWQ experiments.
        if layer_type == "full_attention":
            linears = {name: m for name, m in linears.items() if name.startswith("mlp.")}
        elif layer_type == "linear_attention" and not qwen3_5_quantize_linear_attn:
            linears = {name: m for name, m in linears.items() if name.startswith("mlp.")}
    return linears


def get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVL3":
        layers = model.language_model.model.layers
        # layers = [model.language_model.model.layers, model.vision_model.encoder.layers]
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        root = model.model.language_model
        root.embed_tokens = root.embed_tokens.to(device)
        if hasattr(root, "rotary_emb"):
            root.rotary_emb = root.rotary_emb.to(device)
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        root = model.language_model
        root.embed_tokens = root.embed_tokens.to(device)
        if hasattr(root, "rotary_emb"):
            root.rotary_emb = root.rotary_emb.to(device)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternVL3":
        model.language_model.model.embed_tokens = (
            model.language_model.model.embed_tokens.to(device)
        )
        model.language_model.model.rotary_emb = (
            model.language_model.model.rotary_emb.to(device)
        )
        model.vision_model.embeddings.to(device)
    elif isinstance(model, LlavaLlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif "llavallamamodel" in str(model.__class__).lower():
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def get_model_input_device(model, device=None):
    input_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return input_embeddings.weight.device
    return resolve_device(device)


def _build_dense_causal_mask(input_ids, model):
    batch_size, sequence_length = input_ids.shape
    embedding_module = model.get_input_embeddings()
    mask_dtype = embedding_module.weight.dtype
    mask_device = input_ids.device
    min_value = torch.finfo(mask_dtype).min
    causal_mask = torch.zeros(
        (sequence_length, sequence_length),
        dtype=mask_dtype,
        device=mask_device,
    )
    blocked = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=mask_device),
        diagonal=1,
    )
    causal_mask.masked_fill_(blocked, min_value)
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, sequence_length, sequence_length)


def run_text_forward(model, input_ids):
    if model.__class__.__name__ == "LlavaLlamaModel":
        model.llm(input_ids, use_cache=False)
        return
    if model.__class__.__name__ == "InternVL3":
        model.language_model(input_ids, use_cache=False)
        return
    if getattr(getattr(model, "config", None), "model_type", None) == "qwen2_5_vl" and hasattr(model, "model") and hasattr(model.model, "language_model"):
        dense_causal_mask = _build_dense_causal_mask(input_ids, model.model.language_model)
        attention_mask = {
            "full_attention": dense_causal_mask,
            "sliding_attention": dense_causal_mask,
        }
        model.model.language_model(input_ids, attention_mask=attention_mask, use_cache=False)
        return
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.model.language_model(input_ids, use_cache=False)
        return
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        model.language_model(input_ids, use_cache=False)
        return
    model(input_ids, use_cache=False)


@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    clip_targets="auto",
    qwen3_5_quantize_linear_attn=False,
    # some configs for ablation study
    calib_data="pileval",
    device=None,
    data_path=None,
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    runtime_device = resolve_device(device)

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to(runtime_device)

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen,
        data_path=data_path,
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].to(runtime_device)
    move_embed(model, runtime_device)

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    model_input_device = get_model_input_device(model, device=runtime_device)
    try:
        run_text_forward(model, samples.to(model_input_device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    empty_cache(runtime_device)

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.to(runtime_device)
        named_linears = get_named_linears(
            layer,
            model=model,
            qwen3_5_quantize_linear_attn=qwen3_5_quantize_linear_attn,
        )

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = forward_in_chunks(model, layer, inps, layer_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        empty_cache(runtime_device)

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                model,
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                qwen3_5_quantize_linear_attn=qwen3_5_quantize_linear_attn,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat, device=runtime_device)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        empty_cache(runtime_device)
        # for line in torch.cuda.memory_summary().splitlines():
        #     if "Allocated" in line:
        #         print(line)

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                device=runtime_device,
                model=model,
                clip_targets=clip_targets,
            )
            apply_clip(layer, clip_list, device=runtime_device)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        empty_cache(runtime_device)
        # for line in torch.cuda.memory_summary().splitlines():
        #     if "Allocated" in line:
        #         print(line)

    return awq_results


def apply_awq(model, awq_results, device=None):
    runtime_device = resolve_device(device)
    apply_scale(model, awq_results["scale"], device=runtime_device)
    apply_clip(model, awq_results["clip"], device=runtime_device)
