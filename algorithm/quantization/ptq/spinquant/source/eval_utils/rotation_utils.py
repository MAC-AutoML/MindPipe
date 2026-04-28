# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from algorithm.common.device import empty_cache
from algorithm.common.device import preferred_rotation_dtype
from algorithm.common.device import resolve_device
from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform


def _resolve_text_root_and_prefix(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model, "model.language_model.layers"
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model, "language_model.layers"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "model.layers"
    raise NotImplementedError(f"Unsupported SpinQuant backbone root: {type(model)}")


def _resolve_decoder_config(model, root):
    for config in (
        getattr(root, "config", None),
        getattr(model, "config", None),
    ):
        if config is None:
            continue
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return text_config
        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            return config
    raise AttributeError(f"Cannot resolve decoder config for model={type(model)} root={type(root)}")


def _resolve_attention_head_dim(decoder_config) -> int:
    head_dim = getattr(decoder_config, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    hidden_size = int(getattr(decoder_config, "hidden_size"))
    num_heads = int(getattr(decoder_config, "num_attention_heads"))
    return hidden_size // num_heads


def _resolve_r2_key(checkpoint, layer_index, preferred_prefix):
    key_candidates = [
        f"{preferred_prefix}.{layer_index}.self_attn.R2",
        f"model.layers.{layer_index}.self_attn.R2",
        f"language_model.layers.{layer_index}.self_attn.R2",
        f"model.language_model.layers.{layer_index}.self_attn.R2",
        f"text_backbone.layers.{layer_index}.self_attn.R2",
    ]
    seen = set()
    for candidate in key_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in checkpoint:
            return checkpoint[candidate]
    raise KeyError(f"Missing SpinQuant R2 for layer {layer_index} in rotation checkpoint.")


def _get_extra_rotation_modules(model):
    """Return extra projector-like modules that should follow text-space rotation."""
    multimodal_root = getattr(model, "model", model)
    visual = getattr(multimodal_root, "visual", None)
    merger = getattr(visual, "merger", None)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if (
        model_type == "qwen2_vl"
        and merger is not None
        and hasattr(merger, "mlp")
        and isinstance(merger.mlp[-1], torch.nn.Linear)
    ):
        return [merger.mlp[-1]]
    if (
        model_type == "qwen2_5_vl"
        and merger is not None
        and hasattr(merger, "mlp")
        and isinstance(merger.mlp[-1], torch.nn.Linear)
    ):
        return [merger.mlp[-1]]
    if (
        model_type == "qwen3_vl"
        and merger is not None
        and hasattr(merger, "linear_fc2")
        and isinstance(merger.linear_fc2, torch.nn.Linear)
    ):
        modules = [merger.linear_fc2]
        deepstack_mergers = getattr(visual, "deepstack_merger_list", None)
        if deepstack_mergers is not None:
            for deepstack_merger in deepstack_mergers:
                if hasattr(deepstack_merger, "linear_fc2") and isinstance(
                    deepstack_merger.linear_fc2, torch.nn.Linear
                ):
                    modules.append(deepstack_merger.linear_fc2)
        return modules
    if (
        model_type == "qwen3_5"
        and merger is not None
        and hasattr(merger, "linear_fc2")
        and isinstance(merger.linear_fc2, torch.nn.Linear)
    ):
        return [merger.linear_fc2]
    return []


def _get_token_mixer(layer):
    mixer = getattr(layer, "self_attn", None)
    if mixer is not None:
        return mixer
    mixer = getattr(layer, "linear_attn", None)
    if mixer is not None:
        return mixer
    raise AttributeError(f"Unsupported decoder layer without token mixer: {type(layer)}")


def rotate_extra_modules(modules, R1: torch.Tensor) -> None:
    """Rotate VLM bridge outputs into the same rotated text basis."""
    rotation_dtype = preferred_rotation_dtype(R1.device)
    for module in modules:
        weight_dtype = module.weight.data.dtype
        weight = module.weight.data.to(dtype=rotation_dtype)
        module.weight.data = torch.matmul(R1.to(weight.device).T, weight).to(dtype=weight_dtype)
        if module.bias is not None:
            bias_dtype = module.bias.data.dtype
            bias = module.bias.data.to(dtype=rotation_dtype)
            module.bias.data = torch.matmul(R1.to(bias.device).T, bias).to(dtype=bias_dtype)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    empty_cache(device)
    random_matrix = torch.randn(size, size, dtype=preferred_rotation_dtype(device)).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=None):
    device = resolve_device(device)
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    rotation_dtype = preferred_rotation_dtype(R1.device)
    root, _ = _resolve_text_root_and_prefix(model)
    for W in [root.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(dtype=rotation_dtype)
        W.weight.data = torch.matmul(W_, R1.to(W_.device)).to(dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    rotation_dtype = preferred_rotation_dtype(R1.device)
    token_mixer = _get_token_mixer(layer)
    if hasattr(token_mixer, "q_proj"):
        linears = [token_mixer.q_proj, token_mixer.k_proj, token_mixer.v_proj]
    else:
        linears = [
            token_mixer.in_proj_qkv,
            token_mixer.in_proj_z,
            token_mixer.in_proj_b,
            token_mixer.in_proj_a,
        ]
    for W in linears:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=rotation_dtype)
        W.weight.data = torch.matmul(W_, R1.to(W_.device)).to(dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    token_mixer = _get_token_mixer(layer)
    W = token_mixer.o_proj if hasattr(token_mixer, "o_proj") else token_mixer.out_proj
    rotation_dtype = preferred_rotation_dtype(R1.device)

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=rotation_dtype)
    W.weight.data = torch.matmul(R1.to(W_.device).T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=rotation_dtype)
        W.bias.data = torch.matmul(R1.to(b.device).T, b).to(dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    rotation_dtype = preferred_rotation_dtype(R1.device)
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=rotation_dtype)
        W.weight.data = torch.matmul(W_, R1.to(W_.device)).to(dtype=dtype)


def rotate_mlp_output(layer, R1):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    rotation_dtype = preferred_rotation_dtype(R1.device)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=rotation_dtype)
    W.weight.data = torch.matmul(R1.to(W_.device).T, W_).to(dtype=dtype)
    try:
        apply_exact_had_to_linear(
            W, had_dim=-1, output=False, device=W.weight.device
        )  # apply exact (inverse) hadamard on the weights of mlp output
    except (AssertionError, ValueError):
        pass
    if W.bias is not None:
        b = W.bias.data.to(dtype=rotation_dtype)
        W.bias.data = torch.matmul(R1.to(b.device).T, b).to(dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    root, _ = _resolve_text_root_and_prefix(model)
    embed_tokens = getattr(root, "embed_tokens", None)
    if (
        embed_tokens is not None
        and hasattr(embed_tokens, "weight")
        and embed_tokens.weight is not None
        and embed_tokens.weight.data_ptr() == W.weight.data_ptr()
    ):
        return
    rotation_dtype = preferred_rotation_dtype(R1.device)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=rotation_dtype)
    W.weight.data = torch.matmul(W_, R1.to(W_.device)).to(dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    if not hasattr(layer, "self_attn"):
        return
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2, device=v_proj.weight.device)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2, device=o_proj.weight.device)


@torch.inference_mode()
def rotate_model(model, args):
    root, layer_key_prefix = _resolve_text_root_and_prefix(model)
    decoder_config = _resolve_decoder_config(model, root)
    hidden_size = int(getattr(decoder_config, "hidden_size"))
    num_heads = int(getattr(decoder_config, "num_attention_heads"))

    R1 = get_orthogonal_matrix(hidden_size, args.rotate_mode)
    checkpoint = None
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        checkpoint = torch.load(R_cpk, map_location="cpu")
        model_device = next(iter(model.parameters())).device
        R1 = checkpoint["R1"].to(device=model_device, dtype=preferred_rotation_dtype(model_device))
    head_dim = _resolve_attention_head_dim(decoder_config)

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    rotate_extra_modules(_get_extra_rotation_modules(model), R1)
    utils.cleanup_memory()
    layers = [layer for layer in root.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        # device_map 模式下将 R1 移动到当前层所在设备
        layer_device = next(layer.parameters()).device
        R1_layer = R1.to(device=layer_device)
        rotate_attention_inputs(layers[idx], R1_layer)
        rotate_attention_output(layers[idx], R1_layer)
        rotate_mlp_input(layers[idx], R1_layer)
        rotate_mlp_output(layers[idx], R1_layer)
        if hasattr(layers[idx], "self_attn"):
            if checkpoint is not None:
                R2 = _resolve_r2_key(checkpoint, idx, layer_key_prefix).to(
                    device=layer_device,
                    dtype=preferred_rotation_dtype(layer_device),
                )
            else:
                R2 = get_orthogonal_matrix(head_dim, args.rotate_mode).to(device=layer_device)
            rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        head_dim = int(getattr(config, "head_dim", config.hidden_size // num_heads))
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
# Synchronize quantization device_map support for multi-GPU execution.
