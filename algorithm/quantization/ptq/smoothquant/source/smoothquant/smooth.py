import torch
import torch.nn as nn

from algorithm.common.modeling import get_text_backbone


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not hasattr(ln, "weight") or ln.weight is None:
        raise TypeError(f"SmoothQuant requires a norm module with a learnable weight; got {type(ln)}")
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device = fcs[0].weight.device
    dtype = fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs],
        dim=0,
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    ).clamp(min=1e-5).to(device=device, dtype=dtype)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    supported_model_types = {"llama", "qwen2", "qwen2_5_vl", "minicpm", "minicpmv"}
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in supported_model_types:
        raise NotImplementedError(f"Unsupported SmoothQuant model type: {model_type!r}")

    backbone = get_text_backbone(model)
    for layer_idx, layer in enumerate(backbone.layers):
        layer_name = f"{backbone.prefix}.layers.{layer_idx}"

        attn_ln = layer.input_layernorm
        qkv = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        qkv_input_scales = scales[layer_name + ".self_attn.q_proj"]
        smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

        ffn_ln = layer.post_attention_layernorm
        fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
        fcs_input_scales = scales[layer_name + ".mlp.gate_proj"]
        smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
