import torch
import torch.nn as nn

from algorithm.common.modeling import get_text_backbone


def _compute_smooth_scales(fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert fc.in_features == act_scales.numel()

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
    return fcs, scales


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not hasattr(ln, "weight") or ln.weight is None:
        raise TypeError(f"SmoothQuant requires a norm module with a learnable weight; got {type(ln)}")
    assert ln.weight.numel() == act_scales.numel()
    fcs, scales = _compute_smooth_scales(fcs, act_scales, alpha)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_qwen3_5(ln, fcs, act_scales, alpha=0.5):
    if not hasattr(ln, "weight") or ln.weight is None:
        raise TypeError(f"SmoothQuant requires a norm module with a learnable weight; got {type(ln)}")
    assert ln.weight.numel() == act_scales.numel()
    fcs, scales = _compute_smooth_scales(fcs, act_scales, alpha)

    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    scale = scales.to(torch.float64)
    ln.weight.data = ((1.0 + ln_weight) / scale - 1.0).to(ori_dtype)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    supported_model_types = {
        "llama",
        "qwen2",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_vl",
        "qwen3_5",
        "minicpm",
        "minicpmv",
    }
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in supported_model_types:
        raise NotImplementedError(f"Unsupported SmoothQuant model type: {model_type!r}")

    backbone = get_text_backbone(model)
    for layer_idx, layer in enumerate(backbone.layers):
        layer_name = f"{backbone.prefix}.layers.{layer_idx}"

        if model_type == "qwen3_5":
            attn_ln = layer.input_layernorm
            if getattr(layer, "layer_type", None) == "linear_attention":
                attn_fcs = [
                    layer.linear_attn.in_proj_qkv,
                    layer.linear_attn.in_proj_z,
                    layer.linear_attn.in_proj_a,
                    layer.linear_attn.in_proj_b,
                ]
                attn_input_scales = scales[layer_name + ".linear_attn.in_proj_qkv"]
            else:
                attn_fcs = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
                attn_input_scales = scales[layer_name + ".self_attn.q_proj"]
            smooth_ln_fcs_qwen3_5(attn_ln, attn_fcs, attn_input_scales, alpha)

            ffn_ln = layer.post_attention_layernorm
            ffn_fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
            ffn_input_scales = scales[layer_name + ".mlp.gate_proj"]
            smooth_ln_fcs_qwen3_5(ffn_ln, ffn_fcs, ffn_input_scales, alpha)
            continue

        attn_ln = layer.input_layernorm
        qkv = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        qkv_input_scales = scales[layer_name + ".self_attn.q_proj"]
        smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

        ffn_ln = layer.post_attention_layernorm
        fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
        fcs_input_scales = scales[layer_name + ".mlp.gate_proj"]
        smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
# Adapt SmoothQuant to Qwen3, Qwen3-VL, and Qwen3.5 models.
