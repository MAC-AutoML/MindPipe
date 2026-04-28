from __future__ import annotations

from collections import OrderedDict
from math import inf

import torch

from .int_linear import QuantLinear
from .int_matmul import QuantMatMul
from .transformation import smooth_fc_fc_inplace
from .transformation import smooth_fc_fc_temporary
from .transformation import smooth_ln_fcs_inplace
from .transformation import smooth_ln_fcs_temporary
from .transformation import smooth_q_k_inplace
from .transformation import smooth_q_k_temporary
from .transformation import truncate_number


def let_parameters(model, use_shift: bool = True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for name, param in model.named_parameters():
        if template in name:
            params.append(param)
    return iter(params)


def lwc_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if "bound_factor" in name:
            params.append(param)
    return iter(params)


def get_omni_parameters(model, use_shift: bool = True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for name, param in model.named_parameters():
        if "bound_factor" in name or template in name:
            params.append(param)
    return params


def omni_state_dict(model, destination=None, prefix: str = "", keep_vars: bool = False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if "smooth" in name or "bound_factor" in name:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination


def get_named_linears(module):
    return {name: child for name, child in module.named_modules() if isinstance(child, QuantLinear)}


def register_scales_and_zeros(model):
    for _name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        return max(parameter.grad.detach().abs().max().to(device) for parameter in parameters)
    return torch.norm(
        torch.stack([torch.norm(parameter.grad.detach(), norm_type).to(device) for parameter in parameters]),
        norm_type,
    )


def _project_smoothing_scale(parameter: torch.Tensor, min_scale: float = 1e-2) -> torch.Tensor:
    # LET scales are multiplicative equalization factors. Keep them strictly
    # positive before temporary/in-place reparameterization to avoid sign flips
    # and divisions by tiny negative values during smoothing.
    return truncate_number(parameter.abs(), threshold=min_scale).clamp(min=min_scale)


def smooth_and_quant_temporary(model, args):
    if args.let:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "smooth_scale" in name:
                    parameter.data = _project_smoothing_scale(parameter)
        q_proj_scales = model.self_attn.expand_kv_to_attention(model.qkt_smooth_scale)
        o_proj_scales = model.self_attn.expand_kv_to_attention(model.out_smooth_scale)
        o_proj_shifts = model.self_attn.expand_kv_to_attention(model.out_smooth_shift)
        smooth_ln_fcs_temporary(
            model.input_layernorm,
            [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
            model.qkv_smooth_scale,
            model.qkv_smooth_shift,
        )
        smooth_ln_fcs_temporary(
            model.post_attention_layernorm,
            [model.mlp.up_proj, model.mlp.gate_proj],
            model.fc1_smooth_scale,
            model.fc1_smooth_shift,
        )
        smooth_fc_fc_temporary(
            model.self_attn.v_proj,
            model.self_attn.o_proj,
            model.out_smooth_scale,
            model.out_smooth_shift,
            fc2_scales=o_proj_scales,
            fc2_shifts=o_proj_shifts,
        )
        smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale, q_scales=q_proj_scales)
        model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        model.mlp.down_proj.temp_bias = model.mlp.down_proj.bias
    else:
        for _name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
                module.temp_bias = module.bias

    for _name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            source_weight = module.temp_weight if hasattr(module, "temp_weight") else module.weight
            module.temp_weight = module.weight_quantizer(source_weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter = True


def clear_temp_variable(model):
    for _name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias


@torch.no_grad()
def smooth_and_quant_inplace(model, args):
    if args.let:
        for name, parameter in model.named_parameters():
            if "smooth_scale" in name:
                parameter.data = _project_smoothing_scale(parameter)
        q_proj_scales = model.self_attn.expand_kv_to_attention(model.qkt_smooth_scale)
        o_proj_scales = model.self_attn.expand_kv_to_attention(model.out_smooth_scale)
        o_proj_shifts = model.self_attn.expand_kv_to_attention(model.out_smooth_shift)
        smooth_ln_fcs_inplace(
            model.input_layernorm,
            [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
            model.qkv_smooth_scale,
            model.qkv_smooth_shift,
        )
        smooth_ln_fcs_inplace(
            model.post_attention_layernorm,
            [model.mlp.up_proj, model.mlp.gate_proj],
            model.fc1_smooth_scale,
            model.fc1_smooth_shift,
        )
        smooth_fc_fc_inplace(
            model.self_attn.v_proj,
            model.self_attn.o_proj,
            model.out_smooth_scale,
            model.out_smooth_shift,
            fc2_scales=o_proj_scales,
            fc2_shifts=o_proj_shifts,
        )
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale, q_scales=q_proj_scales)
    for _name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter = False


def set_quant_state(module, weight_quant: bool = False, act_quant: bool = False):
    module.use_weight_quant = weight_quant
    module.use_act_quant = act_quant
    for child in module.modules():
        if isinstance(child, (QuantLinear, QuantMatMul)):
            child.set_quant_state(weight_quant, act_quant)
# Adapt OmniQuant to Qwen2.5, Qwen2.5-VL, LLaMA-family, and MiniCPM models.
