from __future__ import annotations

import torch


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        mask = truncated_tensor.abs() < threshold
        truncated_tensor[mask] = truncated_tensor[mask].sign() * threshold
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


def truncate_number(number, threshold: float = 1e-2):
    return TruncateFunction.apply(number, threshold)


def _set_temp_bias(module, bias: torch.Tensor | None) -> None:
    module.temp_bias = bias


def _replace_or_register_buffer(module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = tensor
        setattr(module, name, tensor)
        return
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, tensor)


def _ensure_bias(module) -> torch.Tensor:
    if getattr(module, "bias", None) is None:
        return torch.zeros(module.weight.shape[0], dtype=module.weight.dtype, device=module.weight.device)
    return module.bias


def smooth_ln_fcs_temporary(ln, fcs, scales, shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]
    if getattr(ln, "bias", None) is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-shifts) / scales
    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        fc_bias = _ensure_bias(fc)
        _set_temp_bias(fc, fc_bias + fc.weight @ shifts)
        fc.temp_weight = fc.weight * scales.view(1, -1)


def smooth_fc_fc_temporary(fc1, fc2, scales, shifts=None, fc2_scales=None, fc2_shifts=None):
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    if fc2_scales is None:
        fc2_scales = scales
    fc1_bias = fc1.temp_bias if hasattr(fc1, "temp_bias") else _ensure_bias(fc1)
    fc1_weight = fc1.temp_weight if hasattr(fc1, "temp_weight") else fc1.weight
    fc1_shift_term = torch.zeros_like(scales) if shifts is None else shifts
    if fc2_shifts is None:
        if shifts is None:
            fc2_shift_term = torch.zeros_like(fc2_scales)
        elif fc2_scales.shape == shifts.shape:
            fc2_shift_term = shifts
        else:
            raise ValueError("fc2_shifts must be provided when fc2_scales shape differs from shifts.")
    else:
        fc2_shift_term = fc2_shifts
    fc1.temp_bias = (fc1_bias - fc1_shift_term) / scales.view(-1)
    fc1.temp_weight = fc1_weight / scales.view(-1, 1)

    fc2_bias = _ensure_bias(fc2)
    _set_temp_bias(fc2, fc2_bias + fc2.weight @ fc2_shift_term)
    fc2.temp_weight = fc2.weight * fc2_scales.view(1, -1)


def smooth_q_k_temporary(q_proj, k_proj, scales, q_scales=None):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    if q_scales is None:
        q_scales = scales
    q_proj.temp_weight = q_proj.temp_weight / q_scales.view(-1, 1)
    if getattr(q_proj, "temp_bias", None) is not None:
        q_proj.temp_bias = q_proj.temp_bias / q_scales.view(-1)
    k_proj.temp_weight = k_proj.temp_weight * scales.view(-1, 1)
    if getattr(k_proj, "temp_bias", None) is not None:
        k_proj.temp_bias = k_proj.temp_bias * scales.view(-1)


def smooth_ln_fcs_inplace(ln, fcs, scales, shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    if getattr(ln, "bias", None) is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        _replace_or_register_buffer(ln, "bias", (-shifts) / scales)
    ln.weight.div_(scales)

    for fc in fcs:
        fc.use_temporary_parameter = False
        if getattr(fc, "bias", None) is not None:
            fc.bias.add_(fc.weight @ shifts)
        else:
            _replace_or_register_buffer(fc, "bias", fc.weight @ shifts)
        fc.weight.mul_(scales.view(1, -1))


def smooth_fc_fc_inplace(fc1, fc2, scales, shifts=None, fc2_scales=None, fc2_shifts=None):
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False
    if fc2_scales is None:
        fc2_scales = scales
    fc1_shift_term = torch.zeros_like(scales) if shifts is None else shifts
    if fc2_shifts is None:
        if shifts is None:
            fc2_shift_term = torch.zeros_like(fc2_scales)
        elif fc2_scales.shape == shifts.shape:
            fc2_shift_term = shifts
        else:
            raise ValueError("fc2_shifts must be provided when fc2_scales shape differs from shifts.")
    else:
        fc2_shift_term = fc2_shifts
    if getattr(fc1, "bias", None) is not None:
        fc1.bias.sub_(fc1_shift_term)
        fc1.bias.div_(scales.view(-1))
    else:
        _replace_or_register_buffer(fc1, "bias", (-fc1_shift_term) / scales.view(-1))
    fc1.weight.div_(scales.view(-1, 1))

    if getattr(fc2, "bias", None) is not None:
        fc2.bias.add_(fc2.weight @ fc2_shift_term)
    else:
        _replace_or_register_buffer(fc2, "bias", fc2.weight @ fc2_shift_term)
    fc2.weight.mul_(fc2_scales.view(1, -1))


def smooth_q_k_inplace(q_proj, k_proj, scales, q_scales=None):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    if q_scales is None:
        q_scales = scales
    q_proj.weight.div_(q_scales.view(-1, 1))
    if getattr(q_proj, "bias", None) is not None:
        q_proj.bias.div_(q_scales.view(-1))
    k_proj.weight.mul_(scales.view(-1, 1))
    if getattr(k_proj, "bias", None) is not None:
        k_proj.bias.mul_(scales.view(-1))
# Adapt OmniQuant to LLaMA-family models with known remaining issues.
