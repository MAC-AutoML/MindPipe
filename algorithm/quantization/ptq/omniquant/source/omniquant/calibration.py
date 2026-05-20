from __future__ import annotations

import functools

import torch
import torch.nn as nn

from algorithm.common.modeling import get_text_backbone


def _model_input_device(model: nn.Module, backbone) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            weight = getattr(embeddings, "weight", None)
            if torch.is_tensor(weight):
                return weight.device
            try:
                return next(embeddings.parameters()).device
            except StopIteration:
                pass
    return next(backbone.layers[0].parameters()).device


def _build_calibration_forward_kwargs(model: nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in {"qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
        return {"attention_mask": torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)}
    return {}


@torch.no_grad()
def run_omniquant_calibration_forward(model: nn.Module, input_ids: torch.Tensor):
    return model(
        input_ids=input_ids,
        use_cache=False,
        **_build_calibration_forward_kwargs(model, input_ids),
    )


@torch.no_grad()
def get_act_scales(model, calibration_batches, device):
    model.eval()
    backbone = get_text_backbone(model)
    act_scales: dict[str, torch.Tensor] = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        incoming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.maximum(act_scales[name], incoming_max)
        else:
            act_scales[name] = incoming_max

    def stat_input_hook(_module, module_inputs, _outputs, name):
        tensor = module_inputs[0] if isinstance(module_inputs, tuple) else module_inputs
        stat_tensor(name, tensor)

    hooks = []
    for name, module in backbone.root.named_modules():
        if isinstance(module, nn.Linear):
            qualified_name = f"{backbone.prefix}.{name}" if name else backbone.prefix
            hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, name=qualified_name)))

    input_device = _model_input_device(model, backbone)
    for token_ids, _labels in calibration_batches:
        run_omniquant_calibration_forward(model, token_ids.to(input_device))

    for hook in hooks:
        hook.remove()
    return act_scales


@torch.no_grad()
def get_act_shifts(model, calibration_batches, device):
    model.eval()
    backbone = get_text_backbone(model)
    act_shifts: dict[str, torch.Tensor] = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()
        incoming_max = torch.max(tensor, dim=0)[0].float().cpu()
        incoming_min = torch.min(tensor, dim=0)[0].float().cpu()
        midpoint = (incoming_max + incoming_min) / 2
        if name in act_shifts:
            act_shifts[name] = 0.99 * act_shifts[name] + 0.01 * midpoint
        else:
            act_shifts[name] = midpoint

    def stat_input_hook(_module, module_inputs, _outputs, name):
        tensor = module_inputs[0] if isinstance(module_inputs, tuple) else module_inputs
        stat_tensor(name, tensor)

    hooks = []
    for name, module in backbone.root.named_modules():
        if isinstance(module, nn.Linear):
            qualified_name = f"{backbone.prefix}.{name}" if name else backbone.prefix
            hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, name=qualified_name)))

    input_device = _model_input_device(model, backbone)
    for token_ids, _labels in calibration_batches:
        run_omniquant_calibration_forward(model, token_ids.to(input_device))

    for hook in hooks:
        hook.remove()
    return act_shifts
# Synchronize quantization device_map support for multi-GPU execution.
