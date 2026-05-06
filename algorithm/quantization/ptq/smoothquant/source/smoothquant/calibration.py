import functools

import torch
import torch.nn as nn

from algorithm.common.modeling import get_text_backbone


def _build_calibration_forward_kwargs(model, input_ids):
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in {"qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
        return {"attention_mask": torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)}
    return {}


@torch.no_grad()
def run_smoothquant_calibration_forward(model, input_ids):
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

    for token_ids, _labels in calibration_batches:
        # device_map 模式下输入数据放到模型所在设备
        run_smoothquant_calibration_forward(model, token_ids.to(next(model.parameters()).device))

    for hook in hooks:
        hook.remove()
    return act_scales
