import torch
import torch.nn as nn
from algorithm.common.device import empty_cache
from .quantizer import pseudo_quantize_tensor
import gc
from ..utils.device import resolve_device
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
except ImportError:
    Qwen3_5DecoderLayer = tuple()  # type: ignore[assignment]

__all__ = ["auto_clip_block"]


def _normalize_clip_targets(clip_targets) -> str:
    value = str(clip_targets or "auto").strip()
    return value or "auto"


def _resolve_explicit_targets(clip_targets: str) -> set[str]:
    if clip_targets in {"auto", "none", "all"}:
        return set()
    return {item.strip() for item in clip_targets.split(",") if item.strip()}


def _should_clip_linear(name: str, module, clip_targets: str) -> bool:
    if clip_targets == "none":
        return False
    if clip_targets == "auto":
        if isinstance(module, Qwen3_5DecoderLayer):
            # Qwen3.5 is highly sensitive to clipping on gate/up projections,
            # and linear-attention blocks do not benefit from down_proj clipping either.
            return getattr(module, "layer_type", None) == "full_attention" and name == "mlp.down_proj"
        return True
    if clip_targets == "all":
        return True
    return name in _resolve_explicit_targets(clip_targets)


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    sample_step = max(1, input_feat.shape[1] // max(1, n_sample_token))
    input_feat = input_feat[:, 0::sample_step]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    # prevent OOM while supporting arbitrary output-channel sizes
    if w.shape[0] >= 256 and w.shape[0] % 256 == 0:
        oc_batch_size = 256
    elif w.shape[0] >= 64 and w.shape[0] % 64 == 0:
        oc_batch_size = 64
    else:
        oc_batch_size = min(64, w.shape[0])
    w_all = w
    total_out_channels = w.shape[0]
    best_max_val_all = []

    for start in range(0, total_out_channels, oc_batch_size):
        end = min(start + oc_batch_size, total_out_channels)
        w = w_all[start:end]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    gc.collect()
    empty_cache(w_all.device)
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat, device=None, model=None, clip_targets="auto"):
    runtime_device = resolve_device(device)
    clip_targets = _normalize_clip_targets(clip_targets)
    named_linears = {
        name: m
        for name, m in module.named_modules()
        if isinstance(m, nn.Linear) and name in input_feat
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        if not _should_clip_linear(name, module, clip_targets):
            continue
        # device_map 模式下不手动移动模块，直接在当前设备上计算
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        clip_list.append((name, max_val))
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list, device=None):
    from ..utils.module import get_op_by_name

    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        # device_map 模式下不手动移动模块，max_val 对齐到权重设备
        max_val = max_val.to(layer.weight.device).to(layer.weight.dtype)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
