"""Weight merging, pseudo-pruning, and real structured pruning for LLM-Pruner."""

from __future__ import annotations

import torch
import torch.nn as nn

from algorithm.common.modeling import (
    get_head_geometry,
    get_mlp_projections,
    get_q_stride,
    supports_head_pruning,
)


# ---------------------------------------------------------------------------
# Head geometry helpers
# ---------------------------------------------------------------------------

def _get_head_geometry(layer):
    """获取 head 几何信息，linear_attention 层返回 None。"""
    return get_head_geometry(layer)


def _expand_attention_mask_to_kv(attn_mask, layer, device):
    """Expand a per-group or per-Q-head keep mask into Q/KV channel masks.

    返回 (q_row_mask, kv_mask, o_col_mask)：
    - q_row_mask: q_proj 的行掩码（包含 query+gate 的 q_stride 步长）
    - kv_mask: k_proj/v_proj 的行掩码（head_dim 步长）
    - o_col_mask: o_proj 的列掩码（head_dim 步长，按 Q head 展开）
    """
    geo = _get_head_geometry(layer)
    if geo is None:
        raise ValueError("linear_attention 层不支持 head 掩码展开")
    num_heads, num_kv_heads, num_kv_groups, head_dim = geo
    q_stride = get_q_stride(layer)

    attn_mask = attn_mask.to(dtype=torch.bool)
    if attn_mask.numel() == num_kv_heads:
        kv_head_mask = attn_mask
        q_head_mask = kv_head_mask.repeat_interleave(num_kv_groups)
    elif attn_mask.numel() == num_heads:
        q_head_mask = attn_mask
        if num_kv_heads == num_heads:
            kv_head_mask = q_head_mask
        else:
            kv_head_mask = q_head_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
    else:
        raise ValueError(
            f"Expected {num_kv_heads} attention groups or {num_heads} Q-head mask elements, "
            f"got {attn_mask.numel()}."
        )
    # q_proj 行掩码（包含 query+gate）
    q_mask = q_head_mask.repeat_interleave(q_stride).to(device)
    # KV 行掩码
    kv_mask = kv_head_mask.repeat_interleave(head_dim).to(device)
    # o_proj 列掩码：按 Q head 展开到 num_heads * head_dim
    o_col_mask = q_head_mask.repeat_interleave(head_dim).to(device)
    return q_mask, kv_mask, o_col_mask


# ---------------------------------------------------------------------------
# Weight merging helpers  (LLM-Pruner core: cosine-similarity merge)
# ---------------------------------------------------------------------------

def _merge_and_prune_out_channels(linear: nn.Linear, remove_idxs: list[int], keep_idxs: list[int]):
    """Prune output channels of a Linear layer with weight merging.

    Removed rows are merged into the most-similar kept row (cosine similarity).
    """
    if not remove_idxs:
        return
    keep_idxs = sorted(keep_idxs)
    remove_idxs = sorted(remove_idxs)

    keep_weight = linear.weight.data[keep_idxs].clone()
    remove_weight = linear.weight.data[remove_idxs]

    sim = torch.mm(remove_weight, keep_weight.t())
    max_indices = torch.argmax(sim, dim=-1)

    keep_weight.index_add_(0, max_indices, remove_weight)
    cnt = torch.ones(keep_weight.size(0), device=keep_weight.device, dtype=keep_weight.dtype)
    cnt.index_add_(
        0,
        max_indices,
        torch.ones_like(max_indices, dtype=keep_weight.dtype),
    )
    keep_weight = keep_weight / cnt.unsqueeze(-1)

    linear.weight = nn.Parameter(keep_weight)
    linear.out_features = len(keep_idxs)

    if linear.bias is not None:
        keep_bias = linear.bias.data[keep_idxs].clone()
        remove_bias = linear.bias.data[remove_idxs]
        keep_bias.index_add_(0, max_indices, remove_bias)
        keep_bias = keep_bias / cnt
        linear.bias = nn.Parameter(keep_bias)


def _merge_and_prune_in_channels(linear: nn.Linear, remove_idxs: list[int], keep_idxs: list[int]):
    """Prune input channels of a Linear layer with weight merging.

    Removed columns are merged into the most-similar kept column.
    """
    if not remove_idxs:
        return
    keep_idxs = sorted(keep_idxs)
    remove_idxs = sorted(remove_idxs)

    keep_weight = linear.weight.data[:, keep_idxs].clone()
    remove_weight = linear.weight.data[:, remove_idxs]

    sim = torch.mm(remove_weight.t(), keep_weight)
    max_indices = torch.argmax(sim, dim=-1)

    keep_weight.index_add_(1, max_indices, remove_weight)

    linear.weight = nn.Parameter(keep_weight)
    linear.in_features = len(keep_idxs)


def _slice_prune_out_channels(linear: nn.Linear, remove_idxs: list[int], keep_idxs: list[int]):
    """Prune output channels of a Linear layer without weight merging."""
    if not remove_idxs:
        return
    keep_idxs = sorted(keep_idxs)
    linear.weight = nn.Parameter(linear.weight.data[keep_idxs].clone())
    linear.out_features = len(keep_idxs)
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[keep_idxs].clone())


def _slice_prune_in_channels(linear: nn.Linear, remove_idxs: list[int], keep_idxs: list[int]):
    """Prune input channels of a Linear layer without weight merging."""
    if not remove_idxs:
        return
    keep_idxs = sorted(keep_idxs)
    linear.weight = nn.Parameter(linear.weight.data[:, keep_idxs].clone())
    linear.in_features = len(keep_idxs)


# ---------------------------------------------------------------------------
# Pseudo-pruning: mask (zero out) without changing tensor shapes
# ---------------------------------------------------------------------------

def pseudo_mask_attention(layer, group_keep_mask: torch.Tensor, device):
    """Zero out pruned attention group weights without changing shapes.

    linear_attention 层会直接跳过（不做任何操作）。
    """
    if not supports_head_pruning(layer):
        return
    q_mask, kv_mask, o_col_mask = _expand_attention_mask_to_kv(group_keep_mask, layer, device)
    attn = layer.self_attn
    attn.q_proj.weight.data *= q_mask.to(attn.q_proj.weight.device).unsqueeze(-1)
    attn.k_proj.weight.data *= kv_mask.to(attn.k_proj.weight.device).unsqueeze(-1)
    attn.v_proj.weight.data *= kv_mask.to(attn.v_proj.weight.device).unsqueeze(-1)
    attn.o_proj.weight.data *= o_col_mask.to(attn.o_proj.weight.device).unsqueeze(0)


def pseudo_mask_mlp(layer, neuron_keep_mask: torch.Tensor, device):
    """Zero out pruned MLP neuron weights without changing shapes."""
    up_proj, gate_proj, down_proj = get_mlp_projections(layer)
    mask = neuron_keep_mask.to(up_proj.weight.device).unsqueeze(-1).to(up_proj.weight.data.dtype)
    up_proj.weight.data *= mask
    gate_proj.weight.data *= mask
    mask_t = neuron_keep_mask.to(down_proj.weight.device).unsqueeze(0).to(down_proj.weight.data.dtype)
    down_proj.weight.data *= mask_t


# ---------------------------------------------------------------------------
# Real structured pruning: reshape weights + update layer attributes
# ---------------------------------------------------------------------------

def prune_attention(layer, remove_groups: list[int], device):
    """Remove attention groups by slicing weights and updating layer attributes.

    linear_attention 层会直接跳过（不做任何操作）。
    """
    if not supports_head_pruning(layer):
        return
    geo = _get_head_geometry(layer)
    if geo is None:
        return
    num_heads, num_kv_heads, num_kv_groups, head_dim = geo
    q_stride = get_q_stride(layer)

    all_groups = list(range(num_kv_heads))
    keep_groups = sorted(set(all_groups) - set(remove_groups))
    remove_groups = sorted(set(remove_groups))

    keep_q_channels = []
    for group_idx in keep_groups:
        start = group_idx * num_kv_groups
        keep_q_channels.extend(range(start, start + num_kv_groups))
    remove_q_channels = []
    for group_idx in remove_groups:
        start = group_idx * num_kv_groups
        remove_q_channels.extend(range(start, start + num_kv_groups))

    # q_proj 行索引（考虑 query+gate 绑定，步长 q_stride）
    keep_q_idxs = []
    for h in keep_q_channels:
        keep_q_idxs.extend(range(h * q_stride, (h + 1) * q_stride))
    remove_q_idxs = []
    for h in remove_q_channels:
        remove_q_idxs.extend(range(h * q_stride, (h + 1) * q_stride))

    # o_proj 列索引（标准 head_dim 步长）
    keep_o_idxs = []
    for h in keep_q_channels:
        keep_o_idxs.extend(range(h * head_dim, (h + 1) * head_dim))
    remove_o_idxs = []
    for h in remove_q_channels:
        remove_o_idxs.extend(range(h * head_dim, (h + 1) * head_dim))

    attn = layer.self_attn

    # o_proj: prune INPUT columns
    if remove_o_idxs:
        _slice_prune_in_channels(attn.o_proj, remove_o_idxs, keep_o_idxs)

    # q_proj: prune OUTPUT rows
    if remove_q_idxs:
        _slice_prune_out_channels(attn.q_proj, remove_q_idxs, keep_q_idxs)

    # --- KV handling (GQA-aware) ---
    kv_keep_heads = keep_groups
    kv_remove_heads = remove_groups

    if kv_remove_heads:
        keep_kv_idxs = []
        for h in kv_keep_heads:
            keep_kv_idxs.extend(range(h * head_dim, (h + 1) * head_dim))
        remove_kv_idxs = []
        for h in kv_remove_heads:
            remove_kv_idxs.extend(range(h * head_dim, (h + 1) * head_dim))

        _slice_prune_out_channels(attn.k_proj, remove_kv_idxs, keep_kv_idxs)
        _slice_prune_out_channels(attn.v_proj, remove_kv_idxs, keep_kv_idxs)

    # Update layer attributes on the attention module itself.
    new_num_heads = len(keep_q_channels)
    new_num_kv_heads = len(kv_keep_heads)
    for attr, val in [
        ("num_heads", new_num_heads),
        ("num_attention_heads", new_num_heads),
        ("num_key_value_heads", new_num_kv_heads),
        ("num_key_value_groups", num_kv_groups),
    ]:
        try:
            setattr(attn, attr, val)
        except AttributeError:
            pass


def prune_mlp(layer, remove_neurons: list[int], device):
    """Remove MLP intermediate neurons by slicing weights."""
    up_proj, gate_proj, down_proj = get_mlp_projections(layer)
    intermediate_size = up_proj.weight.shape[0]
    all_neurons = list(range(intermediate_size))
    keep_neurons = sorted(set(all_neurons) - set(remove_neurons))
    remove_neurons_sorted = sorted(set(remove_neurons))

    if not remove_neurons_sorted:
        return

    # up_proj, gate_proj: prune OUTPUT rows
    _slice_prune_out_channels(up_proj, remove_neurons_sorted, keep_neurons)
    _slice_prune_out_channels(gate_proj, remove_neurons_sorted, keep_neurons)

    # down_proj: prune INPUT columns
    _slice_prune_in_channels(down_proj, remove_neurons_sorted, keep_neurons)

    # Update layer attribute
    layer.mlp.intermediate_size = len(keep_neurons)


# ---------------------------------------------------------------------------
# Config sync
# ---------------------------------------------------------------------------

def sync_config(backbone):
    """Synchronize model config after real pruning.

    Only writes fields that remain uniform across all layers. Per-layer
    structured pruning can legitimately produce heterogeneous dimensions,
    and the shared decoder config cannot encode that exactly.
    """
    head_counts = []
    kv_head_counts = []
    kv_group_counts = []
    intermediate_sizes = []
    for layer in backbone.layers:
        if supports_head_pruning(layer):
            geo = _get_head_geometry(layer)
            if geo is not None:
                num_heads, num_kv_heads, num_kv_groups, _ = geo
                head_counts.append(num_heads)
                kv_head_counts.append(num_kv_heads)
                kv_group_counts.append(num_kv_groups)
        intermediate_sizes.append(int(layer.mlp.intermediate_size))

    decoder_config = backbone.decoder_config
    if len(set(head_counts)) == 1:
        decoder_config.num_attention_heads = head_counts[0]
    if len(set(kv_head_counts)) == 1:
        decoder_config.num_key_value_heads = kv_head_counts[0]
    if hasattr(decoder_config, "num_key_value_groups") and len(set(kv_group_counts)) == 1:
        decoder_config.num_key_value_groups = kv_group_counts[0]
    if len(set(intermediate_sizes)) == 1:
        decoder_config.intermediate_size = intermediate_sizes[0]
