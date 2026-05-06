"""Weight merging, pseudo-pruning, and real structured pruning for LLM-Pruner."""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Head geometry helpers
# ---------------------------------------------------------------------------

def _get_head_geometry(layer):
    attn = layer.self_attn
    config = getattr(attn, "config", None)
    num_heads = int(getattr(attn, "num_heads", getattr(config, "num_attention_heads")))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads", num_heads)))
    hidden_size = int(getattr(attn, "hidden_size", getattr(config, "hidden_size")))
    head_dim = int(getattr(attn, "head_dim", hidden_size // num_heads))
    num_kv_groups = num_heads // num_kv_heads
    return num_heads, num_kv_heads, num_kv_groups, head_dim


def _expand_attention_mask_to_kv(attn_mask, layer, device):
    """Expand a per-group or per-Q-head keep mask into Q/KV channel masks."""
    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_head_geometry(layer)
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
    q_mask = q_head_mask.repeat_interleave(head_dim).to(device)
    kv_mask = kv_head_mask.repeat_interleave(head_dim).to(device)
    return q_mask, kv_mask


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


# ---------------------------------------------------------------------------
# Pseudo-pruning: replay pruning semantics without changing tensor shapes
# ---------------------------------------------------------------------------

def _merge_out_channels_preserve_shape(linear: nn.Linear, remove_idxs: list[int], keep_idxs: list[int]):
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

    linear.weight.data[keep_idxs] = keep_weight
    linear.weight.data[remove_idxs] = 0

    if linear.bias is not None:
        keep_bias = linear.bias.data[keep_idxs].clone()
        remove_bias = linear.bias.data[remove_idxs]
        keep_bias.index_add_(0, max_indices, remove_bias)
        keep_bias = keep_bias / cnt
        linear.bias.data[keep_idxs] = keep_bias
        linear.bias.data[remove_idxs] = 0


def _merge_in_channels_preserve_shape(linear: nn.Linear, remove_idxs: list[int], keep_idxs: list[int]):
    if not remove_idxs:
        return
    keep_idxs = sorted(keep_idxs)
    remove_idxs = sorted(remove_idxs)

    keep_weight = linear.weight.data[:, keep_idxs].clone()
    remove_weight = linear.weight.data[:, remove_idxs]

    sim = torch.mm(remove_weight.t(), keep_weight)
    max_indices = torch.argmax(sim, dim=-1)
    keep_weight.index_add_(1, max_indices, remove_weight)

    linear.weight.data[:, keep_idxs] = keep_weight
    linear.weight.data[:, remove_idxs] = 0


def _attention_channel_indices_from_keep_mask(layer, group_keep_mask: torch.Tensor):
    device = layer.self_attn.q_proj.weight.device
    q_mask, kv_mask = _expand_attention_mask_to_kv(group_keep_mask, layer, device)
    keep_q_idxs = torch.where(q_mask)[0].tolist()
    remove_q_idxs = torch.where(~q_mask)[0].tolist()
    keep_kv_idxs = torch.where(kv_mask)[0].tolist()
    remove_kv_idxs = torch.where(~kv_mask)[0].tolist()
    return keep_q_idxs, remove_q_idxs, keep_kv_idxs, remove_kv_idxs


def pseudo_mask_attention(layer, group_keep_mask: torch.Tensor, device):
    """Replay attention pruning while preserving the original tensor shapes."""
    keep_q_idxs, remove_q_idxs, keep_kv_idxs, remove_kv_idxs = _attention_channel_indices_from_keep_mask(
        layer,
        group_keep_mask,
    )
    attn = layer.self_attn
    _merge_in_channels_preserve_shape(attn.o_proj, remove_q_idxs, keep_q_idxs)
    _merge_out_channels_preserve_shape(attn.q_proj, remove_q_idxs, keep_q_idxs)
    _merge_out_channels_preserve_shape(attn.k_proj, remove_kv_idxs, keep_kv_idxs)
    _merge_out_channels_preserve_shape(attn.v_proj, remove_kv_idxs, keep_kv_idxs)


def pseudo_mask_mlp(layer, neuron_keep_mask: torch.Tensor, device):
    """Replay MLP pruning while preserving the original tensor shapes."""
    keep_neurons = torch.where(neuron_keep_mask.to(dtype=torch.bool))[0].tolist()
    remove_neurons = torch.where(~neuron_keep_mask.to(dtype=torch.bool))[0].tolist()
    _merge_out_channels_preserve_shape(layer.mlp.up_proj, remove_neurons, keep_neurons)
    _merge_out_channels_preserve_shape(layer.mlp.gate_proj, remove_neurons, keep_neurons)
    _merge_in_channels_preserve_shape(layer.mlp.down_proj, remove_neurons, keep_neurons)


# ---------------------------------------------------------------------------
# Real structured pruning: reshape weights + update layer attributes
# ---------------------------------------------------------------------------

def prune_attention(layer, remove_groups: list[int], device):
    """Remove attention groups with weight merging.

    This modifies q/k/v/o_proj weights in-place and updates num_heads / num_kv_heads.
    """
    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_head_geometry(layer)

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

    # Expand to per-channel indices
    keep_q_idxs = []
    for h in keep_q_channels:
        keep_q_idxs.extend(range(h * head_dim, (h + 1) * head_dim))
    remove_q_idxs = []
    for h in remove_q_channels:
        remove_q_idxs.extend(range(h * head_dim, (h + 1) * head_dim))

    attn = layer.self_attn

    # o_proj: output direction is hidden_size, input direction is num_heads*head_dim
    # We prune o_proj's INPUT channels (columns) with merging
    if remove_q_idxs:
        _merge_and_prune_in_channels(attn.o_proj, remove_q_idxs, keep_q_idxs)

    # q_proj: prune OUTPUT rows (which correspond to heads)
    if remove_q_idxs:
        _merge_and_prune_out_channels(attn.q_proj, remove_q_idxs, keep_q_idxs)

    # --- KV handling (GQA-aware) ---
    # Determine which KV heads to keep
    kv_keep_heads = keep_groups
    kv_remove_heads = remove_groups

    if kv_remove_heads:
        keep_kv_idxs = []
        for h in kv_keep_heads:
            keep_kv_idxs.extend(range(h * head_dim, (h + 1) * head_dim))
        remove_kv_idxs = []
        for h in kv_remove_heads:
            remove_kv_idxs.extend(range(h * head_dim, (h + 1) * head_dim))

        _merge_and_prune_out_channels(attn.k_proj, remove_kv_idxs, keep_kv_idxs)
        _merge_and_prune_out_channels(attn.v_proj, remove_kv_idxs, keep_kv_idxs)

    # Update layer attributes on the attention module itself. The shared decoder
    # config cannot represent per-layer heterogeneous head counts.
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
    """Remove MLP intermediate neurons with weight merging.

    This modifies up/gate/down_proj weights in-place and updates intermediate_size.
    """
    mlp = layer.mlp
    intermediate_size = mlp.up_proj.weight.shape[0]
    all_neurons = list(range(intermediate_size))
    keep_neurons = sorted(set(all_neurons) - set(remove_neurons))
    remove_neurons_sorted = sorted(set(remove_neurons))

    if not remove_neurons_sorted:
        return

    # up_proj, gate_proj: prune OUTPUT rows
    _merge_and_prune_out_channels(mlp.up_proj, remove_neurons_sorted, keep_neurons)
    _merge_and_prune_out_channels(mlp.gate_proj, remove_neurons_sorted, keep_neurons)

    # down_proj: prune INPUT columns
    _merge_and_prune_in_channels(mlp.down_proj, remove_neurons_sorted, keep_neurons)

    # Update layer attribute
    mlp.intermediate_size = len(keep_neurons)


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
        num_heads, num_kv_heads, num_kv_groups, _ = _get_head_geometry(layer)
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
# Add the LLM-Pruner method.
