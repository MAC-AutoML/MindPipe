"""Main pruning orchestration for LLM-Pruner."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from algorithm.common.device import empty_cache, resolve_device
from algorithm.common.modeling import (
    find_prunable_linear_layers,
    get_head_geometry,
    supports_head_pruning,
)

from .importance import MagnitudeImportance, TaylorImportance
from .model_ops import (
    pseudo_mask_attention,
    prune_attention,
    pseudo_mask_mlp_group,
    prune_mlp_group,
    sync_config,
)


# ---------------------------------------------------------------------------
# Geometry helpers (local, mirrors FLAP pattern)
# ---------------------------------------------------------------------------

def _get_head_geometry(layer):
    geo = get_head_geometry(layer)
    if geo is None:
        return None
    # 返回旧格式 (num_heads, num_kv_heads, num_kv_groups, head_dim)
    return geo


def _get_mlp_projections(layer):
    from algorithm.common.modeling import get_mlp_projections

    return get_mlp_projections(layer)


# ---------------------------------------------------------------------------
# Attention-group helpers
# ---------------------------------------------------------------------------

def _reduce_attention_importance_to_groups(attn_imp: torch.Tensor, layer) -> torch.Tensor:
    """Reduce per-head importance to per-attention-group importance.

    linear_attention 层直接返回空张量。
    """
    geo = _get_head_geometry(layer)
    if geo is None:
        return attn_imp  # 已经是空张量或不需要 reduce
    num_heads, num_kv_heads, num_kv_groups, _ = geo
    if attn_imp.numel() == num_kv_heads:
        return attn_imp
    if attn_imp.numel() != num_heads:
        raise ValueError(
            f"Expected {num_heads} head importance values or {num_kv_heads} group values, "
            f"got {attn_imp.numel()}."
        )
    return attn_imp.reshape(num_kv_heads, num_kv_groups).sum(dim=1)


# ---------------------------------------------------------------------------
# Min-keep protection
# ---------------------------------------------------------------------------

def _enforce_min_keep(imp: torch.Tensor, keep_mask: torch.Tensor, min_keep: int) -> torch.Tensor:
    """Ensure at least *min_keep* items survive."""
    if min_keep <= 0:
        return keep_mask
    if keep_mask.sum().item() >= min_keep:
        return keep_mask
    _, topk_idxs = torch.topk(imp, k=min(min_keep, imp.numel()))
    keep_mask = keep_mask.clone()
    keep_mask[topk_idxs] = True
    return keep_mask


# ---------------------------------------------------------------------------
# Local mask construction
# ---------------------------------------------------------------------------

def _get_layer_weight_count(layer):
    return sum(m.weight.numel() for m in find_prunable_linear_layers(layer).values())


def _estimate_attention_zero_count(layer, group_keep_mask):
    """Estimate how many weight elements become zero when attention groups are pruned.

    linear_attention 层返回 0。
    """
    if not supports_head_pruning(layer):
        return 0
    geo = _get_head_geometry(layer)
    if geo is None:
        return 0
    num_heads, num_kv_heads, num_kv_groups, head_dim = geo
    attn = layer.self_attn
    q_proj = attn.q_proj
    o_proj = attn.o_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj

    from algorithm.common.modeling import get_q_stride
    q_stride = get_q_stride(layer)

    if group_keep_mask.numel() == num_heads:
        group_keep_mask = group_keep_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
    removed_groups = int((~group_keep_mask).sum().item())
    removed_q_heads = removed_groups * num_kv_groups
    zero_count = 0
    # q_proj rows removed (考虑 query+gate 绑定)
    zero_count += removed_q_heads * q_stride * q_proj.weight.shape[1]
    # o_proj columns removed
    zero_count += removed_q_heads * head_dim * o_proj.weight.shape[0]

    removed_kv_heads = removed_groups
    zero_count += removed_kv_heads * head_dim * k_proj.weight.shape[1]
    zero_count += removed_kv_heads * head_dim * v_proj.weight.shape[1]
    return zero_count


def _estimate_mlp_group_zero_count(group, neuron_keep_mask):
    """估算单个 MLP group 剪掉 neuron 后会置零多少权重。"""
    up_proj = group.up_proj
    gate_proj = group.gate_proj
    down_proj = group.down_proj
    removed = int((~neuron_keep_mask).sum().item())
    zero_count = 0
    zero_count += removed * up_proj.weight.shape[1]
    zero_count += removed * gate_proj.weight.shape[1]
    zero_count += removed * down_proj.weight.shape[0]
    return zero_count


def _estimate_sparsity(layers, attn_masks, mlp_masks):
    """Estimate actual unstructured sparsity from per-layer masks."""
    zero_count = 0
    total_count = 0
    for layer, attn_mask, mlp_mask in zip(layers, attn_masks, mlp_masks):
        total_count += _get_layer_weight_count(layer)
        if attn_mask is not None:
            zero_count += _estimate_attention_zero_count(layer, attn_mask)
        for group, group_mask in mlp_mask:
            zero_count += _estimate_mlp_group_zero_count(group, group_mask)
    return zero_count / max(total_count, 1)


def _build_local_keep_masks(
    layers,
    attn_imp_per_layer,
    mlp_group_imp_per_layer,
    target_ratio,
    min_attention_groups,
    min_mlp_neurons,
):
    """Build local per-layer keep masks.

    linear_attention 层的 attn_mask 为 None（不做 attention 剪枝）。
    """
    attn_masks = []
    mlp_masks = []

    for layer, attn_imp, mlp_group_imp in zip(layers, attn_imp_per_layer, mlp_group_imp_per_layer):
        if not supports_head_pruning(layer) or attn_imp.numel() == 0:
            # linear_attention 层：不做 attention 剪枝
            attn_masks.append(None)
        else:
            geo = _get_head_geometry(layer)
            if geo is None:
                attn_masks.append(None)
            else:
                _, num_kv_heads, _, head_dim = geo
                attn_current_channels = num_kv_heads * head_dim
                attn_n_pruned_channels = attn_current_channels - int(attn_current_channels * (1 - target_ratio))
                attn_n_pruned_groups = min(
                    attn_n_pruned_channels // head_dim,
                    max(0, num_kv_heads - min_attention_groups),
                )
                attn_keep_mask = torch.ones_like(attn_imp, dtype=torch.bool)
                if attn_n_pruned_groups > 0:
                    prune_indices = torch.argsort(attn_imp)[:attn_n_pruned_groups]
                    attn_keep_mask[prune_indices] = False
                attn_keep_mask = _enforce_min_keep(attn_imp, attn_keep_mask, min_attention_groups)
                attn_masks.append(attn_keep_mask)

        group_masks = []
        for group, mlp_imp in mlp_group_imp:
            mlp_width = mlp_imp.numel()
            mlp_n_pruned = min(
                mlp_width - int(mlp_width * (1 - target_ratio)),
                max(0, mlp_width - min_mlp_neurons),
            )
            mlp_keep_mask = torch.ones_like(mlp_imp, dtype=torch.bool)
            if mlp_n_pruned > 0:
                prune_indices = torch.argsort(mlp_imp)[:mlp_n_pruned]
                mlp_keep_mask[prune_indices] = False
            mlp_keep_mask = _enforce_min_keep(mlp_imp, mlp_keep_mask, min_mlp_neurons)
            group_masks.append((group, mlp_keep_mask))
        mlp_masks.append(group_masks)

    return attn_masks, mlp_masks


# ---------------------------------------------------------------------------
# Taylor gradient accumulation (global forward+backward)
# ---------------------------------------------------------------------------

def _accumulate_taylor_gradients(model, calibration_batches, device):
    """Run full-model forward+backward to accumulate gradients on all weights.

    Uses gradient checkpointing to trade compute for memory so that 7B models
    with sequence_length=2048 fit on an 80 GB GPU alongside other processes.
    """
    # Ensure gradients can flow through embeddings
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 开启 gradient checkpointing 以降低激活值显存
    was_checkpointing = getattr(model, "is_gradient_checkpointing", False)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # gradient checkpointing 只有在 self.training=True 时才会生效
    # （见 transformers.modeling_layers.GradientCheckpointingLayer.__call__）。
    # 模型从 method.py 进入时处于 eval 模式，因此必须在梯度累积循环前
    # 切换到 train 模式。本函数末尾的 model.eval() 会恢复回 eval 状态。
    model.train()

    # 清零已有梯度
    model.zero_grad()

    for input_ids, _ in tqdm(calibration_batches, desc="Taylor gradient accumulation"):
        input_ids = input_ids.to(next(model.parameters()).device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        # Don't zero grad — accumulate across batches
        del outputs, loss

    # Restore original checkpointing state
    if not was_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prune_llm_pruner(args, model, tokenizer, device, dataloader):
    """Run LLM-Pruner structured pruning.

    Args:
        args: SimpleNamespace with fields:
            pruner_type: "l2" (magnitude) or "taylor"
            taylor: "param_first" | "param_second" | "param_mix" (only for taylor mode)
            pruning_ratio: target sparsity ratio (0~1)
            unstr: bool, True=pseudo-pruning (mask), False=real pruning
            min_attention_groups: minimum attention groups to keep per layer
            min_mlp_neurons: minimum MLP neurons to keep per layer
        model: HF causal LM model
        tokenizer: HF tokenizer
        device: torch device
        dataloader: list of (input_ids, labels) tuples
    """
    device = resolve_device(device)
    backbone = get_text_backbone(model)
    layers = backbone.layers

    pruner_type = getattr(args, "pruner_type", "taylor")
    target_ratio = getattr(args, "pruning_ratio", 0.2)
    pseudo_pruning = getattr(args, "unstr", True)
    min_attention_groups = getattr(args, "min_attention_groups", 1)
    min_mlp_neurons = getattr(args, "min_mlp_neurons", 8)

    # --- Step 1: Create importance estimator ---
    if pruner_type == "taylor":
        taylor_mode = getattr(args, "taylor", "param_first")
        importance = TaylorImportance(taylor=taylor_mode)
        # Accumulate gradients globally
        _accumulate_taylor_gradients(model, dataloader, device)
    else:
        importance = MagnitudeImportance(p=2)

    # --- Step 2: Compute importance per layer ---
    attn_imp_per_layer = []
    mlp_group_imp_per_layer = []
    for layer in tqdm(layers, desc="Computing importance"):
        attn_imp = _reduce_attention_importance_to_groups(
            importance.compute_attention_importance(layer),
            layer,
        )
        mlp_group_imp = importance.compute_mlp_group_importances(layer)
        attn_imp_per_layer.append(attn_imp)
        mlp_group_imp_per_layer.append(mlp_group_imp)

    # Clean up gradients after reading them
    if pruner_type == "taylor":
        model.zero_grad()

    # --- Step 3: Build local per-layer masks ---
    attn_keep_masks, mlp_keep_masks = _build_local_keep_masks(
        layers,
        attn_imp_per_layer,
        mlp_group_imp_per_layer,
        target_ratio,
        min_attention_groups,
        min_mlp_neurons,
    )
    actual_ratio = _estimate_sparsity(layers, attn_keep_masks, mlp_keep_masks)
    print(
        f"[LLM-Pruner] target_ratio={target_ratio:.4f}  "
        f"selection=local_per_layer  actual_ratio={actual_ratio:.4f}"
    )

    # --- Step 4: Apply pruning ---
    for layer_idx, layer in enumerate(tqdm(layers, desc="Applying pruning")):
        attn_keep_mask = attn_keep_masks[layer_idx]
        mlp_group_keep_masks = mlp_keep_masks[layer_idx]

        if pseudo_pruning:
            # Pseudo-pruning: zero masked channels without changing shapes
            if attn_keep_mask is not None:
                pseudo_mask_attention(layer, attn_keep_mask, device)
            for group, group_keep_mask in mlp_group_keep_masks:
                pseudo_mask_mlp_group(group, group_keep_mask, device)
        else:
            # Real pruning: reshape weights by slicing pruned channels
            if attn_keep_mask is not None:
                remove_groups = sorted(
                    (attn_keep_mask == False).nonzero(as_tuple=False).flatten().tolist()  # noqa: E712
                )
                if remove_groups:
                    prune_attention(layer, remove_groups, device)

            for group, group_keep_mask in mlp_group_keep_masks:
                remove_neurons = sorted(
                    (group_keep_mask == False).nonzero(as_tuple=False).flatten().tolist()  # noqa: E712
                )
                if remove_neurons:
                    prune_mlp_group(group, remove_neurons, device)

    # --- Step 5: Sync config (only needed for real pruning) ---
    if not pseudo_pruning:
        sync_config(backbone)

    empty_cache(device)
    print("[LLM-Pruner] Pruning complete.")
    return {
        "threshold": None,
        "applied_sparsity_ratio": float(actual_ratio),
        "attention_keep_counts": [
            int(mask.sum().item()) if mask is not None else -1
            for mask in attn_keep_masks
        ],
        "mlp_keep_counts": [
            {
                group.name: int(group_mask.sum().item())
                for group, group_mask in group_masks
            }
            for group_masks in mlp_keep_masks
        ],
        "selection_strategy": "local_per_layer",
    }


# 为了兼容，从 common 导入 get_text_backbone
from algorithm.common.modeling import get_text_backbone
# Add pruning support for Qwen3.5.
