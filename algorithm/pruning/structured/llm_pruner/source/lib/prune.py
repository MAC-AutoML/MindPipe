"""Main pruning orchestration for LLM-Pruner."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from algorithm.common.device import empty_cache, resolve_device
from algorithm.common.modeling import get_text_backbone

from .importance import MagnitudeImportance, TaylorImportance
from .model_ops import (
    pseudo_mask_attention,
    pseudo_mask_mlp,
    prune_attention,
    prune_mlp,
    sync_config,
)


# ---------------------------------------------------------------------------
# Geometry helpers (local, mirrors FLAP pattern)
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


def _get_mlp_projections(layer):
    mlp = layer.mlp
    return mlp.up_proj, mlp.gate_proj, mlp.down_proj


# ---------------------------------------------------------------------------
# Attention-group helpers
# ---------------------------------------------------------------------------

def _reduce_attention_importance_to_groups(attn_imp: torch.Tensor, layer) -> torch.Tensor:
    """Reduce per-head importance to per-attention-group importance."""
    num_heads, num_kv_heads, num_kv_groups, _ = _get_head_geometry(layer)
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
    return sum(
        m.weight.numel()
        for m in layer.modules()
        if isinstance(m, nn.Linear)
    )


def _estimate_attention_zero_count(layer, group_keep_mask):
    """Estimate how many weight elements become zero when attention groups are pruned."""
    num_heads, num_kv_heads, num_kv_groups, head_dim = _get_head_geometry(layer)
    q_proj = layer.self_attn.q_proj
    o_proj = layer.self_attn.o_proj
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj

    if group_keep_mask.numel() == num_heads:
        group_keep_mask = group_keep_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
    removed_groups = int((~group_keep_mask).sum().item())
    removed_q_heads = removed_groups * num_kv_groups
    zero_count = 0
    # q_proj rows removed
    zero_count += removed_q_heads * head_dim * q_proj.weight.shape[1]
    # o_proj columns removed
    zero_count += removed_q_heads * head_dim * o_proj.weight.shape[0]

    removed_kv_heads = removed_groups
    zero_count += removed_kv_heads * head_dim * k_proj.weight.shape[1]
    zero_count += removed_kv_heads * head_dim * v_proj.weight.shape[1]
    return zero_count


def _estimate_mlp_zero_count(layer, neuron_keep_mask):
    """Estimate how many weight elements become zero when MLP neurons are pruned."""
    up_proj, gate_proj, down_proj = _get_mlp_projections(layer)
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
        zero_count += _estimate_attention_zero_count(layer, attn_mask)
        zero_count += _estimate_mlp_zero_count(layer, mlp_mask)
    return zero_count / max(total_count, 1)


def _build_local_keep_masks(
    layers,
    attn_imp_per_layer,
    mlp_imp_per_layer,
    target_ratio,
    min_attention_groups,
    min_mlp_neurons,
):
    """Mirror MetaPruner's local per-layer pruning selection.

    Attention follows the upstream GQA path: root on k_proj out_channels with
    ``consecutive_groups=head_dim``. MLP follows local linear out-channel pruning.
    """
    attn_masks = []
    mlp_masks = []

    for layer, attn_imp, mlp_imp in zip(layers, attn_imp_per_layer, mlp_imp_per_layer):
        _, num_kv_heads, _, head_dim = _get_head_geometry(layer)

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
        mlp_masks.append(mlp_keep_mask)

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

    # Enable gradient checkpointing to reduce activation memory
    was_checkpointing = getattr(model, "is_gradient_checkpointing", False)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Zero existing gradients
    model.zero_grad()

    for input_ids, _ in tqdm(calibration_batches, desc="Taylor gradient accumulation"):
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        # Don't zero grad — accumulate across batches
        del outputs, loss
        empty_cache(device)

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
    mlp_imp_per_layer = []
    for layer in tqdm(layers, desc="Computing importance"):
        attn_imp = _reduce_attention_importance_to_groups(
            importance.compute_attention_importance(layer),
            layer,
        )
        mlp_imp = importance.compute_mlp_importance(layer)
        attn_imp_per_layer.append(attn_imp)
        mlp_imp_per_layer.append(mlp_imp)

    # Clean up gradients after reading them
    if pruner_type == "taylor":
        model.zero_grad()

    # --- Step 3: Build local per-layer masks ---
    attn_keep_masks, mlp_keep_masks = _build_local_keep_masks(
        layers,
        attn_imp_per_layer,
        mlp_imp_per_layer,
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
        attn_keep_mask = attn_keep_masks[layer_idx]  # [num_attention_groups]
        mlp_keep_mask = mlp_keep_masks[layer_idx]    # [intermediate_size]

        if pseudo_pruning:
            # Pseudo-pruning: zero masked channels without changing shapes
            pseudo_mask_attention(layer, attn_keep_mask, device)
            pseudo_mask_mlp(layer, mlp_keep_mask, device)
        else:
            # Real pruning: reshape weights by slicing pruned channels
            remove_groups = sorted(
                (attn_keep_mask == False).nonzero(as_tuple=False).flatten().tolist()  # noqa: E712
            )
            remove_neurons = sorted(
                (mlp_keep_mask == False).nonzero(as_tuple=False).flatten().tolist()  # noqa: E712
            )

            if remove_groups:
                prune_attention(layer, remove_groups, device)
            if remove_neurons:
                prune_mlp(layer, remove_neurons, device)

    # --- Step 5: Sync config (only needed for real pruning) ---
    if not pseudo_pruning:
        sync_config(backbone)

    empty_cache(device)
    print("[LLM-Pruner] Pruning complete.")
    return {
        "threshold": None,
        "applied_sparsity_ratio": float(actual_ratio),
        "attention_keep_counts": [int(mask.sum().item()) for mask in attn_keep_masks],
        "mlp_keep_counts": [int(mask.sum().item()) for mask in mlp_keep_masks],
        "selection_strategy": "local_per_layer",
    }
