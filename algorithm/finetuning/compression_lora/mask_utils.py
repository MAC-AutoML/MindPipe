"""Mask helpers for fixed-mask compression-aware LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


_DEFAULT_EXCLUDED_TARGET_PREFIXES = {
    "visual",
    "model.visual",
    "vision_tower",
    "model.vision_tower",
}


def _matches_target(name: str, target_suffixes: set[str] | None) -> bool:
    if not target_suffixes:
        return True
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in target_suffixes)


def _matches_excluded_prefix(name: str, excluded_prefixes: set[str] | None) -> bool:
    if not excluded_prefixes:
        return False
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in excluded_prefixes)


def iter_maskable_linears(
    model: nn.Module,
    target_modules: list[str] | tuple[str, ...] | None = None,
):
    """Yield modules that expose a 2-D weight and match target suffixes."""

    target_suffixes = set(target_modules or [])
    for name, module in model.named_modules():
        if _matches_excluded_prefix(name, _DEFAULT_EXCLUDED_TARGET_PREFIXES):
            continue
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        if not _matches_target(name, target_suffixes):
            continue
        yield name, module


def snapshot_weights(
    model: nn.Module,
    target_modules: list[str] | tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, module in iter_maskable_linears(model, target_modules):
        snapshot[name] = module.weight.detach().cpu().clone()
    return snapshot


def restore_weights(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for name, saved_weight in snapshot.items():
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Cannot restore weight for missing module {name!r}.")
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight):
            raise TypeError(f"Module {name!r} does not expose a tensor weight.")
        if tuple(weight.shape) != tuple(saved_weight.shape):
            raise ValueError(
                f"Weight shape mismatch for {name}: current={tuple(weight.shape)} "
                f"snapshot={tuple(saved_weight.shape)}."
            )
        weight.data.copy_(saved_weight.to(device=weight.device, dtype=weight.dtype))


def extract_masks_from_pruned_model(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    module_map = dict(model.named_modules())
    masks: dict[str, torch.Tensor] = {}
    for name, saved_weight in snapshot.items():
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Cannot extract mask for missing module {name!r}.")
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight):
            raise TypeError(f"Module {name!r} does not expose a tensor weight.")
        if tuple(weight.shape) != tuple(saved_weight.shape):
            raise ValueError(
                f"Mask shape mismatch for {name}: current={tuple(weight.shape)} "
                f"snapshot={tuple(saved_weight.shape)}."
            )
        masks[name] = weight.detach().ne(0).cpu()
    return masks


def save_masks(path: str | Path, masks: dict[str, torch.Tensor], metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata or {},
            "masks": {name: mask.detach().cpu().bool() for name, mask in masks.items()},
        },
        path,
    )
    return path


def load_masks(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "masks" in payload:
        masks = payload["masks"]
        metadata = payload.get("metadata", {})
    elif isinstance(payload, dict):
        masks = payload
        metadata = {}
    else:
        raise TypeError(f"Unsupported pruning mask payload type: {type(payload)!r}.")
    return {str(name): mask.bool() for name, mask in masks.items()}, dict(metadata)


def validate_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for name, mask in masks.items():
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Mask target module not found: {name}")
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight):
            raise TypeError(f"Mask target {name} does not expose tensor weight.")
        if tuple(weight.shape) != tuple(mask.shape):
            raise ValueError(f"Mask shape mismatch for {name}: mask={tuple(mask.shape)} weight={tuple(weight.shape)}")


def apply_masks_to_model(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    validate_masks(model, masks)
    module_map = dict(model.named_modules())
    for name, mask in masks.items():
        weight = module_map[name].weight
        weight.data.mul_(mask.to(device=weight.device, dtype=weight.dtype))


def mask_sparsity(masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    per_layer: dict[str, float] = {}
    total = 0
    kept = 0
    for name, mask in masks.items():
        mask_bool = mask.bool()
        numel = int(mask_bool.numel())
        keep = int(mask_bool.sum().item())
        total += numel
        kept += keep
        per_layer[name] = 1.0 - keep / max(numel, 1)
    return {
        "overall_sparsity": 1.0 - kept / max(total, 1),
        "per_layer_sparsity": per_layer,
    }
