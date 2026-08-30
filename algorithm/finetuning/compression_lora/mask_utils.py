"""Mask helpers for fixed-mask compression-aware LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import errno
import math
import time

import torch
from torch import nn


class PackedMask:
    """CPU bit-packed mask that expands only when a layer consumes it."""

    def __init__(self, packed: torch.Tensor, shape: tuple[int, ...], numel: int | None = None):
        if packed.dtype != torch.uint8 or packed.device.type != "cpu":
            raise ValueError("PackedMask storage must be a CPU uint8 tensor.")
        self.packed = packed.contiguous()
        self.shape = tuple(int(v) for v in shape)
        self.numel = int(numel if numel is not None else math.prod(self.shape))

    def unpack(self, *, device=None) -> torch.Tensor:
        target = torch.device(device) if device is not None else torch.device("cpu")
        if target.type != "cpu":
            packed = self.packed.to(device=target, non_blocking=True)
            shifts = torch.arange(7, -1, -1, device=target, dtype=torch.uint8)
            values = packed.unsqueeze(1).bitwise_right_shift(shifts).bitwise_and_(1).reshape(-1)
            return values[: self.numel].reshape(self.shape).bool()
        values = torch.from_numpy(__import__("numpy").unpackbits(self.packed.numpy()))[: self.numel]
        return values.reshape(self.shape)

    def count_kept(self) -> int:
        return int(__import__("numpy").unpackbits(self.packed.numpy())[: self.numel].sum())

    def __getstate__(self):
        return {"packed": self.packed, "shape": self.shape, "numel": self.numel}

    def __setstate__(self, state):
        self.__init__(state["packed"], tuple(state["shape"]), state.get("numel"))


def _is_expert_mask_name(name: str) -> bool:
    return ".experts." in name and name.rsplit(".", 1)[-1] in {"gate_proj", "up_proj", "down_proj"}


def _pack_bool_mask(mask: torch.Tensor) -> PackedMask:
    values = mask.detach().cpu().bool().flatten().to(torch.uint8)
    padding = (-values.numel()) % 8
    if padding:
        values = torch.cat((values, torch.zeros(padding, dtype=torch.uint8)))
    # numpy is used only for the byte packing primitive; no model-sized copy is retained.
    packed = torch.from_numpy(__import__("numpy").packbits(values.numpy()))
    return PackedMask(packed, tuple(mask.shape), int(mask.numel()))


_DEFAULT_EXCLUDED_TARGET_PREFIXES = {
    "visual",
    "model.visual",
    "vision_tower",
    "model.vision_tower",
}

def _matches_target(name: str, target_suffixes: set[str] | None) -> bool:
    if not target_suffixes:
        return True
    # Mixtral keeps native expert projection names (w1/w3/w2), while the
    # public compression-LoRA recipe uses the shared gate/up/down names.
    aliases = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    if any(name == suffix or name.endswith(f".{suffix}") for suffix in target_suffixes):
        return True
    if ".experts." in name:
        suffix = name.rsplit(".", 1)[-1]
        return aliases.get(suffix) in target_suffixes
    return False


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
    serialized = {
        name: _pack_bool_mask(mask) if _is_expert_mask_name(name) else mask.detach().cpu().bool()
        for name, mask in masks.items()
    }
    torch.save(
        {
            "metadata": {
                **(metadata or {}),
                "expert_masks_bit_packed": any(_is_expert_mask_name(name) for name in masks),
            },
            "masks": serialized,
        },
        path,
    )
    return path


def _load_mask_payload(path: str | Path, map_location: str | torch.device):
    retryable_errors = {errno.EBUSY, getattr(errno, "ESTALE", 116)}
    for attempt in range(6):
        try:
            # The file is an application-owned checkpoint and contains PackedMask metadata.
            return torch.load(path, map_location=map_location, weights_only=False)
        except OSError as exc:
            if exc.errno not in retryable_errors or attempt == 5:
                raise
            time.sleep(2**attempt)


def load_masks(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = _load_mask_payload(path, map_location)
    if isinstance(payload, dict) and "masks" in payload:
        masks = payload["masks"]
        metadata = payload.get("metadata", {})
    elif isinstance(payload, dict):
        masks = payload
        metadata = {}
    else:
        raise TypeError(f"Unsupported pruning mask payload type: {type(payload)!r}.")
    normalized = {}
    for name, mask in masks.items():
        if isinstance(mask, PackedMask):
            normalized[str(name)] = mask
        elif torch.is_tensor(mask):
            normalized[str(name)] = mask.bool()
        else:
            raise TypeError(f"Unsupported mask type for {name!r}: {type(mask)!r}.")
    return normalized, dict(metadata)


def materialize_mask(mask, *, device=None) -> torch.Tensor:
    if isinstance(mask, PackedMask):
        return mask.unpack(device=device)
    return mask.to(device=device) if device is not None else mask


def validate_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for name, mask in masks.items():
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Mask target module not found: {name}")
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight):
            raise TypeError(f"Mask target {name} does not expose tensor weight.")
        mask_shape = mask.shape if isinstance(mask, PackedMask) else tuple(mask.shape)
        if tuple(weight.shape) != tuple(mask_shape):
            raise ValueError(f"Mask shape mismatch for {name}: mask={tuple(mask_shape)} weight={tuple(weight.shape)}")


def apply_masks_to_model(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    validate_masks(model, masks)
    module_map = dict(model.named_modules())
    for name, mask in masks.items():
        weight = module_map[name].weight
        weight.data.mul_(materialize_mask(mask, device=weight.device).to(dtype=weight.dtype))


def mask_sparsity(masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    per_layer: dict[str, float] = {}
    total = 0
    kept = 0
    for name, mask in masks.items():
        if isinstance(mask, PackedMask):
            numel = mask.numel
            keep = mask.count_kept()
        else:
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
