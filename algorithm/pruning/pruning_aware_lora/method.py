"""Pruning-aware LoRA adaptation.

This method is intentionally scoped to a first fixed-mask version:

    W_eff = W_base + delta_W_lora
    W_pruned = W_eff * pruning_mask

A selectable pruning backend builds masks from calibration activations. LoRA is then trained under those masks and finally merged back into ordinary Linear layers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ...common.datasets import get_calibration_and_evaluation_data
from ...common.device import empty_cache
from ...common.device import resolve_device
from ...common.modeling import capture_first_block_inputs
from ...common.modeling import find_prunable_linear_layers
from ...common.modeling import get_layer_device
from ...common.modeling import get_text_backbone
from ...common.modeling import move_tensors_to_device
from ...common.modeling import unwrap_layer_output
from ...common.runtime import prepend_python_path
from ..base import BasePruningMethod


LOGGER = logging.getLogger(__name__)
IGNORE_INDEX = -100
NATIVE_MASK_BACKENDS = {"sparsegpt", "alps", "wanda"}
PSEUDO_PRUNING_BACKENDS = {"flap", "wanda_sp", "llm_pruner", "shortgpt"}
PALORA_BACKENDS = NATIVE_MASK_BACKENDS | PSEUDO_PRUNING_BACKENDS


def _resolve_pruning_pattern(structure_pattern: str) -> tuple[int, int]:
    if structure_pattern == "unstructured":
        return 0, 0
    if ":" not in structure_pattern:
        return 0, 0
    return tuple(int(part) for part in structure_pattern.split(":", maxsplit=1))


def _build_magnitude_mask(
    weight: torch.Tensor,
    *,
    sparsity_ratio: float,
    prune_n: int,
    prune_m: int,
) -> torch.Tensor:
    metric = weight.detach().float().abs()
    if prune_n and prune_m:
        mask = torch.ones_like(metric, dtype=torch.bool)
        columns = metric.shape[1]
        for start in range(0, columns, prune_m):
            end = min(start + prune_m, columns)
            block = metric[:, start:end]
            prune_count = min(prune_n, block.shape[1])
            if prune_count <= 0:
                continue
            prune_indices = torch.topk(block, prune_count, dim=1, largest=False).indices
            mask[:, start:end].scatter_(1, prune_indices, False)
        return mask

    numel = metric.numel()
    prune_count = int(numel * float(sparsity_ratio))
    if prune_count <= 0:
        return torch.ones_like(metric, dtype=torch.bool)
    if prune_count >= numel:
        return torch.zeros_like(metric, dtype=torch.bool)

    threshold = metric.flatten().kthvalue(prune_count).values
    return metric > threshold


def _build_sparsegpt_score_mask(
    weight: torch.Tensor,
    hessian_inv_diag: torch.Tensor,
    *,
    sparsity_ratio: float,
    prune_n: int,
    prune_m: int,
) -> torch.Tensor:
    hessian_inv_diag = hessian_inv_diag.to(device=weight.device, dtype=torch.float32).clamp_min(1e-12)
    metric = weight.detach().float().square() / hessian_inv_diag.reshape((1, -1)).square()
    if prune_n and prune_m:
        mask = torch.ones_like(metric, dtype=torch.bool)
        columns = metric.shape[1]
        for start in range(0, columns, prune_m):
            end = min(start + prune_m, columns)
            block = metric[:, start:end]
            prune_count = min(prune_n, block.shape[1])
            if prune_count <= 0:
                continue
            prune_indices = torch.topk(block, prune_count, dim=1, largest=False).indices
            mask[:, start:end].scatter_(1, prune_indices, False)
        return mask

    numel = metric.numel()
    prune_count = int(numel * float(sparsity_ratio))
    if prune_count <= 0:
        return torch.ones_like(metric, dtype=torch.bool)
    if prune_count >= numel:
        return torch.zeros_like(metric, dtype=torch.bool)

    threshold = metric.flatten().kthvalue(prune_count).values
    return metric > threshold


def _matches_target(module_name: str, target_modules: list[str]) -> bool:
    if not target_modules or "all" in target_modules:
        return True
    normalized_targets = set(target_modules)
    normalized_targets.update(
        target[: -len(".linear")]
        for target in target_modules
        if target.endswith(".linear")
    )
    module_aliases = {module_name, module_name.rsplit(".", maxsplit=1)[-1]}
    if module_name.endswith(".linear"):
        without_linear = module_name[: -len(".linear")]
        module_aliases.add(without_linear)
        module_aliases.add(without_linear.rsplit(".", maxsplit=1)[-1])
    return any(
        alias == target or alias.endswith(f".{target}")
        for alias in module_aliases
        for target in normalized_targets
    )


def _get_child_module(root: torch.nn.Module, qualified_name: str) -> tuple[torch.nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _backbone_layer_attr(backbone) -> str:
    if hasattr(backbone.root, "layers"):
        return "layers"
    if hasattr(backbone.root, "h"):
        return "h"
    raise AttributeError(f"Unsupported backbone root: {type(backbone.root)}")


def _training_input_device(model) -> torch.device:
    backbone = get_text_backbone(model)
    if backbone.embed_tokens is not None:
        try:
            return next(backbone.embed_tokens.parameters()).device
        except StopIteration:
            pass
    return next(backbone.root.parameters()).device


def _expand_layer_kwargs_for_batch(data, batch_size: int):
    if isinstance(data, dict):
        return {key: _expand_layer_kwargs_for_batch(value, batch_size) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_expand_layer_kwargs_for_batch(value, batch_size) for value in data)
    if torch.is_tensor(data):
        if data.dim() > 0 and data.shape[0] == 1 and batch_size > 1:
            return data.expand(batch_size, *data.shape[1:])
        return data
    return data


def _cyclic_batch(tensor: torch.Tensor, start: int, batch_size: int) -> torch.Tensor:
    sample_count = tensor.shape[0]
    end = start + batch_size
    if end <= sample_count:
        return tensor[start:end]
    return torch.cat((tensor[start:], tensor[: end - sample_count]), dim=0)


def _forward_block_samples(
    block: torch.nn.Module,
    input_states: torch.Tensor,
    output_states: torch.Tensor,
    layer_kwargs: dict[str, Any],
    sample_count: int,
) -> None:
    for sample_index in range(sample_count):
        with torch.no_grad():
            output_states[sample_index] = unwrap_layer_output(
                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
            )


def _wanda_return_given_alpha(
    alpha: float,
    sort_values: torch.Tensor,
    metric: torch.Tensor,
    cumulative_metric: torch.Tensor,
    row_sums: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold_cumsum = row_sums * alpha
    selected_counts = (cumulative_metric <= threshold_cumsum.reshape((-1, 1))).sum(dim=1, keepdim=True)
    selected_counts = selected_counts.clamp(min=1, max=sort_values.shape[1])
    thresholds = torch.gather(sort_values, dim=1, index=selected_counts - 1)
    prune_mask = metric <= thresholds
    current_sparsity = prune_mask.sum() / prune_mask.numel()
    return prune_mask, current_sparsity


def _build_wanda_keep_mask(
    weight: torch.Tensor,
    scaler_row: torch.Tensor,
    *,
    sparsity_ratio: float,
    prune_n: int,
    prune_m: int,
    use_variant: bool,
) -> torch.Tensor:
    metric = weight.detach().float().abs() * torch.sqrt(scaler_row.reshape((1, -1)).float().clamp_min(0))
    prune_mask = torch.zeros_like(metric, dtype=torch.bool)

    if prune_n and prune_m:
        columns = metric.shape[1]
        for start in range(0, columns, prune_m):
            end = min(start + prune_m, columns)
            block = metric[:, start:end].float()
            prune_count = min(prune_n, block.shape[1])
            if prune_count <= 0:
                continue
            prune_indices = torch.topk(block, prune_count, dim=1, largest=False).indices
            prune_mask[:, start:end].scatter_(1, prune_indices, True)
        return ~prune_mask

    sort_values, sort_indices = torch.sort(metric, dim=-1, stable=True)
    if use_variant:
        cumulative_metric = torch.cumsum(sort_values, dim=1)
        row_sums = metric.sum(dim=1)
        alpha = 0.4
        alpha_low, alpha_high = 0.0, 0.8
        prune_mask, current_sparsity = _wanda_return_given_alpha(
            alpha,
            sort_values,
            metric,
            cumulative_metric,
            row_sums,
        )
        while (
            torch.abs(current_sparsity - float(sparsity_ratio)) > 0.001
            and (alpha_high - alpha_low) >= 0.001
        ):
            if current_sparsity > float(sparsity_ratio):
                alpha_high = alpha
                alpha = (alpha + alpha_low) / 2.0
            else:
                alpha_low = alpha
                alpha = (alpha + alpha_high) / 2.0
            prune_mask, current_sparsity = _wanda_return_given_alpha(
                alpha,
                sort_values,
                metric,
                cumulative_metric,
                row_sums,
            )
        LOGGER.info("Wanda variant alpha %.6f sparsity %.6f", alpha, float(current_sparsity))
        return ~prune_mask

    prune_count = int(metric.shape[1] * float(sparsity_ratio))
    if prune_count > 0:
        prune_count = min(prune_count, metric.shape[1])
        prune_mask.scatter_(1, sort_indices[:, :prune_count], True)
    return ~prune_mask


def _check_sparsity(model) -> float:
    backbone = get_text_backbone(model)
    zero_count = 0
    total_count = 0
    for layer_index, block in enumerate(backbone.layers):
        layer_zero_count = 0
        layer_total_count = 0
        for linear in find_prunable_linear_layers(block).values():
            weight = linear.weight.data
            layer_zero_count += int((weight == 0).sum().item())
            layer_total_count += int(weight.numel())
        if layer_total_count:
            LOGGER.info("layer %s sparsity %.6f", layer_index, layer_zero_count / layer_total_count)
        zero_count += layer_zero_count
        total_count += layer_total_count
    return zero_count / max(total_count, 1)


def _backend_method(backend: str) -> BasePruningMethod:
    if backend == "flap":
        from ..structured.flap.method import FLAPMethod

        return FLAPMethod()
    if backend == "wanda_sp":
        from ..structured.wanda_sp.method import WandaSPMethod

        return WandaSPMethod()
    if backend == "llm_pruner":
        from ..structured.llm_pruner.method import LLMPrunerMethod

        return LLMPrunerMethod()
    if backend == "shortgpt":
        from ..structured.shortgpt.method import ShortGPTMethod

        return ShortGPTMethod()
    raise ValueError(f"Unsupported pseudo-pruning backend for Pruning-Aware LoRA: {backend}")


class _TokenBatchDataset(Dataset):
    def __init__(self, batches) -> None:
        self.examples = [input_ids.squeeze(0).cpu() for input_ids, _labels in batches]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.examples[index]


@dataclass
class _TokenCollator:
    pad_token_id: int

    def __call__(self, instances: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            instances,
            batch_first=True,
            padding_value=int(self.pad_token_id),
        )
        attention_mask = input_ids.ne(int(self.pad_token_id))
        labels = input_ids.clone()
        labels = labels.masked_fill(~attention_mask, IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class _MaskedWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return dense_weight * mask.to(device=dense_weight.device, dtype=dense_weight.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class PruningAwareLoRALinear(torch.nn.Module):
    """Frozen Linear weight plus trainable LoRA delta under pruning-aware masking."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        mask: torch.Tensor,
        rank: int,
        alpha: int,
        sparsity_ratio: float,
        structure_pattern: str,
        mask_update: str,
        mask_score: str,
        hessian_inv_diag: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("Pruning-Aware LoRA rank must be positive.")
        if linear.weight.shape != mask.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match weight shape {tuple(linear.weight.shape)}."
            )
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.sparsity_ratio = float(sparsity_ratio)
        self.prune_n, self.prune_m = _resolve_pruning_pattern(structure_pattern)
        self.mask_update = str(mask_update)
        self.mask_score = str(mask_score)
        self.register_buffer("pruning_mask", mask.detach().to(device=linear.weight.device, dtype=torch.bool))
        self.register_buffer(
            "hessian_inv_diag",
            None
            if hessian_inv_diag is None
            else hessian_inv_diag.detach().to(device=linear.weight.device, dtype=torch.float32),
        )
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        if self.mask_update == "fixed":
            self.weight.data.masked_fill_(~self.pruning_mask, 0)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = linear.bias
            self.bias.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(
            torch.empty(self.rank, self.in_features, device=linear.weight.device, dtype=torch.float32)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=linear.weight.device, dtype=torch.float32)
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def _dynamic_mask(self, dense_weight: torch.Tensor) -> torch.Tensor:
        if self.mask_score == "sparsegpt" and self.hessian_inv_diag is not None:
            return _build_sparsegpt_score_mask(
                dense_weight,
                self.hessian_inv_diag,
                sparsity_ratio=self.sparsity_ratio,
                prune_n=self.prune_n,
                prune_m=self.prune_m,
            )
        return _build_magnitude_mask(
            dense_weight,
            sparsity_ratio=self.sparsity_ratio,
            prune_n=self.prune_n,
            prune_m=self.prune_m,
        )

    def _current_mask(self, dense_weight: torch.Tensor) -> torch.Tensor:
        if self.mask_update == "dynamic":
            return self._dynamic_mask(dense_weight)
        return self.pruning_mask

    @torch.no_grad()
    def refresh_dynamic_mask(self) -> bool:
        if self.mask_update != "dynamic":
            return False
        dense_weight = self.weight.float() + (self.lora_B @ self.lora_A) * self.scaling
        mask = self._dynamic_mask(dense_weight)
        self.pruning_mask.copy_(mask.to(device=self.pruning_mask.device, dtype=torch.bool))
        return True

    def _merged_weight_fp32(self) -> torch.Tensor:
        base = self.weight.float()
        lora_delta = (self.lora_B @ self.lora_A) * self.scaling
        dense_weight = base + lora_delta
        mask = self._current_mask(dense_weight)
        return dense_weight * mask.float()

    def _dense_merged_weight_fp32(self) -> torch.Tensor:
        return self.weight.float() + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lora_delta = (self.lora_B.to(dtype=input.dtype) @ self.lora_A.to(dtype=input.dtype)) * self.scaling
        if self.mask_update == "dynamic":
            dense_weight = self.weight.to(dtype=input.dtype) + lora_delta
            weight = _MaskedWeightSTE.apply(dense_weight, self.pruning_mask)
        else:
            lora_delta = lora_delta.masked_fill(~self.pruning_mask, 0)
            weight = self.weight.to(dtype=input.dtype) + lora_delta
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, weight, bias)

    @torch.no_grad()
    def to_merged_linear(self, *, apply_mask: bool = True) -> torch.nn.Linear:
        merged = torch.nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        merged_weight = self._merged_weight_fp32() if apply_mask else self._dense_merged_weight_fp32()
        merged.weight.data.copy_(merged_weight.to(dtype=self.weight.dtype))
        if self.bias is not None:
            merged.bias.data.copy_(self.bias.data)
        return merged


class PruningAwareLoRAMethod(BasePruningMethod):
    name = "pruning_aware_lora"
    default_calibration_dataset = "c4"

    def resolve_output_dir(self, args) -> Path:
        from ...common.io import ensure_dir
        from ...common.io import model_slug

        model_name = model_slug(args.model_path)
        pattern = getattr(args, "structure_pattern", "unstructured")
        pattern_suffix = f"_{pattern.replace(':', '-')}" if pattern != "unstructured" else ""
        backend = getattr(args, "palora_backend", "sparsegpt")
        objective = getattr(args, "palora_objective", "lm")
        mask_update = getattr(args, "palora_mask_update", "fixed")
        base_init = getattr(args, "palora_base_init", "dense")
        mask_score = getattr(args, "palora_mask_score", "backend")
        final_compress = bool(getattr(args, "palora_final_backend_compress", False))
        merge_mask = bool(getattr(args, "palora_merge_mask", True))
        final_suffix = "_final_backend" if final_compress else ""
        merge_suffix = "" if merge_mask else "_densemerge"
        run_spec = (
            f"{self.name}_{backend}_s{args.sparsity_ratio}{pattern_suffix}"
            f"_{objective}_{mask_update}"
            f"_{base_init}"
            f"_{mask_score}"
            f"{merge_suffix}"
            f"{final_suffix}"
            f"_r{args.palora_rank}"
            f"_lr{args.palora_learning_rate:g}"
            f"_steps{args.palora_max_steps}"
            f"_train{args.palora_train_samples}"
            f"_seq{args.sequence_length}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, model, tokenizer_bundle, args) -> None:
        if float(args.palora_dropout) != 0.0:
            raise ValueError(
                "Pruning-Aware LoRA v1 implements the exact weight-space form Prune(W + LoRA); "
                "set --palora_dropout 0."
            )
        if args.palora_backend not in PALORA_BACKENDS:
            available = ", ".join(sorted(PALORA_BACKENDS))
            raise ValueError(f"--palora_backend must be one of: {available}.")
        if args.eval_vlm:
            raise ValueError("Pruning-Aware LoRA v1 trains on text tokens only; set --eval_vlm false for the first version.")
        if int(args.palora_train_samples) <= 0:
            raise ValueError("--palora_train_samples must be positive.")
        if int(args.palora_rank) <= 0:
            raise ValueError("--palora_rank must be positive.")
        if int(args.palora_alpha) <= 0:
            raise ValueError("--palora_alpha must be positive.")
        if int(args.palora_per_device_train_batch_size) <= 0:
            raise ValueError("--palora_per_device_train_batch_size must be positive.")
        if int(args.palora_gradient_accumulation_steps) <= 0:
            raise ValueError("--palora_gradient_accumulation_steps must be positive.")
        if int(args.palora_logging_steps) <= 0:
            raise ValueError("--palora_logging_steps must be positive.")
        if not hasattr(args, "palora_gradient_checkpointing"):
            args.palora_gradient_checkpointing = False
        if not hasattr(args, "palora_mask_update"):
            args.palora_mask_update = "fixed"
        if args.palora_mask_update not in {"fixed", "dynamic"}:
            raise ValueError("--palora_mask_update must be fixed or dynamic.")
        if not hasattr(args, "palora_mask_score"):
            args.palora_mask_score = "backend"
        if args.palora_mask_score not in {"backend", "magnitude", "sparsegpt"}:
            raise ValueError("--palora_mask_score must be backend, magnitude, or sparsegpt.")
        if args.palora_mask_score == "sparsegpt" and args.palora_backend != "sparsegpt":
            raise ValueError("--palora_mask_score sparsegpt requires --palora_backend sparsegpt.")
        if not hasattr(args, "palora_base_init"):
            args.palora_base_init = "dense"
        if args.palora_base_init not in {"dense", "compressed"}:
            raise ValueError("--palora_base_init must be dense or compressed.")
        if not hasattr(args, "palora_merge_mask"):
            args.palora_merge_mask = True
        if not hasattr(args, "palora_final_backend_compress"):
            args.palora_final_backend_compress = False
        if bool(args.palora_final_backend_compress) and args.palora_backend not in NATIVE_MASK_BACKENDS:
            raise ValueError("--palora_final_backend_compress currently supports sparsegpt/alps/wanda backends only.")
        if not hasattr(args, "palora_objective"):
            args.palora_objective = "lm"
        if args.palora_objective not in {"lm", "layer_mse"}:
            raise ValueError("--palora_objective must be lm or layer_mse.")
        if args.palora_objective == "layer_mse" and args.palora_backend != "sparsegpt":
            raise ValueError("--palora_objective layer_mse is currently implemented only for --palora_backend sparsegpt.")
        if args.palora_mask_update == "dynamic" and args.palora_backend in PSEUDO_PRUNING_BACKENDS:
            raise ValueError(
                "--palora_mask_update dynamic rebuilds an unstructured magnitude mask and is only valid for "
                "sparsegpt/alps/wanda backends."
            )
        if args.palora_mask_update == "dynamic" and args.palora_mask_score == "backend":
            args.palora_mask_score = "sparsegpt" if args.palora_backend == "sparsegpt" else "magnitude"
        if not hasattr(args, "palora_layer_steps"):
            args.palora_layer_steps = 8
        if int(args.palora_layer_steps) <= 0:
            raise ValueError("--palora_layer_steps must be positive.")
        resolve_device(args.device)
        get_text_backbone(model)

    def _backend_source_root(self, backend: str) -> Path:
        method_group = "unstructured" if backend in NATIVE_MASK_BACKENDS else "structured"
        return Path(__file__).resolve().parents[1] / method_group / backend / "source"

    def _snapshot_prunable_weights(self, model, args) -> dict[str, torch.Tensor]:
        backbone = get_text_backbone(model)
        layer_attr = _backbone_layer_attr(backbone)
        target_modules = list(args.palora_target_modules or ["all"])
        snapshots: dict[str, torch.Tensor] = {}
        for layer_index, block in enumerate(backbone.layers):
            linear_layers = {
                name: linear
                for name, linear in find_prunable_linear_layers(block).items()
                if _matches_target(name, target_modules)
            }
            for name, linear in linear_layers.items():
                full_name = f"{backbone.prefix}.{layer_attr}.{layer_index}.{name}"
                snapshots[full_name] = linear.weight.detach().to(device="cpu", copy=True)
        return snapshots

    @torch.no_grad()
    def _restore_prunable_weights(self, model, snapshots: dict[str, torch.Tensor]) -> int:
        restored = 0
        for full_name, weight in snapshots.items():
            parent, child_name = _get_child_module(model, full_name)
            linear = getattr(parent, child_name)
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"Expected nn.Linear at {full_name}, got {type(linear)!r}.")
            if tuple(linear.weight.shape) != tuple(weight.shape):
                raise ValueError(
                    f"Cannot restore {full_name}: current shape {tuple(linear.weight.shape)} "
                    f"does not match snapshot shape {tuple(weight.shape)}."
                )
            linear.weight.data.copy_(
                weight.to(device=linear.weight.device, dtype=linear.weight.dtype),
                non_blocking=True,
            )
            restored += 1
        return restored

    def _collect_existing_weight_masks(
        self,
        model,
        args,
        before_weights: dict[str, torch.Tensor] | None = None,
        include_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        backbone = get_text_backbone(model)
        layer_attr = _backbone_layer_attr(backbone)
        target_modules = list(args.palora_target_modules or ["all"])
        masks: dict[str, torch.Tensor] = {}
        for layer_index, block in enumerate(backbone.layers):
            linear_layers = {
                name: linear
                for name, linear in find_prunable_linear_layers(block).items()
                if _matches_target(name, target_modules)
            }
            for name, linear in linear_layers.items():
                full_name = f"{backbone.prefix}.{layer_attr}.{layer_index}.{name}"
                after = linear.weight.detach()
                if before_weights is None or full_name not in before_weights:
                    mask = after.ne(0)
                else:
                    before = before_weights[full_name].to(device=after.device)
                    mask = after.ne(0) | before.eq(0)
                mask = mask.to(linear.weight.device, dtype=torch.bool)
                if include_all or bool((~mask).any().item()):
                    masks[full_name] = mask
        return masks

    def _build_pseudo_pruning_masks(self, model, tokenizer_bundle, args) -> dict[str, torch.Tensor]:
        backend_method = _backend_method(args.palora_backend)
        backend_args = copy.copy(args)
        backend_args.pruning = args.palora_backend
        backend_args.algorithm = args.palora_backend
        backend_args.pseudo_pruning = True
        LOGGER.info(
            "building Pruning-Aware LoRA masks with pseudo-pruning backend %s",
            args.palora_backend,
        )
        before_weights = self._snapshot_prunable_weights(model, args)
        summary = backend_method.apply_pruning(model, tokenizer_bundle, backend_args)
        masks = self._collect_existing_weight_masks(model, args, before_weights, include_all=True)
        restored = 0
        if args.palora_base_init == "dense":
            restored = self._restore_prunable_weights(model, before_weights)
            empty_cache(args.device)
        if not masks:
            raise RuntimeError(
                f"Pruning-Aware LoRA backend {args.palora_backend} produced no zeroed weights. "
                "Check backend support, sparsity_ratio, and --palora_target_modules."
            )
        LOGGER.info(
            "pseudo-pruning backend %s produced %s masked linear layers with base_init=%s restored=%s",
            args.palora_backend,
            len(masks),
            args.palora_base_init,
            restored,
        )
        self._last_backend_summary = summary
        return masks

    def _build_pruning_masks(self, model, tokenizer_bundle, args) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        backend = args.palora_backend
        if backend in PSEUDO_PRUNING_BACKENDS:
            return self._build_pseudo_pruning_masks(model, tokenizer_bundle, args)
        source_root = self._backend_source_root(backend)
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )
        backbone = get_text_backbone(model)
        input_states, layer_kwargs = capture_first_block_inputs(
            model=model,
            backbone=backbone,
            calibration_batches=calibration_batches,
            device=args.device,
        )
        output_states = torch.zeros_like(input_states)
        prune_n, prune_m = _resolve_pruning_pattern(args.structure_pattern)
        target_modules = list(args.palora_target_modules or ["all"])
        layer_attr = _backbone_layer_attr(backbone)
        masks: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        before_weights = self._snapshot_prunable_weights(model, args)

        try:
            with prepend_python_path(source_root):
                if backend == "sparsegpt":
                    from sparsegpt import SparseGPT
                elif backend == "alps":
                    from alps import ALPS_prune
                else:
                    from lib.layerwrapper import WrappedGPT

                for layer_index in range(len(backbone.layers)):
                    block = backbone.layers[layer_index]
                    target_device = get_layer_device(backbone, layer_index)
                    input_states = input_states.to(target_device)
                    output_states = output_states.to(target_device)
                    layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
                    linear_layers = {
                        name: linear
                        for name, linear in find_prunable_linear_layers(block).items()
                        if _matches_target(name, target_modules)
                    }
                    if backend == "sparsegpt":
                        backend_states = {name: SparseGPT(linear) for name, linear in linear_layers.items()}
                    elif backend == "alps":
                        backend_states = {
                            name: ALPS_prune(
                                linear,
                                nsamples=args.calibration_samples,
                                seqlen=args.sequence_length,
                                dev=target_device,
                            )
                            for name, linear in linear_layers.items()
                        }
                    else:
                        backend_states = {name: WrappedGPT(linear) for name, linear in linear_layers.items()}

                    def add_batch(name: str):
                        def hook(_module, inputs, outputs):
                            backend_states[name].add_batch(inputs[0].data, outputs.data)

                        return hook

                    handles = [
                        linear_layers[name].register_forward_hook(add_batch(name))
                        for name in linear_layers
                    ]
                    _forward_block_samples(
                        block,
                        input_states,
                        output_states,
                        layer_kwargs,
                        int(args.calibration_samples),
                    )
                    for handle in handles:
                        handle.remove()

                    for name, backend_state in backend_states.items():
                        linear = linear_layers[name]
                        LOGGER.info("building %s mask for layer %s name %s", backend, layer_index, name)
                        if backend == "sparsegpt":
                            backend_state.fasterprune(
                                args.sparsity_ratio,
                                prunen=prune_n,
                                prunem=prune_m,
                                percdamp=args.damp_percent,
                                blocksize=args.block_size,
                            )
                            mask = linear.weight.detach().ne(0)
                            hessian_inv_diag = getattr(backend_state, "last_hessian_inv_diag", None)
                            backend_state.free()
                        elif backend == "alps":
                            backend_state.ALPS_admm(
                                sp=args.sparsity_ratio,
                                nm_n=prune_n,
                                nm_m=prune_m,
                                rho=getattr(args, "rho", 0.1),
                            )
                            mask = linear.weight.detach().ne(0)
                            backend_state.free()
                        else:
                            mask = _build_wanda_keep_mask(
                                linear.weight.data,
                                backend_state.scaler_row,
                                sparsity_ratio=args.sparsity_ratio,
                                prune_n=prune_n,
                                prune_m=prune_m,
                                use_variant=bool(args.use_variant),
                            )
                            linear.weight.data.masked_fill_(~mask, 0)
                        full_name = f"{backbone.prefix}.{layer_attr}.{layer_index}.{name}"
                        if full_name in before_weights:
                            before = before_weights[full_name].to(device=linear.weight.device)
                            mask = mask | before.eq(0)
                        if backend == "sparsegpt" and hessian_inv_diag is not None:
                            masks[full_name] = {
                                "mask": mask.detach().to(linear.weight.device, dtype=torch.bool),
                                "hessian_inv_diag": hessian_inv_diag.detach().to(device="cpu", dtype=torch.float32),
                            }
                        else:
                            masks[full_name] = mask.detach().to(linear.weight.device, dtype=torch.bool)
                    del backend_states

                    _forward_block_samples(
                        block,
                        input_states,
                        output_states,
                        layer_kwargs,
                        int(args.calibration_samples),
                    )

                    input_states, output_states = output_states, input_states
                    empty_cache(args.device)
        finally:
            restored = 0
            if args.palora_base_init == "dense":
                restored = self._restore_prunable_weights(model, before_weights)
                empty_cache(args.device)
            LOGGER.info(
                "%s mask construction complete with base_init=%s restored=%s",
                backend,
                args.palora_base_init,
                restored,
            )
        return masks

    def _inject_lora_layers(self, model, masks: dict[str, torch.Tensor | dict[str, torch.Tensor]], args) -> list[str]:
        injected = []
        for full_name, mask_info in masks.items():
            if isinstance(mask_info, dict):
                mask = mask_info["mask"]
                hessian_inv_diag = mask_info.get("hessian_inv_diag")
            else:
                mask = mask_info
                hessian_inv_diag = None
            parent, child_name = _get_child_module(model, full_name)
            linear = getattr(parent, child_name)
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"Expected nn.Linear at {full_name}, got {type(linear)!r}.")
            wrapped = PruningAwareLoRALinear(
                linear,
                mask=mask,
                rank=int(args.palora_rank),
                alpha=int(args.palora_alpha),
                sparsity_ratio=float(args.sparsity_ratio),
                structure_pattern=args.structure_pattern,
                mask_update=args.palora_mask_update,
                mask_score=args.palora_mask_score,
                hessian_inv_diag=hessian_inv_diag,
            )
            setattr(parent, child_name, wrapped)
            injected.append(full_name)
        return injected

    @torch.no_grad()
    def _merge_lora_layers(self, model, *, apply_mask: bool = True) -> int:
        replacements: list[tuple[torch.nn.Module, str, PruningAwareLoRALinear]] = []

        def collect(parent: torch.nn.Module) -> None:
            for child_name, child in parent.named_children():
                if isinstance(child, PruningAwareLoRALinear):
                    replacements.append((parent, child_name, child))
                else:
                    collect(child)

        collect(model)
        for parent, child_name, wrapped in replacements:
            setattr(parent, child_name, wrapped.to_merged_linear(apply_mask=apply_mask))
        return len(replacements)

    @torch.no_grad()
    def _refresh_dynamic_lora_masks(self, model) -> int:
        refreshed = 0
        for module in model.modules():
            if isinstance(module, PruningAwareLoRALinear) and module.refresh_dynamic_mask():
                refreshed += 1
        return refreshed

    def _apply_final_backend_compression(self, model, tokenizer_bundle, args) -> dict[str, Any] | None:
        if not bool(getattr(args, "palora_final_backend_compress", False)):
            return None
        backend_method = _backend_method(args.palora_backend) if args.palora_backend in PSEUDO_PRUNING_BACKENDS else None
        if args.palora_backend == "sparsegpt":
            from ..unstructured.sparsegpt.method import SparseGPTMethod

            backend_method = SparseGPTMethod()
        elif args.palora_backend == "alps":
            from ..unstructured.alps.method import ALPSMethod

            backend_method = ALPSMethod()
        elif args.palora_backend == "wanda":
            from ..unstructured.wanda.method import WandaMethod

            backend_method = WandaMethod()
        if backend_method is None:
            raise ValueError(f"Unsupported final backend compression: {args.palora_backend}")

        backend_args = copy.copy(args)
        backend_args.pruning = args.palora_backend
        backend_args.algorithm = args.palora_backend
        backend_args.pseudo_pruning = False
        backend_args.palora_final_backend_compress = False
        LOGGER.info("applying final %s compression after LoRA merge", args.palora_backend)
        return backend_method.apply_pruning(model, tokenizer_bundle, backend_args)

    def _train_lora_lm(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        tokenizer = tokenizer_bundle.tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must define eos_token or pad_token for Pruning-Aware LoRA training.")
            tokenizer.pad_token = tokenizer.eos_token
            inner_tokenizer = getattr(tokenizer, "tokenizer", None)
            if inner_tokenizer is not None and getattr(inner_tokenizer, "pad_token_id", None) is None:
                inner_tokenizer.pad_token = tokenizer.eos_token

        train_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=int(args.palora_train_samples),
            seed=int(args.seed) + 1,
            data_path=args.data_path,
        )
        dataset = _TokenBatchDataset(train_batches)
        dataloader = DataLoader(
            dataset,
            batch_size=int(args.palora_per_device_train_batch_size),
            shuffle=True,
            collate_fn=_TokenCollator(pad_token_id=int(tokenizer.pad_token_id)),
        )

        for param in model.parameters():
            param.requires_grad = False
        trainable = []
        for name, param in model.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                param.requires_grad = True
                trainable.append(param)
        if not trainable:
            raise RuntimeError("Pruning-Aware LoRA did not find any trainable LoRA parameters after injection.")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(args.palora_learning_rate),
            weight_decay=float(args.palora_weight_decay),
        )
        input_device = _training_input_device(model)
        model.train()
        previous_use_cache = getattr(model.config, "use_cache", None)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        gradient_checkpointing_enabled = False
        input_require_grads_enabled = False
        input_grad_hook = None
        if bool(getattr(args, "palora_gradient_checkpointing", False)):
            if hasattr(model, "gradient_checkpointing_enable"):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                    input_require_grads_enabled = True
                elif hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
                    def make_inputs_require_grad(_module, _input, output):
                        output.requires_grad_(True)

                    input_grad_hook = model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                model.gradient_checkpointing_enable()
                gradient_checkpointing_enabled = True
            else:
                LOGGER.warning("Model does not expose gradient_checkpointing_enable; continuing without it.")

        total_loss = 0.0
        optimizer_steps = 0
        micro_steps = 0
        max_steps = int(args.palora_max_steps)
        grad_accum = max(1, int(args.palora_gradient_accumulation_steps))
        num_epochs = float(args.palora_num_train_epochs)
        full_epochs = max(1, int(num_epochs))
        if max_steps > 0:
            steps_per_epoch = max(1, (len(dataloader) + grad_accum - 1) // grad_accum)
            full_epochs = max(full_epochs, (max_steps + steps_per_epoch - 1) // steps_per_epoch)

        try:
            for _epoch in range(full_epochs):
                for batch in dataloader:
                    batch = {key: value.to(input_device) for key, value in batch.items()}
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    if loss is None:
                        raise RuntimeError("Model did not return a training loss for Pruning-Aware LoRA.")
                    (loss / grad_accum).backward()
                    total_loss += float(loss.detach().cpu())
                    micro_steps += 1

                    if micro_steps % grad_accum == 0:
                        optimizer.step()
                        refreshed_masks = self._refresh_dynamic_lora_masks(model)
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                        if optimizer_steps % int(args.palora_logging_steps) == 0:
                            LOGGER.info(
                                "Pruning-Aware LoRA step %s loss %.6f refreshed_masks %s",
                                optimizer_steps,
                                total_loss / max(micro_steps, 1),
                                refreshed_masks,
                            )
                        if max_steps > 0 and optimizer_steps >= max_steps:
                            break
                if max_steps > 0 and optimizer_steps >= max_steps:
                    break

            if micro_steps % grad_accum != 0:
                optimizer.step()
                self._refresh_dynamic_lora_masks(model)
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        finally:
            if gradient_checkpointing_enabled and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if input_require_grads_enabled and hasattr(model, "disable_input_require_grads"):
                model.disable_input_require_grads()
            if input_grad_hook is not None:
                input_grad_hook.remove()
            model.eval()
            if previous_use_cache is not None and hasattr(model.config, "use_cache"):
                model.config.use_cache = previous_use_cache
        return {
            "train_examples": len(dataset),
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "mean_loss": total_loss / max(micro_steps, 1),
        }

    def _train_one_layer_mse(
        self,
        block: torch.nn.Module,
        input_states: torch.Tensor,
        target_states: torch.Tensor,
        layer_kwargs: dict[str, Any],
        args,
        *,
        layer_index: int,
    ) -> dict[str, Any]:
        for param in block.parameters():
            param.requires_grad = False
        trainable = []
        for name, param in block.named_parameters():
            if ".lora_A" in name or ".lora_B" in name or name.endswith("lora_A") or name.endswith("lora_B"):
                param.requires_grad = True
                trainable.append(param)
        if not trainable:
            raise RuntimeError(f"Pruning-Aware LoRA found no trainable LoRA parameters in layer {layer_index}.")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(args.palora_learning_rate),
            weight_decay=float(args.palora_weight_decay),
        )
        block.eval()
        sample_count = min(int(args.palora_train_samples), int(input_states.shape[0]))
        batch_size = max(1, int(args.palora_per_device_train_batch_size))
        grad_accum = max(1, int(args.palora_gradient_accumulation_steps))
        optimizer_steps = max(1, int(args.palora_layer_steps))
        total_loss = 0.0
        micro_steps = 0

        for step in range(optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            for accum_index in range(grad_accum):
                offset = (step * grad_accum + accum_index) * batch_size
                inputs = _cyclic_batch(input_states[:sample_count], offset % sample_count, batch_size)
                targets = _cyclic_batch(target_states[:sample_count], offset % sample_count, batch_size)
                batch_kwargs = _expand_layer_kwargs_for_batch(layer_kwargs, int(inputs.shape[0]))
                outputs = unwrap_layer_output(block(inputs, **batch_kwargs))
                loss = F.mse_loss(outputs.float(), targets.float())
                (loss / grad_accum).backward()
                total_loss += float(loss.detach().cpu())
                micro_steps += 1

            optimizer.step()
            if (step + 1) % int(args.palora_logging_steps) == 0:
                LOGGER.info(
                    "Pruning-Aware LoRA layer_mse layer %s step %s loss %.6f",
                    layer_index,
                    step + 1,
                    total_loss / max(micro_steps, 1),
                )

        for param in trainable:
            param.requires_grad = False
        return {
            "layer_index": layer_index,
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "mean_loss": total_loss / max(micro_steps, 1),
        }

    def _train_lora_layer_mse(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        source_root = Path(__file__).resolve().parents[1] / "unstructured" / "sparsegpt" / "source"
        sample_count = max(int(args.calibration_samples), int(args.palora_train_samples))
        train_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=sample_count,
            seed=args.seed,
            data_path=args.data_path,
        )
        backbone = get_text_backbone(model)
        input_states, layer_kwargs = capture_first_block_inputs(
            model=model,
            backbone=backbone,
            calibration_batches=train_batches,
            device=args.device,
        )
        output_states = torch.zeros_like(input_states)
        prune_n, prune_m = _resolve_pruning_pattern(args.structure_pattern)
        target_modules = list(args.palora_target_modules or ["all"])
        layer_attr = _backbone_layer_attr(backbone)
        calibration_count = min(int(args.calibration_samples), sample_count)
        mask_count = 0
        injected_layers: list[str] = []
        merged_count = 0
        layer_metrics: list[dict[str, Any]] = []

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        previous_use_cache = getattr(model.config, "use_cache", None)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        try:
            with prepend_python_path(source_root):
                from sparsegpt import SparseGPT

                for layer_index in range(len(backbone.layers)):
                    block = backbone.layers[layer_index]
                    target_device = get_layer_device(backbone, layer_index)
                    input_states = input_states.to(target_device)
                    output_states = output_states.to(target_device)
                    layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
                    linear_layers = {
                        name: linear
                        for name, linear in find_prunable_linear_layers(block).items()
                        if _matches_target(name, target_modules)
                    }
                    gpt_states = {name: SparseGPT(linear) for name, linear in linear_layers.items()}

                    def add_batch(name: str):
                        def hook(_module, inputs, outputs):
                            gpt_states[name].add_batch(inputs[0].data, outputs.data)

                        return hook

                    handles = [
                        linear_layers[name].register_forward_hook(add_batch(name))
                        for name in linear_layers
                    ]
                    for sample_index in range(calibration_count):
                        with torch.no_grad():
                            output_states[sample_index] = unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            )
                    for handle in handles:
                        handle.remove()

                    for sample_index in range(calibration_count, sample_count):
                        with torch.no_grad():
                            output_states[sample_index] = unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            )

                    layer_masks: dict[str, torch.Tensor] = {}
                    for name, gpt_state in gpt_states.items():
                        linear = linear_layers[name]
                        LOGGER.info("building SparseGPT mask for layer %s name %s", layer_index, name)
                        gpt_state.fasterprune(
                            args.sparsity_ratio,
                            prunen=prune_n,
                            prunem=prune_m,
                            percdamp=args.damp_percent,
                            blocksize=args.block_size,
                        )
                        full_name = f"{backbone.prefix}.{layer_attr}.{layer_index}.{name}"
                        layer_masks[full_name] = linear.weight.detach().ne(0).to(linear.weight.device)
                        gpt_state.free()
                    del gpt_states

                    if layer_masks:
                        mask_count += len(layer_masks)
                        injected_layers.extend(self._inject_lora_layers(model, layer_masks, args))
                        layer_metrics.append(
                            self._train_one_layer_mse(
                                block,
                                input_states,
                                output_states,
                                layer_kwargs,
                                args,
                                layer_index=layer_index,
                            )
                        )
                        merged_count += self._merge_lora_layers(block)

                    for sample_index in range(sample_count):
                        with torch.no_grad():
                            output_states[sample_index] = unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            )

                    input_states, output_states = output_states, input_states
                    empty_cache(args.device)
        finally:
            model.eval()
            if previous_use_cache is not None and hasattr(model.config, "use_cache"):
                model.config.use_cache = previous_use_cache

        total_optimizer_steps = sum(item["optimizer_steps"] for item in layer_metrics)
        total_micro_steps = sum(item["micro_steps"] for item in layer_metrics)
        weighted_loss = sum(item["mean_loss"] * item["micro_steps"] for item in layer_metrics)
        return {
            "train_metrics": {
                "objective": "layer_mse",
                "train_examples": int(args.palora_train_samples),
                "calibration_examples": calibration_count,
                "sample_buffer_examples": sample_count,
                "layer_steps": int(args.palora_layer_steps),
                "micro_steps": total_micro_steps,
                "optimizer_steps": total_optimizer_steps,
                "mean_loss": weighted_loss / max(total_micro_steps, 1),
                "layer_metrics": layer_metrics,
            },
            "mask_count": mask_count,
            "injected_lora_layers": injected_layers,
            "merged_lora_layer_count": merged_count,
        }

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(model, tokenizer_bundle, args)
        resolved = resolve_device(args.device)
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        output_dir = self.resolve_output_dir(args)
        if args.palora_objective == "layer_mse":
            layer_mse_result = self._train_lora_layer_mse(model, tokenizer_bundle, args)
            train_metrics = layer_mse_result["train_metrics"]
            mask_count = layer_mse_result["mask_count"]
            injected_layers = layer_mse_result["injected_lora_layers"]
            merged_count = layer_mse_result["merged_lora_layer_count"]
            final_backend_summary = None
        else:
            masks = self._build_pruning_masks(model, tokenizer_bundle, args)
            if not masks:
                raise RuntimeError(
                    "Pruning-Aware LoRA did not build any masks. Check --palora_target_modules "
                    "or the model's prunable Linear module names."
                )
            mask_count = len(masks)
            injected_layers = self._inject_lora_layers(model, masks, args)
            masks.clear()
            empty_cache(args.device)
            train_metrics = self._train_lora_lm(model, tokenizer_bundle, args)
            merge_with_mask = bool(args.palora_merge_mask) and not bool(args.palora_final_backend_compress)
            merged_count = self._merge_lora_layers(model, apply_mask=merge_with_mask)
            final_backend_summary = self._apply_final_backend_compression(model, tokenizer_bundle, args)
        observed_sparsity = (
            float(final_backend_summary["observed_sparsity_ratio"])
            if final_backend_summary and "observed_sparsity_ratio" in final_backend_summary
            else _check_sparsity(model)
        )

        return {
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": args.structure_pattern,
            "palora_config": {
                "backend": args.palora_backend,
                "rank": int(args.palora_rank),
                "alpha": int(args.palora_alpha),
                "learning_rate": float(args.palora_learning_rate),
                "weight_decay": float(args.palora_weight_decay),
                "num_train_epochs": float(args.palora_num_train_epochs),
                "max_steps": int(args.palora_max_steps),
                "train_samples": int(args.palora_train_samples),
                "target_modules": list(args.palora_target_modules or ["all"]),
                "mask_update": args.palora_mask_update,
                "objective": args.palora_objective,
                "layer_steps": int(args.palora_layer_steps),
                "base_init": args.palora_base_init,
                "merge_mask": bool(args.palora_merge_mask),
                "final_backend_compress": bool(args.palora_final_backend_compress),
                "gradient_checkpointing": bool(args.palora_gradient_checkpointing),
            },
            "train_metrics": train_metrics,
            "mask_count": mask_count,
            "final_backend_summary": final_backend_summary,
            "injected_lora_layers": injected_layers,
            "merged_lora_layer_count": merged_count,
            "output_dir": str(output_dir),
            "_updated_model": model,
            "_updated_tokenizer_bundle": tokenizer_bundle,
        }
