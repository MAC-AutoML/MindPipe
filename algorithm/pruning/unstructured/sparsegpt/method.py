"""Unified SparseGPT runner."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import find_prunable_linear_layers
from ....common.modeling import get_layer_device
from ....common.modeling import get_text_backbone
from ....common.modeling import move_tensors_to_device
from ....common.modeling import unwrap_layer_output
from ....common.runtime import prepend_python_path
from ...base import BasePruningMethod


def _resolve_pruning_pattern(structure_pattern: str) -> tuple[int, int]:
    if structure_pattern == "unstructured":
        return 0, 0
    return tuple(int(part) for part in structure_pattern.split(":", maxsplit=1))


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
            print(f"layer {layer_index} sparsity {layer_zero_count / layer_total_count:.6f}")
        zero_count += layer_zero_count
        total_count += layer_total_count
    return zero_count / max(total_count, 1)


class SparseGPTMethod(BasePruningMethod):
    name = "sparsegpt"
    default_calibration_dataset = "c4"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        resolved = resolve_device(args.device)
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        source_root = Path(__file__).resolve().parent / "source"
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
        layer_step = float(os.environ.get("SPARSEGPT_LAYER_STEP", "0") or "0")
        if layer_step > 0:
            layer_count = len(backbone.layers)
            first_sparsity = args.sparsity_ratio - (layer_step * (layer_count - 1)) / 2
            sparsity_rates = [first_sparsity + layer_index * layer_step for layer_index in range(layer_count)]
            print(
                f"SparseGPT layer-wise sparsity enabled: mean={args.sparsity_ratio:.6f}, "
                f"step={layer_step:.6f}, first={sparsity_rates[0]:.6f}, last={sparsity_rates[-1]:.6f}"
            )
        else:
            sparsity_rates = [args.sparsity_ratio] * len(backbone.layers)

        with prepend_python_path(source_root):
            from sparsegpt import SparseGPT

            pruned_linear_layers = []
            for layer_index in range(len(backbone.layers)):
                block = backbone.layers[layer_index]
                target_device = get_layer_device(backbone, layer_index)
                input_states = input_states.to(target_device)
                output_states = output_states.to(target_device)
                layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
                linear_layers = find_prunable_linear_layers(block)

                # SparseGPT: collect statistics for every target Linear in one
                # layer pass. Running one calibration pass per projection group
                # is prohibitively slow after Qwen3.5/3.6 MoE experts are
                # unfused into per-expert Linear modules.
                gpt_states = {
                    name: SparseGPT(linear)
                    for name, linear in linear_layers.items()
                }

                def add_batch(name: str):
                    def hook(_module, inputs, outputs):
                        gpt_states[name].add_batch(inputs[0].data, outputs.data)

                    return hook

                handles = [
                    linear_layers[name].register_forward_hook(add_batch(name))
                    for name in linear_layers
                ]
                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        )
                for handle in handles:
                    handle.remove()

                for name, gpt_state in gpt_states.items():
                    print(f"pruning layer {layer_index} name {name} sparsity {sparsity_rates[layer_index]:.6f}")
                    gpt_state.fasterprune(
                        sparsity_rates[layer_index],
                        prunen=prune_n,
                        prunem=prune_m,
                        percdamp=args.damp_percent,
                        blocksize=args.block_size,
                    )
                    gpt_state.free()
                    pruned_linear_layers.append(f"{backbone.prefix}.layers.{layer_index}.{name}")
                del gpt_states

                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        )

                input_states, output_states = output_states, input_states

        observed_sparsity = _check_sparsity(model)
        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": args.structure_pattern,
            "pruned_linear_count": len(pruned_linear_layers),
            "pruned_linear_layers": pruned_linear_layers,
        }
# Migrate pruning to device_map loading for future multi-GPU support.