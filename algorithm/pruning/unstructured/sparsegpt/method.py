"""Unified SparseGPT runner."""

from __future__ import annotations

from pathlib import Path

import torch

from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import build_decoder_layer_groups
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import find_linear_layers
from ....common.modeling import get_text_backbone
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
        for linear in find_linear_layers(block).values():
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

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
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

        with prepend_python_path(source_root):
            from sparsegpt import SparseGPT

            pruned_linear_layers = []
            for layer_index, block in enumerate(backbone.layers):
                block = block.to(args.device)
                linear_layers = find_linear_layers(block)
                layer_groups = build_decoder_layer_groups(block, set(linear_layers))

                for group in layer_groups:
                    subset = {name: linear_layers[name] for name in group}
                    gpt_states = {name: SparseGPT(linear) for name, linear in subset.items()}

                    def add_batch(name: str):
                        def hook(_module, inputs, outputs):
                            gpt_states[name].add_batch(inputs[0].data, outputs.data)

                        return hook

                    handles = [
                        subset[name].register_forward_hook(add_batch(name))
                        for name in subset
                    ]
                    for sample_index in range(args.calibration_samples):
                        with torch.no_grad():
                            output_states[sample_index] = unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            )
                    for handle in handles:
                        handle.remove()

                    for name, gpt_state in gpt_states.items():
                        print(f"pruning layer {layer_index} name {name}")
                        gpt_state.fasterprune(
                            args.sparsity_ratio,
                            prunen=prune_n,
                            prunem=prune_m,
                            percdamp=args.damp_percent,
                            blocksize=args.block_size,
                        )
                        gpt_state.free()
                        pruned_linear_layers.append(f"{backbone.prefix}.layers.{layer_index}.{name}")
                    del gpt_states
                    torch.cuda.empty_cache()

                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        )

                backbone.layers[layer_index] = block.cpu()
                del block
                torch.cuda.empty_cache()
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
