"""Unified ALPS runner."""

from __future__ import annotations

from pathlib import Path

import torch

from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import find_linear_layers
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
        for linear in find_linear_layers(block).values():
            weight = linear.weight.data
            layer_zero_count += int((weight == 0).sum().item())
            layer_total_count += int(weight.numel())
        if layer_total_count:
            print(f"layer {layer_index} sparsity {layer_zero_count / layer_total_count:.6f}")
        zero_count += layer_zero_count
        total_count += layer_total_count
    return zero_count / max(total_count, 1)


class ALPSMethod(BasePruningMethod):
    name = "alps"
    npu_ready = True
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

        rho = getattr(args, "rho", 0.1)

        with prepend_python_path(source_root):
            from alps import ALPS_prune

            pruned_linear_layers = []
            for layer_index in range(len(backbone.layers)):
                block = backbone.layers[layer_index]
                target_device = get_layer_device(backbone, layer_index)
                input_states = input_states.to(target_device)
                output_states = output_states.to(target_device)
                layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
                linear_layers = find_linear_layers(block)

                # ALPS: collect statistics for ALL linear layers in one pass,
                # then prune all — same semantics as the original implementation.
                all_names = list(linear_layers.keys())
                subset = {n: linear_layers[n] for n in all_names}

                alps_states = {
                    name: ALPS_prune(
                        linear,
                        nsamples=args.calibration_samples,
                        seqlen=args.sequence_length,
                        dev=target_device,
                    )
                    for name, linear in subset.items()
                }

                def add_batch(name: str):
                    def hook(_module, inputs, outputs):
                        alps_states[name].add_batch(inputs[0].data, outputs.data)

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

                for name, alps_state in alps_states.items():
                    print(f"pruning layer {layer_index} name {name}")
                    alps_state.ALPS_admm(
                        sp=args.sparsity_ratio,
                        nm_n=prune_n,
                        nm_m=prune_m,
                        rho=rho,
                    )
                    alps_state.free()
                    pruned_linear_layers.append(f"{backbone.prefix}.layers.{layer_index}.{name}")
                del alps_states

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
# Maintenance touch for repository metadata refresh.
