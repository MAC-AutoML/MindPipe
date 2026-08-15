"""Unified SparseGPT runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def _check_sparsity(model, max_layers: int | None = None) -> float:
    backbone = get_text_backbone(model)
    zero_count = 0
    total_count = 0
    layer_count = len(backbone.layers) if max_layers is None else min(len(backbone.layers), int(max_layers))
    for layer_index, block in enumerate(backbone.layers[:layer_count]):
        layer_zero_count = 0
        layer_total_count = 0
        for linear in find_prunable_linear_layers(block).values():
            weight = linear.weight.data
            if getattr(weight, "is_meta", False):
                continue
            layer_zero_count += int((weight == 0).sum().item())
            layer_total_count += int(weight.numel())
        if layer_total_count:
            print(f"layer {layer_index} sparsity {layer_zero_count / layer_total_count:.6f}")
        zero_count += layer_zero_count
        total_count += layer_total_count
    return zero_count / max(total_count, 1)


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def _capture_multimodal_first_block_inputs(model, tokenizer_bundle, backbone, args):
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
    if model_type != "qwen2_5_vl":
        raise NotImplementedError(
            "SparseGPT multimodal calibration currently supports Qwen2.5-VL only, "
            f"got {model_type!r}."
        )
    processor = getattr(tokenizer_bundle, "processor", None)
    if processor is None:
        raise ValueError("Qwen2.5-VL SparseGPT multimodal calibration requires a processor.")

    # Reuse the tested Qwen2.5-VL prompt/image preparation used by GPTQ.
    from ....quantization.ptq.gptq.method import GPTQMethod

    helper_args = args
    helper_args.gptq_vlm_calib_num = int(
        getattr(args, "pruning_vlm_calib_num", None) or args.calibration_samples
    )
    calibration_inputs = GPTQMethod()._build_qwen2_vlm_calibration_inputs(
        processor=processor,
        dataset_name=str(args.pruning_vlm_dataset_name),
        args=helper_args,
    )

    decoder_config = backbone.decoder_config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False
    blocks = backbone.layers
    captured_samples: list[tuple[torch.Tensor, dict[str, Any]]] = []

    class _CaptureComplete(Exception):
        pass

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, hidden_states, **kwargs):
            captured_samples.append((_to_cpu(hidden_states), _to_cpu(kwargs)))
            raise _CaptureComplete

    source_model = getattr(model, "_source_model", model)
    target_device = resolve_device(args.device)
    blocks[0] = Catcher(blocks[0])
    try:
        source_model.eval()
        for prepared_inputs in calibration_inputs:
            moved_inputs = GPTQMethod._move_inputs_to_device(prepared_inputs, target_device)
            try:
                GPTQMethod()._run_multimodal_forward(
                    source_model=source_model,
                    prepared_inputs=moved_inputs,
                )
            except _CaptureComplete:
                pass
    finally:
        blocks[0] = blocks[0].module
        decoder_config.use_cache = use_cache

    if len(captured_samples) != len(calibration_inputs):
        raise RuntimeError(
            "SparseGPT multimodal capture count mismatch: "
            f"captured {len(captured_samples)} of {len(calibration_inputs)} samples."
        )
    return captured_samples


class SparseGPTMethod(BasePruningMethod):
    name = "sparsegpt"
    default_calibration_dataset = "c4"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        resolved = resolve_device(args.device)
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        source_root = Path(__file__).resolve().parent / "source"
        backbone = get_text_backbone(model)
        vlm_dataset_name = getattr(args, "pruning_vlm_dataset_name", None)
        if vlm_dataset_name:
            multimodal_samples = _capture_multimodal_first_block_inputs(
                model, tokenizer_bundle, backbone, args
            )
            input_states = [sample[0] for sample in multimodal_samples]
            sample_kwargs = [sample[1] for sample in multimodal_samples]
            sample_count = len(input_states)
            variable_length_inputs = True
        else:
            calibration_batches, _ = get_calibration_and_evaluation_data(
                tokenizer=tokenizer_bundle.tokenizer,
                dataset_name=args.calibration_dataset,
                sequence_length=args.sequence_length,
                sample_count=args.calibration_samples,
                seed=args.seed,
                data_path=args.data_path,
            )
            input_states, layer_kwargs = capture_first_block_inputs(
                model=model,
                backbone=backbone,
                calibration_batches=calibration_batches,
                device=args.device,
            )
            output_states = torch.zeros_like(input_states)
            sample_count = int(args.calibration_samples)
            variable_length_inputs = False
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
            layer_count = len(backbone.layers)
            pruning_max_layers = getattr(args, "pruning_max_layers", None)
            if pruning_max_layers is not None:
                if pruning_max_layers <= 0:
                    raise ValueError("--pruning_max_layers must be positive when provided.")
                layer_count = min(layer_count, int(pruning_max_layers))
                print(
                    f"SparseGPT pruning layer cap enabled: pruning first {layer_count} "
                    f"of {len(backbone.layers)} decoder layers"
                )
            layer_indices = list(range(layer_count))
            for layer_position, layer_index in enumerate(layer_indices):
                block = backbone.layers[layer_index]
                target_device = get_layer_device(backbone, layer_index)
                if not variable_length_inputs:
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
                first_pass_outputs = [] if variable_length_inputs else None
                for sample_index in range(sample_count):
                    with torch.no_grad():
                        if variable_length_inputs:
                            current_input = input_states[sample_index].to(target_device)
                            current_kwargs = move_tensors_to_device(sample_kwargs[sample_index], target_device)
                            current_output = unwrap_layer_output(block(current_input, **current_kwargs))
                            first_pass_outputs.append(current_output.detach().cpu())
                        else:
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

                if layer_position + 1 >= len(layer_indices):
                    continue

                next_input_states = [] if variable_length_inputs else None
                for sample_index in range(sample_count):
                    with torch.no_grad():
                        if variable_length_inputs:
                            current_input = input_states[sample_index].to(target_device)
                            current_kwargs = move_tensors_to_device(sample_kwargs[sample_index], target_device)
                            current_output = unwrap_layer_output(block(current_input, **current_kwargs))
                            next_input_states.append(current_output.detach().cpu())
                        else:
                            output_states[sample_index] = unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            )

                if variable_length_inputs:
                    input_states = next_input_states
                else:
                    input_states, output_states = output_states, input_states

        observed_sparsity = _check_sparsity(
            model,
            max_layers=getattr(args, "pruning_max_layers", None),
        )
        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": args.structure_pattern,
            "pruning_max_layers": getattr(args, "pruning_max_layers", None),
            "pruned_linear_count": len(pruned_linear_layers),
            "pruned_linear_layers": pruned_linear_layers,
            "multimodal_calibration": (
                {
                    "dataset_name": str(vlm_dataset_name),
                    "sample_count": sample_count,
                    "scope": "language_layers",
                }
                if vlm_dataset_name
                else None
            ),
        }
# Migrate pruning to device_map loading for future multi-GPU support.
