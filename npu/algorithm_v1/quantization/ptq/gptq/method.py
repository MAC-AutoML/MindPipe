"""Unified GPTQ runner."""

from __future__ import annotations

from pathlib import Path

import torch

from ....common.datasets import get_calibration_and_evaluation_data
from ....common.device import empty_cache
from ....common.modeling import build_decoder_layer_groups
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import find_linear_layers
from ....common.modeling import get_text_backbone
from ....common.modeling import unwrap_layer_output
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


class GPTQMethod(BaseQuantizationMethod):
    name = "gptq"
    quantization_block_size = 32

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
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

        with prepend_python_path(source_root):
            from gptq import GPTQ
            from quant import Quantizer

            quantizer_artifacts = {}
            for layer_index, block in enumerate(backbone.layers):
                block = block.to(args.device)
                linear_layers = find_linear_layers(block)
                layer_groups = build_decoder_layer_groups(block, set(linear_layers))

                for group in layer_groups:
                    subset = {name: linear_layers[name] for name in group}
                    gptq_states = {}
                    for name, linear in subset.items():
                        gptq_state = GPTQ(linear)
                        gptq_state.quantizer = Quantizer()
                        gptq_state.quantizer.configure(
                            args.weight_bits,
                            perchannel=True,
                            sym=args.weight_symmetric,
                            mse=False,
                        )
                        gptq_states[name] = gptq_state

                    def add_batch(name: str):
                        def hook(_module, inputs, outputs):
                            gptq_states[name].add_batch(inputs[0].data, outputs.data)

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

                    for name, gptq_state in gptq_states.items():
                        gptq_state.fasterquant(
                            blocksize=self.quantization_block_size,
                            percdamp=args.damp_percent,
                            groupsize=args.weight_group_size,
                            actorder=args.use_activation_order,
                            static_groups=args.static_groups,
                        )
                        quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{name}"] = {
                            "bits": args.weight_bits,
                            "group_size": args.weight_group_size,
                            "symmetric": args.weight_symmetric,
                        }
                        gptq_state.free()
                    del gptq_states
                    empty_cache(args.device)

                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        )

                backbone.layers[layer_index] = block.cpu()
                del block
                empty_cache(args.device)
                input_states, output_states = output_states, input_states

        return {
            "source_root": str(source_root),
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
