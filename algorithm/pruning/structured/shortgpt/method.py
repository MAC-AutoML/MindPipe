"""ShortGPT structured pruning: layer-wise pseudo pruning via weight zeroing."""

from __future__ import annotations

import logging
from pathlib import Path

from ...base import BasePruningMethod
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import find_linear_layers
from ....common.modeling import get_text_backbone
from ....common.runtime import prepend_python_path

logger = logging.getLogger(__name__)


class ShortGPTMethod(BasePruningMethod):
    name = "shortgpt"
    npu_ready = True
    default_calibration_dataset = "pg19"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict:
        tokenizer = tokenizer_bundle.tokenizer
        device = args.device
        sparsity_ratio = args.sparsity_ratio

        # 1. Load calibration data via generic loader
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )
        logger.info("ShortGPT: loaded %d calibration batches from %s",
                     len(calibration_batches), args.calibration_dataset)

        # 2. Compute layer importances
        model.eval()

        source_root = Path(__file__).resolve().parent / "source"
        with prepend_python_path(source_root):
            from lib.shortgpt import compute_layer_importances

            importances = compute_layer_importances(
                model=model,
                calibration_batches=calibration_batches,
                device=device,
            )

        n_layers = len(importances)
        logger.info("ShortGPT: layer importances = %s",
                     [f"{v:.4f}" for v in importances])

        # 3. Determine layers to prune
        n_prune = int(n_layers * sparsity_ratio)
        ranked = sorted(range(n_layers), key=lambda i: importances[i])
        layers_to_prune = sorted(ranked[:n_prune])
        logger.info("ShortGPT: pruning %d/%d layers (sparsity_ratio=%.2f): %s",
                     n_prune, n_layers, sparsity_ratio, layers_to_prune)

        if not args.pseudo_pruning:
            raise NotImplementedError(
                "ShortGPT real layer removal is not yet implemented. "
                "Use --pseudo_pruning true for weight zeroing."
            )

        # 4. Zero out all Linear layers in the pruned layers
        backbone = get_text_backbone(model)
        zeroed_params = 0
        for layer_idx in layers_to_prune:
            block = backbone.layers[layer_idx]
            for linear in find_linear_layers(block).values():
                linear.weight.data.zero_()
                zeroed_params += linear.weight.data.numel()
                if linear.bias is not None:
                    linear.bias.data.zero_()
                    zeroed_params += linear.bias.data.numel()

        logger.info("ShortGPT: zeroed %d parameters across %d layers",
                     zeroed_params, len(layers_to_prune))

        # 5. Compute observed sparsity
        total_params = 0
        zero_params = 0
        for block in backbone.layers:
            for linear in find_linear_layers(block).values():
                w = linear.weight.data
                zero_params += int((w == 0).sum().item())
                total_params += w.numel()
        observed_sparsity = zero_params / max(total_params, 1)

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": "layer-removal",
            "n_prune_layers": n_prune,
            "total_layers": n_layers,
            "pruned_layer_indices": layers_to_prune,
            "layer_importances": importances,
            "pseudo_pruning": True,
        }
