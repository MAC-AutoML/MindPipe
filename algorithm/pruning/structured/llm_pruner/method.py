"""LLM-Pruner pruning method adapter for MindPipe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ...base import BasePruningMethod
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import find_linear_layers, get_text_backbone
from ....common.runtime import prepend_python_path


def _linear_weight_stats(model) -> tuple[int, int]:
    backbone = get_text_backbone(model)
    zero_count = 0
    total_count = 0
    for block in backbone.layers:
        for linear in find_linear_layers(block).values():
            weight = linear.weight.data
            zero_count += int((weight == 0).sum().item())
            total_count += int(weight.numel())
    return zero_count, total_count


class LLMPrunerMethod(BasePruningMethod):
    name = "llm_pruner"
    default_calibration_dataset = "c4"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        source_args = SimpleNamespace(
            model=args.model_path,
            seed=args.seed,
            pruning_ratio=args.sparsity_ratio,
            pruner_type=args.llmpruner_pruner_type,
            taylor=getattr(args, "llmpruner_taylor", "param_first"),
            unstr=args.pseudo_pruning,
            min_attention_groups=getattr(args, "llmpruner_min_attention_heads", 1),
            min_mlp_neurons=getattr(args, "llmpruner_min_mlp_neurons", 8),
        )

        model.seqlen = args.sequence_length
        model.eval()
        _, linear_weight_count_before = _linear_weight_stats(model)

        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )

        with prepend_python_path(source_root):
            from lib.prune import prune_llm_pruner

            pruning_summary = prune_llm_pruner(
                source_args,
                model,
                tokenizer_bundle.tokenizer,
                args.device,
                dataloader=calibration_batches,
            )
            zero_weight_count_after, linear_weight_count_after = _linear_weight_stats(model)

        observed_zero_sparsity = zero_weight_count_after / max(linear_weight_count_after, 1)
        observed_param_reduction = 1.0 - (
            linear_weight_count_after / max(linear_weight_count_before, 1)
        )
        observed_sparsity = (
            observed_zero_sparsity if args.pseudo_pruning else observed_param_reduction
        )

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "observed_zero_sparsity_ratio": observed_zero_sparsity,
            "observed_param_reduction_ratio": observed_param_reduction,
            "estimated_effective_sparsity_ratio": pruning_summary["applied_sparsity_ratio"],
            "threshold": pruning_summary["threshold"],
            "pruner_type": source_args.pruner_type,
            "taylor_mode": source_args.taylor,
            "pseudo_pruning": source_args.unstr,
        }
# Migrate pruning to device_map loading for future multi-GPU support.
