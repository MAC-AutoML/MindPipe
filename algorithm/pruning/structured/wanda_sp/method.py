"""Unified structured Wanda runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ...base import BasePruningMethod
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import find_linear_layers
from ....common.modeling import get_text_backbone
from ....common.runtime import prepend_python_path


def _check_sparsity(model) -> float:
    backbone = get_text_backbone(model)
    zero_count = 0
    total_count = 0
    for block in backbone.layers:
        for linear in find_linear_layers(block).values():
            weight = linear.weight.data
            zero_count += int((weight == 0).sum().item())
            total_count += int(weight.numel())
    return zero_count / max(total_count, 1)


class WandaSPMethod(BasePruningMethod):
    name = "wanda_sp"
    default_calibration_dataset = "c4"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        source_args = SimpleNamespace(
            model=args.model_path,
            seed=args.seed,
            nsamples=args.calibration_samples,
            pruning_ratio=args.sparsity_ratio,
            metrics="N/A",
            structure="N/A",
            prune_method="wanda_sp",
            cache_dir="llm_weights",
            unstr=args.pseudo_pruning,
            eval=False,
            save_model=None,
            data_path=args.data_path,
        )

        model.seqlen = args.sequence_length
        model.to(args.device)
        model.eval()

        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )

        with prepend_python_path(source_root):
            from lib.prune import prune_wanda_sp

            prune_wanda_sp(source_args, model, tokenizer_bundle.tokenizer, args.device, dataloader=calibration_batches)
            observed_sparsity = _check_sparsity(model)

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": "uniform-structured",
            "calibration_dataset": args.calibration_dataset,
            "calibration_samples": source_args.nsamples,
            "pseudo_pruning": source_args.unstr,
        }
