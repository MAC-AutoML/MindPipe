"""Unified structured Wanda runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ...base import BasePruningMethod
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

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        if args.structure_pattern != "unstructured":
            raise ValueError(
                "`wanda_sp` does not use --structure_pattern; leave it as 'unstructured'."
            )

        source_root = Path(__file__).resolve().parent / "source"
        source_args = SimpleNamespace(
            model=args.model_path,
            seed=args.seed,
            nsamples=args.calibration_samples,
            pruning_ratio=args.sparsity_ratio,
            remove_heads=args.flap_remove_heads,
            metrics="N/A",
            structure="N/A",
            prune_method="wanda_sp",
            cache_dir="llm_weights",
            unstr=args.pseudo_pruning,
            eval=False,
            save_model=None,
        )

        model.seqlen = args.sequence_length
        model.to(args.device)
        model.eval()

        with prepend_python_path(source_root):
            from lib.prune import prune_wanda_sp

            prune_wanda_sp(source_args, model, tokenizer_bundle.tokenizer, args.device)
            observed_sparsity = _check_sparsity(model)

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": "uniform-structured",
            "calibration_dataset": "c4",
            "calibration_samples": source_args.nsamples,
            "pseudo_pruning": source_args.unstr,
        }
