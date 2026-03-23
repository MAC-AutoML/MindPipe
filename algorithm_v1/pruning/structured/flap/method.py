"""Unified FLAP runner."""

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


class FLAPMethod(BasePruningMethod):
    name = "flap"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        source_args = SimpleNamespace(
            model=args.model_path,
            seed=args.seed,
            nsamples=args.calibration_samples,
            pruning_ratio=args.sparsity_ratio,
            remove_heads=args.flap_remove_heads,
            metrics=args.flap_metrics,
            structure=args.structure_pattern if args.structure_pattern != "unstructured" else "AL-AM",
            prune_method="flap",
            cache_dir="llm_weights",
            unstr=args.pseudo_pruning,
            eval=False,
            save_model=None,
        )

        model.seqlen = args.sequence_length
        model.to(args.device)
        model.eval()

        with prepend_python_path(source_root):
            from lib.prune import prune_flap

            prune_flap(source_args, model, tokenizer_bundle.tokenizer, args.device)
            observed_sparsity = _check_sparsity(model)

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": source_args.structure,
            "flap_metrics": source_args.metrics,
            "flap_remove_heads": source_args.remove_heads,
            "pseudo_pruning": source_args.unstr,
        }
