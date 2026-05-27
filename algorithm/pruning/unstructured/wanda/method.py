"""Unified Wanda runner."""

from __future__ import annotations

import torch

from pathlib import Path

from ...base import BasePruningMethod
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.device import resolve_device
from ....common.runtime import prepend_python_path
from ....common.runtime import purge_conflicting_modules


class WandaMethod(BasePruningMethod):
    name = "wanda"
    default_calibration_dataset = "c4"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        resolved = resolve_device(args.device)
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )

        source_root = Path(__file__).resolve().parent / "source"
        with prepend_python_path(source_root):
            purge_conflicting_modules("lib", allowed_root=source_root)
            from lib.prune import check_sparsity
            from lib.prune import prune_wanda

            prune_n = 0
            prune_m = 0
            if args.structure_pattern != "unstructured":
                prune_n, prune_m = map(int, args.structure_pattern.split(":"))
            args.nsamples = args.calibration_samples
            prune_wanda(
                args=args,
                model=model,
                tokenizer=tokenizer_bundle.tokenizer,
                device=args.device,
                prune_n=prune_n,
                prune_m=prune_m,
                dataloader=calibration_batches,
            )
            observed_sparsity = check_sparsity(model)

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": args.sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": args.structure_pattern,
        }
# feat: integrate ShortGPT layer pruning and unify calibration data loading.
