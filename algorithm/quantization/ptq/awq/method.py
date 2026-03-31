"""Unified AWQ runner."""

from __future__ import annotations

from pathlib import Path

import torch

from ....common.io import ensure_dir
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


class AWQMethod(BaseQuantizationMethod):
    name = "awq"

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        awq_state_path = output_dir / "awq_search.pt"
        quantization_config = {
            "zero_point": not args.weight_symmetric,
            "q_group_size": args.weight_group_size,
        }
        if str(args.device).startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device))

        with prepend_python_path(source_root):
            from awq.quantize.pre_quant import run_awq
            from awq.quantize.quantizer import pseudo_quantize_model_weight

            awq_state = None
            if args.awq_search:
                awq_state = run_awq(
                    model,
                    tokenizer_bundle.tokenizer,
                    w_bit=args.weight_bits,
                    q_config=quantization_config,
                    n_samples=args.calibration_samples,
                    seqlen=args.sequence_length,
                    calib_data=args.calibration_dataset,
                    device=args.device,
                )
                torch.save(awq_state, awq_state_path)

            pseudo_quantize_model_weight(
                model,
                w_bit=args.weight_bits,
                q_config=quantization_config,
                device=args.device,
            )

        artifacts = {
            "source_root": str(source_root),
            "quantization_config": quantization_config,
        }
        if awq_state_path.exists():
            artifacts["awq_search_path"] = str(awq_state_path)
        return artifacts
