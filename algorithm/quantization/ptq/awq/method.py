"""Unified AWQ runner."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ....common.device import backend_module
from ....common.device import resolve_device
from ....common.io import ensure_dir
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod

logger = logging.getLogger(__name__)


class AWQMethod(BaseQuantizationMethod):
    name = "awq"
    default_calibration_dataset = "pileval"

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        awq_state_path = output_dir / "awq_search.pt"
        awq_search_sequence_length = int(getattr(args, "awq_search_sequence_length", 512) or 512)
        awq_auto_scale = bool(getattr(args, "awq_auto_scale", True))
        awq_mse_range = bool(getattr(args, "awq_mse_range", True))
        awq_clip_targets = str(getattr(args, "awq_clip_targets", "auto") or "auto")
        if awq_search_sequence_length <= 0:
            raise ValueError(
                f"awq_search_sequence_length must be positive, got {awq_search_sequence_length}."
            )
        quantization_config = {
            "zero_point": not args.weight_symmetric,
            "q_group_size": args.weight_group_size,
        }
        runtime_device = resolve_device(args.device)
        if runtime_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        runtime_backend = backend_module(runtime_device)
        if runtime_backend is not None and hasattr(runtime_backend, "set_device"):
            runtime_backend.set_device(runtime_device)

        with prepend_python_path(source_root):
            from awq.quantize.pre_quant import run_awq
            from awq.quantize.quantizer import pseudo_quantize_model_weight

            awq_search_enabled = bool(args.awq_search)
            awq_state = None
            if awq_search_enabled:
                if awq_search_sequence_length != int(args.sequence_length):
                    logger.info(
                        "AWQ search uses sequence length %s while the global sequence_length remains %s.",
                        awq_search_sequence_length,
                        args.sequence_length,
                    )
                awq_state = run_awq(
                    model,
                    tokenizer_bundle.tokenizer,
                    w_bit=args.weight_bits,
                    q_config=quantization_config,
                    n_samples=args.calibration_samples,
                    seqlen=awq_search_sequence_length,
                    auto_scale=awq_auto_scale,
                    mse_range=awq_mse_range,
                    clip_targets=awq_clip_targets,
                    calib_data=args.calibration_dataset,
                    device=args.device,
                    data_path=args.data_path,
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
            "awq_search_enabled": awq_search_enabled,
            "awq_search_sequence_length": awq_search_sequence_length,
            "awq_auto_scale": awq_auto_scale,
            "awq_mse_range": awq_mse_range,
            "awq_clip_targets": awq_clip_targets,
        }
        if awq_state_path.exists():
            artifacts["awq_search_path"] = str(awq_state_path)
        return artifacts
