"""Unified FlatQuant runner."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from ....common.device import empty_cache
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


LOGGER = logging.getLogger(__name__)


class FlatQuantMethod(BaseQuantizationMethod):
    name = "flatquant"

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        if (
            args.weight_bits == 16
            and args.activation_bits == 16
            and args.query_bits == 16
            and args.key_bits == 16
            and args.value_bits == 16
        ):
            run_spec = f"{self.name}_w16a16_seq{args.sequence_length}"
        elif (
            args.weight_bits == 4
            and args.activation_bits == 16
            and args.query_bits == 16
            and args.key_bits == 16
            and args.value_bits == 16
        ):
            run_spec = f"{self.name}_w4a16_seq{args.sequence_length}"
        elif (
            args.weight_bits == 4
            and args.activation_bits == 4
            and args.query_bits == 16
            and args.key_bits == 16
            and args.value_bits == 16
        ):
            run_spec = f"{self.name}_w4a4_q16k16v16_seq{args.sequence_length}"
        elif (
            args.weight_bits == 4
            and args.activation_bits == 4
            and args.query_bits == 16
            and args.key_bits == 4
            and args.value_bits == 4
        ):
            run_spec = f"{self.name}_w4a4_seq{args.sequence_length}"
        else:
            run_spec = (
                f"{self.name}_w{args.weight_bits}a{args.activation_bits}"
                f"_q{args.query_bits}k{args.key_bits}v{args.value_bits}"
                f"_seq{args.sequence_length}"
            )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _build_source_args(self, args, output_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            a_asym=not args.activation_symmetric,
            a_bits=args.activation_bits,
            a_groupsize=-1,
            act_order=args.use_activation_order,
            add_diag=args.flatquant_add_diag,
            cali_bsz=args.flatquant_calibration_batch_size,
            cali_dataset=args.calibration_dataset,
            cali_trans=args.flatquant_cali_trans,
            deactive_amp=args.flatquant_deactive_amp,
            diag_alpha=args.flatquant_diag_alpha,
            diag_init=args.flatquant_diag_init,
            direct_inv=args.flatquant_direct_inv,
            epochs=args.flatquant_epochs,
            exp_dir=str(output_dir),
            exp_name="default",
            flat_lr=args.flatquant_lr,
            gptq=args.weight_method == "gptq",
            gptq_mse=False,
            hf_token=args.hf_token,
            k_asym=not args.key_symmetric,
            k_bits=args.key_bits,
            k_groupsize=args.kv_group_size if args.key_bits < 16 else -1,
            lac=args.flatquant_lac,
            lwc=args.flatquant_lwc,
            model=args.model_path,
            model_name=model_slug(args.model_path),
            nsamples=args.calibration_samples,
            output_dir=str(output_dir),
            percdamp=args.damp_percent,
            q_asym=not args.query_symmetric,
            q_bits=args.query_bits,
            q_groupsize=-1,
            quantize=(
                args.weight_bits < 16
                or args.activation_bits < 16
                or args.query_bits < 16
                or args.key_bits < 16
                or args.value_bits < 16
            ),
            separate_vtrans=args.flatquant_separate_vtrans,
            seed=args.seed,
            v_asym=not args.value_symmetric,
            v_bits=args.value_bits,
            v_groupsize=args.kv_group_size if args.value_bits < 16 else -1,
            w_asym=not args.weight_symmetric,
            w_bits=args.weight_bits,
            w_groupsize=args.weight_group_size if args.weight_method == "gptq" else -1,
            warmup=args.flatquant_warmup,
        )

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        source_args = self._build_source_args(args, output_dir)
        flat_parameters_path = output_dir / "flat_parameters.pth"

        if not source_args.quantize:
            return {
                "source_root": str(source_root),
                "flatquant_config": {
                    "weight_bits": source_args.w_bits,
                    "activation_bits": source_args.a_bits,
                    "query_bits": source_args.q_bits,
                    "key_bits": source_args.k_bits,
                    "value_bits": source_args.v_bits,
                    "calibration_samples": source_args.nsamples,
                    "calibration_batch_size": source_args.cali_bsz,
                    "epochs": 0,
                    "flat_lr": 0.0,
                    "lwc": False,
                    "lac": False,
                    "cali_trans": False,
                    "add_diag": False,
                    "weight_quantizer": "none",
                },
                "quantized_linear_count": 0,
                "quantized_linear_layers": {},
                "skipped_reason": "all quantization bit-widths are 16; keep baseline model unchanged",
            }

        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
        )

        with prepend_python_path(source_root):
            import torch

            import gptq_utils
            import flatquant.utils as ref_utils
            from flatquant.flat_utils import reparameterize_model
            from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
            from flatquant.train_utils import cali_flat_quant

            ref_utils.DEV = torch.device(args.device)
            model = apply_flatquant_to_qwen(source_args, model)
            LOGGER.info("Applied FlatQuant wrappers to model")

            if source_args.cali_trans or source_args.add_diag or source_args.lwc or source_args.lac:
                cali_flat_quant(source_args, model, calibration_batches, args.device, LOGGER)

            reparameterize_model(model)
            LOGGER.info("Reparameterized FlatQuant model")

            quantizer_artifacts = {}
            weight_quantizer_name = "none"
            if source_args.w_bits < 16:
                if source_args.gptq:
                    try:
                        quantizers = gptq_utils.gptq_fwrd(model, calibration_batches, args.device, source_args)
                        weight_quantizer_name = "gptq"
                    except RuntimeError as error:
                        LOGGER.warning(
                            "FlatQuant GPTQ weight quantization failed; fallback to RTN. Error: %s",
                            error,
                        )
                        fallback_args = SimpleNamespace(**vars(source_args))
                        fallback_args.w_groupsize = -1
                        quantizers = gptq_utils.rtn_fwrd(model, args.device, fallback_args)
                        weight_quantizer_name = "rtn_fallback"
                else:
                    quantizers = gptq_utils.rtn_fwrd(model, args.device, source_args)
                    weight_quantizer_name = "rtn"
                quantizer_artifacts = {
                    name: {
                        "bits": source_args.w_bits,
                        "group_size": source_args.w_groupsize,
                        "symmetric": not source_args.w_asym,
                    }
                    for name in quantizers
                }
                empty_cache(args.device)

        artifacts = {
            "source_root": str(source_root),
            "flatquant_config": {
                "weight_bits": source_args.w_bits,
                "activation_bits": source_args.a_bits,
                "query_bits": source_args.q_bits,
                "key_bits": source_args.k_bits,
                "value_bits": source_args.v_bits,
                "calibration_samples": source_args.nsamples,
                "calibration_batch_size": source_args.cali_bsz,
                "epochs": source_args.epochs,
                "flat_lr": source_args.flat_lr,
                "lwc": source_args.lwc,
                "lac": source_args.lac,
                "cali_trans": source_args.cali_trans,
                "add_diag": source_args.add_diag,
                "weight_quantizer": weight_quantizer_name,
            },
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
        if flat_parameters_path.exists():
            artifacts["flat_parameters_path"] = str(flat_parameters_path)
        return artifacts
