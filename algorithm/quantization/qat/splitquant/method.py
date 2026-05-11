"""Unified SplitQuant runner."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


LOGGER = logging.getLogger(__name__)


def _purge_conflicting_modules(module_name: str, allowed_root: Path) -> None:
    import sys

    allowed_root = allowed_root.resolve()
    prefix = f"{module_name}."
    for name, module in list(sys.modules.items()):
        if name != module_name and not name.startswith(prefix):
            continue

        module_file = getattr(module, "__file__", None)
        module_path = getattr(module, "__path__", None)
        candidates: list[Path] = []
        if module_file:
            candidates.append(Path(module_file))
        if module_path:
            candidates.extend(Path(path) for path in module_path)

        if any(path.resolve().is_relative_to(allowed_root) for path in candidates):
            continue
        sys.modules.pop(name, None)


def _resolve_split_group_size(args) -> int:
    group_sizes: list[int] = []
    if args.weight_bits < 16:
        group_sizes.append(int(args.weight_group_size))
    if args.activation_bits < 16:
        group_sizes.append(int(args.activation_group_size))
    if not group_sizes:
        return -1
    if any(size <= 0 for size in group_sizes):
        return -1
    first = group_sizes[0]
    if any(size != first for size in group_sizes[1:]):
        return -1
    return first


class SplitQuantMethod(BaseQuantizationMethod):
    name = "splitquant"
    npu_ready = True  # NPU path is exercised in-tree; Hadamard uses the non-CUDA fallback implementation
    default_calibration_dataset = "pileval"

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = (
            f"{self.name}_w{args.weight_bits}a{args.activation_bits}"
            f"_q{args.query_bits}k{args.key_bits}v{args.value_bits}"
            f"_seq{args.sequence_length}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, args) -> None:
        split_group_size = _resolve_split_group_size(args)
        if (args.weight_bits < 16 or args.activation_bits < 16) and split_group_size <= 0:
            raise ValueError(
                "SplitQuant requires a positive and consistent activation/weight group size when weight or activation quantization is enabled."
            )

        if args.weight_bits < 16 and args.activation_bits < 16 and args.weight_group_size != args.activation_group_size:
            raise ValueError(
                "SplitQuant requires --weight_group_size and --activation_group_size to match when both weights and activations are quantized."
            )

        model_type = getattr(getattr(args, "model_config", None), "model_type", None)
        if model_type == "qwen3_5" and (args.query_bits < 16 or args.key_bits < 16 or args.value_bits < 16):
            raise ValueError(
                "SplitQuant Qwen3.5 support currently covers weight/activation quantization only; keep --query_bits/--key_bits/--value_bits at 16."
            )

    def _build_source_args(self, args, output_dir: Path) -> SimpleNamespace:
        split_group_size = _resolve_split_group_size(args)
        return SimpleNamespace(
            a_asym=not args.activation_symmetric,
            a_bits=args.activation_bits,
            a_groupsize=args.activation_group_size if args.activation_bits < 16 else -1,
            act_order=args.use_activation_order,
            add_diag=args.splitquant_add_diag,
            cali_bsz=args.splitquant_calibration_batch_size,
            cali_dataset=args.calibration_dataset,
            cali_trans=args.splitquant_cali_trans,
            deactive_amp=args.splitquant_deactive_amp,
            diag_alpha=args.splitquant_diag_alpha,
            diag_init=args.splitquant_diag_init,
            epochs=args.splitquant_epochs,
            exp_dir=str(output_dir),
            exp_name="default",
            lr=args.splitquant_lr,
            gptq=args.weight_method == "gptq",
            gptq_mse=False,
            hf_token=args.hf_token,
            k_asym=not args.key_symmetric,
            k_bits=args.key_bits,
            k_groupsize=args.kv_group_size if args.key_bits < 16 else -1,
            lac=args.splitquant_lac,
            lwc=args.splitquant_lwc,
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
            reload_matrix=bool(args.splitquant_reload_matrix_from),
            resume=bool(args.splitquant_resume_from),
            separate_vtrans=args.splitquant_separate_vtrans,
            save_matrix=args.splitquant_save_matrix,
            seed=args.seed,
            split_group_size=split_group_size,
            v_asym=not args.value_symmetric,
            v_bits=args.value_bits,
            v_groupsize=args.kv_group_size if args.value_bits < 16 else -1,
            w_asym=not args.weight_symmetric,
            w_bits=args.weight_bits,
            w_groupsize=args.weight_group_size if args.weight_bits < 16 else -1,
            warmup=args.splitquant_warmup,
        )

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(args)
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        source_args = self._build_source_args(args, output_dir)
        parameter_checkpoint_path = output_dir / "splitquant_parameters.pth"
        config_artifact = {
            "weight_bits": source_args.w_bits,
            "activation_bits": source_args.a_bits,
            "query_bits": source_args.q_bits,
            "key_bits": source_args.k_bits,
            "value_bits": source_args.v_bits,
            "calibration_samples": source_args.nsamples,
            "calibration_batch_size": source_args.cali_bsz,
            "epochs": source_args.epochs,
            "lr": source_args.lr,
            "weight_group_size": source_args.w_groupsize,
            "activation_group_size": source_args.a_groupsize,
            "kv_group_size": max(source_args.k_groupsize, source_args.v_groupsize),
            "lwc": source_args.lwc,
            "lac": source_args.lac,
            "cali_trans": source_args.cali_trans,
            "add_diag": source_args.add_diag,
            "diag_init": source_args.diag_init,
            "diag_alpha": source_args.diag_alpha,
            "warmup": source_args.warmup,
            "deactive_amp": source_args.deactive_amp,
            "separate_vtrans": source_args.separate_vtrans,
            "save_matrix": source_args.save_matrix,
            "split_group_size": source_args.split_group_size,
            "weight_quantizer": "none",
        }

        if not source_args.quantize:
            return {
                "source_root": str(source_root),
                "splitquant_config": {**config_artifact, "epochs": 0, "lr": 0.0, "lwc": False, "lac": False, "cali_trans": False, "add_diag": False},
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
            data_path=args.data_path,
        )

        with prepend_python_path(source_root):
            import importlib
            import torch

            importlib.invalidate_caches()
            _purge_conflicting_modules("splitquant", source_root / "splitquant")
            _purge_conflicting_modules("gptq_utils", source_root)

            import gptq_utils
            import splitquant.utils as ref_utils
            from splitquant.model_tools.llama31_utils import apply_splitquant_to_llama_31
            from splitquant.model_tools.llama_utils import apply_splitquant_to_llama
            from splitquant.split_utils import load_splitquant_matrices
            from splitquant.split_utils import load_splitquant_parameters
            from splitquant.split_utils import reparameterize_splitquant_model
            from splitquant.split_utils import save_splitquant_matrices
            from splitquant.model_tools.minicpm_split_utils import apply_splitquant_to_minicpm
            from splitquant.model_tools.qwen3_split_utils import apply_splitquant_to_qwen3
            from splitquant.model_tools.qwen_split_utils import apply_splitquant_to_qwen
            from splitquant.train_utils import cali_split_quant
            from splitquant.backbone_utils import get_decoder_layers as splitquant_layers

            resolved_dev = resolve_device(args.device)
            ref_utils.DEV = resolved_dev
            if resolved_dev.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

            model_type = getattr(model.config, "model_type", None)
            rope_scaling = getattr(model.config, "rope_scaling", None) or {}
            rope_type = rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None
            if model_type == "llama":
                apply_wrapper = apply_splitquant_to_llama_31 if rope_type == "llama3" else apply_splitquant_to_llama
            elif model_type in {"minicpm", "minicpmv"}:
                apply_wrapper = apply_splitquant_to_minicpm
            elif model_type in {"qwen2", "qwen2_5_vl"}:
                apply_wrapper = apply_splitquant_to_qwen
            elif model_type in {"qwen3", "qwen3_vl"}:
                apply_wrapper = apply_splitquant_to_qwen3
            elif model_type == "qwen3_5":
                from splitquant.model_tools.qwen3_5_split_utils import apply_splitquant_to_qwen3_5

                apply_wrapper = apply_splitquant_to_qwen3_5
            elif model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
                from splitquant.model_tools.qwen3_5_split_utils import apply_splitquant_to_qwen3_5_moe

                apply_wrapper = apply_splitquant_to_qwen3_5_moe
            else:
                raise NotImplementedError(
                    f"SplitQuant currently supports LLaMA-, MiniCPM-, and Qwen-style models only; got model_type={model_type!r}."
                )

            original_layer_devices = [next(layer.parameters()).device for layer in splitquant_layers(model)]
            model = apply_wrapper(source_args, model)
            LOGGER.info("Applied SplitQuant wrappers to model")

            # 将 wrapper 新建的参数移到原始层所在设备（覆盖 resume/reload/cali 全路径）
            for layer, layer_device in zip(splitquant_layers(model), original_layer_devices):
                layer.to(layer_device)

            if source_args.resume:
                load_splitquant_parameters(source_args, model, path=args.splitquant_resume_from)
                LOGGER.info("Loaded SplitQuant parameters from %s", args.splitquant_resume_from)
            elif source_args.reload_matrix:
                load_splitquant_matrices(source_args, model, path=args.splitquant_reload_matrix_from)
                LOGGER.info("Loaded SplitQuant matrices from %s", args.splitquant_reload_matrix_from)
            elif source_args.cali_trans or source_args.add_diag or source_args.lwc or source_args.lac:
                cali_split_quant(source_args, model, calibration_batches, args.device, LOGGER)
            if source_args.save_matrix:
                save_splitquant_matrices(source_args, model)
                LOGGER.info("Saved SplitQuant matrices to %s", output_dir)

            reparameterize_splitquant_model(model)
            LOGGER.info("Reparameterized SplitQuant model")

            quantizer_artifacts = {}
            weight_quantizer_name = "none"
            if source_args.w_bits < 16:
                if source_args.gptq:
                    quantizers = gptq_utils.gptq_fwrd(model, calibration_batches, args.device, source_args)
                    weight_quantizer_name = "gptq"
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

        config_artifact["weight_quantizer"] = weight_quantizer_name
        artifacts = {
            "source_root": str(source_root),
            "splitquant_config": copy.deepcopy(config_artifact),
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
        if parameter_checkpoint_path.exists():
            artifacts["splitquant_parameters_path"] = str(parameter_checkpoint_path)
        return artifacts
