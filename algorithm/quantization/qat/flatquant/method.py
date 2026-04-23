"""Unified FlatQuant runner."""

from __future__ import annotations

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


def _infer_direct_inv_from_checkpoint(checkpoint_root: str | None, checkpoint_name: str) -> bool | None:
    if not checkpoint_root:
        return None
    checkpoint_path = Path(checkpoint_root) / checkpoint_name
    if not checkpoint_path.exists():
        return None

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not checkpoint:
        return None
    first_layer_state = checkpoint[next(iter(checkpoint))]
    keys = tuple(first_layer_state.keys())
    if any(".linear_left.weight" in key or ".linear_right.weight" in key for key in keys):
        return True
    if any(".linear_u_left.weight" in key or ".linear_u_right.weight" in key for key in keys):
        return False
    if any(".linear.weight" in key and ".o_trans." in key for key in keys):
        return True
    if any(".linear_u.weight" in key or ".linear_v.weight" in key for key in keys):
        return False
    return None


class FlatQuantMethod(BaseQuantizationMethod):
    name = "flatquant"
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
            reload_matrix=bool(args.flatquant_reload_matrix_from),
            resume=bool(args.flatquant_resume_from),
            separate_vtrans=args.flatquant_separate_vtrans,
            save_matrix=args.flatquant_save_matrix,
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
        model_type = getattr(getattr(args, "model_config", None), "model_type", None)
        if model_type == "qwen3_5" and (args.query_bits < 16 or args.key_bits < 16 or args.value_bits < 16):
            raise ValueError(
                "FlatQuant Qwen3.5 support currently covers weight/activation quantization only; keep --query_bits/--key_bits/--value_bits at 16."
            )

        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        source_args = self._build_source_args(args, output_dir)
        flat_parameters_path = output_dir / "flat_parameters.pth"

        inferred_direct_inv = None
        if source_args.resume:
            inferred_direct_inv = _infer_direct_inv_from_checkpoint(args.flatquant_resume_from, "flat_parameters.pth")
        elif source_args.reload_matrix:
            inferred_direct_inv = _infer_direct_inv_from_checkpoint(
                args.flatquant_reload_matrix_from,
                "flat_matrices.pth",
            )
        if inferred_direct_inv is not None and inferred_direct_inv != source_args.direct_inv:
            LOGGER.warning(
                "FlatQuant checkpoint structure implies direct_inv=%s; overriding requested direct_inv=%s",
                inferred_direct_inv,
                source_args.direct_inv,
            )
            source_args.direct_inv = inferred_direct_inv

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
                    "direct_inv": source_args.direct_inv,
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
            data_path=args.data_path,
        )

        with prepend_python_path(source_root):
            import importlib
            import torch

            importlib.invalidate_caches()
            _purge_conflicting_modules("flatquant", source_root / "flatquant")
            _purge_conflicting_modules("gptq_utils", source_root)

            import gptq_utils
            import flatquant.utils as ref_utils
            from flatquant.flat_utils import load_flat_matrices
            from flatquant.flat_utils import load_flat_parameters
            from flatquant.flat_utils import reparameterize_model
            from flatquant.flat_utils import save_flat_matrices
            from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
            from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
            from flatquant.model_tools.minicpm_utils import apply_flatquant_to_minicpm
            from flatquant.model_tools.qwen3_utils import apply_flatquant_to_qwen3
            from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
            from flatquant.train_utils import cali_flat_quant

            resolved_dev = resolve_device(args.device)
            ref_utils.DEV = resolved_dev
            if resolved_dev.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

            model_type = getattr(model.config, "model_type", None)
            rope_scaling = getattr(model.config, "rope_scaling", None) or {}
            rope_type = rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None
            if model_type == "llama":
                apply_wrapper = apply_flatquant_to_llama_31 if rope_type == "llama3" else apply_flatquant_to_llama
            elif model_type in {"minicpm", "minicpmv"}:
                apply_wrapper = apply_flatquant_to_minicpm
            elif model_type in {"qwen2", "qwen2_5_vl"}:
                apply_wrapper = apply_flatquant_to_qwen
            elif model_type in {"qwen3", "qwen3_vl"}:
                apply_wrapper = apply_flatquant_to_qwen3
            elif model_type == "qwen3_5":
                from flatquant.model_tools.qwen3_5_utils import apply_flatquant_to_qwen3_5

                apply_wrapper = apply_flatquant_to_qwen3_5
            else:
                raise NotImplementedError(
                    f"FlatQuant does not support model_type={model_type!r} in the unified launcher yet."
                )

            model = apply_wrapper(source_args, model)
            LOGGER.info("Applied FlatQuant wrappers to model")

            if source_args.resume:
                load_flat_parameters(source_args, model, path=args.flatquant_resume_from)
                LOGGER.info("Loaded FlatQuant parameters from %s", args.flatquant_resume_from)
            elif source_args.reload_matrix:
                load_flat_matrices(source_args, model, path=args.flatquant_reload_matrix_from)
                LOGGER.info("Loaded FlatQuant matrices from %s", args.flatquant_reload_matrix_from)
            elif source_args.cali_trans or source_args.add_diag or source_args.lwc or source_args.lac:
                cali_flat_quant(source_args, model, calibration_batches, args.device, LOGGER)
            if source_args.save_matrix:
                save_flat_matrices(source_args, model)
                LOGGER.info("Saved FlatQuant matrices to %s", output_dir)

            reparameterize_model(model)
            LOGGER.info("Reparameterized FlatQuant model")

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
                "direct_inv": source_args.direct_inv,
                "weight_quantizer": weight_quantizer_name,
            },
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
        if flat_parameters_path.exists():
            artifacts["flat_parameters_path"] = str(flat_parameters_path)
        return artifacts
