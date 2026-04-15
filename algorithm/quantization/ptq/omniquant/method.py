"""Unified OmniQuant runner."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODEL_TYPES = {"llama"}


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


def _format_group_suffix(group_size: int | None) -> str:
    if group_size is None:
        return ""
    return f"g{group_size}"


class OmniQuantMethod(BaseQuantizationMethod):
    name = "omniquant"
    npu_ready = False
    default_calibration_dataset = "wikitext2"

    def _resolve_weight_symmetric(self, args) -> bool:
        if getattr(args, "omniquant_weight_symmetric", None) is not None:
            return bool(args.omniquant_weight_symmetric)
        # Upstream OmniQuant defaults to asymmetric weight quantization unless explicitly overridden.
        return False

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = (
            f"{self.name}_w{args.weight_bits}a{args.activation_bits}"
            f"{_format_group_suffix(self._resolve_weight_group_size(args))}"
            f"_seq{args.sequence_length}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _resolve_weight_group_size(self, args) -> int | None:
        if int(args.weight_bits) >= 16:
            return None
        group_size = int(args.weight_group_size)
        return None if group_size <= 0 else group_size

    def _validate_args(self, model, args) -> None:
        model_type = getattr(model.config, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                "OmniQuant currently supports LLaMA-style text decoders only; "
                f"got model_type={model_type!r}."
            )
        if int(args.query_bits) < 16 or int(args.key_bits) < 16 or int(args.value_bits) < 16:
            raise ValueError(
                "OmniQuant follows upstream semantics and does not expose independent Q/K/V cache quantization in MindPipe; "
                "keep --query_bits/--key_bits/--value_bits at 16."
            )
        if int(args.activation_bits) < 16 and bool(args.activation_symmetric):
            raise ValueError(
                "OmniQuant activation quantization follows upstream asymmetric per-token quantization; "
                "set --activation_symmetric false when --activation_bits < 16."
            )
        if int(args.activation_bits) < 16 and int(args.activation_group_size) != int(args.group_size):
            raise ValueError(
                "OmniQuant only applies group size to weights. Keep --activation_group_size aligned with --group_size "
                "instead of using it as an independent activation grouping knob."
            )
        if int(args.weight_bits) < 16 and int(args.weight_group_size) != int(args.group_size):
            raise ValueError(
                "OmniQuant uses --group_size as the weight grouping knob. Keep --weight_group_size aligned with --group_size."
            )
        if args.omniquant_epochs > 0 and not (args.omniquant_lwc or args.omniquant_let):
            raise ValueError("OmniQuant requires --omniquant_lwc or --omniquant_let when --omniquant_epochs > 0.")

    def _build_source_args(self, args, output_dir: Path) -> SimpleNamespace:
        weight_group_size = self._resolve_weight_group_size(args)
        weight_symmetric = self._resolve_weight_symmetric(args)
        effective_deactive_amp = bool(args.omniquant_deactive_amp) or any(
            8 <= int(bits) < 16 for bits in (args.weight_bits, args.activation_bits)
        )
        return SimpleNamespace(
            abits=int(args.activation_bits),
            act_quant_params={
                "n_bits": int(args.activation_bits),
                "per_channel_axes": [],
                "symmetric": False,
                "dynamic_method": "per_token",
            },
            alpha=float(args.omniquant_alpha),
            aug_loss=bool(args.omniquant_aug_loss),
            batch_size=int(args.batch_size),
            deactive_amp=effective_deactive_amp,
            disable_zero_point=bool(args.omniquant_disable_zero_point),
            epochs=int(args.omniquant_epochs),
            let=bool(args.omniquant_let),
            let_lr=float(args.omniquant_let_lr),
            lwc=bool(args.omniquant_lwc),
            lwc_lr=float(args.omniquant_lwc_lr),
            nsamples=int(args.calibration_samples),
            output_dir=str(output_dir),
            save_diagnostics=bool(args.omniquant_save_diagnostics),
            p_quant_params={
                "n_bits": 16,
                "metric": "fix0to1",
            },
            q_quant_params={
                "n_bits": int(args.activation_bits),
                "per_channel_axes": [],
                "symmetric": False,
                "dynamic_method": "per_token",
            },
            resume=args.omniquant_resume_from,
            k_quant_params={
                "n_bits": int(args.activation_bits),
                "per_channel_axes": [],
                "symmetric": False,
                "dynamic_method": "per_token",
            },
            v_quant_params={
                "n_bits": int(args.activation_bits),
                "per_channel_axes": [],
                "symmetric": False,
                "dynamic_method": "per_token",
            },
            wbits=int(args.weight_bits),
            wd=float(args.omniquant_weight_decay),
            weight_group_size=weight_group_size,
            weight_quant_params={
                "n_bits": int(args.weight_bits),
                "per_channel_axes": [0],
                "symmetric": weight_symmetric,
                "dynamic_method": "per_channel",
                "group_size": weight_group_size,
                "lwc": bool(args.omniquant_lwc),
                "disable_zero_point": bool(args.omniquant_disable_zero_point),
            },
        )

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(model, args)
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        source_args = self._build_source_args(args, output_dir)
        weight_symmetric = self._resolve_weight_symmetric(args)
        runtime_device = resolve_device(args.device)
        if runtime_device.type != "cuda":
            raise NotImplementedError("OmniQuant in MindPipe currently requires CUDA execution.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if source_args.wbits >= 16 and source_args.abits >= 16:
            return {
                "source_root": str(source_root),
                "omniquant_config": {
                    "weight_bits": source_args.wbits,
                    "activation_bits": source_args.abits,
                    "group_size": source_args.weight_group_size,
                    "epochs": 0,
                    "let": False,
                    "lwc": False,
                    "weight_symmetric": weight_symmetric,
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

        generated_act_scales_path = output_dir / "act_scales.pt"
        generated_act_shifts_path = output_dir / "act_shifts.pt"
        parameter_checkpoint_path = output_dir / "omni_parameters.pth"
        act_scales_path = None
        act_shifts_path = None

        with prepend_python_path(source_root):
            import importlib

            importlib.invalidate_caches()
            _purge_conflicting_modules("omniquant", source_root / "omniquant")

            from omniquant.calibration import get_act_scales
            from omniquant.calibration import get_act_shifts
            from omniquant.train_utils import cali_omni_quant

            act_scales = None
            act_shifts = None
            if source_args.let:
                if args.omniquant_act_scales_from:
                    act_scales_path = Path(args.omniquant_act_scales_from)
                    act_scales = torch.load(act_scales_path, map_location="cpu")
                    LOGGER.info("Loaded OmniQuant activation scales from %s", act_scales_path)
                else:
                    model.to(runtime_device)
                    act_scales = get_act_scales(model, calibration_batches, runtime_device)
                    LOGGER.info(
                        "Collected OmniQuant activation scales from %s calibration samples",
                        args.calibration_samples,
                    )
                    if args.omniquant_save_act_stats:
                        act_scales_path = generated_act_scales_path
                        torch.save(act_scales, act_scales_path)
                        LOGGER.info("Saved OmniQuant activation scales to %s", act_scales_path)

                if args.omniquant_act_shifts_from:
                    act_shifts_path = Path(args.omniquant_act_shifts_from)
                    act_shifts = torch.load(act_shifts_path, map_location="cpu")
                    LOGGER.info("Loaded OmniQuant activation shifts from %s", act_shifts_path)
                else:
                    model.to(runtime_device)
                    act_shifts = get_act_shifts(model, calibration_batches, runtime_device)
                    LOGGER.info(
                        "Collected OmniQuant activation shifts from %s calibration samples",
                        args.calibration_samples,
                    )
                    if args.omniquant_save_act_stats:
                        act_shifts_path = generated_act_shifts_path
                        torch.save(act_shifts, act_shifts_path)
                        LOGGER.info("Saved OmniQuant activation shifts to %s", act_shifts_path)

                model.to("cpu")
                empty_cache(runtime_device)

            quantizer_artifacts = cali_omni_quant(
                source_args,
                model,
                calibration_batches,
                runtime_device,
                act_scales,
                act_shifts,
                LOGGER,
            )

        artifacts = {
            "source_root": str(source_root),
            "omniquant_config": {
                "weight_bits": source_args.wbits,
                "activation_bits": source_args.abits,
                "group_size": source_args.weight_group_size,
                "epochs": source_args.epochs,
                "alpha": source_args.alpha,
                "let": source_args.let,
                "let_lr": source_args.let_lr,
                "lwc": source_args.lwc,
                "lwc_lr": source_args.lwc_lr,
                "aug_loss": source_args.aug_loss,
                "weight_symmetric": weight_symmetric,
                "activation_symmetric": False,
            },
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
        if act_scales_path is not None:
            artifacts["act_scales_path"] = str(act_scales_path)
        if act_shifts_path is not None:
            artifacts["act_shifts_path"] = str(act_shifts_path)
        if parameter_checkpoint_path.exists():
            artifacts["omni_parameters_path"] = str(parameter_checkpoint_path)
        diagnostics_path = output_dir / "layer_diagnostics.json"
        if diagnostics_path.exists():
            artifacts["diagnostics_path"] = str(diagnostics_path)
        return artifacts
