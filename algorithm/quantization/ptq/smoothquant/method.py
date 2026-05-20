"""Unified SmoothQuant runner."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODEL_TYPES = {
    "llama",
    "qwen2",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "minicpm",
    "minicpmv",
}
SUPPORTED_BIT_CONFIGS = {(8, 8)}


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


def _format_alpha(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


class SmoothQuantMethod(BaseQuantizationMethod):
    name = "smoothquant"
    npu_ready = True
    default_calibration_dataset = "pileval"

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = (
            f"{self.name}_w{args.weight_bits}a{args.activation_bits}"
            f"_seq{args.sequence_length}_alpha{_format_alpha(float(args.smoothquant_alpha))}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, model, args) -> None:
        model_type = getattr(model.config, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                "SmoothQuant currently supports LLaMA-, Qwen2/Qwen3/Qwen3.5/Qwen3.5-MoE-, and MiniCPM-style text decoders only; "
                f"got model_type={model_type!r}."
            )
        bit_config = (int(args.weight_bits), int(args.activation_bits))
        if bit_config not in SUPPORTED_BIT_CONFIGS:
            LOGGER.warning(
                "SmoothQuant non-W8A8 configurations are disabled in MindPipe; requested W%sA%s.",
                bit_config[0],
                bit_config[1],
            )
            raise ValueError(
                "SmoothQuant currently supports only W8A8 in MindPipe; "
                f"got W{bit_config[0]}A{bit_config[1]}."
            )
        if any(int(bits) < 16 for bits in (args.query_bits, args.key_bits, args.value_bits)):
            raise ValueError(
                "SmoothQuant fake-quant integration does not currently quantize Q/K/V caches; keep --query_bits/--key_bits/--value_bits at 16."
            )

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(model, args)
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        generated_act_scales_path = output_dir / "act_scales.pt"

        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )

        runtime_device = resolve_device(args.device)
        if runtime_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        with prepend_python_path(source_root):
            import importlib

            importlib.invalidate_caches()
            _purge_conflicting_modules("smoothquant", source_root / "smoothquant")

            from smoothquant.calibration import get_act_scales
            from smoothquant.fake_quant import quantize_model
            from smoothquant.smooth import smooth_lm

            act_scales_path = None
            if args.smoothquant_act_scales_from:
                act_scales_path = Path(args.smoothquant_act_scales_from)
                act_scales = torch.load(act_scales_path, map_location="cpu")
                LOGGER.info("Loaded SmoothQuant activation scales from %s", act_scales_path)
            else:
                # device_map 模式下不手动移动模型，由 dispatch_model 管理
                act_scales = get_act_scales(model, calibration_batches, runtime_device)
                LOGGER.info(
                    "Collected SmoothQuant activation scales from %s calibration samples",
                    args.calibration_samples,
                )
                # device_map 模式下不手动移动到 cpu
                if args.smoothquant_save_act_scales:
                    act_scales_path = generated_act_scales_path
                    torch.save(act_scales, act_scales_path)
                    LOGGER.info("Saved SmoothQuant activation scales to %s", act_scales_path)

            smooth_lm(model, act_scales, alpha=float(args.smoothquant_alpha))
            LOGGER.info("Applied SmoothQuant smoothing with alpha=%s", args.smoothquant_alpha)

            quantized_linear_names = quantize_model(
                model,
                weight_bits=int(args.weight_bits),
                activation_bits=int(args.activation_bits),
                weight_quant="per_channel",
                act_quant="per_token",
                quantize_bmm_input=True,
            )
            LOGGER.info(
                "Replaced %s linear layers with SmoothQuant fake-quant modules (W%sA%s)",
                len(quantized_linear_names),
                args.weight_bits,
                args.activation_bits,
            )

        artifacts: dict[str, object] = {
            "source_root": str(source_root),
            "smoothquant_config": {
                "alpha": float(args.smoothquant_alpha),
                "weight_bits": int(args.weight_bits),
                "activation_bits": int(args.activation_bits),
                "weight_quant": "per_channel",
                "activation_quant": "per_token",
                "quantize_bmm_input": True,
                "model_type": getattr(model.config, "model_type", None),
            },
            "quantized_linear_count": len(quantized_linear_names),
            "quantized_linear_layers": {
                name: {
                    "weight_bits": int(args.weight_bits),
                    "activation_bits": int(args.activation_bits),
                }
                for name in quantized_linear_names
            },
        }
        if act_scales_path is not None:
            artifacts["act_scales_path"] = str(act_scales_path)
        return artifacts
