"""MQuant adapter for multimodal VLM quantization."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


class MQuantMethod(BaseQuantizationMethod):
    """Run MQuant-style RTN quantization on supported multimodal VLM backbones."""

    name = "mquant"
    npu_ready = False

    _DEFAULT_MQUANT_ROOT = Path("/mnt/42_store/zy/HUAWEI/work1/MQuant")
    _SUPPORTED_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl", "internvl_chat", "minicpmv"}

    @staticmethod
    def _resolve_source_model(model):
        return getattr(model, "_source_model", model)

    @staticmethod
    def _resolve_model_type(model) -> str:
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", "")
        return str(model_type)

    def _resolve_family(self, source_model) -> str:
        model_type = self._resolve_model_type(source_model)
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return "qwen2vl"
        if model_type == "internvl_chat":
            return "internvl"
        if model_type == "minicpmv":
            return "minicpmv"
        raise NotImplementedError(
            f"MQuant currently supports {sorted(self._SUPPORTED_MODEL_TYPES)}, got model_type={model_type!r}."
        )

    @staticmethod
    def _build_qwen2vl_compat_root(source_model):
        """Build a lightweight view matching the module layout expected by MQuant scripts."""
        multimodal_root = getattr(source_model, "model", source_model)
        visual = getattr(multimodal_root, "visual", None)
        text_root = (
            getattr(multimodal_root, "language_model", None)
            or getattr(multimodal_root, "model", None)
            or getattr(source_model, "language_model", None)
            or getattr(source_model, "model", None)
        )
        lm_head = getattr(source_model, "lm_head", None) or getattr(multimodal_root, "lm_head", None)
        text_config = getattr(text_root, "config", None) if text_root is not None else None
        model_config = text_config or getattr(source_model, "config", None)

        if visual is None or text_root is None or lm_head is None or model_config is None:
            raise AttributeError(
                "Failed to build Qwen2-VL compatibility root. "
                "Expected visual/text/lm_head/config are missing."
            )
        return SimpleNamespace(visual=visual, model=text_root, lm_head=lm_head, config=model_config)

    @staticmethod
    def _infer_module_device(module: Any) -> torch.device:
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    @classmethod
    def _resolve_mquant_root(cls) -> Path:
        root = Path(os.environ.get("MQUANT_ROOT", str(cls._DEFAULT_MQUANT_ROOT))).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"MQuant repo root does not exist: {root}. "
                "Set `MQUANT_ROOT` or prepare `/mnt/42_store/zy/HUAWEI/work1/MQuant`."
            )
        return root

    @staticmethod
    def _build_mquant_args(args) -> SimpleNamespace:
        activation_group_size = int(args.activation_group_size)
        return SimpleNamespace(
            # Core run flags
            seed=int(args.seed),
            quant=True,
            quant_llm=True,
            quant_visual_clip=True,
            quant_cross_attention=True,
            # Fuse/rotation
            not_fuse_layer_norms=False,
            no_fuse_visual_clip=False,
            no_fuse_visual_cross_attn=False,
            no_fuse_llm=False,
            rotate=True,
            rotate_visual_clip=True,
            rotate_visual_cross_attn=True,
            rotate_llm=True,
            rotate_mode=str(args.rotation_mode),
            # Weight quantization
            visual_w_bits=int(args.weight_bits),
            llm_w_bits=int(args.weight_bits),
            w_groupsize=int(args.weight_group_size),
            w_asym=not bool(args.weight_symmetric),
            visual_w_rtn=True,
            llm_w_rtn=True,
            visual_w_clip=bool(args.weight_bits <= 4),
            llm_w_clip=bool(args.weight_bits <= 4),
            percdamp=float(args.damp_percent),
            act_order=bool(args.use_activation_order),
            # Activation quantization
            visual_a_bits=int(args.activation_bits),
            llm_a_bits=int(args.activation_bits),
            a_groupsize=activation_group_size,
            a_asym=not bool(args.activation_symmetric),
            a_clip_ratio=1.0,
            visual_static=False,
            llm_static=False,
            act_per_tensor=False,
            # Online hadamard (defer to follow-up tuning; keep disabled by default)
            online_llm_hadamard=False,
            online_visual_hadamard=False,
            fp32_had=False,
            llm_split=False,
            visual_split=False,
            # Calibration placeholders
            dataset_name="",
            nsamples=int(args.calibration_samples),
            calib_num=int(args.calibration_samples),
            skip_names=[],
        )

    @staticmethod
    def _adapt_mquant_args_for_model(mquant_args: SimpleNamespace, source_model) -> SimpleNamespace:
        """Patch MQuant flags for model-specific architectural differences."""
        model_type = getattr(getattr(source_model, "config", None), "model_type", "")
        if str(model_type) == "qwen2_5_vl":
            # Qwen2.5-VL visual MLP uses gate/up/down projections instead of fc1/fc2.
            # Keep text-side fusion/rotation enabled, but skip visual-side steps that
            # assume the old fc1/fc2 layout used by Qwen2-VL.
            mquant_args.no_fuse_visual_clip = True
            mquant_args.rotate_visual_clip = False
        return mquant_args

    @staticmethod
    def _configure_activation_quantizers(
        quant_utils,
        modules: list,
        *,
        bits: int,
        groupsize: int,
        symmetric: bool,
        clip_ratio: float,
        skip_names: list[str],
    ) -> int:
        configured = 0
        for module in modules:
            qlayers = quant_utils.find_qlayers(module, layers=[quant_utils.ActQuantWrapper])
            for name, layer in qlayers.items():
                if any(pattern in name for pattern in skip_names):
                    continue
                layer.quantizer.configure(
                    bits=bits,
                    groupsize=groupsize,
                    sym=symmetric,
                    clip_ratio=clip_ratio,
                    act_per_tensor=False,
                    static=False,
                    observer_type="minmax",
                )
                configured += 1
        return configured

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        if args.weight_method != "rtn":
            raise NotImplementedError(
                "MindPipe MQuant adapter currently supports `--weight_method rtn` only. "
                "GPTQ path needs dataset-driven VLM wrappers and will be added separately."
            )

        source_model = self._resolve_source_model(model)
        family = self._resolve_family(source_model)
        mquant_root = self._resolve_mquant_root()
        mquant_args = self._build_mquant_args(args)
        mquant_args = self._adapt_mquant_args_for_model(mquant_args, source_model)
        if family == "qwen2vl":
            mquant_root_model = self._build_qwen2vl_compat_root(source_model)
        else:
            mquant_root_model = source_model
        proxy = SimpleNamespace(model=mquant_root_model)

        quantizer_map: dict[str, Any] = {}
        configured_act_quantizers = 0

        with prepend_python_path(mquant_root):
            try:
                from fake_quant import gptq as mquant_gptq
            except ModuleNotFoundError as exc:
                if exc.name == "unfoldNd":
                    raise ModuleNotFoundError(
                        "Missing MQuant dependency `unfoldNd`. "
                        "Install it in the current environment, e.g. `pip install unfoldNd`."
                    ) from exc
                raise
            from fake_quant import quant_utils as mquant_quant_utils
            from fake_quant import rotation_utils as mquant_rotation_utils
            from fake_quant import qwen2vl_rotation as mquant_qwen2vl_rotation
            from fake_quant.internvl_rotation import fuse_internvl_layer_norms
            from fake_quant.internvl_rotation import rotate_internvl2_model
            from fake_quant.minicpmv_rotation import fuse_minicpmv_layer_norms
            from fake_quant.minicpmv_rotation import rotate_minicpmv_model
            from fake_quant.qwen2vl_rotation import fuse_qwen2vl_layer_norms
            from fake_quant.qwen2vl_rotation import rotate_qwen2vl_model

            if family == "qwen2vl":
                if not mquant_args.not_fuse_layer_norms:
                    fuse_qwen2vl_layer_norms(proxy, mquant_args)
                if mquant_args.rotate:
                    original_get_orthogonal_matrix = mquant_qwen2vl_rotation.get_orthogonal_matrix

                    def _device_aware_get_orthogonal_matrix(size, mode):
                        visual_device = self._infer_module_device(mquant_root_model.visual.patch_embed.proj)
                        return mquant_rotation_utils.get_orthogonal_matrix(
                            size, mode, device=visual_device
                        )

                    mquant_qwen2vl_rotation.get_orthogonal_matrix = _device_aware_get_orthogonal_matrix
                    try:
                        rotate_qwen2vl_model(mquant_root_model, mquant_args)
                    finally:
                        mquant_qwen2vl_rotation.get_orthogonal_matrix = original_get_orthogonal_matrix
                if int(args.activation_bits) < 16:
                    mquant_quant_utils.qwen2vl_add_act_qaunt(proxy, mquant_args)
                quantizer_map = mquant_gptq.qwen2vl_rtn_gptq_fwrd_plus(
                    proxy,
                    dataset=None,
                    dev=args.device,
                    dataset_name="",
                    args=mquant_args,
                )
                if int(args.activation_bits) < 16:
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[mquant_root_model.visual, mquant_root_model.model],
                        bits=int(args.activation_bits),
                        groupsize=int(args.activation_group_size),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        skip_names=list(mquant_args.skip_names),
                    )

            elif family == "internvl":
                mquant_quant_utils.fuse_internvl(proxy)
                if not mquant_args.not_fuse_layer_norms:
                    fuse_internvl_layer_norms(proxy, mquant_args)
                if mquant_args.rotate:
                    rotate_internvl2_model(source_model, mquant_args)
                if int(args.activation_bits) < 16:
                    mquant_quant_utils.internvl_add_act_qaunt(proxy, mquant_args)
                quantizer_map = mquant_gptq.internvl_rtn_gptq_fwrd_plus(
                    proxy,
                    dataset=None,
                    dev=args.device,
                    dataset_name="",
                    args=mquant_args,
                )
                if int(args.activation_bits) < 16:
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[source_model.vision_model, source_model.mlp1, source_model.language_model],
                        bits=int(args.activation_bits),
                        groupsize=int(args.activation_group_size),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        skip_names=list(mquant_args.skip_names),
                    )

            elif family == "minicpmv":
                if not mquant_args.not_fuse_layer_norms:
                    fuse_minicpmv_layer_norms(source_model, mquant_args)
                if mquant_args.rotate:
                    rotate_minicpmv_model(source_model, mquant_args)
                if int(args.activation_bits) < 16:
                    mquant_quant_utils.minicpmv_add_act_qaunt(source_model, mquant_args)
                quantizer_map = mquant_gptq.minicpmv_rtn_gptq_fwrd_plus(
                    proxy,
                    dataset=None,
                    dev=args.device,
                    dataset_name="",
                    args=mquant_args,
                )
                if int(args.activation_bits) < 16:
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[source_model.vpm, source_model.resampler, source_model],
                        bits=int(args.activation_bits),
                        groupsize=int(args.activation_group_size),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        skip_names=list(mquant_args.skip_names),
                    )

        quantized_names = sorted(quantizer_map.keys())
        return {
            "mquant_root": str(mquant_root),
            "mquant_family": family,
            "weight_method": "rtn",
            "rotation_mode": str(args.rotation_mode),
            "quantized_linear_count": len(quantized_names),
            "quantized_linear_layers": quantized_names,
            "configured_act_quantizer_count": int(configured_act_quantizers),
            "mquant_config": {
                "llm_w_bits": int(mquant_args.llm_w_bits),
                "visual_w_bits": int(mquant_args.visual_w_bits),
                "llm_a_bits": int(mquant_args.llm_a_bits),
                "visual_a_bits": int(mquant_args.visual_a_bits),
                "w_groupsize": int(mquant_args.w_groupsize),
                "a_groupsize": int(mquant_args.a_groupsize),
                "rotate": bool(mquant_args.rotate),
                "rotate_visual_clip": bool(mquant_args.rotate_visual_clip),
                "rotate_visual_cross_attn": bool(mquant_args.rotate_visual_cross_attn),
                "rotate_llm": bool(mquant_args.rotate_llm),
            },
        }
