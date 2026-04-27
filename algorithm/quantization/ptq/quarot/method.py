"""Unified QuaRot runner."""

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import build_decoder_layer_groups
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import get_layer_device
from ....common.modeling import get_text_backbone
from ....common.modeling import load_model_and_tokenizer
from ....common.modeling import move_tensors_to_device
from ....common.modeling import unwrap_layer_output
from ....common.runtime import prepend_python_path

from ...base import BaseQuantizationMethod


class QuaRotMethod(BaseQuantizationMethod):
    name = "quarot"
    npu_ready = False  # Hadamard fallback 需在 NPU 上验证
    default_calibration_dataset = "c4"

    @staticmethod
    def _is_minicpm_like(model) -> bool:
        return getattr(getattr(model, "config", None), "model_type", None) == "minicpmv"

    @staticmethod
    def _supports_llama_style_backbone(model) -> bool:
        try:
            backbone = get_text_backbone(model)
            if backbone.embed_tokens is None:
                return False
            first_layer = backbone.layers[0]
        except Exception:
            return False

        if not hasattr(first_layer, "self_attn") or not hasattr(first_layer, "mlp"):
            return False

        attn = first_layer.self_attn
        mlp = first_layer.mlp
        required_attn = ("q_proj", "k_proj", "v_proj", "o_proj")
        required_mlp = ("up_proj", "gate_proj", "down_proj")
        if not all(hasattr(attn, name) for name in required_attn):
            return False
        if not all(hasattr(mlp, name) for name in required_mlp):
            return False
        return hasattr(first_layer, "input_layernorm") and hasattr(first_layer, "post_attention_layernorm")

    @staticmethod
    def _get_decoder_config_value(model, key: str):
        decoder_config = get_text_backbone(model).decoder_config
        if hasattr(decoder_config, key):
            return getattr(decoder_config, key)
        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, key):
            return getattr(model_config, key)
        raise AttributeError(f"Neither decoder config nor model config exposes `{key}`.")

    def _sync_decoder_config_to_model_config(self, model) -> None:
        model_config = getattr(model, "config", None)
        if model_config is None:
            return
        decoder_config = get_text_backbone(model).decoder_config
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "max_position_embeddings",
        ):
            if not hasattr(model_config, key) and hasattr(decoder_config, key):
                setattr(model_config, key, getattr(decoder_config, key))

    @staticmethod
    def _rotation_guard_reason(model) -> str | None:
        return None

    @staticmethod
    def _default_disable_hidden_hadamard(model) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        return model_type == "qwen3_5"

    @staticmethod
    def _try_get_hadK(hadamard_utils, size: int):
        try:
            return hadamard_utils.get_hadK(size)
        except (AssertionError, ValueError):
            return None, None

    def load_resources(self, args):
        return load_model_and_tokenizer(args.model_path,dtype=args.dtype,attn_implementation=args.attn_implementation)

    @staticmethod
    def _looks_like_qwen_model_path(model_path: str) -> bool:
        return "qwen" in str(model_path).lower()

    @staticmethod
    def _quantization_root(model):
        try:
            return get_text_backbone(model).root
        except Exception:
            if model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
                return model.language_model
        return model

    @staticmethod
    def _is_qwen2_5_vl(model) -> bool:
        return getattr(getattr(model, "config", None), "model_type", None) == "qwen2_5_vl"

    @staticmethod
    def _get_qwen2_5_vl_visual_root(model):
        multimodal_root = getattr(model, "model", model)
        return getattr(multimodal_root, "visual", None)

    @staticmethod
    def _get_qwen2_5_vl_visual_prefix(model) -> str:
        return "model.visual" if hasattr(getattr(model, "model", None), "visual") else "visual"

    @staticmethod
    def _resolve_qwen2_5_vl_visual_weight_bits(args) -> int:
        override = os.environ.get("QUAROT_QWEN2_5_VL_VISUAL_WEIGHT_BITS", "").strip()
        if override:
            return int(override)
        return int(args.weight_bits)

    @staticmethod
    def _resolve_qwen2_5_vl_visual_activation_bits(args) -> int:
        override = os.environ.get("QUAROT_QWEN2_5_VL_VISUAL_ACT_BITS", "").strip()
        if override:
            return int(override)
        if hasattr(args, "activation_bits"):
            return int(args.activation_bits)
        return int(args.a_bits)

    @staticmethod
    def _get_token_mixer(layer):
        mixer = getattr(layer, "self_attn", None)
        if mixer is not None:
            return mixer
        mixer = getattr(layer, "linear_attn", None)
        if mixer is not None:
            return mixer
        raise AttributeError(f"Unsupported decoder layer without token mixer: {type(layer)}")

    @classmethod
    def _get_token_mixer_input_linears(cls, layer) -> list[torch.nn.Linear]:
        mixer = cls._get_token_mixer(layer)
        if hasattr(mixer, "q_proj"):
            return [mixer.q_proj, mixer.k_proj, mixer.v_proj]
        return [
            mixer.in_proj_qkv,
            mixer.in_proj_z,
            mixer.in_proj_b,
            mixer.in_proj_a,
        ]

    @staticmethod
    def _is_linear_attn_mixer(token_mixer) -> bool:
        return hasattr(token_mixer, "in_proj_qkv") and hasattr(token_mixer, "out_proj")

    @staticmethod
    def _get_token_mixer_output_groupsize(token_mixer, decoder_config, default_head_dim: int) -> int:
        if not QuaRotMethod._is_linear_attn_mixer(token_mixer):
            return default_head_dim
        value_head_dim = getattr(token_mixer, "head_v_dim", None)
        if value_head_dim is None:
            value_head_dim = getattr(decoder_config, "linear_value_head_dim", None)
        return int(value_head_dim) if value_head_dim is not None else default_head_dim

    @staticmethod
    def _get_linear_attn_k_quant_overrides(token_mixer, decoder_config, default_head_dim: int) -> dict[str, int]:
        key_head_dim = getattr(token_mixer, "head_k_dim", None)
        if key_head_dim is None:
            key_head_dim = getattr(decoder_config, "linear_key_head_dim", None)
        value_heads = getattr(token_mixer, "num_v_heads", None)
        if value_heads is None:
            value_heads = getattr(decoder_config, "linear_num_value_heads", None)

        overrides = {"qk_head_dim": int(key_head_dim) if key_head_dim is not None else default_head_dim}
        if value_heads is not None:
            overrides["qk_num_heads"] = int(value_heads)
        return overrides

    def _build_source_args(self, args) -> SimpleNamespace:
        # Keep QuaRot defaults aligned with upstream fake_quant parser:
        # w_clip=False, a/k/v clip ratios = 1.0.
        default_w_clip = bool(getattr(args, "w_clip", False))
        default_a_clip_ratio = float(getattr(args, "a_clip_ratio", 1.0))
        default_k_clip_ratio = float(getattr(args, "k_clip_ratio", 1.0))
        default_v_clip_ratio = float(getattr(args, "v_clip_ratio", 1.0))
        return SimpleNamespace(
            model=args.model_path,
            seed=args.seed,
            hf_token=args.hf_token,
            rotate=True,
            rotate_mode=args.rotation_mode,
            optimized_rotation_path=args.rotation_checkpoint,
            fp32_had=bool(getattr(args, "fp32_had", False)),
            quarot_disable_hidden_hadamard=getattr(args, "quarot_disable_hidden_hadamard", None),
            w_bits=args.weight_bits,
            w_groupsize=args.weight_group_size,
            w_asym=not args.weight_symmetric,
            w_rtn=args.weight_method == "rtn",
            w_clip=bool(getattr(args, "weight_clip", default_w_clip)),
            nsamples=args.calibration_samples,
            percdamp=args.damp_percent,
            act_order=args.use_activation_order,
            int8_down_proj=False,
            a_bits=args.activation_bits,
            a_groupsize=args.activation_group_size,
            a_asym=not args.activation_symmetric,
            a_clip_ratio=float(getattr(args, "activation_clip_ratio", default_a_clip_ratio)),
            k_bits=args.key_bits,
            k_groupsize=args.kv_group_size,
            k_asym=not args.key_symmetric,
            k_clip_ratio=float(getattr(args, "key_clip_ratio", default_k_clip_ratio)),
            k_pre_rope=bool(getattr(args, "quarot_k_pre_rope", False)),
            k_tokenwise_per_head=bool(getattr(args, "quarot_k_tokenwise_per_head", False)),
            k_hadamard=bool(getattr(args, "quarot_k_hadamard", True)),
            k_per_head_channel=bool(getattr(args, "quarot_k_per_head_channel", False)),
            k_equalize=bool(getattr(args, "quarot_k_equalize", False)),
            k_equalize_alpha=float(getattr(args, "quarot_k_equalize_alpha", 1.0)),
            k_equalize_max_scale=float(getattr(args, "quarot_k_equalize_max_scale", 8.0)),
            k_equalize_with_q=bool(getattr(args, "quarot_k_equalize_with_q", False)),
            k_equalize_q_power=float(getattr(args, "quarot_k_equalize_q_power", 1.0)),
            v_bits=args.value_bits,
            v_groupsize=args.kv_group_size,
            v_asym=not args.value_symmetric,
            v_clip_ratio=float(getattr(args, "value_clip_ratio", default_v_clip_ratio)),
            quarot_qwen2_5_vl_rotate_visual_branch=bool(
                getattr(args, "quarot_qwen2_5_vl_rotate_visual_branch", True)
            ),
            quarot_qwen2_5_vl_rotate_merger_output=bool(
                getattr(args, "quarot_qwen2_5_vl_rotate_merger_output", True)
            ),
            quarot_qwen2_5_vl_disable_hidden_hadamard=bool(
                getattr(args, "quarot_qwen2_5_vl_disable_hidden_hadamard", False)
            ),
            quarot_qwen2_5_vl_center_merger_output=bool(
                getattr(args, "quarot_qwen2_5_vl_center_merger_output", False)
            ),
            quarot_qwen2_5_vl_first_layer_activation_bits=getattr(
                args, "quarot_qwen2_5_vl_first_layer_activation_bits", None
            ),
        )

    @staticmethod
    def _bind_rotation_device(rotation_utils, device: str) -> None:
        original_get_orthogonal_matrix = rotation_utils.get_orthogonal_matrix

        def get_orthogonal_matrix(size, mode, device_override=None):
            resolved_device = torch.device(device if device_override is None else device_override)
            return original_get_orthogonal_matrix(size, mode, device=resolved_device)

        rotation_utils.get_orthogonal_matrix = get_orthogonal_matrix

    @staticmethod
    def _ensure_forward_global(module, function_name: str) -> None:
        function_object = getattr(module, "forward").__func__
        if function_name in function_object.__globals__:
            return
        source_module = importlib.import_module(function_object.__module__)
        if not hasattr(source_module, function_name):
            raise KeyError(f"Missing {function_name} in {function_object.__module__}.")
        function_object.__globals__[function_name] = getattr(source_module, function_name)

    def _patch_model_utils(self, model_utils, model):
        qwen_model_types: tuple[type, ...] = ()
        qwen_model_type_names = {"qwen2", "qwen2_5_vl", "qwen2_vl", "qwen3", "qwen3_vl", "qwen3_5"}

        for module_name, class_name in (
            ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
            ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
            ("transformers.models.qwen2_vl.modeling_qwen2_vl", "Qwen2VLForConditionalGeneration"),
            ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
            ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLForConditionalGeneration"),
            ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ):
            try:
                module = importlib.import_module(module_name)
                qwen_model_types += (getattr(module, class_name),)
            except Exception:
                continue

        def _is_qwen_like(model) -> bool:
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            if model_type in qwen_model_type_names:
                return True
            return isinstance(model, qwen_model_types)

        def _is_llama_like(model) -> bool:
            return (
                _is_qwen_like(model)
                or self._is_minicpm_like(model)
                or isinstance(model, model_utils.LLAMA_MODEL)
                or self._supports_llama_style_backbone(model)
            )

        def model_type_extractor(model):
            if _is_llama_like(model):
                return model_utils.LLAMA_MODEL
            if isinstance(model, model_utils.OPT_MODEL):
                return model_utils.OPT_MODEL
            raise ValueError(f"Unknown model type {model}")

        def get_model_type(model):
            if _is_llama_like(model):
                return model_utils.LLAMA_MODEL
            if isinstance(model, model_utils.OPT_MODEL):
                return model_utils.OPT_MODEL
            raise ValueError(f"Unknown model type {model}")

        def get_rope_function_name(model):
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            if model_type in {"qwen2_5_vl", "qwen2_vl"}:
                return "apply_multimodal_rotary_pos_emb"
            if _is_llama_like(model):
                return "apply_rotary_pos_emb"
            raise NotImplementedError

        def get_layers(model):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                return backbone.layers
            return model_utils.get_layers.__wrapped__(model)  # type: ignore[attr-defined]

        def get_embeddings(model, model_type):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                if backbone.embed_tokens is not None:
                    return [backbone.embed_tokens]
            return model_utils.get_embeddings.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        def get_transformer_layers(model, model_type):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                return list(backbone.layers)
            return model_utils.get_transformer_layers.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        def get_lm_head(model, model_type):
            if _is_llama_like(model):
                return model.lm_head
            return model_utils.get_lm_head.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        def get_pre_head_layernorm(model, model_type):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                if backbone.final_norm is not None:
                    return backbone.final_norm
            return model_utils.get_pre_head_layernorm.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        model_utils.model_type_extractor = functools.wraps(model_utils.model_type_extractor)(model_type_extractor)
        model_utils.get_model_type = functools.wraps(model_utils.get_model_type)(get_model_type)
        model_utils.get_rope_function_name = functools.wraps(model_utils.get_rope_function_name)(get_rope_function_name)
        model_utils.get_layers = functools.wraps(model_utils.get_layers)(get_layers)
        model_utils.get_embeddings = functools.wraps(model_utils.get_embeddings)(get_embeddings)
        model_utils.get_transformer_layers = functools.wraps(model_utils.get_transformer_layers)(get_transformer_layers)
        model_utils.get_lm_head = functools.wraps(model_utils.get_lm_head)(get_lm_head)
        model_utils.get_pre_head_layernorm = functools.wraps(model_utils.get_pre_head_layernorm)(get_pre_head_layernorm)
        extra_norm_types = []
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

            extra_norm_types.append(Qwen2RMSNorm)
        except Exception:
            pass
        try:
            extra_norm_types.append(get_text_backbone(model).layers[0].input_layernorm.__class__)
        except Exception:
            pass
        return tuple(dict.fromkeys(extra_norm_types))

    def _fuse_layer_norms(
        self,
        model,
        model_utils,
        rotation_utils,
        extra_norm_types=(),
        *,
        fuse_qwen2_5_vl_visual_branch: bool = True,
    ):
        model_type = model_utils.get_model_type(model)
        kwargs = {"model": model, "model_type": model_type}

        for embedding in model_utils.get_embeddings(**kwargs):
            weight = embedding.weight.data.double()
            embedding.weight.data = (weight - weight.mean(dim=-1, keepdim=True)).to(embedding.weight.data.dtype)

        for layer in model_utils.get_transformer_layers(**kwargs):
            rotation_utils.fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            self._neutralize_norm(layer.post_attention_layernorm)
            rotation_utils.fuse_ln_linear(
                layer.input_layernorm,
                self._get_token_mixer_input_linears(layer),
            )
            self._neutralize_norm(layer.input_layernorm)

        pre_head_norm = model_utils.get_pre_head_layernorm(**kwargs)
        rotation_utils.fuse_ln_linear(pre_head_norm, [model_utils.get_lm_head(**kwargs)])
        self._neutralize_norm(pre_head_norm)

        model_type_name = getattr(getattr(model, "config", None), "model_type", None)
        if model_type_name in {"qwen2", "qwen2_5_vl", "qwen2_vl"}:
            import transformers

            hidden_size = int(self._get_decoder_config_value(model, "hidden_size"))
            norm_types = (transformers.models.llama.modeling_llama.LlamaRMSNorm, *extra_norm_types)
            replace_root = get_text_backbone(model).root
            model_utils.replace_modules(
                replace_root,
                norm_types,
                lambda _: model_utils.RMSN(hidden_size),
                replace_layers=False,
            )

        if self._is_qwen2_5_vl(model) and fuse_qwen2_5_vl_visual_branch:
            self._fuse_qwen2_5_vl_visual_layer_norms(model, model_utils, rotation_utils)

    def _fuse_qwen2_5_vl_visual_layer_norms(self, model, model_utils, rotation_utils) -> None:
        visual = self._get_qwen2_5_vl_visual_root(model)
        if visual is None or not hasattr(visual, "blocks") or len(visual.blocks) == 0:
            return

        if hasattr(visual, "patch_embed") and hasattr(visual.patch_embed, "proj"):
            rotation_utils.bake_mean_into_conv(visual.patch_embed.proj)

        for layer in visual.blocks:
            rotation_utils.fuse_ln_linear(layer.norm1, [layer.attn.qkv])
            rotation_utils.fuse_ln_linear(layer.norm2, [layer.mlp.gate_proj, layer.mlp.up_proj])
            rotation_utils.bake_mean_into_linear(layer.attn.proj)
            rotation_utils.bake_mean_into_linear(layer.mlp.down_proj)

        if hasattr(visual, "merger") and hasattr(visual.merger, "ln_q") and hasattr(visual.merger, "mlp"):
            first_linear = visual.merger.mlp[0] if len(visual.merger.mlp) > 0 else None
            if isinstance(first_linear, torch.nn.Linear):
                rotation_utils.fuse_merger_linear(visual.merger.ln_q, [first_linear])

        visual_norm = visual.blocks[0].norm1
        visual_hidden_size = int(getattr(visual_norm.weight, "shape", [0])[0])
        visual_eps = float(getattr(visual_norm, "variance_epsilon", getattr(visual_norm, "eps", 1e-6)))
        visual_norm_type = visual_norm.__class__
        model_utils.replace_modules(
            visual,
            visual_norm_type,
            lambda _: model_utils.RMSN(visual_hidden_size, eps=visual_eps),
            replace_layers=False,
        )

    def _configure_activation_quantizers(
        self,
        model,
        source_args,
        quant_utils,
        hadamard_utils,
        rotation_utils,
        model_utils,
        ref_utils,
        *,
        configure_hadamard: bool = True,
        configure_input_quant: bool = True,
        configure_k_quant: bool = True,
    ) -> None:
        quantization_root = self._quantization_root(model)
        decoder_config = get_text_backbone(model).decoder_config
        intermediate_size = int(self._get_decoder_config_value(model, "intermediate_size"))
        num_heads = int(self._get_decoder_config_value(model, "num_attention_heads"))
        hidden_size = int(self._get_decoder_config_value(model, "hidden_size"))
        qlayers = quant_utils.find_qlayers(quantization_root)
        disable_hidden_hadamard = self._disable_hidden_hadamard_requested(model, source_args)
        disable_runtime_hadamard = (
            getattr(getattr(model, "config", None), "model_type", None) == "qwen2_vl"
        )
        head_dim = int(getattr(decoder_config, "head_dim", hidden_size // num_heads))
        if configure_hadamard and source_args.rotate and not disable_hidden_hadamard and not disable_runtime_hadamard:
            for layer_name, qlayer in qlayers.items():
                if "down_proj" in layer_name:
                    had_k, k_value = self._try_get_hadK(hadamard_utils, intermediate_size)
                    if k_value is not None:
                        qlayer.online_full_had = True
                        qlayer.had_K = had_k
                        qlayer.K = k_value
                        qlayer.fp32_had = source_args.fp32_had
                if "o_proj" in layer_name:
                    had_k, k_value = hadamard_utils.get_hadK(num_heads)
                    qlayer.online_partial_had = True
                    qlayer.had_K = had_k
                    qlayer.K = k_value
                    qlayer.had_dim = head_dim
                    qlayer.fp32_had = source_args.fp32_had

        if configure_input_quant and (source_args.a_bits < 16 or source_args.v_bits < 16):
            down_proj_groupsize = -1
            if source_args.a_groupsize > 0:
                down_proj_groupsize = ref_utils.llama_down_proj_groupsize(quantization_root, source_args.a_groupsize)

            act_qlayers = quant_utils.find_qlayers(quantization_root, layers=[quant_utils.ActQuantWrapper])
            first_layer_act_bits = getattr(
                source_args, "quarot_qwen2_5_vl_first_layer_activation_bits", None
            )
            for layer_name, qlayer in act_qlayers.items():
                layer_input_bits = source_args.a_bits
                layer_groupsize = source_args.a_groupsize
                if (
                    self._is_qwen2_5_vl(model)
                    and first_layer_act_bits is not None
                    and ("layers.0." in layer_name or layer_name.startswith("layers.0."))
                ):
                    layer_input_bits = int(first_layer_act_bits)
                if "v_proj" in layer_name and source_args.v_bits < 16:
                    qlayer.out_quantizer.configure(
                        bits=source_args.v_bits,
                        groupsize=source_args.v_groupsize,
                        sym=not source_args.v_asym,
                        clip_ratio=source_args.v_clip_ratio,
                    )
                if "o_proj" in layer_name or "linear_attn.out_proj" in layer_name:
                    layer_groupsize = head_dim
                if "lm_head" in layer_name:
                    layer_input_bits = 16
                if "down_proj" in layer_name:
                    layer_groupsize = down_proj_groupsize
                qlayer.quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=not source_args.a_asym,
                    clip_ratio=source_args.a_clip_ratio,
                )

        if configure_k_quant and source_args.k_bits < 16:
            rope_function_name = model_utils.get_rope_function_name(model)
            backbone = get_text_backbone(model)
            k_quant_config = {
                "k_bits": source_args.k_bits,
                "k_groupsize": source_args.k_groupsize,
                "k_sym": not source_args.k_asym,
                "k_clip_ratio": source_args.k_clip_ratio,
                "k_pre_rope": source_args.k_pre_rope,
                "k_tokenwise_per_head": source_args.k_tokenwise_per_head,
                "k_hadamard": source_args.k_hadamard,
                "k_per_head_channel": source_args.k_per_head_channel,
                "k_equalize": getattr(source_args, "k_equalize", False),
                "k_equalize_alpha": getattr(source_args, "k_equalize_alpha", 1.0),
                "k_equalize_max_scale": getattr(source_args, "k_equalize_max_scale", 8.0),
                "k_equalize_with_q": getattr(source_args, "k_equalize_with_q", False),
                "k_equalize_q_power": getattr(source_args, "k_equalize_q_power", 1.0),
            }
            for layer in backbone.layers:
                token_mixer = self._get_token_mixer(layer)
                if hasattr(token_mixer, "q_proj"):
                    self._ensure_forward_global(token_mixer, rope_function_name)
                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        token_mixer,
                        rope_function_name,
                        config=decoder_config,
                        **k_quant_config,
                    )
                    continue

                if self._is_linear_attn_mixer(token_mixer):
                    linear_k_quant_config = dict(k_quant_config)
                    linear_k_quant_config.update(
                        self._get_linear_attn_k_quant_overrides(token_mixer, decoder_config, head_dim)
                    )
                    rotation_utils.add_qk_rotation_wrapper_to_linear_attn(
                        token_mixer,
                        config=decoder_config,
                        **linear_k_quant_config,
                    )

    def _configure_qwen2_5_vl_visual_activation_quantizers(
        self,
        model,
        source_args,
        quant_utils,
    ) -> None:
        visual_activation_bits = self._resolve_qwen2_5_vl_visual_activation_bits(source_args)
        if visual_activation_bits >= 16 or not self._is_qwen2_5_vl(model):
            return

        visual = self._get_qwen2_5_vl_visual_root(model)
        if visual is None:
            return

        act_qlayers = quant_utils.find_qlayers(visual, layers=[quant_utils.ActQuantWrapper])
        for layer_name, qlayer in act_qlayers.items():
            module = getattr(qlayer, "module", None)
            in_features = getattr(module, "in_features", None)
            layer_groupsize = source_args.a_groupsize
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                layer_groupsize = -1
            if (
                in_features is None
                or source_args.a_groupsize <= 0
                or in_features % source_args.a_groupsize != 0
            ):
                layer_groupsize = -1

            qlayer.quantizer.configure(
                bits=visual_activation_bits,
                groupsize=layer_groupsize,
                sym=not source_args.a_asym,
                clip_ratio=source_args.a_clip_ratio,
            )

    @staticmethod
    def _get_qk_wrapper_attr_name(model, model_utils) -> str:
        return f"{model_utils.get_rope_function_name(model)}_qk_rotation_wrapper"

    def _get_qk_wrappers(self, model, model_utils):
        wrapper_attr = self._get_qk_wrapper_attr_name(model, model_utils)
        wrappers = []
        for layer in get_text_backbone(model).layers:
            token_mixer = getattr(layer, "self_attn", None)
            if token_mixer is None:
                token_mixer = getattr(layer, "linear_attn", None)
            if token_mixer is None:
                continue
            wrapper = getattr(token_mixer, wrapper_attr, None)
            if wrapper is not None:
                wrappers.append(wrapper)
                continue
            linear_wrapper = getattr(token_mixer, "linear_attn_qk_rotation_wrapper", None)
            if linear_wrapper is not None:
                wrappers.append(linear_wrapper)
        return wrappers

    def _collect_k_equalize_scales(self, model, tokenizer_bundle, args, model_utils, ref_utils) -> None:
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )

        wrappers = self._get_qk_wrappers(model, model_utils)
        if not wrappers:
            return

        active_wrappers = []
        for wrapper in wrappers:
            if hasattr(wrapper, "start_k_equalize_calibration"):
                wrapper.start_k_equalize_calibration()
                active_wrappers.append(wrapper)
        if not active_wrappers:
            return

        try:
            # device_map 模式下不手动移动模型，输入数据放到模型所在设备
            input_device = next(model.parameters()).device
            with torch.no_grad():
                for input_ids, _labels in calibration_batches:
                    model(input_ids=input_ids.to(input_device), use_cache=False)
        finally:
            # device_map 模式下不手动移动到 cpu
            for wrapper in active_wrappers:
                wrapper.finish_k_equalize_calibration()
            ref_utils.cleanup_memory(verbos=False)

    @staticmethod
    def _neutralize_norm(norm) -> None:
        if hasattr(norm, "weight") and norm.weight is not None:
            fill_value = 0 if norm.__class__.__name__.startswith("Qwen3_5RMSNorm") else 1
            norm.weight.data.fill_(fill_value)
        if hasattr(norm, "bias") and norm.bias is not None:
            norm.bias.data.zero_()

    @staticmethod
    def _disable_hidden_hadamard_requested(model, source_args) -> bool:
        requested = getattr(source_args, "quarot_disable_hidden_hadamard", None)
        if requested is not None:
            return bool(requested)
        if QuaRotMethod._default_disable_hidden_hadamard(model):
            return True
        return (
            getattr(getattr(model, "config", None), "model_type", None) == "qwen2_5_vl"
            and bool(getattr(source_args, "quarot_qwen2_5_vl_disable_hidden_hadamard", False))
        )

    @staticmethod
    def _untie_lm_head_if_shared(model, model_utils) -> bool:
        try:
            model_type = model_utils.get_model_type(model)
            kwargs = {"model": model, "model_type": model_type}
            lm_head = model_utils.get_lm_head(**kwargs)
            if lm_head is None or not hasattr(lm_head, "weight") or lm_head.weight is None:
                return False
            for embedding in model_utils.get_embeddings(**kwargs):
                if not hasattr(embedding, "weight") or embedding.weight is None:
                    continue
                if embedding.weight.data_ptr() == lm_head.weight.data_ptr():
                    lm_head.weight = torch.nn.Parameter(lm_head.weight.detach().clone())
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _patch_qwen2_vl_rotation_ops(rotation_utils, model_utils) -> None:
        from hadamard_utils import apply_exact_had_to_linear

        original_rotate_mlp_output = rotation_utils.rotate_mlp_output
        original_rotate_ov_proj = rotation_utils.rotate_ov_proj

        def rotate_mlp_output(layer, q_matrix, model_type):
            # Align with the MQuant Qwen2-VL default: keep the LLM down_proj in the
            # orthogonal basis only and do not pair it with online Hadamard wrappers.
            if model_type != model_utils.LLAMA_MODEL:
                return original_rotate_mlp_output(layer, q_matrix, model_type)
            weight = layer.mlp.down_proj
            weight_device = weight.weight.data.device
            rotation_dtype = rotation_utils.preferred_rotation_dtype(weight_device)
            dtype = weight.weight.data.dtype
            rotated = weight.weight.data.to(dtype=rotation_dtype)
            weight.weight.data = torch.matmul(q_matrix.T.to(device=weight_device, dtype=rotation_dtype), rotated).to(dtype=dtype)
            if weight.bias is not None:
                bias = weight.bias.data.to(dtype=rotation_dtype)
                weight.bias.data = torch.matmul(q_matrix.T.to(device=weight_device, dtype=rotation_dtype), bias).to(dtype=dtype)

        def rotate_ov_proj(layer, model_type, head_num, head_dim):
            if model_type != model_utils.LLAMA_MODEL:
                return original_rotate_ov_proj(layer, model_type, head_num, head_dim)
            v_proj = layer.self_attn.v_proj
            o_proj = layer.self_attn.o_proj
            apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, device=rotation_utils.utils.DEV)
            apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, device=rotation_utils.utils.DEV)

        rotation_utils.rotate_mlp_output = rotate_mlp_output
        rotation_utils.rotate_ov_proj = rotate_ov_proj

    @staticmethod
    def _patch_no_hidden_hadamard(rotation_utils, model_utils) -> None:
        original_rotate_mlp_output = rotation_utils.rotate_mlp_output
        original_rotate_ov_proj = rotation_utils.rotate_ov_proj

        def rotate_mlp_output(layer, q_matrix, model_type):
            if model_type != model_utils.LLAMA_MODEL:
                return original_rotate_mlp_output(layer, q_matrix, model_type)
            weight = layer.mlp.down_proj
            weight_device = weight.weight.data.device
            rotation_dtype = rotation_utils.preferred_rotation_dtype(weight_device)
            dtype = weight.weight.data.dtype
            rotated = weight.weight.data.to(dtype=rotation_dtype)
            weight.weight.data = torch.matmul(q_matrix.T.to(device=weight_device, dtype=rotation_dtype), rotated).to(dtype=dtype)
            if weight.bias is not None:
                bias = weight.bias.data.to(dtype=rotation_dtype)
                weight.bias.data = torch.matmul(q_matrix.T.to(device=weight_device, dtype=rotation_dtype), bias).to(dtype=dtype)

        def rotate_ov_proj(layer, model_type, head_num, head_dim):
            if model_type != model_utils.LLAMA_MODEL:
                return original_rotate_ov_proj(layer, model_type, head_num, head_dim)
            return None

        rotation_utils.rotate_mlp_output = rotate_mlp_output
        rotation_utils.rotate_ov_proj = rotate_ov_proj

    @staticmethod
    def _patch_qwen2_5_vl_merger_output_centering(rotation_utils) -> None:
        def rotate_extra_modules(modules, q_matrix: torch.Tensor) -> None:
            for module in modules:
                weight_device = module.weight.data.device
                rotation_dtype = rotation_utils.preferred_rotation_dtype(weight_device)
                dtype = module.weight.data.dtype
                q_transposed = q_matrix.T.to(device=weight_device, dtype=rotation_dtype)
                weight = module.weight.data.to(dtype=rotation_dtype)
                centered_weight = weight - weight.mean(dim=0, keepdim=True)
                module.weight.data = torch.matmul(q_transposed, centered_weight).to(dtype=dtype)
                if module.bias is not None:
                    bias = module.bias.data.to(dtype=rotation_dtype)
                    centered_bias = bias - bias.mean()
                    module.bias.data = torch.matmul(q_transposed, centered_bias).to(dtype=dtype)

        rotation_utils.rotate_extra_modules = rotate_extra_modules

    def _apply_rtn_quantization(self, backbone, quant_utils, args, *, w_clip: bool) -> dict[str, object]:
        quantizer_artifacts: dict[str, object] = {}
        int8_down_proj = bool(getattr(args, "int8_down_proj", False))
        for layer_index, block in enumerate(backbone.layers):
            # device_map 模式下不手动移动 block，由 dispatch_model 管理
            # Keep RTN behavior aligned with upstream QuaRot: quantize true Linear modules
            # (including those wrapped by ActQuantWrapper.module).
            linear_layers = quant_utils.find_qlayers(block, layers=[torch.nn.Linear])
            for layer_name, linear_layer in linear_layers.items():
                quantizer = quant_utils.WeightQuantizer()
                layer_weight_bits = args.weight_bits
                if int8_down_proj and "down_proj" in layer_name:
                    layer_weight_bits = 8

                quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=args.weight_symmetric,
                    mse=w_clip,
                    weight_groupsize=args.weight_group_size,
                )
                weights = linear_layer.weight.data
                quantizer.find_params(weights)
                linear_layer.weight.data = quantizer.quantize(weights).to(weights.dtype)

                # Collapse wrapper-internal suffix for artifact readability.
                normalized_name = layer_name[:-7] if layer_name.endswith(".module") else layer_name
                quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{normalized_name}"] = {
                    "bits": layer_weight_bits,
                    "group_size": args.weight_group_size,
                    "symmetric": args.weight_symmetric,
                }
            # device_map 模式下不手动移动到 cpu
        return quantizer_artifacts

    def _apply_qwen2_5_vl_visual_rtn_quantization(
        self,
        model,
        quant_utils,
        args,
        *,
        w_clip: bool,
    ) -> dict[str, object]:
        visual = self._get_qwen2_5_vl_visual_root(model)
        if visual is None:
            return {}

        # device_map 模式下不手动移动 visual
        prefix = self._get_qwen2_5_vl_visual_prefix(model)
        visual_weight_bits = self._resolve_qwen2_5_vl_visual_weight_bits(args)
        quantizer_artifacts: dict[str, object] = {}
        modules = quant_utils.find_qlayers(visual, layers=[torch.nn.Linear, torch.nn.Conv3d])
        for layer_name, module in modules.items():
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                visual_weight_bits,
                perchannel=True,
                sym=args.weight_symmetric,
                mse=w_clip,
            )
            weights = module.weight.data
            quantizer.find_params(weights)
            module.weight.data = quantizer.quantize(weights).to(weights.dtype)
            quantizer_artifacts[f"{prefix}.{layer_name}"] = {
                "bits": visual_weight_bits,
                "group_size": -1,
                "symmetric": args.weight_symmetric,
            }

        # device_map 模式下不手动移动到 cpu
        return quantizer_artifacts

    def _apply_gptq_quantization(
        self,
        model,
        backbone,
        calibration_batches,
        gptq_utils,
        quant_utils,
        args,
        *,
        w_clip: bool,
    ) -> dict[str, object]:
        input_states, layer_kwargs = capture_first_block_inputs(
            model=model,
            backbone=backbone,
            calibration_batches=calibration_batches,
            device=args.device,
        )
        output_states = torch.zeros_like(input_states)
        quantizer_artifacts: dict[str, object] = {}
        int8_down_proj = bool(getattr(args, "int8_down_proj", False))
        sample_count = input_states.shape[0]
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        enable_qwen3_vl_down_proj_rtn_fallback = bool(
            getattr(args, "quarot_qwen3_vl_down_proj_rtn_fallback", False)
        )

        for layer_index, block in enumerate(backbone.layers):
            target_device = get_layer_device(backbone, layer_index)
            input_states = input_states.to(target_device)
            output_states = output_states.to(target_device)
            layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
            # device_map 模式下不手动移动 block，由 dispatch_model 管理
            full_linears = quant_utils.find_qlayers(block, layers=[torch.nn.Linear])
            act_wrappers = quant_utils.find_qlayers(block, layers=[quant_utils.ActQuantWrapper])
            normalized_to_actual = {}
            for layer_name in full_linears:
                normalized_name = layer_name[:-7] if layer_name.endswith(".module") else layer_name
                normalized_to_actual[normalized_name] = layer_name
            sequential = build_decoder_layer_groups(block, set(normalized_to_actual))

            for group in sequential:
                missing = [name for name in group if name not in normalized_to_actual]
                if missing:
                    raise KeyError(
                        f"QuaRot GPTQ sequential group not found in layer {layer_index}: {missing}"
                    )
                subset = {
                    name: full_linears[normalized_to_actual[name]]
                    for name in group
                }
                gptq_subset = {}
                rtn_subset = {}
                for layer_name, linear_module in subset.items():
                    wrapper = act_wrappers.get(layer_name)
                    use_rtn_fallback = (
                        enable_qwen3_vl_down_proj_rtn_fallback
                        and model_type == "qwen3_vl"
                        and "down_proj" in layer_name
                        and wrapper is not None
                        and getattr(wrapper, "online_full_had", False)
                    )
                    if use_rtn_fallback:
                        rtn_subset[layer_name] = linear_module
                    else:
                        gptq_subset[layer_name] = linear_module

                gptq_states = {}
                for layer_name, linear_module in gptq_subset.items():
                    layer_weight_bits = args.weight_bits
                    if int8_down_proj and "down_proj" in layer_name:
                        layer_weight_bits = 8
                    gptq_state = gptq_utils.GPTQ(linear_module)
                    gptq_state.quantizer = quant_utils.WeightQuantizer()
                    gptq_state.quantizer.configure(
                        layer_weight_bits,
                        perchannel=True,
                        sym=args.weight_symmetric,
                        mse=w_clip,
                    )
                    gptq_states[layer_name] = gptq_state

                def add_batch(layer_name: str):
                    def hook(_module, inputs, outputs):
                        gptq_states[layer_name].add_batch(inputs[0].data, outputs.data)

                    return hook

                handles = [
                    gptq_subset[layer_name].register_forward_hook(add_batch(layer_name))
                    for layer_name in gptq_subset
                ]
                if gptq_subset:
                    for sample_index in range(sample_count):
                        with torch.no_grad():
                            hidden_states = unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            )
                            if hidden_states.dim() == 3 and hidden_states.shape[0] == 1:
                                hidden_states = hidden_states[0]
                            output_states[sample_index] = hidden_states
                for handle in handles:
                    handle.remove()

                for layer_name, gptq_state in gptq_states.items():
                    layer_weight_bits = args.weight_bits
                    if int8_down_proj and "down_proj" in layer_name:
                        layer_weight_bits = 8
                    gptq_state.fasterquant(
                        percdamp=args.damp_percent,
                        groupsize=args.weight_group_size,
                        actorder=args.use_activation_order,
                        static_groups=False,
                    )
                    quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{layer_name}"] = {
                        "bits": layer_weight_bits,
                        "group_size": args.weight_group_size,
                        "symmetric": args.weight_symmetric,
                    }
                    gptq_state.free()
                del gptq_states

                for layer_name, linear_module in rtn_subset.items():
                    layer_weight_bits = args.weight_bits
                    if int8_down_proj and "down_proj" in layer_name:
                        layer_weight_bits = 8
                    quantizer = quant_utils.WeightQuantizer()
                    quantizer.configure(
                        layer_weight_bits,
                        perchannel=True,
                        sym=args.weight_symmetric,
                        mse=w_clip,
                        weight_groupsize=args.weight_group_size,
                    )
                    weights = linear_module.weight.data
                    quantizer.find_params(weights)
                    linear_module.weight.data = quantizer.quantize(weights).to(weights.dtype)
                    quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{layer_name}"] = {
                        "bits": layer_weight_bits,
                        "group_size": args.weight_group_size,
                        "symmetric": args.weight_symmetric,
                    }
                empty_cache(args.device)

            for sample_index in range(sample_count):
                with torch.no_grad():
                    hidden_states = unwrap_layer_output(
                        block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                    )
                    if hidden_states.dim() == 3 and hidden_states.shape[0] == 1:
                        hidden_states = hidden_states[0]
                    output_states[sample_index] = hidden_states

            # device_map 模式下不手动移动到 cpu
            input_states, output_states = output_states, input_states

        return quantizer_artifacts

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        source_args = self._build_source_args(args)
        backbone = get_text_backbone(model)
        self._sync_decoder_config_to_model_config(model)
        rotation_guard_reason = self._rotation_guard_reason(model)
        if source_args.rotate and rotation_guard_reason is not None:
            source_args.rotate = False

        with prepend_python_path(source_root):
            import gptq_utils
            import hadamard_utils
            import model_utils
            import quant_utils
            import rotation_utils
            import utils as ref_utils

            ref_utils.DEV = torch.device(args.device)
            if ref_utils.DEV.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            extra_norm_types = self._patch_model_utils(model_utils, model)
            self._bind_rotation_device(rotation_utils, args.device)
            model_type_name = getattr(getattr(model, "config", None), "model_type", None)
            if model_type_name == "qwen2_vl":
                self._patch_qwen2_vl_rotation_ops(rotation_utils, model_utils)
            elif self._disable_hidden_hadamard_requested(model, source_args):
                self._patch_no_hidden_hadamard(rotation_utils, model_utils)
            if model_type_name == "qwen2_5_vl" and getattr(
                source_args, "quarot_qwen2_5_vl_center_merger_output", False
            ):
                self._patch_qwen2_5_vl_merger_output_centering(rotation_utils)

            lm_head_untied = False
            if source_args.rotate:
                lm_head_untied = self._untie_lm_head_if_shared(model, model_utils)
                self._fuse_layer_norms(
                    model,
                    model_utils,
                    rotation_utils,
                    extra_norm_types,
                    fuse_qwen2_5_vl_visual_branch=bool(
                        getattr(source_args, "quarot_qwen2_5_vl_rotate_visual_branch", True)
                    ),
                )
                rotation_utils.rotate_model(model, source_args)
                ref_utils.cleanup_memory(verbos=False)

            quant_utils.add_actquant(self._quantization_root(model))
            if self._is_qwen2_5_vl(model):
                visual_root = self._get_qwen2_5_vl_visual_root(model)
                if visual_root is not None:
                    quant_utils.add_actquant(visual_root)
            self._configure_activation_quantizers(
                model,
                source_args,
                quant_utils,
                hadamard_utils,
                rotation_utils,
                model_utils,
                ref_utils,
                configure_hadamard=True,
                configure_input_quant=False,
                configure_k_quant=False,
            )

            quantizer_artifacts: dict[str, object] = {}
            weight_quantizer_name = "none"
            visual_weight_quantizer_name = "none"
            if args.weight_bits < 16:
                if args.weight_method == "rtn":
                    quantizer_artifacts = self._apply_rtn_quantization(
                        backbone,
                        quant_utils,
                        args,
                        w_clip=source_args.w_clip,
                    )
                    weight_quantizer_name = "rtn"
                else:
                    calibration_batches, _ = get_calibration_and_evaluation_data(
                        tokenizer=tokenizer_bundle.tokenizer,
                        dataset_name=args.calibration_dataset,
                        sequence_length=args.sequence_length,
                        sample_count=args.calibration_samples,
                        seed=args.seed,
                        data_path=args.data_path,
                    )
                    quantizer_artifacts = self._apply_gptq_quantization(
                        model,
                        backbone,
                        calibration_batches,
                        gptq_utils,
                        quant_utils,
                        args,
                        w_clip=source_args.w_clip,
                        )
                    weight_quantizer_name = "gptq"

                if self._is_qwen2_5_vl(model):
                    visual_quantizer_artifacts = self._apply_qwen2_5_vl_visual_rtn_quantization(
                        model,
                        quant_utils,
                        args,
                        w_clip=source_args.w_clip,
                    )
                    quantizer_artifacts.update(visual_quantizer_artifacts)
                    visual_weight_quantizer_name = "rtn"
                    if args.weight_method != "rtn":
                        visual_weight_quantizer_name = "rtn_fallback"

            # Keep QuaRot ordering stable: rotate/Hadamard setup first, then weight quantization,
            # finally enable input & key activation quantization wrappers for evaluation.
            self._configure_activation_quantizers(
                model,
                source_args,
                quant_utils,
                hadamard_utils,
                rotation_utils,
                model_utils,
                ref_utils,
                configure_hadamard=False,
                configure_input_quant=True,
                configure_k_quant=True,
            )
            if self._is_qwen2_5_vl(model):
                self._configure_qwen2_5_vl_visual_activation_quantizers(
                    model,
                    source_args,
                    quant_utils,
                )
            if source_args.k_bits < 16 and getattr(source_args, "k_equalize", False):
                self._collect_k_equalize_scales(model, tokenizer_bundle, args, model_utils, ref_utils)

        return {
            "source_root": str(source_root),
            "quarot_config": {
                "rotate": source_args.rotate,
                "rotation_mode": source_args.rotate_mode,
                "rotation_checkpoint": source_args.optimized_rotation_path,
                "weight_bits": source_args.w_bits,
                "activation_bits": source_args.a_bits,
                "key_bits": source_args.k_bits,
                "value_bits": source_args.v_bits,
                "weight_group_size": source_args.w_groupsize,
                "activation_group_size": source_args.a_groupsize,
                "kv_group_size": source_args.k_groupsize,
                "weight_clip": source_args.w_clip,
                "activation_clip_ratio": source_args.a_clip_ratio,
                "key_clip_ratio": source_args.k_clip_ratio,
                "value_clip_ratio": source_args.v_clip_ratio,
                "fp32_had": source_args.fp32_had,
                "quarot_disable_hidden_hadamard": self._disable_hidden_hadamard_requested(
                    model, source_args
                ),
                "quarot_disable_hidden_hadamard_requested": getattr(
                    source_args, "quarot_disable_hidden_hadamard", None
                ),
                "k_pre_rope": source_args.k_pre_rope,
                "k_tokenwise_per_head": source_args.k_tokenwise_per_head,
                "k_hadamard": source_args.k_hadamard,
                "k_per_head_channel": source_args.k_per_head_channel,
                "k_equalize": source_args.k_equalize,
                "k_equalize_alpha": source_args.k_equalize_alpha,
                "k_equalize_max_scale": source_args.k_equalize_max_scale,
                "k_equalize_with_q": source_args.k_equalize_with_q,
                "k_equalize_q_power": source_args.k_equalize_q_power,
                "qwen2_5_vl_disable_hidden_hadamard": getattr(
                    source_args, "quarot_qwen2_5_vl_disable_hidden_hadamard", False
                ),
                "qwen2_5_vl_center_merger_output": getattr(
                    source_args, "quarot_qwen2_5_vl_center_merger_output", False
                ),
                "qwen2_5_vl_first_layer_activation_bits": getattr(
                    source_args, "quarot_qwen2_5_vl_first_layer_activation_bits", None
                ),
                "qwen3_vl_down_proj_rtn_fallback": bool(
                    getattr(args, "quarot_qwen3_vl_down_proj_rtn_fallback", False)
                ),
                "weight_quantizer": weight_quantizer_name,
                "visual_weight_quantizer": visual_weight_quantizer_name,
                "visual_weight_bits": self._resolve_qwen2_5_vl_visual_weight_bits(args),
                "visual_activation_bits": self._resolve_qwen2_5_vl_visual_activation_bits(args),
                "calibration_samples": source_args.nsamples,
                "rotation_guard_reason": rotation_guard_reason,
                "lm_head_untied_before_rotation": lm_head_untied,
            },
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
