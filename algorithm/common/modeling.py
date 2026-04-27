"""Shared modeling helpers for decoder-only text backbones."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from types import MethodType
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - older/newer transformers variants
    AutoModelForImageTextToText = None
try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - older/newer transformers variants
    AutoModelForVision2Seq = None
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
from transformers.dynamic_module_utils import get_class_from_dynamic_module
try:
    from transformers.cache_utils import Cache
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - older/newer transformers variants
    Cache = None
    DynamicCache = None

from .device import empty_cache
from .device import resolve_device


def _load_vision_text_model(model_path: str, **model_kwargs):
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
    raise ImportError("Neither AutoModelForImageTextToText nor AutoModelForVision2Seq is available in this transformers build.")


def _load_config_with_fallback(model_path: str, trust_remote_code: bool = True):
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        # Qwen3.5-VL checkpoints currently expose `model_type=qwen3_5`, while some
        # transformers builds only register the equivalent Qwen3-VL config classes.
        if "model type `qwen3_5`" not in str(exc):
            raise
        from transformers import Qwen3VLConfig

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)

        config_dict["model_type"] = "qwen3_vl"
        text_cfg = config_dict.get("text_config")
        if isinstance(text_cfg, dict):
            text_cfg["model_type"] = "qwen3_vl_text"
            if text_cfg.get("rope_scaling") is None and isinstance(text_cfg.get("rope_parameters"), dict):
                text_cfg["rope_scaling"] = dict(text_cfg["rope_parameters"])
        vision_cfg = config_dict.get("vision_config")
        if isinstance(vision_cfg, dict):
            vision_cfg["model_type"] = "qwen3_vl"
        architectures = config_dict.get("architectures")
        if isinstance(architectures, list):
            config_dict["architectures"] = [
                "Qwen3VLForConditionalGeneration" if name == "Qwen3_5ForConditionalGeneration" else name
                for name in architectures
            ]
        return Qwen3VLConfig.from_dict(config_dict)


@dataclass
class TokenizerBundle:
    tokenizer: Any
    processor: Any | None = None

    def save_pretrained(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)
        if self.processor is not None:
            self.processor.save_pretrained(path)

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)


class MiniCPMTokenizerAdapter:
    """Expose MiniCPM-specific tokenizer fields on top of a standard HF tokenizer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"

    @property
    def bos_id(self) -> int:
        return int(self.tokenizer.bos_token_id)

    @property
    def eos_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def unk_id(self) -> int:
        return int(self.tokenizer.unk_token_id)

    @property
    def im_start_id(self) -> int:
        return int(self.tokenizer.convert_tokens_to_ids(self.im_start))

    @property
    def im_end_id(self) -> int:
        return int(self.tokenizer.convert_tokens_to_ids(self.im_end))

    def save_pretrained(self, path: str, *args, **kwargs) -> None:
        self.tokenizer.save_pretrained(path, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)


class TextModelAdapter(nn.Module):
    """Expose a multimodal model's text decoder through a causal-LM interface."""

    def __init__(self, text_model: nn.Module, source_model: nn.Module | None = None):
        super().__init__()
        self.text_model = text_model
        object.__setattr__(self, "_source_model", source_model if source_model is not None else text_model)

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)

    def save_pretrained(self, path: str, *args, **kwargs) -> None:
        self._source_model.save_pretrained(path, *args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.text_model, name)


@dataclass
class TextBackbone:
    model: nn.Module
    root: nn.Module
    prefix: str

    @property
    def layers(self):
        if hasattr(self.root, "layers"):
            return self.root.layers
        if hasattr(self.root, "h"):
            return self.root.h
        raise AttributeError(f"Unsupported backbone root: {type(self.root)}")

    @property
    def decoder_config(self):
        config_candidates = [
            getattr(self.root, "config", None),
            getattr(self.model, "config", None),
        ]
        for config in config_candidates:
            if config is None:
                continue
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                return text_config
            if hasattr(config, "hidden_size"):
                return config
        raise AttributeError(f"Unsupported decoder config for backbone root: {type(self.root)}")

    @property
    def hidden_size(self) -> int:
        return int(self.decoder_config.hidden_size)

    @property
    def embed_tokens(self) -> nn.Module | None:
        for attr_name in ("embed_tokens", "tok_embeddings", "wte", "word_embeddings", "embed_in"):
            if hasattr(self.root, attr_name):
                return getattr(self.root, attr_name)
        return None

    @property
    def final_norm(self) -> nn.Module | None:
        for attr_name in ("norm", "final_layer_norm", "ln_f"):
            if hasattr(self.root, attr_name):
                return getattr(self.root, attr_name)
        return None

    def move_front_modules(self, device: str | torch.device) -> None:
        for attr_name in (
            "embed_tokens",
            "tok_embeddings",
            "embed_positions",
            "rotary_emb",
            "rotary_pos_emb",
            "word_embeddings",
            "word_embeddings_layernorm",
            "emb_dropout",
            "wte",
            "wpe",
        ):
            if hasattr(self.root, attr_name):
                getattr(self.root, attr_name).to(device)

    def move_back_modules(self, device: str | torch.device) -> None:
        if self.final_norm is not None:
            self.final_norm.to(device)
        head = get_output_head(self.model)
        if head is not None:
            head.to(device)


def resolve_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return "auto"


def _ensure_transformers_remote_code_compat() -> None:
    """Patch small API removals that older HF remote code may still import."""
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")


def _normalize_legacy_remote_config(config) -> None:
    model_type = getattr(config, "model_type", None)
    if model_type not in {"minicpm", "minicpmv"}:
        return

    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return
    if "type" in rope_scaling and "factor" in rope_scaling:
        return

    rope_type = rope_scaling.get("rope_type")
    factor = rope_scaling.get("factor")
    if rope_type in (None, "default"):
        config.rope_scaling = None
        return
    if rope_type in {"linear", "dynamic"} and factor is not None:
        config.rope_scaling = {"type": rope_type, "factor": float(factor)}
        return
    config.rope_scaling = None


def _patch_minicpmv_remote_class(model_path: str) -> None:
    _ensure_transformers_remote_code_compat()
    minicpmv_cls = get_class_from_dynamic_module("modeling_minicpmv.MiniCPMV", model_path)
    if (
        getattr(minicpmv_cls, "_mindpipe_post_init_patched", False)
        and getattr(minicpmv_cls, "_mindpipe_chat_patched", False)
    ):
        return

    original_init = minicpmv_cls.__init__
    original_chat = getattr(minicpmv_cls, "chat", None)

    def patched_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    def patched_chat(
        self,
        image,
        msgs,
        context,
        tokenizer,
        vision_hidden_states=None,
        max_new_tokens=2048,
        sampling=False,
        **kwargs,
    ):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        # Keep the original remote-code implementation for non-string prompts.
        if original_chat is not None and any(not isinstance(msg.get("content"), str) for msg in msgs):
            return original_chat(
                self,
                image=image,
                msgs=msgs,
                context=context,
                tokenizer=tokenizer,
                vision_hidden_states=vision_hidden_states,
                max_new_tokens=max_new_tokens,
                sampling=sampling,
                **kwargs,
            )

        prompt = ""
        for index, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported MiniCPM chat role: {role}")
            if index == 0:
                if role != "user":
                    raise ValueError("The role of first MiniCPM message should be user")
                content = tokenizer.im_start + tokenizer.unk_token * self.config.query_num + tokenizer.im_end + "\n" + content
            prompt += "<用户>" if role == "user" else "<AI>"
            prompt += content
        prompt += "<AI>"

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.6,
                "do_sample": True,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        # The upstream implementation only forwards a hard-coded subset of kwargs.
        # Preserve the defaults above, but allow callers to pass any valid generate()
        # argument such as `use_cache=False`.
        generation_config.update(kwargs)

        with torch.inference_mode():
            responses, vision_hidden_states = self.generate(
                data_list=[prompt],
                max_inp_length=2048,
                img_list=[[image]],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config,
            )
        answer = responses[0]
        next_context = list(msgs)
        next_context.append({"role": "assistant", "content": answer})
        return answer, next_context, generation_config

    if not getattr(minicpmv_cls, "_mindpipe_post_init_patched", False):
        minicpmv_cls.__init__ = patched_init
    if original_chat is not None and not getattr(minicpmv_cls, "_mindpipe_chat_patched", False):
        minicpmv_cls.chat = patched_chat
    minicpmv_cls._mindpipe_post_init_patched = True
    minicpmv_cls._mindpipe_chat_patched = True


def _prepare_minicpm_tokenizer_env() -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _refresh_minicpm_rotary_embeddings(model: nn.Module) -> None:
    backbone = get_text_backbone(model)
    for layer_idx, layer in enumerate(backbone.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "_init_rope"):
            continue

        q_proj = getattr(attn, "q_proj", None)
        q_weight = getattr(q_proj, "weight", None)
        target_device = q_weight.device if q_weight is not None else None

        attn._init_rope()
        rotary_emb = getattr(attn, "rotary_emb", None)
        if rotary_emb is None:
            raise RuntimeError(f"MiniCPM attention layer {layer_idx} did not rebuild rotary embeddings.")
        if target_device is not None:
            rotary_emb.to(device=target_device)

        for buffer_name in ("inv_freq", "cos_cached", "sin_cached"):
            buffer_value = getattr(rotary_emb, buffer_name, None)
            if buffer_value is None or not torch.is_tensor(buffer_value):
                raise RuntimeError(f"MiniCPM rotary buffer `{buffer_name}` is missing at layer {layer_idx}.")
            if not torch.isfinite(buffer_value).all():
                raise RuntimeError(f"MiniCPM rotary buffer `{buffer_name}` is non-finite at layer {layer_idx}.")


def ensure_generation_compat(model: nn.Module) -> nn.Module:
    """Restore `.generate()` for remote-code models that no longer inherit GenerationMixin."""

    _ensure_dynamic_cache_compat()
    if hasattr(model, "generate"):
        return model
    if not hasattr(model, "prepare_inputs_for_generation"):
        return model

    patched_class = type(
        f"Patched{model.__class__.__name__}",
        (model.__class__, GenerationMixin),
        {},
    )
    model.__class__ = patched_class
    if getattr(model, "generation_config", None) is None and hasattr(model, "config"):
        model.generation_config = GenerationConfig.from_model_config(model.config)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    if hasattr(model, "config"):
        model.config.use_cache = False
    if "MiniCPMForCausalLM" in model.__class__.__name__:
        model.prepare_inputs_for_generation = MethodType(
            _patched_minicpm_prepare_inputs_for_generation,
            model,
        )
    return model


def _ensure_dynamic_cache_compat() -> None:
    if DynamicCache is None:
        return

    def _dynamic_cache_seen_tokens(self):
        return self.get_seq_length(0)

    def _dynamic_cache_get_max_length(self, *args, **kwargs):
        return None

    def _dynamic_cache_get_usable_length(self, new_seq_length=0, layer_idx=0):
        return self.get_seq_length(layer_idx)

    @classmethod
    def _dynamic_cache_from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is None:
            return cache
        if Cache is not None and isinstance(past_key_values, Cache):
            return past_key_values
        if (
            isinstance(past_key_values, (list, tuple))
            and len(past_key_values) == 2
            and all(torch.is_tensor(item) or item is None for item in past_key_values)
        ):
            past_key_values = (past_key_values,)
        for layer_idx, layer_past in enumerate(past_key_values):
            if layer_past is None:
                continue
            if torch.is_tensor(layer_past):
                raise ValueError(
                    "DynamicCache.from_legacy_cache expected per-layer cache tuples, "
                    f"but received a tensor at layer {layer_idx} with shape {tuple(layer_past.shape)}."
                )
            if not isinstance(layer_past, (list, tuple)) or len(layer_past) < 2:
                raise ValueError(
                    "DynamicCache.from_legacy_cache expected per-layer cache tuples of "
                    f"(key, value), but received {type(layer_past).__name__} at layer {layer_idx}."
                )
            key_states, value_states = layer_past[:2]
            cache.update(key_states, value_states, layer_idx)
        return cache

    def _dynamic_cache_to_legacy_cache(self):
        legacy_cache = []
        for layer in getattr(self, "layers", []):
            key_states = getattr(layer, "keys", None)
            value_states = getattr(layer, "values", None)
            sliding_window = getattr(layer, "_sliding_window_tensor", None)
            if key_states is None or value_states is None:
                legacy_cache.append(None)
                continue
            if sliding_window is not None:
                legacy_cache.append((key_states, value_states, sliding_window))
            else:
                legacy_cache.append((key_states, value_states))
        return tuple(legacy_cache)

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(_dynamic_cache_seen_tokens)
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = _dynamic_cache_get_max_length
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = _dynamic_cache_get_usable_length
    if not hasattr(DynamicCache, "from_legacy_cache"):
        DynamicCache.from_legacy_cache = _dynamic_cache_from_legacy_cache
    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = _dynamic_cache_to_legacy_cache


def _patched_minicpm_prepare_inputs_for_generation(
    self,
    input_ids,
    next_sequence_length=None,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    is_first_iteration: bool | None = False,
    **kwargs,
):
    return GenerationMixin.prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=next_sequence_length,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        is_first_iteration=is_first_iteration,
        **kwargs,
    )


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "auto",
    attn_implementation: str | None = None,
    device_map: str | dict | None = None,
):
    _ensure_transformers_remote_code_compat()
    config = _load_config_with_fallback(model_path, trust_remote_code=True)
    _normalize_legacy_remote_config(config)
    if attn_implementation is None:
        raise ValueError("attn_implementation must be specified explicitly when loading a model.")
    architectures = set(getattr(config, "architectures", []) or [])
    model_kwargs = {
        "torch_dtype": resolve_dtype(dtype),
        "config": config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    is_qwen3_5 = config.model_type == "qwen3_5" or "Qwen3_5ForConditionalGeneration" in architectures
    is_qwen3_vl = config.model_type == "qwen3_vl" or "Qwen3VLForConditionalGeneration" in architectures
    is_qwen2_5_vl = config.model_type == "qwen2_5_vl" or "Qwen2_5_VLForConditionalGeneration" in architectures
    is_qwen2_vl = config.model_type == "qwen2_vl" or "Qwen2VLForConditionalGeneration" in architectures
    is_llava = config.model_type == "llava" or "LlavaForConditionalGeneration" in architectures

    if is_qwen3_5:
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        except Exception:
            model = _load_vision_text_model(model_path, **model_kwargs)
        if hasattr(model, "language_model"):
            ensure_generation_compat(model.language_model)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif is_qwen3_vl:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        except Exception:
            model = _load_vision_text_model(model_path, **model_kwargs)
        if hasattr(model, "language_model"):
            ensure_generation_compat(model.language_model)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif is_qwen2_5_vl:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif is_qwen2_vl:
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif config.model_type == "internvl_chat" or "InternVLChatModel" in architectures:
        multimodal_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not hasattr(multimodal_model, "language_model"):
            raise AttributeError(f"InternVL model from {model_path} does not expose a `language_model` decoder.")
        ensure_generation_compat(multimodal_model.language_model)
        model = TextModelAdapter(text_model=multimodal_model.language_model, source_model=multimodal_model)
        processor = None
    elif is_llava:
        model = _load_vision_text_model(model_path, **model_kwargs)
        if hasattr(model, "language_model"):
            ensure_generation_compat(model.language_model)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif config.model_type == "minicpmv" or "MiniCPMV" in architectures:
        _patch_minicpmv_remote_class(model_path)
        multimodal_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not hasattr(multimodal_model, "llm"):
            raise AttributeError(f"MiniCPM-V model from {model_path} does not expose an `llm` decoder.")
        ensure_generation_compat(multimodal_model.llm)
        model = TextModelAdapter(text_model=multimodal_model.llm, source_model=multimodal_model)
        processor = None
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        processor = None

    tokenizer = getattr(processor, "tokenizer", None)
    if config.model_type in {"minicpm", "minicpmv"} or "MiniCPMV" in architectures:
        _refresh_minicpm_rotary_embeddings(model)
        _prepare_minicpm_tokenizer_env()
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
            )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        backbone = get_text_backbone(model)
        max_position_embeddings = getattr(backbone.decoder_config, "max_position_embeddings", 2048)
        model.seqlen = min(int(max_position_embeddings), 2048)
    except Exception:
        max_position_embeddings = getattr(model.config, "max_position_embeddings", 2048)
        model.seqlen = min(int(max_position_embeddings), 2048)
    model.eval()
    return model, TokenizerBundle(tokenizer=tokenizer, processor=processor)


def get_text_backbone(model: nn.Module) -> TextBackbone:
    if isinstance(model, TextModelAdapter):
        if hasattr(model.text_model, "model") and hasattr(model.text_model.model, "layers"):
            return TextBackbone(model=model, root=model.text_model.model, prefix="text_model.model")
        if hasattr(model.text_model, "layers"):
            return TextBackbone(model=model, root=model.text_model, prefix="text_model")
    if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "layers"):
        return TextBackbone(model=model, root=model.llm.model, prefix="llm.model")
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            # Qwen2.5-VL 等模型的 language_model 直接有 .layers
            return TextBackbone(model=model, root=language_model, prefix="model.language_model")
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            # LLaVA 等模型的 language_model 是 CausalLM 包装，layers 在其 .model 内
            return TextBackbone(model=model, root=language_model.model, prefix="model.language_model.model")
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return TextBackbone(model=model, root=model.language_model, prefix="language_model")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return TextBackbone(model=model, root=model.model, prefix="model")
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return TextBackbone(model=model, root=model.model.decoder, prefix="model.decoder")
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return TextBackbone(model=model, root=model.transformer, prefix="transformer")
    raise NotImplementedError(f"Unsupported model backbone: {type(model)}")


def unwrap_layer_output(layer_output):
    if torch.is_tensor(layer_output):
        return layer_output
    if isinstance(layer_output, (tuple, list)):
        return layer_output[0]
    if hasattr(layer_output, "last_hidden_state"):
        return layer_output.last_hidden_state
    raise TypeError(f"Unsupported layer output type: {type(layer_output)}")


def get_output_head(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(model, "llm") and hasattr(model.llm, "lm_head"):
        return model.llm.lm_head
    for attr_name in ("lm_head", "embed_out"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return None


def find_linear_layers(module: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    result: dict[str, nn.Linear] = {}
    for child_name, child in module.named_children():
        qualified_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            result[qualified_name] = child
            continue
        result.update(find_linear_layers(child, qualified_name))
    return result


def build_decoder_layer_groups(layer: nn.Module, available_names: set[str]) -> list[list[str]]:
    if hasattr(layer, "self_attn") and hasattr(layer, "mlp"):
        groups = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
        compact_groups = []
        for group in groups:
            present = [name for name in group if name in available_names]
            if present:
                compact_groups.append(present)
        if compact_groups:
            return compact_groups
    if hasattr(layer, "linear_attn") and hasattr(layer, "mlp"):
        groups = [
            [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
            ],
            ["linear_attn.out_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
        compact_groups = []
        for group in groups:
            present = [name for name in group if name in available_names]
            if present:
                compact_groups.append(present)
        if compact_groups:
            return compact_groups
    return [sorted(available_names)]


# ── 混合注意力层抽象（Qwen3.5 等混合架构） ──


def get_attn_output_proj(layer):
    """获取注意力层的输出投影。

    兼容标准 self_attn.o_proj 和 Qwen3.5 linear_attn.out_proj。
    """
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn.o_proj
    if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
        return layer.linear_attn.out_proj
    return None


def supports_head_pruning(layer) -> bool:
    """判断该层是否支持 attention head 结构化剪枝。

    linear_attention 层（如 Qwen3.5 的 GatedDeltaNet）没有标准的 head 概念，
    不适合做 head 级别的结构化剪枝，应仅剪 MLP。
    """
    return hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj")


def get_head_geometry(layer):
    """获取 attention head 的几何信息。

    返回 (num_heads, num_kv_heads, num_kv_groups, head_dim)，
    linear_attention 层返回 None。
    """
    if not supports_head_pruning(layer):
        return None
    attn = layer.self_attn
    config = getattr(attn, "config", None)
    num_heads = int(getattr(attn, "num_heads", getattr(config, "num_attention_heads")))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads", num_heads)))
    hidden_size = int(getattr(attn, "hidden_size", getattr(config, "hidden_size")))
    head_dim = int(getattr(attn, "head_dim", getattr(config, "head_dim", hidden_size // num_heads)))
    num_kv_groups = num_heads // num_kv_heads
    return num_heads, num_kv_heads, num_kv_groups, head_dim


def get_q_stride(layer) -> int | None:
    """获取 q_proj 中每个 head 占的行数。

    标准 attention 是 head_dim；
    Qwen3.5 full-attn 是 head_dim * 2（query+gate 绑定）。
    linear_attention 层返回 None。
    """
    geo = get_head_geometry(layer)
    if geo is None:
        return None
    num_heads, _, _, head_dim = geo
    q_proj = layer.self_attn.q_proj
    expected_q_size = num_heads * head_dim
    actual_q_size = q_proj.out_features
    return head_dim * 2 if actual_q_size == expected_q_size * 2 else head_dim


def get_attn_projections(layer):
    """获取注意力层的所有投影层 (q_proj, k_proj, v_proj, o_proj)。

    linear_attention 层返回 None。
    """
    if not supports_head_pruning(layer):
        return None
    attn = layer.self_attn
    return attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj


def get_mlp_projections(layer):
    """获取 MLP 的三个投影层 (up_proj, gate_proj, down_proj)。"""
    return layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj


# ── 多卡并行设备管理 helpers ──


def move_tensors_to_device(data, device):
    """将字典中的张量递归移动到指定设备。"""
    if isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        moved = [move_tensors_to_device(v, device) for v in data]
        return type(data)(moved)
    if torch.is_tensor(data):
        return data.to(device)
    return data


def get_layer_device(backbone: TextBackbone, layer_index: int) -> torch.device:
    """获取指定层所在的设备。device_map 分片时直接从层参数获取。"""
    return next(backbone.layers[layer_index].parameters()).device


@torch.no_grad()
def capture_first_block_inputs(
    model: nn.Module,
    backbone: TextBackbone,
    calibration_batches,
    device: str | torch.device,
):
    decoder_config = backbone.decoder_config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False
    blocks = backbone.layers

    # device_map 模式下，直接从模型参数获取 embedding 所在设备
    capture_device = next(model.parameters()).device

    dtype = next(iter(model.parameters())).dtype
    sample_count = len(calibration_batches)
    sequence_length = calibration_batches[0][0].shape[1]
    inputs = torch.zeros(
        sample_count,
        sequence_length,
        backbone.hidden_size,
        dtype=dtype,
        device=capture_device,
    )
    cached_kwargs: dict[str, Any] = {}
    input_index = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, hidden_states, **kwargs):
            nonlocal input_index
            inputs[input_index] = hidden_states
            input_index += 1
            cached_kwargs.clear()
            cached_kwargs.update(kwargs)
            raise ValueError

    blocks[0] = Catcher(blocks[0])
    for token_ids, _labels in calibration_batches:
        try:
            with torch.no_grad():
                model(input_ids=token_ids.to(capture_device), use_cache=False)
        except ValueError:
            pass

    blocks[0] = blocks[0].module
    decoder_config.use_cache = use_cache
    return inputs, dict(cached_kwargs)
