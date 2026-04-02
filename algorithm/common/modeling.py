"""Shared modeling helpers for decoder-only text backbones."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
try:
    from transformers.cache_utils import Cache
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - older/newer transformers variants
    Cache = None
    DynamicCache = None

from .device import empty_cache
from .device import resolve_device


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
    def hidden_size(self) -> int:
        return int(getattr(self.root.config, "hidden_size", self.model.config.hidden_size))

    @property
    def embed_tokens(self) -> nn.Module | None:
        for attr_name in ("embed_tokens", "wte", "word_embeddings", "embed_in"):
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
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self, *args, **kwargs: None
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *args, **kwargs: self.get_seq_length()


def _patched_minicpm_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    cache_length = 0
    if past_key_values is not None:
        if Cache is not None and isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = getattr(past_key_values, "seen_tokens", cache_length)
            get_max_length = getattr(past_key_values, "get_max_length", None)
            max_cache_length = get_max_length() if callable(get_max_length) else None
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if inputs_embeds is not None and (past_key_values is None or cache_length == 0):
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "auto",
    force_eager: bool = False,
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if force_eager:
        if hasattr(config, "_attn_implementation_internal"):
            config._attn_implementation_internal = "eager"
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    model_kwargs = {
        "torch_dtype": resolve_dtype(dtype),
        "config": config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if force_eager:
        model_kwargs["attn_implementation"] = "eager"
    architectures = set(getattr(config, "architectures", []) or [])
    if config.model_type == "qwen2_5_vl" or "Qwen2_5_VLForConditionalGeneration" in architectures:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif config.model_type == "minicpmv" or "MiniCPMV" in architectures:
        multimodal_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not hasattr(multimodal_model, "llm"):
            raise AttributeError(f"MiniCPM-V model from {model_path} does not expose an `llm` decoder.")
        ensure_generation_compat(multimodal_model.llm)
        model = TextModelAdapter(text_model=multimodal_model.llm, source_model=multimodal_model)
        processor = None
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        processor = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        backbone = get_text_backbone(model)
        max_position_embeddings = getattr(backbone.root.config, "max_position_embeddings", 2048)
        model.seqlen = min(int(max_position_embeddings), 2048)
    except Exception:
        max_position_embeddings = getattr(model.config, "max_position_embeddings", 2048)
        model.seqlen = min(int(max_position_embeddings), 2048)
    model.eval()
    return model, TokenizerBundle(tokenizer=tokenizer, processor=processor)


def get_text_backbone(model: nn.Module) -> TextBackbone:
    if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "layers"):
        return TextBackbone(model=model, root=model.llm.model, prefix="llm.model")
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            return TextBackbone(model=model, root=language_model, prefix="model.language_model")
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
    return [sorted(available_names)]


@torch.no_grad()
def capture_first_block_inputs(
    model: nn.Module,
    backbone: TextBackbone,
    calibration_batches,
    device: str | torch.device,
):
    device = resolve_device(device)
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    blocks = backbone.layers
    backbone.move_front_modules(device)
    blocks[0] = blocks[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    sample_count = len(calibration_batches)
    sequence_length = calibration_batches[0][0].shape[1]
    inputs = torch.zeros(
        sample_count,
        sequence_length,
        backbone.hidden_size,
        dtype=dtype,
        device=device,
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
                model(input_ids=token_ids.to(device), use_cache=False)
        except ValueError:
            pass

    blocks[0] = blocks[0].module
    blocks[0] = blocks[0].cpu()
    backbone.move_front_modules("cpu")
    empty_cache(device)
    model.config.use_cache = use_cache
    return inputs, dict(cached_kwargs)
