from __future__ import annotations

from typing import Any

import torch


def get_decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            return language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer
    raise NotImplementedError(f"Unsupported decoder backbone: {type(model)}")


def get_decoder_layers(model):
    root = get_decoder_root(model)
    if hasattr(root, "layers"):
        return root.layers
    if hasattr(root, "h"):
        return root.h
    raise AttributeError(f"Unsupported decoder root: {type(root)}")


def move_front_modules(model, device):
    root = get_decoder_root(model)
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
        if hasattr(root, attr_name):
            getattr(root, attr_name).to(device)


def move_back_modules(model, device):
    root = get_decoder_root(model)
    for attr_name in ("norm", "final_layer_norm", "ln_f"):
        if hasattr(root, attr_name):
            getattr(root, attr_name).to(device)
            break

    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            head.to(device)
            return
    for attr_name in ("lm_head", "embed_out"):
        if hasattr(model, attr_name):
            getattr(model, attr_name).to(device)
            return


def unwrap_layer_output(layer_output):
    if torch.is_tensor(layer_output):
        return layer_output
    if isinstance(layer_output, (tuple, list)):
        return layer_output[0]
    if hasattr(layer_output, "last_hidden_state"):
        return layer_output.last_hidden_state
    raise TypeError(f"Unsupported layer output type: {type(layer_output)}")


def _repeat_singleton_batch_dim(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size == 1 or value.ndim == 0:
        return value
    if value.shape[0] == 1:
        return value.repeat(batch_size, *([1] * (value.ndim - 1)))
    if value.ndim > 1 and value.shape[1] == 1 and value.shape[0] in (2, 3):
        repeats = [1] * value.ndim
        repeats[1] = batch_size
        return value.repeat(*repeats)
    return value


def build_batched_layer_kwargs(layer_kwargs: dict[str, Any], batch_size: int) -> dict[str, Any]:
    repeated_kwargs: dict[str, Any] = {}
    for name, value in layer_kwargs.items():
        if torch.is_tensor(value):
            repeated_value = _repeat_singleton_batch_dim(value, batch_size)
            repeated_kwargs[name] = repeated_value
            continue
        if isinstance(value, tuple):
            repeated_kwargs[name] = tuple(
                _repeat_singleton_batch_dim(item, batch_size) if torch.is_tensor(item) else item
                for item in value
            )
            continue
        repeated_kwargs[name] = value
    return repeated_kwargs
