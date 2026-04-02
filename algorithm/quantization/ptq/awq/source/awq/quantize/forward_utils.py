from __future__ import annotations

import inspect

import torch


def _get_decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported decoder backbone: {type(model)}")


def _module_device(module):
    if module is None:
        return None
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return None


def _infer_batch_size(hidden_states, layer_kwargs):
    for key in ("attention_mask", "position_ids"):
        value = layer_kwargs.get(key)
        if not torch.is_tensor(value):
            continue
        if key == "position_ids" and value.ndim >= 3 and value.shape[0] in (3, 4):
            return int(value.shape[1])
        if value.ndim >= 1:
            return int(value.shape[0])
    return int(hidden_states.shape[0])


def _slice_tensor_for_batch(value, batch_start, batch_end, batch_size):
    if not torch.is_tensor(value):
        return value
    if value.ndim >= 1 and value.shape[0] == batch_size:
        return value[batch_start:batch_end]
    if value.ndim >= 2 and value.shape[1] == batch_size:
        return value[:, batch_start:batch_end]
    return value


def _build_qwen2_5_vl_position_ids(hidden_states, position_ids, cache_position):
    if position_ids is not None:
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            return position_ids[1:]
        if position_ids.ndim == 3:
            return position_ids
        if position_ids.ndim == 2:
            return position_ids.unsqueeze(0).expand(3, position_ids.shape[0], -1)
    if cache_position is None:
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    if cache_position.ndim == 1:
        return cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
    if cache_position.ndim == 2:
        return cache_position.unsqueeze(0).expand(3, -1, -1)
    raise ValueError(f"Unsupported cache_position shape for Qwen2.5-VL: {tuple(cache_position.shape)}")


def _filter_kwargs_for_layer(layer, kwargs):
    signature = inspect.signature(layer.forward)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    accepted_names = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {name: value for name, value in kwargs.items() if name in accepted_names}


def build_layer_forward_kwargs(model, layer, hidden_states, layer_kwargs, batch_start=0, total_batch_size=None):
    batch_size = total_batch_size if total_batch_size is not None else _infer_batch_size(hidden_states, layer_kwargs)
    batch_end = batch_start + hidden_states.shape[0]
    kwargs = {}
    for key, value in layer_kwargs.items():
        if key == "position_embeddings":
            continue
        kwargs[key] = _slice_tensor_for_batch(value, batch_start, batch_end, batch_size)

    decoder_root = _get_decoder_root(model)
    layer_rotary_embedding = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
    decoder_rotary_embedding = getattr(decoder_root, "rotary_emb", None)
    rotary_embedding = decoder_rotary_embedding or layer_rotary_embedding
    rotary_device = _module_device(rotary_embedding)
    if rotary_embedding is not None and rotary_device is not None and rotary_device != hidden_states.device:
        rotary_embedding = rotary_embedding.to(hidden_states.device)

    if rotary_embedding is None:
        return _filter_kwargs_for_layer(layer, kwargs)

    model_type = getattr(model.config, "model_type", None)
    if model_type == "minicpmv":
        kwargs.pop("cache_position", None)
        kwargs.pop("position_embeddings", None)
        kwargs["use_cache"] = False
        return _filter_kwargs_for_layer(layer, kwargs)
    position_ids = kwargs.get("position_ids")
    cache_position = kwargs.get("cache_position")
    rotary_position_ids = position_ids

    if model_type == "qwen2_5_vl":
        rotary_position_ids = _build_qwen2_5_vl_position_ids(hidden_states, position_ids, cache_position)
        if position_ids is None or (position_ids.ndim == 3 and position_ids.shape[0] == 3):
            kwargs["position_ids"] = None
    elif rotary_position_ids is None:
        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if cache_position.ndim == 1:
            rotary_position_ids = cache_position.view(1, -1).expand(hidden_states.shape[0], -1)
        elif cache_position.ndim == 2:
            rotary_position_ids = cache_position

    if rotary_position_ids is not None:
        kwargs["position_embeddings"] = rotary_embedding(hidden_states, rotary_position_ids)

    return _filter_kwargs_for_layer(layer, kwargs)


def forward_in_chunks(model, layer, inputs, kwargs, chunk_size=1):
    outputs = []
    batch_start = 0
    total_batch_size = inputs.shape[0]
    for chunk in torch.split(inputs, chunk_size, dim=0):
        chunk_kwargs = build_layer_forward_kwargs(
            model,
            layer,
            chunk,
            kwargs,
            batch_start=batch_start,
            total_batch_size=total_batch_size,
        )
        chunk_output = layer(chunk, **chunk_kwargs)
        if isinstance(chunk_output, tuple):
            chunk_output = chunk_output[0]
        outputs.append(chunk_output)
        batch_start += chunk.shape[0]
    return torch.cat(outputs, dim=0)
