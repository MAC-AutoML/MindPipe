import torch


def get_module_device(module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def move_tensor_tree_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tensor_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_tensor_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_tensor_tree_to_device(item, device) for key, item in value.items()}
    return value


def align_attention_auxiliary_tensors(
    device: torch.device,
    *,
    attention_mask=None,
    position_ids=None,
    cache_position=None,
    position_embeddings=None,
):
    return (
        move_tensor_tree_to_device(attention_mask, device),
        move_tensor_tree_to_device(position_ids, device),
        move_tensor_tree_to_device(cache_position, device),
        move_tensor_tree_to_device(position_embeddings, device),
    )


def build_cache_kwargs_on_device(device: torch.device, **kwargs):
    return {
        key: move_tensor_tree_to_device(value, device)
        for key, value in kwargs.items()
        if value is not None
    }
