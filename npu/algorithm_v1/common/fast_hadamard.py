"""Portable Hadamard transform fallback used when CUDA extensions are unavailable."""

from __future__ import annotations

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    width = int(x.shape[-1])
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard transform requires a power-of-two last dimension, got {width}.")

    original_shape = x.shape
    result = x.reshape(-1, width)
    block = 1
    while block < width:
        result = result.reshape(-1, width // (block * 2), block * 2)
        left = result[..., :block]
        right = result[..., block : block * 2]
        result = torch.cat((left + right, left - right), dim=-1)
        block *= 2
    return result.reshape(original_shape) * scale
