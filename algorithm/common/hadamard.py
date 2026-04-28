"""Unified Hadamard transform backend.

Provides ``hadamard_transform`` that dispatches to the fast CUDA kernel
when ``fast_hadamard_transform`` is available, and falls back to a
pure-PyTorch butterfly algorithm otherwise.

Drop-in replacement for ``fast_hadamard_transform.hadamard_transform``.
"""

from __future__ import annotations

import math

import torch


_FAST_HADAMARD_AVAILABLE = False
try:
    import fast_hadamard_transform
    _FAST_HADAMARD_AVAILABLE = True
except ImportError:
    pass


def _pytorch_hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-PyTorch butterfly Hadamard transform on the last dimension.

    The last dimension must be a power of 2.
    """
    n = x.shape[-1]
    original_shape = x.shape
    x = x.clone().reshape(-1, n)
    for bit in range(int(math.log2(n))):
        half = 1 << bit
        x = x.reshape(x.shape[0], -1, 2 * half)
        left, right = x[..., :half], x[..., half:]
        x = torch.cat([left + right, left - right], dim=-1)
        x = x.reshape(x.shape[0], -1)
    return x.reshape(original_shape) * scale


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Drop-in replacement for ``fast_hadamard_transform.hadamard_transform``."""
    if _FAST_HADAMARD_AVAILABLE and x.is_cuda:
        return fast_hadamard_transform.hadamard_transform(x.contiguous(), scale=scale)
    return _pytorch_hadamard_transform(x, scale)
# Maintenance touch for repository metadata refresh.
