"""Local QuaRot runtime with optional CUDA kernels."""

from __future__ import annotations

import importlib
import logging

import torch

from .functional.quantization import pack_i4
from .functional.quantization import unpack_i4

logger = logging.getLogger(__name__)

try:
    _CUDA = importlib.import_module(f"{__name__}._CUDA")
except Exception as exc:  # pragma: no cover - environment dependent
    _CUDA = None
    logger.warning("QuaRot CUDA extension unavailable, falling back to torch ops: %s", exc)


def _flatten_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape_excl_last = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]), shape_excl_last


class PackedQuantizedTensor:
    def __init__(self, quantized_x: torch.Tensor, scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()

    @property
    def device(self):
        return self.quantized_x.device

    @property
    def dtype(self):
        return self.quantized_x.dtype


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] % 32 == 0, f"a.shape[-1]={a.shape[-1]} must be divisible by 32"
    a_flat, a_shape = _flatten_last_dim(a)
    b_flat, b_shape = _flatten_last_dim(b)
    if _CUDA is not None and a_flat.is_cuda and b_flat.is_cuda:
        out = _CUDA.matmul(a_flat.contiguous(), b_flat.contiguous())
    else:
        lhs = unpack_i4(a_flat).to(torch.float32)
        rhs = unpack_i4(b_flat).to(torch.float32)
        out = torch.matmul(lhs, rhs.t()).to(torch.int32)
    return out.view(*a_shape, *b_shape)


def sym_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_flat, x_shape = _flatten_last_dim(x)
    scale_flat = scale.reshape(-1)
    if _CUDA is not None and x_flat.is_cuda and scale_flat.is_cuda:
        return _CUDA.sym_quant(x_flat.contiguous(), scale_flat.contiguous()).view(*x_shape, -1)
    q = torch.clamp(
        torch.round(x_flat / scale_flat.unsqueeze(1)),
        min=-8,
        max=7,
    ).to(torch.int8)
    return pack_i4(q).view(*x_shape, -1)


def sym_dequant(
    q: torch.Tensor,
    scale_row: torch.Tensor,
    scale_col: torch.Tensor,
    bits: int = 32,
) -> torch.Tensor:
    if bits != 32:
        raise NotImplementedError(f"Only int32 accumulation dequant is supported, got bits={bits}.")
    q_flat, q_shape = _flatten_last_dim(q)
    row_scale = scale_row.reshape(-1, 1).to(torch.float32)
    col_scale = scale_col.reshape(1, -1).to(torch.float32)
    if _CUDA is not None and q_flat.is_cuda and scale_row.is_cuda and scale_col.is_cuda:
        out = _CUDA.sym_dequant(
            q_flat.contiguous(),
            scale_row.reshape(-1).contiguous(),
            scale_col.contiguous(),
            bits,
        )
    else:
        out = (q_flat.to(torch.float32) * row_scale * col_scale).to(torch.float16)
    return out.view(*q_shape, -1)


from . import functional  # noqa: E402
from . import nn  # noqa: E402

__all__ = [
    "PackedQuantizedTensor",
    "functional",
    "matmul",
    "nn",
    "pack_i4",
    "sym_dequant",
    "sym_quant",
    "unpack_i4",
]
# Refactor the project structure and clarify the evaluation entrypoint.
