"""Shared real-quant export artifact types."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RealQuantLinearArtifact:
    """Integer weight and qparams captured from a quantized Linear layer."""

    name: str
    bits: int
    group_size: int
    symmetric: bool
    original_shape: tuple[int, int]
    int_weight: torch.Tensor
    scale: torch.Tensor
