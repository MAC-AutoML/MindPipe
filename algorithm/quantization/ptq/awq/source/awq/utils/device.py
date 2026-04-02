from __future__ import annotations

import torch

from algorithm.common.device import default_accelerator_device
from algorithm.common.device import resolve_device as resolve_runtime_device


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return default_accelerator_device()
    return resolve_runtime_device(device)
