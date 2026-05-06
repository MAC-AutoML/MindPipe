from __future__ import annotations

import torch

from algorithm.common.device import resolve_device as resolve_runtime_device


def resolve_device(device: str | torch.device) -> torch.device:
    return resolve_runtime_device(device)
# Fix GPU/NPU adaptation bugs and use a unified device abstraction.
