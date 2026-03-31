from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(device)
