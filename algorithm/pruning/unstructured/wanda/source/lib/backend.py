from __future__ import annotations

import torch

from algorithm.common.device import resolve_device


def resolve_runtime_device(device: str | torch.device | int | None) -> torch.device:
    return resolve_device(device)


def move_optional_tensor(value, device: str | torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device)


def sparsity_threshold(
    metric: torch.Tensor,
    sparsity: float,
    compute_device: str | torch.device | int | None,
) -> torch.Tensor:
    if metric.numel() == 0:
        raise ValueError("Cannot compute a sparsity threshold for an empty tensor.")

    runtime_device = resolve_runtime_device(compute_device)
    flattened_metric = metric.reshape(-1)
    if flattened_metric.device != runtime_device:
        flattened_metric = flattened_metric.to(runtime_device)

    threshold_index = int(flattened_metric.numel() * sparsity)
    threshold_index = min(max(threshold_index, 0), flattened_metric.numel() - 1)
    threshold = torch.sort(flattened_metric)[0][threshold_index]
    return threshold.to(metric.device)
# Maintenance touch for repository metadata refresh.
