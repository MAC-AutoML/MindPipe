"""Helpers for best-effort reproducible execution."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int, device: str | None = None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from algorithm.common.device import manual_seed_all
    from algorithm.common.device import resolve_device

    if device is not None:
        resolved = resolve_device(device)
        manual_seed_all(seed, resolved)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as error:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to enable deterministic algorithms: %s", error)

    try:
        import transformers

        transformers.set_seed(seed)
    except Exception as error:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to propagate seed to transformers: %s", error)

    LOGGER.info("Global seed set to %s with deterministic backends enabled", seed)
