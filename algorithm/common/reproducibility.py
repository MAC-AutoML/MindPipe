"""Helpers for best-effort reproducible execution."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
