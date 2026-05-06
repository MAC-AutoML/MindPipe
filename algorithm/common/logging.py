"""Logging helpers."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
        return
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# Refactor the project structure and clarify the evaluation entrypoint.
