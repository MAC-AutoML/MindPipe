"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def write_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def model_slug(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name

# Refactor the project structure and clarify the evaluation entrypoint.
