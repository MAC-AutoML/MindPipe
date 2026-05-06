"""Runtime helpers."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def prepend_python_path(path: str | Path):
    resolved_path = str(Path(path).resolve())
    sys.path.insert(0, resolved_path)
    try:
        yield
    finally:
        try:
            sys.path.remove(resolved_path)
        except ValueError:
            pass

# Refactor the project structure and clarify the evaluation entrypoint.
