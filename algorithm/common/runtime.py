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


def purge_conflicting_modules(module_name: str, allowed_root: str | Path) -> None:
    """Remove already-imported modules that shadow a source_root injection.

    Many upstream algorithm implementations ship as a top-level package named `lib`.
    When we load multiple algorithms in the same Python process, `sys.modules` caching
    can cause later stages to silently reuse an earlier algorithm's `lib.*` modules
    even if we prepend a different `source_root` to `sys.path`.

    This helper drops `module_name` and its children from `sys.modules` unless their
    origin is under `allowed_root`, forcing a clean import from the intended source.
    """

    allowed_root = Path(allowed_root).resolve()
    prefix = f"{module_name}."
    for name, module in list(sys.modules.items()):
        if name != module_name and not name.startswith(prefix):
            continue

        module_file = getattr(module, "__file__", None)
        module_path = getattr(module, "__path__", None)
        candidates: list[Path] = []
        if module_file:
            candidates.append(Path(module_file))
        if module_path:
            candidates.extend(Path(path) for path in module_path)

        try:
            if any(path.resolve().is_relative_to(allowed_root) for path in candidates):
                continue
        except Exception:
            # If we can't reason about the module origin, assume it is unsafe to keep.
            pass
        sys.modules.pop(name, None)

# Refactor the project structure and clarify the evaluation entrypoint.
