"""Pruning method registry."""

from __future__ import annotations

from .structured.flap.method import FLAPMethod
from .unstructured.sparsegpt.method import SparseGPTMethod
from .unstructured.wanda.method import WandaMethod


METHOD_REGISTRY = {
    "flap": FLAPMethod,
    "sparsegpt": SparseGPTMethod,
    "wanda": WandaMethod,
}


def get_method(name: str):
    method_cls = METHOD_REGISTRY.get(name)
    if method_cls is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(f"Unknown pruning method '{name}'. Available methods: {available}")
    return method_cls()

