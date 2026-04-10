"""Pruning method registry."""

from __future__ import annotations

from .structured.flap.method import FLAPMethod
from .structured.wanda_sp.method import WandaSPMethod
from .unstructured.alps.method import ALPSMethod
from .unstructured.sparsegpt.method import SparseGPTMethod
from .unstructured.wanda.method import WandaMethod


METHOD_REGISTRY = {
    "alps": ALPSMethod,
    "flap": FLAPMethod,
    "sparsegpt": SparseGPTMethod,
    "wanda": WandaMethod,
    "wanda_sp": WandaSPMethod,
}


def get_method(name: str):
    method_cls = METHOD_REGISTRY.get(name)
    if method_cls is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(f"Unknown pruning method '{name}'. Available methods: {available}")
    return method_cls()
