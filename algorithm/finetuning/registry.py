"""Finetuning method registry."""

from __future__ import annotations

from .compression_lora.method import CompressionLoRAMethod


METHOD_REGISTRY = {
    "compression_lora": CompressionLoRAMethod,
}


def get_method(name: str):
    method_cls = METHOD_REGISTRY.get(name)
    if method_cls is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(f"Unknown finetuning method '{name}'. Available methods: {available}")
    return method_cls()

