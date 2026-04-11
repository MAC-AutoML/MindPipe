"""Quantization method registry."""

from __future__ import annotations

from .ptq.awq.method import AWQMethod
from .ptq.gptq.method import GPTQMethod
from .ptq.quarot.method import QuaRotMethod
from .ptq.smoothquant.method import SmoothQuantMethod
from .ptq.spinquant.method import SpinQuantMethod
from .qat.flatquant.method import FlatQuantMethod
from .qat.splitquant.method import SplitQuantMethod


METHOD_REGISTRY = {
    "awq": AWQMethod,
    "flatquant": FlatQuantMethod,
    "gptq": GPTQMethod,
    "quarot": QuaRotMethod,
    "smoothquant": SmoothQuantMethod,
    "splitquant": SplitQuantMethod,
    "spinquant": SpinQuantMethod,
}


def get_method(name: str):
    method_cls = METHOD_REGISTRY.get(name)
    if method_cls is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(f"Unknown quantization method '{name}'. Available methods: {available}")
    return method_cls()
