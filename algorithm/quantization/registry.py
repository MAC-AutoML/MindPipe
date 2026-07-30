"""Quantization method registry."""

from __future__ import annotations

from importlib import import_module


class _LazyMethod:
    """Callable proxy that resolves a quantization method on first use."""

    __slots__ = ("_method_cls", "_path")

    def __init__(self, path: str):
        self._path = path
        self._method_cls = None

    def resolve(self):
        if self._method_cls is None:
            self._method_cls = _load_method_cls(self._path)
        return self._method_cls

    def __call__(self, *args, **kwargs):
        return self.resolve()(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._path!r})"


METHOD_REGISTRY = {
    "awq": _LazyMethod("algorithm.quantization.ptq.awq.method:AWQMethod"),
    "flatquant": _LazyMethod(
        "algorithm.quantization.qat.flatquant.method:FlatQuantMethod"
    ),
    "gptq": _LazyMethod("algorithm.quantization.ptq.gptq.method:GPTQMethod"),
    "mquant": _LazyMethod("algorithm.quantization.ptq.mquant.method:MQuantMethod"),
    "omniquant": _LazyMethod(
        "algorithm.quantization.ptq.omniquant.method:OmniQuantMethod"
    ),
    "qalora": _LazyMethod("algorithm.quantization.qat.qalora.method:QALoRAMethod"),
    "qlora": _LazyMethod("algorithm.quantization.qat.qlora.method:QLoRAMethod"),
    "quarot": _LazyMethod("algorithm.quantization.ptq.quarot.method:QuaRotMethod"),
    "sliderquant": _LazyMethod(
        "algorithm.quantization.qat.sliderquant.method:SliderQuantMethod"
    ),
    "smoothquant": _LazyMethod(
        "algorithm.quantization.ptq.smoothquant.method:SmoothQuantMethod"
    ),
    "spinquant": _LazyMethod("algorithm.quantization.ptq.spinquant.method:SpinQuantMethod"),
    "splitquant": _LazyMethod(
        "algorithm.quantization.qat.splitquant.method:SplitQuantMethod"
    ),
}


def _load_method_cls(path: str):
    module_name, class_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def get_method(name: str):
    method_factory = METHOD_REGISTRY.get(name)
    if method_factory is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(
            f"Unknown quantization method '{name}'. Available methods: {available}"
        )
    return method_factory()
