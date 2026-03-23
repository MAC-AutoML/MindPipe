"""Device helpers shared across CUDA/NPU-compatible paths."""

from __future__ import annotations

from typing import Any

import torch


_NPU_COMPAT_ENABLED = False
_DEFAULT_NPU_DEVICE = torch.device("npu:0")


def _device_type(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def _normalize_npu_device(device: Any = None) -> torch.device:
    if device is None:
        return _DEFAULT_NPU_DEVICE
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return torch.device("npu", device.index or 0)
        return device
    if isinstance(device, int):
        return torch.device("npu", device)
    if isinstance(device, str) and device.startswith("cuda"):
        if ":" in device:
            return torch.device(f"npu:{device.split(':', maxsplit=1)[1]}")
        return _DEFAULT_NPU_DEVICE
    return torch.device(device)


def is_npu_device(device: str | torch.device | None) -> bool:
    return _device_type(device) == "npu"


def accelerator_available() -> bool:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return True
    return torch.cuda.is_available()


def default_accelerator_device(index: int | None = 0) -> torch.device:
    resolved_index = 0 if index is None else index
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu", resolved_index)
    if torch.cuda.is_available():
        return torch.device("cuda", resolved_index)
    return torch.device("cpu")


def backend_module(device: str | torch.device | None):
    device_type = _device_type(device)
    if device_type == "npu" and hasattr(torch, "npu"):
        return torch.npu
    if device_type == "cuda":
        return torch.cuda
    return None


def empty_cache(device: str | torch.device | None = None) -> None:
    backend = backend_module(device)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def synchronize(device: str | torch.device | None = None) -> None:
    backend = backend_module(device)
    if backend is not None and hasattr(backend, "synchronize"):
        target_device = device
        if is_npu_device(device):
            target_device = _normalize_npu_device(device)
        if target_device is None:
            backend.synchronize()
        else:
            backend.synchronize(target_device)


def memory_reserved(device: str | torch.device | None = None) -> int:
    backend = backend_module(device)
    if backend is None or not hasattr(backend, "memory_reserved"):
        return 0
    if device is None:
        return int(backend.memory_reserved())
    return int(backend.memory_reserved(_normalize_npu_device(device) if is_npu_device(device) else device))


def device_count(device: str | torch.device | None = None) -> int:
    backend = backend_module(device)
    if backend is None or not hasattr(backend, "device_count"):
        return 0
    return int(backend.device_count())


def manual_seed_all(seed: int, device: str | torch.device | None = None) -> None:
    backend = backend_module(device)
    if backend is not None:
        if hasattr(backend, "manual_seed"):
            backend.manual_seed(seed)
        if hasattr(backend, "manual_seed_all"):
            backend.manual_seed_all(seed)


def preferred_rotation_dtype(device: str | torch.device | None) -> torch.dtype:
    return torch.float32 if is_npu_device(device) else torch.float64


def enable_device_compat(device: str | torch.device | None) -> None:
    global _NPU_COMPAT_ENABLED, _DEFAULT_NPU_DEVICE
    if not is_npu_device(device) or _NPU_COMPAT_ENABLED:
        return

    try:
        import torch_npu  # noqa: F401
    except Exception:
        return

    _DEFAULT_NPU_DEVICE = _normalize_npu_device(device)

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to(
            _normalize_npu_device(device),
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    def module_cuda(self, device=None):
        return self.to(_normalize_npu_device(device))

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda

    torch.cuda.empty_cache = lambda: empty_cache(_DEFAULT_NPU_DEVICE)
    torch.cuda.synchronize = lambda device=None: synchronize(_normalize_npu_device(device))
    torch.cuda.current_device = lambda: torch.npu.current_device()
    torch.cuda.device_count = lambda: torch.npu.device_count()
    torch.cuda.memory_reserved = lambda device=None: memory_reserved(_normalize_npu_device(device))
    torch.cuda.memory_allocated = lambda device=None: int(
        torch.npu.memory_allocated(_normalize_npu_device(device))
    )
    torch.cuda.set_device = lambda device=None: torch.npu.set_device(_normalize_npu_device(device))
    torch.cuda.manual_seed = lambda seed: manual_seed_all(seed, _DEFAULT_NPU_DEVICE)
    torch.cuda.manual_seed_all = lambda seed: manual_seed_all(seed, _DEFAULT_NPU_DEVICE)

    _NPU_COMPAT_ENABLED = True
