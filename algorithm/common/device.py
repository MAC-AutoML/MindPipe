"""Strict device helpers shared across CUDA/NPU backends."""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any

import torch


_ACCELERATOR_PATTERN = re.compile(r"^(cuda|npu)(?::(\d+))?$", re.IGNORECASE)
_TORCH_NPU_IMPORT_ATTEMPTED = False
_NPU_COMPAT_ENABLED = False
_DEFAULT_ACCELERATOR_DEVICE: torch.device | None = None
_PENDING_ADAPTATION_NOTICES: set[str] = set()


def _ensure_torch_npu_loaded() -> bool:
    global _TORCH_NPU_IMPORT_ATTEMPTED
    if hasattr(torch, "npu"):
        return True
    if _TORCH_NPU_IMPORT_ATTEMPTED:
        return False
    _TORCH_NPU_IMPORT_ATTEMPTED = True
    try:
        importlib.import_module("torch_npu")
    except Exception:
        return False
    return hasattr(torch, "npu")


def npu_available() -> bool:
    if not _ensure_torch_npu_loaded():
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def accelerator_available() -> bool:
    return npu_available() or cuda_available()


def active_accelerator_backend(preferred: str | None = None) -> str | None:
    normalized_preferred = None if preferred is None else preferred.lower()
    if normalized_preferred == "npu":
        return "npu" if npu_available() else None
    if normalized_preferred == "cuda":
        return "cuda" if cuda_available() else None
    if npu_available():
        return "npu"
    if cuda_available():
        return "cuda"
    return None


def _coerce_requested_device(device: str | torch.device | int | None) -> tuple[str, int | None]:
    if device is None:
        return "auto", None
    if isinstance(device, int):
        return "auto", device
    if isinstance(device, torch.device):
        if device.index is None:
            return device.type, None
        return device.type, int(device.index)
    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized in {"", "auto"}:
            return "auto", None
        if normalized == "cpu":
            return "cpu", None
        match = _ACCELERATOR_PATTERN.fullmatch(normalized)
        if match is not None:
            backend = match.group(1)
            index_text = match.group(2)
            return backend, 0 if index_text is None else int(index_text)
        parsed_device = torch.device(device)
        if parsed_device.index is None:
            return parsed_device.type, None
        return parsed_device.type, int(parsed_device.index)
    raise TypeError(f"Unsupported device specifier: {type(device)}")


def _require_backend(backend: str) -> None:
    normalized_backend = backend.lower()
    if normalized_backend == "cuda":
        if not cuda_available():
            raise RuntimeError("Requested CUDA execution, but CUDA is not available.")
        return
    if normalized_backend == "npu":
        if not _ensure_torch_npu_loaded():
            raise RuntimeError("Requested NPU execution, but torch_npu could not be imported.")
        if not npu_available():
            raise RuntimeError("Requested NPU execution, but no NPU device is available.")
        return
    raise ValueError(f"Unsupported accelerator backend: {backend}")


def default_accelerator_device(index: int | None = 0) -> torch.device:
    resolved_index = 0 if index is None else int(index)
    backend = active_accelerator_backend()
    if backend is None:
        raise RuntimeError(
            "No accelerator backend is available. Specify --device cpu explicitly only if CPU execution is intended."
        )
    return torch.device(backend, resolved_index)


def resolve_device(device: str | torch.device | int | None) -> torch.device:
    requested_backend, requested_index = _coerce_requested_device(device)
    if requested_backend == "cpu":
        return torch.device("cpu")
    if requested_backend == "auto":
        return default_accelerator_device(requested_index)
    if requested_backend in {"cuda", "npu"}:
        _require_backend(requested_backend)
        return torch.device(requested_backend, 0 if requested_index is None else requested_index)
    if requested_index is None:
        return torch.device(requested_backend)
    return torch.device(requested_backend, requested_index)


def resolve_device_string(device: str | torch.device | int | None) -> str:
    return str(resolve_device(device))


def _device_type(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    return resolve_device(device).type


def is_npu_device(device: str | torch.device | None) -> bool:
    return _device_type(device) == "npu"


def backend_module(device: str | torch.device | None):
    device_type = _device_type(device)
    if device_type == "npu" and npu_available():
        return torch.npu
    if device_type == "cuda" and cuda_available():
        return torch.cuda
    return None


def empty_cache(device: str | torch.device | None = None) -> None:
    backend = backend_module(device)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def synchronize(device: str | torch.device | None = None) -> None:
    backend = backend_module(device)
    if backend is None or not hasattr(backend, "synchronize"):
        return
    if device is None:
        backend.synchronize()
        return
    backend.synchronize(resolve_device(device))


def memory_reserved(device: str | torch.device | None = None) -> int:
    backend = backend_module(device)
    if backend is None or not hasattr(backend, "memory_reserved"):
        return 0
    if device is None:
        return int(backend.memory_reserved())
    return int(backend.memory_reserved(resolve_device(device)))


def device_count(device: str | torch.device | None = None) -> int:
    backend = backend_module(device)
    if backend is None or not hasattr(backend, "device_count"):
        return 0
    return int(backend.device_count())


def manual_seed_all(seed: int, device: str | torch.device | None = None) -> None:
    backend = backend_module(device)
    if backend is None:
        return
    if hasattr(backend, "manual_seed"):
        backend.manual_seed(seed)
    if hasattr(backend, "manual_seed_all"):
        backend.manual_seed_all(seed)


def preferred_rotation_dtype(device: str | torch.device | None) -> torch.dtype:
    return torch.float32 if is_npu_device(device) else torch.float64


def _warn_pending_cpu_offload(feature: str, source_device: torch.device, operation: str) -> None:
    notice_key = f"{feature}:{source_device}:{operation}"
    if notice_key in _PENDING_ADAPTATION_NOTICES:
        return
    _PENDING_ADAPTATION_NOTICES.add(notice_key)
    warnings.warn(
        f"[pending-adaptation] {feature}: {operation} is explicitly offloaded from {source_device} to CPU; "
        "native NPU support is still pending.",
        RuntimeWarning,
        stacklevel=3,
    )


def maybe_offload_hessian_to_cpu(
    hessian: torch.Tensor,
    feature: str,
    cuda_threshold: int | None = None,
) -> torch.Tensor:
    if hessian.device.type == "npu":
        _warn_pending_cpu_offload(feature, hessian.device, "Hessian Cholesky solve")
        return hessian.cpu()
    if (
        cuda_threshold is not None
        and hessian.device.type == "cuda"
        and hessian.shape[-1] >= cuda_threshold
    ):
        return hessian.cpu()
    return hessian


def distributed_backend(device: str | torch.device | None = None) -> str:
    device_type = _device_type(device)
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def enable_device_compat(device: str | torch.device | None) -> str:
    global _DEFAULT_ACCELERATOR_DEVICE, _NPU_COMPAT_ENABLED
    resolved_device = resolve_device(device)
    if resolved_device.type != "npu":
        return str(resolved_device)

    _require_backend("npu")
    _DEFAULT_ACCELERATOR_DEVICE = resolved_device
    if _NPU_COMPAT_ENABLED:
        return str(resolved_device)

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        target_device = _DEFAULT_ACCELERATOR_DEVICE if device is None else resolve_device(device)
        return self.to(
            target_device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    def module_cuda(self, device=None):
        target_device = _DEFAULT_ACCELERATOR_DEVICE if device is None else resolve_device(device)
        return self.to(target_device)

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda

    torch.cuda.empty_cache = lambda: empty_cache(_DEFAULT_ACCELERATOR_DEVICE)
    torch.cuda.synchronize = lambda device=None: synchronize(
        _DEFAULT_ACCELERATOR_DEVICE if device is None else resolve_device(device)
    )
    torch.cuda.current_device = lambda: torch.npu.current_device()
    torch.cuda.device_count = lambda: torch.npu.device_count()
    torch.cuda.memory_reserved = lambda device=None: memory_reserved(
        _DEFAULT_ACCELERATOR_DEVICE if device is None else resolve_device(device)
    )
    torch.cuda.memory_allocated = lambda device=None: int(
        torch.npu.memory_allocated(_DEFAULT_ACCELERATOR_DEVICE if device is None else resolve_device(device))
    )
    torch.cuda.set_device = lambda device=None: torch.npu.set_device(
        _DEFAULT_ACCELERATOR_DEVICE if device is None else resolve_device(device)
    )
    torch.cuda.manual_seed = lambda seed: manual_seed_all(seed, _DEFAULT_ACCELERATOR_DEVICE)
    torch.cuda.manual_seed_all = lambda seed: manual_seed_all(seed, _DEFAULT_ACCELERATOR_DEVICE)

    _NPU_COMPAT_ENABLED = True
    return str(resolved_device)
