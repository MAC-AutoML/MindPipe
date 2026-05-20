"""NPU-compatible fallback for the CUDA-only causal-conv1d package.

The upstream package builds CUDA extensions and is not installable on Ascend
NPU. This shim exposes the two symbols imported by FLA/HF Qwen3.5 and routes
them to Triton-Ascend kernels when practical, otherwise to torch operations.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn.functional as F

__version__ = "1.6.2.post1+npu"


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_vllm_ascend_update():
    extra_path = os.environ.get("MINDPIPE_QWEN35_VLLM_SRC")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    try:
        from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu
    except Exception:
        return None
    return causal_conv1d_update_npu


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    **_: Any,
):
    if seq_idx is not None:
        raise NotImplementedError("causal_conv1d shim does not support seq_idx.")
    dtype = x.dtype
    width = weight.shape[-1]
    x_work = x.to(weight.dtype)
    if initial_states is not None:
        x_work = torch.cat([initial_states, x_work], dim=-1)
        padding = 0
    else:
        padding = width - 1
    out = F.conv1d(x_work, weight.unsqueeze(1), bias, padding=padding, groups=x.shape[1])
    out = out[:, :, : x.shape[-1]]
    if activation in {"silu", "swish"}:
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported causal_conv1d activation: {activation!r}")
    out = out.to(dtype)
    if return_final_states:
        final_states = F.pad(x_work[:, :, -width + 1 :], (max(width - 1 - x_work.shape[-1], 0), 0)).to(dtype)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
            final_states = final_states_out
        return out, final_states
    return out


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    **_: Any,
) -> torch.Tensor:
    del cache_seqlens
    if (
        hidden_states.device.type == "npu"
        and _env_enabled("MINDPIPE_CAUSAL_CONV1D_SHIM_USE_VLLM_ASCEND", False)
        and conv_state_indices is not None
    ):
        update = _load_vllm_ascend_update()
        if update is not None:
            try:
                x_arg = hidden_states.squeeze(-1) if hidden_states.ndim == 3 and hidden_states.shape[-1] == 1 else hidden_states
                return update(
                    x_arg.contiguous(),
                    conv_state,
                    weight.contiguous(),
                    bias.contiguous() if bias is not None else None,
                    activation,
                    conv_state_indices=conv_state_indices,
                    pad_slot_id=-1,
                    validate_data=False,
                )
            except Exception:
                pass

    dtype = hidden_states.dtype
    state_len = conv_state.shape[-1]
    if hidden_states.ndim == 2:
        hidden_states = hidden_states.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_states.shape[1])
    out = out[:, :, -hidden_states.shape[-1] :]
    if activation in {None, "silu", "swish"}:
        out = F.silu(out) if activation is not None else out
    else:
        raise ValueError(f"Unsupported causal_conv1d activation: {activation!r}")
    out = out.to(dtype)
    return out.squeeze(-1) if squeeze else out


__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
