"""Ascend NPU fast-path patch for HuggingFace Qwen3.5 linear attention."""

from __future__ import annotations

import importlib
import os
import sys
import time
import types
import warnings
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_WARNED: set[str] = set()
_PROFILE_COUNTS: dict[str, int] = {}
_PROFILE_SECONDS: dict[str, float] = {}
_SHIM_ROOT = os.path.join(os.path.dirname(__file__), "npu_shims")


def _ensure_npu_shim_path() -> None:
    if _SHIM_ROOT not in sys.path:
        sys.path.insert(0, _SHIM_ROOT)


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _record_profile(key: str, elapsed: float) -> None:
    if not _env_enabled("MINDPIPE_QWEN35_NPU_PROFILE_PATCH", False):
        return
    _PROFILE_COUNTS[key] = _PROFILE_COUNTS.get(key, 0) + 1
    _PROFILE_SECONDS[key] = _PROFILE_SECONDS.get(key, 0.0) + elapsed
    interval = int(os.environ.get("MINDPIPE_QWEN35_NPU_PROFILE_INTERVAL", "128"))
    count = _PROFILE_COUNTS[key]
    if interval > 0 and count % interval == 0:
        print(
            f"[MindPipe Qwen3.5 NPU patch] {key}: count={count} total={_PROFILE_SECONDS[key]:.6f}s "
            f"avg={_PROFILE_SECONDS[key] / count:.6f}s",
            flush=True,
        )


def _load_torch_npu_recurrent_gated_delta_rule() -> Callable[..., Any] | None:
    try:
        import torch_npu
    except Exception:
        return None
    return getattr(torch_npu, "npu_recurrent_gated_delta_rule", None)


def _load_torch_npu_attention_ops() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    try:
        import torch_npu
    except Exception:
        return None, None
    return (
        getattr(torch_npu, "npu_fusion_attention", None),
        getattr(torch_npu, "npu_incre_flash_attention", None),
    )


def _load_torch_npu_grouped_matmul() -> Callable[..., Any] | None:
    try:
        import torch_npu
    except Exception:
        return None
    return getattr(torch_npu, "npu_grouped_matmul", None)


def _load_torch_npu_dense_ops() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None, Callable[..., Any] | None]:
    try:
        import torch_npu
    except Exception:
        return None, None, None
    return (
        getattr(torch_npu, "npu_rms_norm", None),
        getattr(torch_npu, "npu_rotary_mul", None),
        getattr(torch_npu, "npu_swiglu", None),
    )


def _load_vllm_packed_decode() -> Callable[..., Any] | None:
    extra_path = os.environ.get("MINDPIPE_QWEN35_VLLM_SRC")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    try:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    except Exception as exc:
        _warn_once(
            "vllm_packed_decode_import",
            f"Qwen3.5 NPU vLLM packed decode path is unavailable: {exc!r}",
        )
        return None
    return fused_recurrent_gated_delta_rule_packed_decode


def _load_vllm_ascend_fused_sigmoid_update() -> Callable[..., Any] | None:
    extra_path = os.environ.get("MINDPIPE_QWEN35_VLLM_SRC")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    try:
        from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_sigmoid_gating_delta_rule_update
    except Exception as exc:
        _warn_once(
            "vllm_ascend_fused_sigmoid_import",
            f"Qwen3.5 NPU vLLM-Ascend fused sigmoid update path is unavailable: {exc!r}",
        )
        return None
    return fused_sigmoid_gating_delta_rule_update


def _load_vllm_ascend_causal_conv1d_update() -> Callable[..., Any] | None:
    extra_path = os.environ.get("MINDPIPE_QWEN35_VLLM_SRC")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    try:
        from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu
    except Exception as exc:
        _warn_once(
            "vllm_ascend_conv_update_import",
            f"Qwen3.5 NPU vLLM-Ascend causal conv update path is unavailable: {exc!r}",
        )
        return None
    return causal_conv1d_update_npu


def _ensure_mindspeed_lite_alias() -> bool:
    """Map ms-swift's old mindspeed.lite imports to MindSpeed-core's ops path."""

    if not _env_enabled("MINDPIPE_QWEN35_NPU_ALIAS_MINDSPEED_LITE", True):
        return False
    try:
        import mindspeed.ops.triton as ms_triton
    except Exception as exc:
        _warn_once(
            "mindspeed_ops_import",
            f"Qwen3.5 NPU MindSpeed ops path is unavailable: {exc!r}",
        )
        return False

    for name in ("mindspeed.lite", "mindspeed.lite.ops"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["mindspeed.lite.ops.triton"] = ms_triton

    for submodule in ("chunk_delta_h", "chunk_o", "chunk_scaled_dot_kkt", "cumsum", "solve_tril", "utils", "wy_fast"):
        try:
            sys.modules[f"mindspeed.lite.ops.triton.{submodule}"] = importlib.import_module(
                f"mindspeed.ops.triton.{submodule}"
            )
        except Exception as exc:
            _warn_once(
                f"mindspeed_lite_alias_{submodule}",
                f"Qwen3.5 NPU MindSpeed lite alias failed for {submodule}: {exc!r}",
            )
            return False
    return True


def _load_mindspeed_chunk_gated_delta_rule() -> Callable[..., Any] | None:
    default_enabled = not torch.__version__.startswith("2.9.")
    if not _env_enabled("MINDPIPE_QWEN35_NPU_USE_MINDSPEED_CHUNK", default_enabled):
        return None

    _ensure_mindspeed_lite_alias()
    candidates = (
        "swift.model.chunk_gated_delta_rule",
        "mindspeed.core.ssm.chunk_gated_delta_rule",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            _warn_once(
                f"mindspeed_chunk_import_{module_name}",
                f"Qwen3.5 NPU MindSpeed chunk path {module_name} is unavailable: {exc!r}",
            )
            continue
        chunk_fn = getattr(module, "chunk_gated_delta_rule", None)
        if chunk_fn is not None:
            return chunk_fn
    return None


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    if not hasattr(torch, "npu"):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _is_qwen3_5_config(config: Any | None) -> bool:
    if config is None:
        return True
    model_types = {
        str(getattr(config, "model_type", "")),
        str(getattr(getattr(config, "text_config", None), "model_type", "")),
    }
    if any(model_type in {"qwen3_5", "qwen3_5_moe"} or model_type.startswith(("qwen3_5_", "qwen3_5_moe_")) for model_type in model_types):
        return True
    architectures = set(getattr(config, "architectures", []) or [])
    return any("Qwen3_5" in name for name in architectures)


def _load_fla_ops() -> dict[str, Callable[..., Any]] | None:
    _ensure_npu_shim_path()
    try:
        from fla.modules.convolution import causal_conv1d_fwd
        from fla.modules.convolution import causal_conv1d_update as fla_causal_conv1d_update
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    except Exception as exc:
        _warn_once(
            "fla_import",
            f"Qwen3.5 NPU linear-attention patch is disabled because FLA import failed: {exc!r}",
        )
        return None
    mindspeed_chunk_gated_delta_rule = _load_mindspeed_chunk_gated_delta_rule()
    return {
        "causal_conv1d_fwd": causal_conv1d_fwd,
        "causal_conv1d_update": fla_causal_conv1d_update,
        "chunk_gated_delta_rule": fla_chunk_gated_delta_rule,
        "mindspeed_chunk_gated_delta_rule": mindspeed_chunk_gated_delta_rule,
        "fused_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
    }


def _torch_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **_: Any,
) -> torch.Tensor:
    dtype = x.dtype
    channels = x.shape[1]
    out = F.conv1d(
        x.to(weight.dtype),
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[-1] - 1,
        groups=channels,
    )
    out = out[:, :, : x.shape[-1]]
    if activation in {"silu", "swish"}:
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported Qwen3.5 causal conv activation on NPU: {activation!r}")
    return out.to(dtype)


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    dtype = hidden_states.dtype
    hidden_size = hidden_states.shape[1]
    seq_len = hidden_states.shape[-1]
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation in {None, "silu", "swish"}:
        out = F.silu(out)
    else:
        raise ValueError(f"Unsupported Qwen3.5 causal conv activation on NPU: {activation!r}")
    return out.to(dtype)


def _make_causal_conv1d_fn(fla_ops: dict[str, Callable[..., Any]]) -> Callable[..., torch.Tensor]:
    fla_causal_conv1d_fwd = fla_ops["causal_conv1d_fwd"]

    def causal_conv1d_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        seq_idx: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not _env_enabled("MINDPIPE_QWEN35_NPU_USE_FLA_CONV_FWD", False):
            return _torch_causal_conv1d_fn(x, weight, bias=bias, activation=activation, **kwargs)
        if seq_idx is not None:
            _warn_once("conv_seq_idx", "Qwen3.5 NPU causal_conv1d received seq_idx; using torch fallback.")
            return _torch_causal_conv1d_fn(x, weight, bias=bias, activation=activation, **kwargs)
        try:
            y, _ = fla_causal_conv1d_fwd(
                x=x.transpose(1, 2).contiguous(),
                weight=weight.contiguous(),
                bias=bias,
                residual=None,
                activation=activation,
            )
            return y.transpose(1, 2).contiguous()
        except Exception as exc:
            _warn_once("conv_fwd_fallback", f"Qwen3.5 NPU causal_conv1d FLA path failed; using torch fallback: {exc!r}")
            return _torch_causal_conv1d_fn(x, weight, bias=bias, activation=activation, **kwargs)

    return causal_conv1d_fn


def _make_causal_conv1d_update(fla_ops: dict[str, Callable[..., Any]]) -> Callable[..., torch.Tensor]:
    fla_causal_conv1d_update = fla_ops["causal_conv1d_update"]
    vllm_ascend_causal_conv1d_update = _load_vllm_ascend_causal_conv1d_update()

    def causal_conv1d_update(
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ) -> torch.Tensor:
        if (
            vllm_ascend_causal_conv1d_update is not None
            and _env_enabled("MINDPIPE_QWEN35_NPU_VLLM_ASCEND_CONV_UPDATE", False)
            and hidden_states.device.type == "npu"
            and hidden_states.dtype in {torch.float16, torch.bfloat16}
            and hidden_states.ndim == 3
            and conv_state.ndim == 3
        ):
            try:
                start = time.perf_counter()
                batch_size = hidden_states.shape[0]
                state_backing = conv_state.transpose(1, 2).contiguous()
                state_view = state_backing.transpose(1, 2)
                state_indices = torch.arange(batch_size, device=hidden_states.device, dtype=torch.int32)
                if hidden_states.shape[-1] == 1:
                    x_arg = hidden_states.squeeze(-1).contiguous()
                else:
                    x_arg = hidden_states.transpose(1, 2).contiguous()
                out = vllm_ascend_causal_conv1d_update(
                    x_arg,
                    state_view,
                    weight.contiguous(),
                    bias.contiguous() if bias is not None else None,
                    activation,
                    conv_state_indices=state_indices,
                    pad_slot_id=-1,
                    validate_data=True,
                )
                if out.ndim == 2:
                    out = out.unsqueeze(-1)
                elif out.shape[1] != hidden_states.shape[1]:
                    out = out.transpose(1, 2).contiguous()
                conv_state.copy_(state_backing.transpose(1, 2).contiguous())
                _record_profile("conv_update_vllm_ascend", time.perf_counter() - start)
                return out.to(hidden_states.dtype)
            except Exception as exc:
                _warn_once(
                    "conv_update_vllm_ascend_fallback",
                    f"Qwen3.5 NPU vLLM-Ascend causal_conv1d_update failed; using FLA path: {exc!r}",
                )
        try:
            start = time.perf_counter()
            y, _ = fla_causal_conv1d_update(
                x=hidden_states.transpose(1, 2).contiguous(),
                cache=conv_state,
                weight=weight.contiguous(),
                bias=bias,
                activation=activation,
            )
            _record_profile("conv_update_fla", time.perf_counter() - start)
            return y.transpose(1, 2).contiguous()
        except Exception as exc:
            start = time.perf_counter()
            _warn_once(
                "conv_update_fallback",
                f"Qwen3.5 NPU causal_conv1d_update FLA path failed; using torch fallback: {exc!r}",
            )
            out = _torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=bias, activation=activation)
            _record_profile("conv_update_torch", time.perf_counter() - start)
            return out

    return causal_conv1d_update


def _native_recurrent_gated_delta_rule(
    native_recurrent_gated_delta_rule: Callable[..., Any] | None,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if native_recurrent_gated_delta_rule is None or initial_state is None or beta is None:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None
    batch_size, seq_len, num_heads, head_dim = query.shape
    if seq_len > 8:
        return None
    if key.shape[:3] != query.shape[:3] or value.shape[:3] != query.shape[:3]:
        return None
    if initial_state.ndim != 4:
        return None

    value_dim = value.shape[-1]
    if initial_state.shape[0] < batch_size or initial_state.shape[1:] != (num_heads, head_dim, value_dim):
        return None

    query_3d = query.reshape(batch_size * seq_len, num_heads, head_dim)
    key_3d = key.reshape(batch_size * seq_len, num_heads, head_dim)
    value_3d = value.reshape(batch_size * seq_len, num_heads, value_dim)
    beta_2d = beta.reshape(batch_size * seq_len, num_heads).to(torch.bfloat16)
    g_2d = g.reshape(batch_size * seq_len, num_heads).to(torch.float32) if g is not None else None
    if use_qk_l2norm_in_kernel:
        query_3d = F.normalize(query_3d, p=2, dim=-1)
        key_3d = F.normalize(key_3d, p=2, dim=-1)

    state = initial_state[:batch_size].transpose(-1, -2).contiguous().to(torch.bfloat16)
    actual_seq_lengths = torch.full((batch_size,), seq_len, device=query.device, dtype=torch.int32)
    ssm_state_indices = torch.arange(batch_size, device=query.device, dtype=torch.int32).repeat_interleave(seq_len)
    num_accepted_tokens = actual_seq_lengths

    out = native_recurrent_gated_delta_rule(
        query_3d,
        key_3d,
        value_3d,
        state,
        beta=beta_2d,
        scale=head_dim**-0.5 if scale is None else scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        g=g_2d,
        gk=None,
    )
    out = out.reshape(batch_size, seq_len, num_heads, value_dim)
    final_state = state.transpose(-1, -2).contiguous().to(initial_state.dtype)
    if initial_state.shape[0] != batch_size:
        padded_state = initial_state.clone()
        padded_state[:batch_size].copy_(final_state)
        final_state = padded_state
    return out, final_state if output_final_state else None


def _make_chunk_gated_delta_rule(
    module: Any,
    fla_ops: dict[str, Callable[..., Any]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    torch_chunk_gated_delta_rule = module.torch_chunk_gated_delta_rule
    fla_chunk_gated_delta_rule = fla_ops["chunk_gated_delta_rule"]
    mindspeed_chunk_gated_delta_rule = fla_ops.get("mindspeed_chunk_gated_delta_rule")
    fla_recurrent_gated_delta_rule = fla_ops["fused_recurrent_gated_delta_rule"]
    default_fla_recurrent_chunk = not torch.__version__.startswith("2.9.")

    def chunk_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        nonlocal mindspeed_chunk_gated_delta_rule, fla_recurrent_gated_delta_rule
        if mindspeed_chunk_gated_delta_rule is not None and _env_enabled("MINDPIPE_QWEN35_NPU_USE_MINDSPEED_CHUNK", True):
            try:
                start = time.perf_counter()
                out = mindspeed_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu_seqlens,
                    **kwargs,
                )
                _record_profile("chunk_mindspeed", time.perf_counter() - start)
                return out
            except Exception as exc:
                _warn_once(
                    "chunk_mindspeed_fallback",
                    f"Qwen3.5 NPU MindSpeed chunk gated-delta failed; using next path: {exc!r}",
                )
                if _env_enabled("MINDPIPE_QWEN35_NPU_DISABLE_BROKEN_MINDSPEED_CHUNK", True):
                    mindspeed_chunk_gated_delta_rule = None
        if _env_enabled("MINDPIPE_QWEN35_NPU_USE_FLA_CHUNK", False):
            try:
                start = time.perf_counter()
                out = fla_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu_seqlens,
                    **kwargs,
                )
                _record_profile("chunk_fla", time.perf_counter() - start)
                return out
            except Exception as exc:
                _warn_once(
                    "chunk_fla_fallback",
                    f"Qwen3.5 NPU FLA chunk gated-delta failed; using recurrent path: {exc!r}",
                )
        if fla_recurrent_gated_delta_rule is None or not _env_enabled(
            "MINDPIPE_QWEN35_NPU_USE_FLA_RECURRENT_CHUNK", default_fla_recurrent_chunk
        ):
            start = time.perf_counter()
            out = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=int(kwargs.get("chunk_size", 64)),
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            _record_profile("chunk_torch", time.perf_counter() - start)
            return out
        try:
            start = time.perf_counter()
            out = fla_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
            _record_profile("chunk_recurrent_fla", time.perf_counter() - start)
            return out
        except Exception as exc:
            start = time.perf_counter()
            _warn_once(
                "chunk_recurrent_fallback",
                f"Qwen3.5 NPU recurrent gated-delta failed; using torch chunk fallback: {exc!r}",
            )
            if _env_enabled("MINDPIPE_QWEN35_NPU_DISABLE_BROKEN_FLA_RECURRENT_CHUNK", True):
                fla_recurrent_gated_delta_rule = None
            out = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=int(kwargs.get("chunk_size", 64)),
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            _record_profile("chunk_torch", time.perf_counter() - start)
            return out

    return chunk_gated_delta_rule


def _make_recurrent_gated_delta_rule(
    module: Any,
    fla_ops: dict[str, Callable[..., Any]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    torch_recurrent_gated_delta_rule = module.torch_recurrent_gated_delta_rule
    fla_recurrent_gated_delta_rule = fla_ops["fused_recurrent_gated_delta_rule"]
    native_recurrent_gated_delta_rule = _load_torch_npu_recurrent_gated_delta_rule()

    def recurrent_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _env_enabled("MINDPIPE_QWEN35_NPU_USE_NATIVE_RECURRENT", True):
            try:
                start = time.perf_counter()
                native_result = _native_recurrent_gated_delta_rule(
                    native_recurrent_gated_delta_rule,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    scale,
                    initial_state,
                    output_final_state,
                    use_qk_l2norm_in_kernel,
                )
                if native_result is not None:
                    _record_profile("recurrent_native", time.perf_counter() - start)
                    return native_result
            except Exception as exc:
                _warn_once(
                    "native_recurrent_fallback",
                    f"Qwen3.5 NPU native recurrent gated-delta failed; using FLA path: {exc!r}",
                )
        try:
            start = time.perf_counter()
            out = fla_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                gk=gk,
                gv=gv,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
            _record_profile("recurrent_fla", time.perf_counter() - start)
            return out
        except Exception as exc:
            start = time.perf_counter()
            _warn_once(
                "recurrent_fallback",
                f"Qwen3.5 NPU recurrent gated-delta FLA path failed; using torch fallback: {exc!r}",
            )
            out = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            _record_profile("recurrent_torch", time.perf_counter() - start)
            return out

    return recurrent_gated_delta_rule


def _make_qwen3_5_gdn_forward(
    original_forward: Callable[..., Any],
    module_globals: Any,
    packed_decode: Callable[..., Any],
    fused_sigmoid_update: Callable[..., Any] | None,
) -> Callable[..., torch.Tensor]:
    apply_mask_to_padding_states = module_globals.apply_mask_to_padding_states

    def qwen3_5_gdn_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not _env_enabled("MINDPIPE_QWEN35_NPU_VLLM_PACKED_DECODE", True)
            or hidden_states.device.type != "npu"
            or hidden_states.dtype not in {torch.float16, torch.bfloat16}
            or cache_params is None
            or hidden_states.ndim != 3
            or hidden_states.shape[1] != 1
            or not cache_params.has_previous_state(self.layer_idx)
        ):
            return original_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)

        try:
            start = time.perf_counter()
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
            batch_size, seq_len, _ = hidden_states.shape
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

            mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
            z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)

            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            if (
                fused_sigmoid_update is not None
                and _env_enabled("MINDPIPE_QWEN35_NPU_VLLM_ASCEND_FUSED_SIGMOID_UPDATE", True)
            ):
                mixed_qkv_t = mixed_qkv.transpose(1, 2).contiguous()
                query, key, value = torch.split(mixed_qkv_t, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim).contiguous()
                key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim).contiguous()
                value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim).contiguous()
                state_indices = torch.arange(batch_size, device=hidden_states.device, dtype=torch.int32)
                cu_seqlens = torch.arange(batch_size + 1, device=hidden_states.device, dtype=torch.int32)
                core_attn_out = fused_sigmoid_update(
                    A_log=self.A_log.contiguous(),
                    a=a.reshape(batch_size * seq_len, self.num_v_heads).contiguous(),
                    dt_bias=self.dt_bias.contiguous(),
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query,
                    k=key,
                    v=value,
                    b=b.reshape(batch_size * seq_len, self.num_v_heads).contiguous(),
                    initial_state_source=recurrent_state,
                    initial_state_indices=state_indices,
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                )
                core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
                z = z.reshape(-1, self.head_v_dim)
                core_attn_out = self.norm(core_attn_out, z)
                core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
                _record_profile("gdn_forward_vllm_ascend_fused_sigmoid", time.perf_counter() - start)
                return self.out_proj(core_attn_out)

            packed_qkv = mixed_qkv.transpose(1, 2).reshape(batch_size, -1).contiguous()
            cache_id = id(cache_params)
            state_v_k = getattr(self, "_mindpipe_recurrent_state_v_k", None)
            if (
                not _env_enabled("MINDPIPE_QWEN35_NPU_KEEP_VLLM_STATE_LAYOUT", False)
                or getattr(self, "_mindpipe_recurrent_state_cache_id", None) != cache_id
                or state_v_k is None
                or state_v_k.shape != (batch_size, self.num_v_heads, self.head_v_dim, self.head_k_dim)
                or state_v_k.device != hidden_states.device
            ):
                state_v_k = recurrent_state.transpose(-1, -2).contiguous()
            core_attn_out = hidden_states.new_empty(batch_size, 1, self.num_v_heads, self.head_v_dim)
            state_indices = torch.arange(batch_size, device=hidden_states.device, dtype=torch.int32)
            core_attn_out, state_v_k = packed_decode(
                packed_qkv,
                a.reshape(batch_size, self.num_v_heads).contiguous(),
                b.reshape(batch_size, self.num_v_heads).contiguous(),
                self.A_log.contiguous(),
                self.dt_bias.contiguous(),
                self.head_k_dim**-0.5,
                state_v_k,
                core_attn_out,
                state_indices,
                True,
            )
            if _env_enabled("MINDPIPE_QWEN35_NPU_KEEP_VLLM_STATE_LAYOUT", False):
                self._mindpipe_recurrent_state_cache_id = cache_id
                self._mindpipe_recurrent_state_v_k = state_v_k
            else:
                cache_params.update_recurrent_state(state_v_k.transpose(-1, -2).contiguous(), self.layer_idx)

            core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
            _record_profile("gdn_forward_vllm_packed_decode", time.perf_counter() - start)
            return self.out_proj(core_attn_out)
        except Exception as exc:
            _warn_once(
                "gdn_forward_vllm_packed_decode_fallback",
                f"Qwen3.5 NPU vLLM packed decode forward failed; using HF forward: {exc!r}",
            )
            return original_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)

    return qwen3_5_gdn_forward


def _patch_qwen3_5_gdn_forward(module: Any) -> bool:
    if not _env_enabled("MINDPIPE_QWEN35_NPU_VLLM_PACKED_DECODE", True):
        return False
    gdn_cls = getattr(module, "Qwen3_5GatedDeltaNet", None)
    if gdn_cls is None:
        return False
    if getattr(gdn_cls.forward, "_mindpipe_qwen3_5_vllm_packed_decode", False):
        return True
    packed_decode = _load_vllm_packed_decode()
    if packed_decode is None:
        return False
    fused_sigmoid_update = _load_vllm_ascend_fused_sigmoid_update()
    original_forward = gdn_cls.forward
    patched_forward = _make_qwen3_5_gdn_forward(original_forward, module, packed_decode, fused_sigmoid_update)
    patched_forward._mindpipe_qwen3_5_vllm_packed_decode = True  # type: ignore[attr-defined]
    patched_forward._mindpipe_original_forward = original_forward  # type: ignore[attr-defined]
    gdn_cls.forward = patched_forward
    return True


def _make_npu_attention_forward(fallback: Callable[..., Any]) -> Callable[..., tuple[torch.Tensor, None]]:
    npu_fusion_attention, npu_incre_flash_attention = _load_torch_npu_attention_ops()

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

    def npu_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if (
            query.device.type != "npu"
            or query.dtype not in {torch.float16, torch.bfloat16}
            or dropout not in {0, 0.0}
            or kwargs.get("output_attentions", False)
        ):
            return fallback(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        scale = query.shape[-1] ** -0.5 if scaling is None else scaling
        num_heads = query.shape[1]
        num_key_value_heads = key.shape[1]
        batch_size, _, query_len, head_dim = query.shape
        key_len = key.shape[2]

        try:
            if query_len == 1 and npu_incre_flash_attention is not None:
                q_bsh = query.transpose(1, 2).reshape(batch_size, query_len, num_heads * head_dim).contiguous()
                k_bsh = key.transpose(1, 2).reshape(batch_size, key_len, num_key_value_heads * head_dim).contiguous()
                v_bsh = value.transpose(1, 2).reshape(batch_size, key_len, num_key_value_heads * value.shape[-1]).contiguous()
                actual_seq_lengths = [key_len] * batch_size
                out = npu_incre_flash_attention(
                    q_bsh,
                    k_bsh,
                    v_bsh,
                    atten_mask=None,
                    actual_seq_lengths=actual_seq_lengths,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    input_layout="BSH",
                    scale_value=scale,
                )
                out = out.reshape(batch_size, query_len, num_heads, value.shape[-1]).contiguous()
                return out, None

            if npu_fusion_attention is not None:
                if num_key_value_heads != num_heads:
                    key = repeat_kv(key, num_heads // num_key_value_heads)
                    value = repeat_kv(value, num_heads // num_key_value_heads)
                q_bsnd = query.transpose(1, 2).contiguous()
                k_bsnd = key.transpose(1, 2).contiguous()
                v_bsnd = value.transpose(1, 2).contiguous()
                atten_mask = attention_mask
                if atten_mask is not None and atten_mask.dtype != torch.bool:
                    atten_mask = torch.logical_not(atten_mask.bool()).to(query.device)
                out = npu_fusion_attention(
                    q_bsnd,
                    k_bsnd,
                    v_bsnd,
                    num_heads,
                    "BSND",
                    atten_mask=atten_mask,
                    scale=scale,
                    keep_prob=1.0,
                    next_tockens=0,
                    sparse_mode=0,
                )[0]
                return out.contiguous(), None
        except Exception as exc:
            _warn_once("npu_attention_fallback", f"Qwen3.5 NPU attention op failed; using fallback: {exc!r}")

        return fallback(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

    return npu_attention_forward


def _make_npu_qwen3_5_rmsnorm_cls(original_cls: type[nn.Module]) -> type[nn.Module] | None:
    npu_rms_norm, _, _ = _load_torch_npu_dense_ops()
    if npu_rms_norm is None:
        return None

    class NpuQwen3_5RMSNorm(original_cls):  # type: ignore[misc, valid-type]
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.device.type != "npu" or x.dtype not in {torch.float16, torch.bfloat16}:
                return super().forward(x)
            scale = (1.0 + self.weight).to(dtype=x.dtype)
            return npu_rms_norm(x, scale, epsilon=self.eps)[0]

    NpuQwen3_5RMSNorm.__name__ = original_cls.__name__
    return NpuQwen3_5RMSNorm


def _make_npu_apply_rotary_pos_emb(fallback: Callable[..., Any]) -> Callable[..., Any]:
    _, npu_rotary_mul, _ = _load_torch_npu_dense_ops()

    def npu_apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if npu_rotary_mul is None or q.device.type != "npu" or k.device.type != "npu":
            return fallback(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        if isinstance(position_ids, int) and unsqueeze_dim == 1:
            unsqueeze_dim = position_ids
        try:
            cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
            sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)
            rotary_dim = cos_unsqueezed.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_embed = npu_rotary_mul(q_rot, cos_unsqueezed, sin_unsqueezed)
            k_embed = npu_rotary_mul(k_rot, cos_unsqueezed, sin_unsqueezed)
            if q_pass.numel() > 0:
                q_embed = torch.cat([q_embed, q_pass], dim=-1)
            if k_pass.numel() > 0:
                k_embed = torch.cat([k_embed, k_pass], dim=-1)
            return q_embed, k_embed
        except Exception as exc:
            _warn_once("npu_rotary_fallback", f"Qwen3.5 NPU rotary op failed; using fallback: {exc!r}")
            return fallback(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    return npu_apply_rotary_pos_emb


def _make_npu_swiglu_forward(fallback: Callable[..., Any]) -> Callable[..., Any]:
    _, _, npu_swiglu = _load_torch_npu_dense_ops()

    def npu_swiglu_forward(self: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
        if npu_swiglu is None or hidden_state.device.type != "npu" or hidden_state.dtype not in {torch.float16, torch.bfloat16}:
            return fallback(self, hidden_state)
        try:
            gate = self.gate_proj(hidden_state)
            up = self.up_proj(hidden_state)
            return self.down_proj(npu_swiglu(torch.cat((gate, up), dim=-1), dim=-1))
        except Exception as exc:
            _warn_once("npu_swiglu_fallback", f"Qwen3.5 NPU SwiGLU op failed; using fallback: {exc!r}")
            return fallback(self, hidden_state)

    return npu_swiglu_forward


def _patch_qwen3_5_dense_ops(module: Any) -> bool:
    patched = False

    rmsnorm_cls = getattr(module, "Qwen3_5RMSNorm", None) or getattr(module, "Qwen3_5MoeRMSNorm", None)
    if rmsnorm_cls is not None and not getattr(rmsnorm_cls, "_mindpipe_qwen3_5_npu_rmsnorm", False):
        npu_rmsnorm_cls = _make_npu_qwen3_5_rmsnorm_cls(rmsnorm_cls)
        if npu_rmsnorm_cls is not None:
            npu_rmsnorm_cls._mindpipe_qwen3_5_npu_rmsnorm = True  # type: ignore[attr-defined]
            setattr(module, rmsnorm_cls.__name__, npu_rmsnorm_cls)
            patched = True

    apply_rotary = getattr(module, "apply_rotary_pos_emb", None)
    if apply_rotary is not None and not getattr(apply_rotary, "_mindpipe_qwen3_5_npu_rotary", False):
        patched_rotary = _make_npu_apply_rotary_pos_emb(apply_rotary)
        patched_rotary._mindpipe_qwen3_5_npu_rotary = True  # type: ignore[attr-defined]
        setattr(module, "apply_rotary_pos_emb", patched_rotary)
        patched = True

    for mlp_name in ("Qwen3_5MLP", "Qwen3_5MoeMLP"):
        mlp_cls = getattr(module, mlp_name, None)
        if mlp_cls is None or getattr(mlp_cls.forward, "_mindpipe_qwen3_5_npu_swiglu", False):
            continue
        original_forward = mlp_cls.forward
        patched_forward = _make_npu_swiglu_forward(original_forward)
        patched_forward._mindpipe_qwen3_5_npu_swiglu = True  # type: ignore[attr-defined]
        patched_forward._mindpipe_original_forward = original_forward  # type: ignore[attr-defined]
        mlp_cls.forward = patched_forward
        patched = True

    return patched


def _make_npu_qwen3_5_moe_experts_forward(fallback: Callable[..., Any]) -> Callable[..., torch.Tensor]:
    grouped_matmul = _load_torch_npu_grouped_matmul()

    def npu_qwen3_5_moe_experts_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            grouped_matmul is None
            or not _env_enabled("MINDPIPE_QWEN35_NPU_MOE_GROUPED_MATMUL", False)
            or hidden_states.device.type != "npu"
            or hidden_states.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or hidden_states.ndim != 2
            or top_k_index.ndim != 2
        ):
            return fallback(self, hidden_states, top_k_index, top_k_weights)

        try:
            start = time.perf_counter()
            num_experts = int(getattr(self, "num_experts"))
            final_hidden_states = torch.zeros_like(hidden_states)
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            if expert_hit.numel() == 0:
                _record_profile("moe_grouped_empty", time.perf_counter() - start)
                return final_hidden_states
            if expert_hit.numel() > 128:
                return fallback(self, hidden_states, top_k_index, top_k_weights)

            current_states: list[torch.Tensor] = []
            gate_up_weights: list[torch.Tensor] = []
            down_weights: list[torch.Tensor] = []
            token_indices: list[torch.Tensor] = []
            top_k_positions: list[torch.Tensor] = []

            for expert_index_tensor in expert_hit:
                expert_index = expert_index_tensor[0]
                top_k_pos, token_index = torch.where(expert_mask[expert_index])
                if token_index.numel() == 0:
                    continue
                current_states.append(hidden_states[token_index])
                gate_up_weights.append(self.gate_up_proj[expert_index].transpose(0, 1))
                down_weights.append(self.down_proj[expert_index].transpose(0, 1))
                token_indices.append(token_index)
                top_k_positions.append(top_k_pos)

            if not current_states:
                _record_profile("moe_grouped_empty", time.perf_counter() - start)
                return final_hidden_states

            gate_up_outputs = grouped_matmul(current_states, gate_up_weights, group_type=-1)
            intermediate_states = []
            for gate_up_output in gate_up_outputs:
                gate, up = gate_up_output.chunk(2, dim=-1)
                intermediate_states.append(self.act_fn(gate) * up)
            down_outputs = grouped_matmul(intermediate_states, down_weights, group_type=-1)

            for token_index, top_k_pos, down_output in zip(token_indices, top_k_positions, down_outputs):
                weighted_output = down_output * top_k_weights[token_index, top_k_pos, None]
                final_hidden_states.index_add_(0, token_index, weighted_output.to(final_hidden_states.dtype))

            _record_profile("moe_grouped", time.perf_counter() - start)
            return final_hidden_states
        except Exception as exc:
            _warn_once("moe_grouped_fallback", f"Qwen3.5 NPU grouped MoE experts failed; using fallback: {exc!r}")
            return fallback(self, hidden_states, top_k_index, top_k_weights)

    return npu_qwen3_5_moe_experts_forward


def _patch_qwen3_5_moe_experts(module: Any) -> bool:
    experts_cls = getattr(module, "Qwen3_5MoeExperts", None)
    if experts_cls is None:
        return False
    if getattr(experts_cls.forward, "_mindpipe_qwen3_5_npu_moe_grouped", False):
        return True
    original_forward = experts_cls.forward
    patched_forward = _make_npu_qwen3_5_moe_experts_forward(original_forward)
    patched_forward._mindpipe_qwen3_5_npu_moe_grouped = True  # type: ignore[attr-defined]
    patched_forward._mindpipe_original_forward = original_forward  # type: ignore[attr-defined]
    experts_cls.forward = patched_forward
    return True


def _patch_modeling_module(module_name: str, fla_ops: dict[str, Callable[..., Any]]) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False

    module.causal_conv1d_fn = _make_causal_conv1d_fn(fla_ops)
    module.causal_conv1d_update = _make_causal_conv1d_update(fla_ops)
    module.chunk_gated_delta_rule = _make_chunk_gated_delta_rule(module, fla_ops)
    module.fused_recurrent_gated_delta_rule = _make_recurrent_gated_delta_rule(module, fla_ops)
    module.FusedRMSNormGated = None
    module.is_fast_path_available = True
    _patch_qwen3_5_dense_ops(module)
    _patch_qwen3_5_gdn_forward(module)
    _patch_qwen3_5_moe_experts(module)
    module._mindpipe_qwen3_5_npu_linear_attn_patch = True
    return True


def _patch_attention_registry() -> bool:
    if not _env_enabled("MINDPIPE_QWEN35_NPU_FULL_ATTN", True):
        return False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return False
    for name in ("sdpa", "flash_attention_2"):
        fallback = ALL_ATTENTION_FUNCTIONS.get(name)
        if fallback is not None and not getattr(fallback, "_mindpipe_qwen3_5_npu_attention", False):
            patched = _make_npu_attention_forward(fallback)
            patched._mindpipe_qwen3_5_npu_attention = True  # type: ignore[attr-defined]
            ALL_ATTENTION_FUNCTIONS[name] = patched
    return True


def maybe_enable_qwen3_5_npu_linear_attention(config: Any | None = None) -> dict[str, Any]:
    """Patch HF Qwen3.5 linear-attention globals before model instantiation."""

    result: dict[str, Any] = {"enabled": False, "reason": None, "patched_modules": []}
    if not _env_enabled("MINDPIPE_QWEN35_NPU_LINEAR_ATTN", True):
        result["reason"] = "disabled_by_env"
        return result
    if not _is_qwen3_5_config(config):
        result["reason"] = "not_qwen3_5"
        return result
    if not _npu_available():
        result["reason"] = "npu_unavailable"
        return result

    fla_ops = _load_fla_ops()
    if fla_ops is None:
        result["reason"] = "fla_unavailable"
        return result

    for module_name in (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    ):
        if _patch_modeling_module(module_name, fla_ops):
            result["patched_modules"].append(module_name)

    result["patched_attention_registry"] = _patch_attention_registry()
    result["enabled"] = bool(result["patched_modules"])
    result["reason"] = "patched" if result["enabled"] else "no_qwen3_5_module"
    return result
