from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from algorithm.common.device import empty_cache
from algorithm.common.device import resolve_device
from algorithm.common.modeling import get_text_backbone
from algorithm.common.modeling import unwrap_layer_output
from algorithm.common.modeling import move_tensors_to_device
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask as create_qwen3_5_causal_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask as create_qwen3_5_moe_causal_mask

from omniquant.calibration import run_omniquant_calibration_forward
from omniquant.model_tools.llama_utils import QuantLlamaDecoderLayer
from omniquant.model_tools.llama_utils import initialize_omni_parameters as initialize_llama_omni_parameters
from omniquant.model_tools.minicpm_utils import QuantMiniCPMDecoderLayer
from omniquant.model_tools.minicpm_utils import initialize_omni_parameters as initialize_minicpm_omni_parameters
from omniquant.model_tools.qwen3_utils import QuantQwen3DecoderLayer
from omniquant.model_tools.qwen3_utils import QuantQwen3MoeDecoderLayer
from omniquant.model_tools.qwen3_utils import QuantQwen3_5DecoderLayer
from omniquant.model_tools.qwen3_utils import QuantQwen3_5MoeDecoderLayer
from omniquant.model_tools.qwen3_utils import initialize_qwen3_5_omni_parameters
from omniquant.model_tools.qwen3_utils import initialize_qwen3_5_moe_omni_parameters
from omniquant.model_tools.qwen3_utils import initialize_qwen3_omni_parameters
from omniquant.model_tools.qwen_utils import QuantQwenDecoderLayer
from omniquant.model_tools.qwen_utils import initialize_omni_parameters as initialize_qwen_omni_parameters
from omniquant.utils import ampscaler_get_grad_norm
from omniquant.utils import clear_temp_variable
from omniquant.utils import get_named_linears
from omniquant.utils import get_named_packed_moe_experts
from omniquant.utils import get_omni_parameters
from omniquant.utils import let_parameters
from omniquant.utils import lwc_parameters
from omniquant.utils import omni_state_dict
from omniquant.utils import register_scales_and_zeros
from omniquant.utils import set_quant_state
from omniquant.utils import smooth_and_quant_inplace
from omniquant.utils import smooth_and_quant_temporary


def _cast_fp_tensors(value, dtype):
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_cast_fp_tensors(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_fp_tensors(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_fp_tensors(item, dtype) for key, item in value.items()}
    return value


def _expand_for_batch(value, batch_size: int):
    if torch.is_tensor(value):
        if value.dim() > 0 and value.shape[0] == 1 and batch_size > 1:
            repeat_dims = [batch_size] + [1] * (value.dim() - 1)
            return value.repeat(*repeat_dims)
        if value.dim() > 1 and value.shape[1] == 1 and value.shape[0] in (2, 3) and batch_size > 1:
            repeat_dims = [1] * value.dim()
            repeat_dims[1] = batch_size
            return value.repeat(*repeat_dims)
        return value
    if isinstance(value, tuple):
        return tuple(_expand_for_batch(item, batch_size) for item in value)
    if isinstance(value, list):
        return [_expand_for_batch(item, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: _expand_for_batch(item, batch_size) for key, item in value.items()}
    return value


def _autocast_context(enabled: bool, device_type: str, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def _max_memory_allocated_mib(device: torch.device) -> float | None:
    if device.type == "cuda" and hasattr(torch.cuda, "max_memory_allocated"):
        return float(torch.cuda.max_memory_allocated(device) / 1024**2)
    if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "max_memory_allocated"):
        return float(torch.npu.max_memory_allocated(device) / 1024**2)
    return None


@torch.no_grad()
def _capture_first_block_inputs(model, backbone, calibration_batches, device):
    device = resolve_device(device)
    decoder_config = backbone.decoder_config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False
    blocks = backbone.layers
    # device_map 模式下不手动移动 front modules 和 blocks[0]

    dtype = next(iter(model.parameters())).dtype
    sample_count = len(calibration_batches)
    sequence_length = calibration_batches[0][0].shape[1]
    # device_map 模式下输入放到 blocks[0] 所在设备
    block0_device = next(blocks[0].parameters()).device
    inputs = torch.zeros(
        sample_count,
        sequence_length,
        backbone.hidden_size,
        dtype=dtype,
        device=block0_device,
    )
    cached_kwargs: dict[str, object] = {}
    input_index = 0

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, hidden_states, **kwargs):
            nonlocal input_index
            inputs[input_index] = hidden_states
            input_index += 1
            cached_kwargs.clear()
            cached_kwargs.update(kwargs)
            raise ValueError

    blocks[0] = Catcher(blocks[0])
    for token_ids, _labels in calibration_batches:
        try:
            run_omniquant_calibration_forward(model, token_ids.to(block0_device))
        except ValueError:
            pass

    blocks[0] = blocks[0].module
    # device_map 模式下不手动移动到 cpu
    empty_cache(device)
    decoder_config.use_cache = use_cache
    return inputs, dict(cached_kwargs)


def _resolve_omniquant_impl(model_type: str):
    if model_type == "llama":
        return QuantLlamaDecoderLayer, initialize_llama_omni_parameters
    if model_type in {"qwen2", "qwen2_5_vl"}:
        return QuantQwenDecoderLayer, initialize_qwen_omni_parameters
    if model_type in {"qwen3", "qwen3_vl"}:
        return QuantQwen3DecoderLayer, initialize_qwen3_omni_parameters
    if model_type == "qwen3_moe":
        return QuantQwen3MoeDecoderLayer, initialize_qwen3_5_moe_omni_parameters
    if model_type == "qwen3_5":
        return QuantQwen3_5DecoderLayer, initialize_qwen3_5_omni_parameters
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return QuantQwen3_5MoeDecoderLayer, initialize_qwen3_5_moe_omni_parameters
    if model_type in {"minicpm", "minicpmv"}:
        return QuantMiniCPMDecoderLayer, initialize_minicpm_omni_parameters
    raise NotImplementedError(
        f"OmniQuant currently supports LLaMA-, Qwen-, and MiniCPM-style decoders only; got model_type={model_type!r}."
    )


def _build_qwen3_5_layer_kwargs(
    backbone,
    inputs: torch.Tensor,
    layer_kwargs: dict[str, object],
    *,
    model_type: str,
) -> dict[str, dict[str, object]]:
    position_ids = layer_kwargs.get("position_ids")
    if position_ids is None:
        seq_len = inputs.shape[1]
        position_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)

    full_attention_kwargs = dict(layer_kwargs)
    create_causal_mask = (
        create_qwen3_5_moe_causal_mask
        if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}
        else create_qwen3_5_causal_mask
    )
    full_attention_kwargs["attention_mask"] = create_causal_mask(
        config=backbone.decoder_config,
        inputs_embeds=inputs[:1],
        attention_mask=torch.ones((1, inputs.shape[1]), dtype=torch.long, device=inputs.device),
        past_key_values=full_attention_kwargs.get("past_key_values"),
        position_ids=position_ids,
    )

    linear_attention_kwargs = dict(layer_kwargs)
    linear_attention_kwargs["attention_mask"] = None
    return {
        "full_attention": full_attention_kwargs,
        "linear_attention": linear_attention_kwargs,
    }


def _tensor_mse(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(lhs.float(), rhs.float()).detach().cpu())


def _tensor_summary(name: str, tensor: torch.Tensor) -> dict[str, object]:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = int(detached.numel())
    summary: dict[str, object] = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "total_count": total_count,
        "finite_count": finite_count,
        "nonfinite_count": total_count - finite_count,
    }
    if detached.is_floating_point() or detached.is_complex():
        summary["nan_count"] = int(torch.isnan(detached).sum().item())
        summary["inf_count"] = int(torch.isinf(detached).sum().item())
    else:
        summary["nan_count"] = 0
        summary["inf_count"] = 0
    if finite_count > 0:
        finite_values = detached[finite_mask].float()
        summary.update(
            {
                "finite_min": float(finite_values.min().item()),
                "finite_max": float(finite_values.max().item()),
                "finite_mean": float(finite_values.mean().item()),
                "finite_abs_max": float(finite_values.abs().max().item()),
            }
        )
    return summary


def _value_summary(name: str, value):
    if torch.is_tensor(value):
        return _tensor_summary(name, value)
    if isinstance(value, dict):
        return {key: _value_summary(f"{name}.{key}", item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_value_summary(f"{name}[{index}]", item) for index, item in enumerate(value)]
    if isinstance(value, list):
        return [_value_summary(f"{name}[{index}]", item) for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _omni_parameter_summaries(model, use_shift: bool) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    template = "smooth" if use_shift else "smooth_scale"
    for name, parameter in model.named_parameters():
        if "bound_factor" not in name and template not in name:
            continue
        summaries[name] = _tensor_summary(name, parameter)
    return summaries


def _write_nonfinite_diagnostics(
    output_dir: str | Path,
    *,
    layer_index: int,
    phase: str,
    payload: dict[str, object],
    epoch: int | None = None,
    batch_start: int | None = None,
    batch_end: int | None = None,
) -> Path:
    filename_parts = [f"nonfinite_layer{layer_index}", phase]
    if epoch is not None:
        filename_parts.append(f"epoch{epoch}")
    if batch_start is not None and batch_end is not None:
        filename_parts.append(f"batch{batch_start}_{batch_end}")
    diagnostics_path = Path(output_dir) / ("_".join(filename_parts) + ".json")
    diagnostics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return diagnostics_path


def _raise_nonfinite_error(
    output_dir: str | Path,
    *,
    layer_index: int,
    phase: str,
    message: str,
    payload: dict[str, object],
    epoch: int | None = None,
    batch_start: int | None = None,
    batch_end: int | None = None,
):
    diagnostics_path = _write_nonfinite_diagnostics(
        output_dir,
        layer_index=layer_index,
        phase=phase,
        payload=payload,
        epoch=epoch,
        batch_start=batch_start,
        batch_end=batch_end,
    )
    raise RuntimeError(f"{message} Diagnostics saved to {diagnostics_path}.")


def _collect_layer_outputs(layer, inputs, layer_kwargs, autocast_enabled: bool, batch_size: int):
    outputs = torch.zeros_like(inputs)
    model_dtype = next(iter(layer.parameters())).dtype
    device_type = inputs.device.type
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            batch_kwargs = _expand_for_batch(layer_kwargs, end - start)
            with _autocast_context(autocast_enabled, device_type, model_dtype):
                outputs[start:end] = unwrap_layer_output(layer(inputs[start:end], **batch_kwargs))
    return outputs


def _collect_temporary_quant_outputs(layer, args, inputs, layer_kwargs, autocast_enabled: bool, batch_size: int):
    outputs = torch.zeros_like(inputs)
    model_dtype = next(iter(layer.parameters())).dtype
    device_type = inputs.device.type
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            batch_kwargs = _expand_for_batch(layer_kwargs, end - start)
            with _autocast_context(autocast_enabled, device_type, model_dtype):
                smooth_and_quant_temporary(layer, args)
                outputs[start:end] = unwrap_layer_output(layer(inputs[start:end], **batch_kwargs))
            clear_temp_variable(layer)
    return outputs


def cali_omni_quant(args, model, calibration_batches, device, act_scales, act_shifts, logger):
    device = resolve_device(device)
    backbone = get_text_backbone(model)
    model_type = getattr(model.config, "model_type", None)
    quant_decoder_layer_cls, initialize_omni_parameters = _resolve_omniquant_impl(model_type)
    model_dtype = next(model.parameters()).dtype
    device_type = device.type
    quantized_linear_artifacts: dict[str, dict[str, object]] = {}
    diagnostics_enabled = bool(getattr(args, "save_diagnostics", False))
    diagnostics_path = Path(args.output_dir) / "layer_diagnostics.json"
    diagnostics_records: list[dict[str, object]] = []
    input_states, layer_kwargs = _capture_first_block_inputs(
        model=model,
        backbone=backbone,
        calibration_batches=calibration_batches,
        device=device,
    )
    layer_kwargs_by_type = None
    if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"}:
        layer_kwargs_by_type = _build_qwen3_5_layer_kwargs(
            backbone,
            input_states,
            layer_kwargs,
            model_type=model_type,
        )
    if args.deactive_amp:
        input_states = input_states.float()
        layer_kwargs = _cast_fp_tensors(layer_kwargs, torch.float32)
        if layer_kwargs_by_type is not None:
            layer_kwargs_by_type = {
                layer_type: _cast_fp_tensors(kwargs, torch.float32)
                for layer_type, kwargs in layer_kwargs_by_type.items()
            }

    quant_inps = input_states
    fp_inps = input_states.clone()
    fp_inps_2 = input_states.clone() if args.aug_loss else None
    autocast_enabled = not args.deactive_amp
    mse_loss = torch.nn.MSELoss()

    resume_parameters = None
    if args.resume:
        resume_parameters = torch.load(args.resume, map_location="cpu")

    saved_parameters: dict[int, dict[str, torch.Tensor]] = {}
    parameter_checkpoint_path = Path(args.output_dir) / "omni_parameters.pth"

    for layer_index, layer in enumerate(backbone.layers):
        layer_start_time = time.perf_counter()
        logger.info("========= Layer %s =========", layer_index)
        active_layer_kwargs = layer_kwargs
        if layer_kwargs_by_type is not None:
            active_layer_kwargs = layer_kwargs_by_type.get(getattr(layer, "layer_type", None), layer_kwargs)
        if not torch.isfinite(quant_inps).all():
            _raise_nonfinite_error(
                args.output_dir,
                layer_index=layer_index,
                phase="layer_input",
                message=f"OmniQuant received non-finite quantized inputs at layer {layer_index}.",
                payload={
                    "layer_index": layer_index,
                    "phase": "layer_input",
                    "autocast_enabled": autocast_enabled,
                    "quant_inps": _tensor_summary("quant_inps", quant_inps),
                    "fp_inps": _tensor_summary("fp_inps", fp_inps),
                    "layer_kwargs": _value_summary("layer_kwargs", active_layer_kwargs),
                },
            )
        # device_map 模式下不手动移动 layer，在当前设备上操作
        # 将输入数据移到当前层设备
        layer_device = next(layer.parameters()).device
        fp_inps = fp_inps.to(layer_device)
        quant_inps = quant_inps.to(layer_device)
        active_layer_kwargs = move_tensors_to_device(active_layer_kwargs, layer_device)
        qlayer = quant_decoder_layer_cls(layer, args)
        # 将 wrapper 及其新建的可学习参数移到当前层设备
        qlayer = qlayer.to(layer_device)
        if args.deactive_amp:
            qlayer = qlayer.float()
        qlayer.eval()
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        layer_diagnostics: dict[str, object] | None = None
        if diagnostics_enabled:
            layer_diagnostics = {
                "layer_index": layer_index,
                "input_mse": _tensor_mse(fp_inps, quant_inps),
            }

        fp_outs = None
        fp_outs_2 = None
        if args.epochs > 0 or diagnostics_enabled:
            fp_outs = _collect_layer_outputs(
                qlayer,
                fp_inps,
                active_layer_kwargs,
                autocast_enabled=autocast_enabled,
                batch_size=int(args.batch_size),
            )
            if args.aug_loss:
                fp_outs_2 = _collect_layer_outputs(
                    qlayer,
                    quant_inps,
                    active_layer_kwargs,
                    autocast_enabled=autocast_enabled,
                    batch_size=int(args.batch_size),
                )

        use_shift = bool(getattr(args, "use_shift", False))
        initialize_omni_parameters(
            qlayer,
            f"{backbone.prefix}.layers.{layer_index}",
            args,
            act_scales,
            act_shifts,
            use_shift=use_shift,
        )
        if resume_parameters is not None:
            qlayer.load_state_dict(resume_parameters[layer_index], strict=False)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)

        if args.epochs > 0:
            qlayer = qlayer.float()
            if diagnostics_enabled and layer_diagnostics is not None and fp_outs is not None:
                pre_train_outputs = _collect_temporary_quant_outputs(
                    qlayer,
                    args,
                    quant_inps,
                    active_layer_kwargs,
                    autocast_enabled=autocast_enabled,
                    batch_size=int(args.batch_size),
                )
                layer_diagnostics["pre_train_temporary_mse"] = _tensor_mse(fp_outs, pre_train_outputs)
            parameter_groups = []
            if args.let:
                params = list(let_parameters(qlayer, use_shift=use_shift))
                if params:
                    parameter_groups.append({"params": params, "lr": args.let_lr})
            if args.lwc:
                params = list(lwc_parameters(qlayer))
                if params:
                    parameter_groups.append({"params": params, "lr": args.lwc_lr})
            if not parameter_groups:
                raise ValueError("No OmniQuant trainable parameters were created for layer calibration.")

            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.wd)
            use_grad_scaler = bool(autocast_enabled and device_type == "cuda")
            grad_scaler = torch.cuda.amp.GradScaler(enabled=True) if use_grad_scaler else None

            for epoch in range(args.epochs):
                iter_start_time = time.perf_counter()
                loss_values = []
                norm_values = []
                for start in range(0, args.nsamples, args.batch_size):
                    end = min(start + args.batch_size, args.nsamples)
                    batch_kwargs = _expand_for_batch(active_layer_kwargs, end - start)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast_context(autocast_enabled, device_type, model_dtype):
                        smooth_and_quant_temporary(qlayer, args)
                        quant_out = unwrap_layer_output(qlayer(quant_inps[start:end], **batch_kwargs))
                        loss = mse_loss(fp_outs[start:end], quant_out)
                        if args.aug_loss and fp_outs_2 is not None:
                            loss = loss + mse_loss(fp_outs_2[start:end], quant_out)
                    if not torch.isfinite(quant_out).all():
                        _raise_nonfinite_error(
                            args.output_dir,
                            layer_index=layer_index,
                            phase="temporary_forward",
                            message=f"OmniQuant temporary forward produced non-finite outputs at layer {layer_index}.",
                            payload={
                                "layer_index": layer_index,
                                "phase": "temporary_forward",
                                "epoch": epoch,
                                "batch_range": [start, end],
                                "autocast_enabled": autocast_enabled,
                                "quant_inps": _tensor_summary("quant_inps", quant_inps[start:end]),
                                "fp_outs": _tensor_summary("fp_outs", fp_outs[start:end]),
                                "quant_out": _tensor_summary("quant_out", quant_out),
                                "layer_kwargs": _value_summary("layer_kwargs", batch_kwargs),
                                "trainable_parameters": _omni_parameter_summaries(qlayer, use_shift=use_shift),
                            },
                            epoch=epoch,
                            batch_start=start,
                            batch_end=end,
                        )
                    if not math.isfinite(loss.item()):
                        _raise_nonfinite_error(
                            args.output_dir,
                            layer_index=layer_index,
                            phase="loss",
                            message=f"OmniQuant loss became non-finite at layer {layer_index}, epoch {epoch}.",
                            payload={
                                "layer_index": layer_index,
                                "phase": "loss",
                                "epoch": epoch,
                                "batch_range": [start, end],
                                "autocast_enabled": autocast_enabled,
                                "loss": float(loss.detach().float().cpu().item()),
                                "quant_inps": _tensor_summary("quant_inps", quant_inps[start:end]),
                                "fp_outs": _tensor_summary("fp_outs", fp_outs[start:end]),
                                "quant_out": _tensor_summary("quant_out", quant_out),
                                "layer_kwargs": _value_summary("layer_kwargs", batch_kwargs),
                                "trainable_parameters": _omni_parameter_summaries(qlayer, use_shift=use_shift),
                            },
                            epoch=epoch,
                            batch_start=start,
                            batch_end=end,
                        )
                    if use_grad_scaler and grad_scaler is not None:
                        grad_scaler.scale(loss).backward()
                        grad_scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    grad_norm = ampscaler_get_grad_norm(get_omni_parameters(qlayer, use_shift=use_shift)).detach().cpu()
                    if use_grad_scaler and grad_scaler is not None:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    loss_values.append(loss.detach().cpu())
                    norm_values.append(grad_norm)
                max_memory_mib = _max_memory_allocated_mib(device)
                memory_fragment = (
                    f" max_memory_allocated {max_memory_mib:.2f} MiB"
                    if max_memory_mib is not None
                    else ""
                )
                logger.info(
                    "layer %s iter %s loss:%s norm:%s time:%.3fs%s",
                    layer_index,
                    epoch,
                    torch.stack(loss_values).mean().item(),
                    torch.stack(norm_values).mean().item(),
                    time.perf_counter() - iter_start_time,
                    memory_fragment,
                )
            if diagnostics_enabled and layer_diagnostics is not None and fp_outs is not None:
                post_train_outputs = _collect_temporary_quant_outputs(
                    qlayer,
                    args,
                    quant_inps,
                    active_layer_kwargs,
                    autocast_enabled=autocast_enabled,
                    batch_size=int(args.batch_size),
                )
                layer_diagnostics["post_train_temporary_mse"] = _tensor_mse(fp_outs, post_train_outputs)
            clear_temp_variable(qlayer)
            del optimizer
            fp_inps = fp_outs
            if args.aug_loss:
                fp_inps_2 = fp_outs_2
        elif diagnostics_enabled and layer_diagnostics is not None and fp_outs is not None:
            pre_train_outputs = _collect_temporary_quant_outputs(
                qlayer,
                args,
                quant_inps,
                active_layer_kwargs,
                autocast_enabled=autocast_enabled,
                batch_size=int(args.batch_size),
            )
            layer_diagnostics["pre_train_temporary_mse"] = _tensor_mse(fp_outs, pre_train_outputs)

        if not args.deactive_amp:
            qlayer = qlayer.to(dtype=model_dtype)
        smooth_and_quant_inplace(qlayer, args)

        quant_outs = None
        if args.epochs > 0 or diagnostics_enabled:
            quant_outs = _collect_layer_outputs(
                qlayer,
                quant_inps,
                active_layer_kwargs,
                autocast_enabled=autocast_enabled,
                batch_size=int(args.batch_size),
            )
            if not torch.isfinite(quant_outs).all():
                _raise_nonfinite_error(
                    args.output_dir,
                    layer_index=layer_index,
                    phase="post_inplace",
                    message=f"OmniQuant post-inplace outputs became non-finite at layer {layer_index}.",
                    payload={
                        "layer_index": layer_index,
                        "phase": "post_inplace",
                        "autocast_enabled": autocast_enabled,
                        "quant_inps": _tensor_summary("quant_inps", quant_inps),
                        "fp_outs": _tensor_summary("fp_outs", fp_outs) if fp_outs is not None else None,
                        "quant_outs": _tensor_summary("quant_outs", quant_outs),
                        "layer_kwargs": _value_summary("layer_kwargs", active_layer_kwargs),
                        "trainable_parameters": _omni_parameter_summaries(qlayer, use_shift=use_shift),
                    },
                )
            if diagnostics_enabled and layer_diagnostics is not None and fp_outs is not None:
                layer_diagnostics["post_inplace_mse"] = _tensor_mse(fp_outs, quant_outs)
                diagnostics_records.append(layer_diagnostics)
                diagnostics_path.write_text(json.dumps(diagnostics_records, indent=2), encoding="utf-8")
                logger.info(
                    "layer %s diagnostics input_mse=%s pre_train_temporary_mse=%s post_train_temporary_mse=%s post_inplace_mse=%s",
                    layer_index,
                    layer_diagnostics.get("input_mse"),
                    layer_diagnostics.get("pre_train_temporary_mse"),
                    layer_diagnostics.get("post_train_temporary_mse"),
                    layer_diagnostics.get("post_inplace_mse"),
                )
        if args.epochs > 0 and quant_outs is not None:
            quant_inps = quant_outs
            register_scales_and_zeros(qlayer)
            saved_parameters[layer_index] = omni_state_dict(qlayer)
            torch.save(saved_parameters, parameter_checkpoint_path)
        else:
            register_scales_and_zeros(qlayer)

        for name in get_named_linears(qlayer):
            quantized_linear_artifacts[f"{backbone.prefix}.layers.{layer_index}.{name}"] = {
                "weight_bits": args.wbits,
                "activation_bits": args.abits,
                "group_size": args.weight_group_size,
                "weight_symmetric": bool(args.weight_quant_params["symmetric"]),
                "activation_symmetric": False,
            }
        for name in get_named_packed_moe_experts(qlayer):
            for weight_name in ("gate_up_proj", "down_proj"):
                quantized_linear_artifacts[f"{backbone.prefix}.layers.{layer_index}.{name}.{weight_name}"] = {
                    "weight_bits": args.wbits,
                    "activation_bits": args.abits,
                    "group_size": args.weight_group_size,
                    "weight_symmetric": bool(args.weight_quant_params["symmetric"]),
                    "activation_symmetric": False,
                }

        backbone.layers[layer_index] = qlayer.to(dtype=model_dtype)
        del layer
        empty_cache(device)
        logger.info("layer %s done time:%.3fs", layer_index, time.perf_counter() - layer_start_time)

    return quantized_linear_artifacts
# Synchronize quantization device_map support for multi-GPU execution.
