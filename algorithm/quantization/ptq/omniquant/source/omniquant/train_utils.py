from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from algorithm.common.device import empty_cache
from algorithm.common.modeling import capture_first_block_inputs
from algorithm.common.modeling import get_text_backbone
from algorithm.common.modeling import unwrap_layer_output

from omniquant.model_tools.llama_utils import QuantLlamaDecoderLayer
from omniquant.model_tools.llama_utils import initialize_omni_parameters
from omniquant.utils import ampscaler_get_grad_norm
from omniquant.utils import clear_temp_variable
from omniquant.utils import get_named_linears
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
        return value
    if isinstance(value, tuple):
        return tuple(_expand_for_batch(item, batch_size) for item in value)
    if isinstance(value, list):
        return [_expand_for_batch(item, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: _expand_for_batch(item, batch_size) for key, item in value.items()}
    return value


def _autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.cuda.amp.autocast()


def _tensor_mse(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(lhs.float(), rhs.float()).detach().cpu())


def _collect_layer_outputs(layer, inputs, layer_kwargs, autocast_enabled: bool, batch_size: int):
    outputs = torch.zeros_like(inputs)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            batch_kwargs = _expand_for_batch(layer_kwargs, end - start)
            with _autocast_context(autocast_enabled):
                outputs[start:end] = unwrap_layer_output(layer(inputs[start:end], **batch_kwargs))
    return outputs


def _collect_temporary_quant_outputs(layer, args, inputs, layer_kwargs, autocast_enabled: bool, batch_size: int):
    outputs = torch.zeros_like(inputs)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            batch_kwargs = _expand_for_batch(layer_kwargs, end - start)
            with _autocast_context(autocast_enabled):
                smooth_and_quant_temporary(layer, args)
                outputs[start:end] = unwrap_layer_output(layer(inputs[start:end], **batch_kwargs))
            clear_temp_variable(layer)
    return outputs


def cali_omni_quant(args, model, calibration_batches, device, act_scales, act_shifts, logger):
    backbone = get_text_backbone(model)
    model_dtype = next(model.parameters()).dtype
    quantized_linear_artifacts: dict[str, dict[str, object]] = {}
    diagnostics_enabled = bool(getattr(args, "save_diagnostics", False))
    diagnostics_path = Path(args.output_dir) / "layer_diagnostics.json"
    diagnostics_records: list[dict[str, object]] = []
    input_states, layer_kwargs = capture_first_block_inputs(
        model=model,
        backbone=backbone,
        calibration_batches=calibration_batches,
        device=device,
    )
    if args.deactive_amp:
        input_states = input_states.float()
        layer_kwargs = _cast_fp_tensors(layer_kwargs, torch.float32)

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
        logger.info("========= Layer %s =========", layer_index)
        layer = layer.to(device)
        qlayer = QuantLlamaDecoderLayer(layer, args).to(device)
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
                layer_kwargs,
                autocast_enabled=autocast_enabled,
                batch_size=int(args.batch_size),
            )
            if args.aug_loss:
                fp_outs_2 = _collect_layer_outputs(
                    qlayer,
                    quant_inps,
                    layer_kwargs,
                    autocast_enabled=autocast_enabled,
                    batch_size=int(args.batch_size),
                )

        # Upstream OmniQuant disables learned shift updates for LLaMA-family models.
        use_shift = False
        initialize_omni_parameters(
            qlayer,
            f"{backbone.prefix}.layers.{layer_index}",
            args,
            act_scales,
            act_shifts,
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
                    layer_kwargs,
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
            grad_scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

            for epoch in range(args.epochs):
                loss_values = []
                norm_values = []
                for start in range(0, args.nsamples, args.batch_size):
                    end = min(start + args.batch_size, args.nsamples)
                    batch_kwargs = _expand_for_batch(layer_kwargs, end - start)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast_context(autocast_enabled):
                        smooth_and_quant_temporary(qlayer, args)
                        quant_out = unwrap_layer_output(qlayer(quant_inps[start:end], **batch_kwargs))
                        loss = mse_loss(fp_outs[start:end], quant_out)
                        if args.aug_loss and fp_outs_2 is not None:
                            loss = loss + mse_loss(fp_outs_2[start:end], quant_out)
                    if not math.isfinite(loss.item()):
                        raise RuntimeError(f"OmniQuant loss became non-finite at layer {layer_index}, epoch {epoch}.")
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    grad_norm = ampscaler_get_grad_norm(get_omni_parameters(qlayer, use_shift=use_shift)).detach().cpu()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    loss_values.append(loss.detach().cpu())
                    norm_values.append(grad_norm)
                logger.info(
                    "layer %s iter %s loss:%s norm:%s max_memory_allocated %.2f MiB",
                    layer_index,
                    epoch,
                    torch.stack(loss_values).mean().item(),
                    torch.stack(norm_values).mean().item(),
                    torch.cuda.max_memory_allocated(device) / 1024**2,
                )
            if diagnostics_enabled and layer_diagnostics is not None and fp_outs is not None:
                post_train_outputs = _collect_temporary_quant_outputs(
                    qlayer,
                    args,
                    quant_inps,
                    layer_kwargs,
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
                layer_kwargs,
                autocast_enabled=autocast_enabled,
                batch_size=int(args.batch_size),
            )
            layer_diagnostics["pre_train_temporary_mse"] = _tensor_mse(fp_outs, pre_train_outputs)

        qlayer = qlayer.to(dtype=model_dtype)
        smooth_and_quant_inplace(qlayer, args)

        quant_outs = None
        if args.epochs > 0 or diagnostics_enabled:
            quant_outs = _collect_layer_outputs(
                qlayer,
                quant_inps,
                layer_kwargs,
                autocast_enabled=autocast_enabled,
                batch_size=int(args.batch_size),
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

        backbone.layers[layer_index] = qlayer.to(dtype=model_dtype).cpu()
        del layer
        empty_cache(device)

    return quantized_linear_artifacts
