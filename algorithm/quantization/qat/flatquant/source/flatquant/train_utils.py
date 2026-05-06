import os
import time
import gc
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformers

from algorithm.common.device import empty_cache
from algorithm.common.modeling import get_text_backbone
from algorithm.common.modeling import move_tensors_to_device
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask as create_qwen3_5_causal_mask

from flatquant.backbone_utils import build_batched_layer_kwargs
from flatquant.backbone_utils import get_decoder_config
from flatquant.backbone_utils import get_decoder_layers
from flatquant.backbone_utils import move_front_modules
from flatquant.backbone_utils import unwrap_layer_output
from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad
from flatquant.quant_utils import set_quantizer_state


def _build_calibration_forward_kwargs(model, sample):
    model_type = getattr(model.config, "model_type", None)
    if model_type in {"qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
        # Keep an explicit all-ones mask during capture so Qwen3.5-family models
        # stay on the expected masking path before the first decoder block.
        return {"attention_mask": torch.ones_like(sample, dtype=torch.long, device=sample.device)}
    return {}


def _resolve_calibration_input_device(model):
    try:
        backbone = get_text_backbone(model)
        embed_tokens = backbone.embed_tokens
        if embed_tokens is not None:
            for tensor in tuple(embed_tokens.parameters()) + tuple(embed_tokens.buffers()):
                return tensor.device
    except Exception:
        pass
    return next(model.parameters()).device


def _build_qwen3_5_layer_kwargs(decoder_config, inputs, layer_kwargs):
    position_ids = layer_kwargs.get("position_ids")
    if position_ids is None:
        seq_len = inputs.shape[1]
        position_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)

    full_attention_kwargs = dict(layer_kwargs)
    full_attention_kwargs["attention_mask"] = create_qwen3_5_causal_mask(
        config=decoder_config,
        inputs_embeds=inputs[:1],
        attention_mask=torch.ones((1, inputs.shape[1]), dtype=torch.long, device=inputs.device),
        cache_position=full_attention_kwargs.get("cache_position"),
        past_key_values=full_attention_kwargs.get("past_key_values"),
        position_ids=position_ids,
    )

    linear_attention_kwargs = dict(layer_kwargs)
    linear_attention_kwargs["attention_mask"] = None
    return {
        "full_attention": full_attention_kwargs,
        "linear_attention": linear_attention_kwargs,
    }


def _select_layer_kwargs(layer, layer_kwargs, layer_kwargs_by_type):
    if layer_kwargs_by_type is None:
        return layer_kwargs
    return layer_kwargs_by_type.get(getattr(layer, "layer_type", None), layer_kwargs)


def _reset_square_linear_to_identity(linear):
    if hasattr(linear, "parametrizations") and hasattr(linear.parametrizations, "weight"):
        weight = linear.parametrizations.weight.original
    else:
        weight = linear.weight
    eye = torch.eye(weight.shape[0], weight.shape[1], device=weight.device, dtype=weight.dtype)
    weight.data.copy_(eye)


def _reset_transformation_module(module):
    if module is None:
        return
    for attr_name in (
        "linear",
        "linear_left",
        "linear_right",
        "linear_u",
        "linear_v",
        "linear_u_left",
        "linear_v_left",
        "linear_u_right",
        "linear_v_right",
    ):
        if hasattr(module, attr_name):
            _reset_square_linear_to_identity(getattr(module, attr_name))
    for attr_name in ("linear_diag", "linear_diag_left", "linear_diag_right", "diag_scale"):
        if hasattr(module, attr_name):
            getattr(module, attr_name).data.fill_(1)
    if hasattr(module, "use_diag"):
        module.use_diag = False


def _stabilize_flatquant_layer(layer):
    for attr_name in ("ln_trans", "o_trans", "kcache_trans", "vcache_trans"):
        _reset_transformation_module(getattr(layer.self_attn, attr_name, None))
    for attr_name in ("_ln_trans",):
        _reset_transformation_module(getattr(layer.self_attn, attr_name, None))
    for attr_name in ("up_gate_trans", "down_trans"):
        _reset_transformation_module(getattr(layer.mlp, attr_name, None))
    for attr_name in ("_up_gate_trans", "_down_trans"):
        _reset_transformation_module(getattr(layer.mlp, attr_name, None))


def cali_flat_quant(args, model, dataloader, dev, logger):
    model.eval()
    decoder_config = get_decoder_config(model)
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = next(iter(model.parameters())).dtype
        _device_type = torch.device(dev).type
        traincast = functools.partial(torch.amp.autocast, device_type=_device_type, dtype=dtype)

    # device_map 模式下不手动移动 front modules 和 layer[0]，由 dispatch_model 管理
    layers = get_decoder_layers(model)

    # catch the first layer input
    layer0_device = next(layers[0].parameters()).device
    capture_device = _resolve_calibration_input_device(model)
    logger.info(
        "FlatQuant calibration device placement: capture_device=%s layer0_device=%s first_param_device=%s",
        capture_device,
        layer0_device,
        next(model.parameters()).device,
    )
    inps = torch.zeros(
        (args.nsamples, model.seqlen, decoder_config.hidden_size), dtype=dtype, device=layer0_device
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = dict(kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                sample = sample.to(capture_device)
                model(sample, use_cache=False, **_build_calibration_forward_kwargs(model, sample))
            except ValueError:
                pass
    layer_kwargs = dict(cache["layer_kwargs"])
    layer_kwargs_by_type = None
    batched_layer_kwargs_by_type = None
    if getattr(model.config, "model_type", None) == "qwen3_5":
        layer_kwargs_by_type = _build_qwen3_5_layer_kwargs(decoder_config, inps, layer_kwargs)
        batched_layer_kwargs_by_type = {
            layer_type: build_batched_layer_kwargs(kwargs, args.cali_bsz)
            for layer_type, kwargs in layer_kwargs_by_type.items()
        }
    batched_layer_kwargs = build_batched_layer_kwargs(layer_kwargs, args.cali_bsz)
    
    # device_map 模式下不手动移动到 cpu
    layers[0] = layers[0].module
    empty_cache(dev)

    # same input of first layer for fp model and quant model
    fp_inps = inps   # take output of fp model as input
    fp_outs = torch.zeros_like(inps)   # take output of fp model as input

    loss_func = torch.nn.MSELoss()
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    mse_dict = {}
    for i in range(num_train_layer):
        logger.info(f"========= Layer {i} =========")
        dtype_dict = {}
        # device_map 模式下不手动移动 layer，在当前设备上操作
        layer = layers[i]
        # 将输入数据移到当前层设备
        layer_dev = next(layer.parameters()).device
        fp_inps = fp_inps.to(layer_dev)
        fp_outs = fp_outs.to(layer_dev)
        layer_kwargs = move_tensors_to_device(layer_kwargs, layer_dev)
        batched_layer_kwargs = move_tensors_to_device(batched_layer_kwargs, layer_dev)
        if layer_kwargs_by_type is not None:
            layer_kwargs_by_type = {
                layer_type: move_tensors_to_device(kwargs, layer_dev)
                for layer_type, kwargs in layer_kwargs_by_type.items()
            }
        if batched_layer_kwargs_by_type is not None:
            batched_layer_kwargs_by_type = {
                layer_type: move_tensors_to_device(kwargs, layer_dev)
                for layer_type, kwargs in batched_layer_kwargs_by_type.items()
            }
        active_layer_kwargs = _select_layer_kwargs(layer, layer_kwargs, layer_kwargs_by_type)
        active_batched_layer_kwargs = _select_layer_kwargs(layer, batched_layer_kwargs, batched_layer_kwargs_by_type)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = unwrap_layer_output(layer(fp_inps[j].unsqueeze(0), **active_layer_kwargs))
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False
        if args.diag_init == "sq_style":
            layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.diag_alpha)
        elif args.diag_init == "one_style":
            pass
        else:
            raise NotImplementedError

        # 将 layer（含新建的 diag 参数）移到当前层设备
        layer = layer.to(layer_dev)
        # device_map 模式下不手动移动 layer，保持在当前设备
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": args.flat_lr})
            paras_name.append("trans.linear")
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": args.flat_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.nsamples // args.cali_bsz), eta_min=args.flat_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        # check_params_grad(layer)
        # set_quantizer_state(layer, False)
        unstable_layer = False
        for epoch in range(args.epochs):
            mse = 0
            start_tick = time.time()
            with traincast():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    try:
                        quant_out = unwrap_layer_output(layer(fp_inps[index:index+args.cali_bsz,], **active_batched_layer_kwargs))
                    except RuntimeError as error:
                        error_text = str(error).lower()
                        if "singular" in error_text or "linalg" in error_text or "inverse" in error_text:
                            logger.warning(
                                "Layer %s became numerically unstable at epoch %s; stop optimizing this layer. Error: %s",
                                i,
                                epoch,
                                error,
                            )
                            _stabilize_flatquant_layer(layer)
                            unstable_layer = True
                            break
                        raise
                    loss = loss_func(fp_outs[index:index+args.cali_bsz,], quant_out)
                    if not torch.isfinite(loss):
                        logger.warning(
                            "Layer %s produced a non-finite loss at epoch %s; stop optimizing this layer.",
                            i,
                            epoch,
                        )
                        _stabilize_flatquant_layer(layer)
                        unstable_layer = True
                        break
                    mse += loss.detach().cpu()
                    loss = loss / loss.detach().clamp_min(1e-12)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            if unstable_layer:
                break
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {float(mse):.8f}" )

        fp_inps, fp_outs = fp_outs, fp_inps
        # device_map 模式下不手动移动到 cpu，保持在当前设备
        flat_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters.pth"))
        logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters.pth")))
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        empty_cache(dev)

    del inps, fp_inps, fp_outs
    gc.collect()
    empty_cache(dev)
    decoder_config.use_cache = use_cache
    return model
