import torch
import torch.nn as nn
from algorithm.common.modeling import capture_first_block_inputs
from algorithm.common.modeling import find_prunable_linear_layers
from algorithm.common.modeling import get_layer_device
from algorithm.common.modeling import get_text_backbone
from algorithm.common.modeling import move_tensors_to_device
from algorithm.common.modeling import unwrap_layer_output
from .layerwrapper import WrappedGPT
from .backend import resolve_runtime_device


def find_layers(module, layers=[nn.Linear], name=''):
    """递归查找模块中的特定类型层。"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_prunable_layers(module):
    """查找统一剪枝策略允许处理的 Linear 层。"""
    return find_prunable_linear_layers(module)


def _move_layer_kwargs(layer_kwargs, device):
    """将 layer_kwargs 中的张量移动到指定设备。"""
    moved = {}
    for key, value in layer_kwargs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, tuple):
            moved[key] = tuple(
                item.to(device) if torch.is_tensor(item) else item
                for item in value
            )
        else:
            moved[key] = value
    return moved


def check_sparsity(model):
    """检查模型权重的稀疏度。"""
    backbone = get_text_backbone(model)
    decoder_config = backbone.decoder_config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False

    layers = backbone.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_prunable_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    decoder_config.use_cache = use_cache
    return float(count)/total_params


def prepare_calibration_input(model, dataloader, device):
    """
    准备校准输入数据。

    复用公共 Catcher 路径，和 SparseGPT 保持一致；其中已处理
    Qwen3-MoE 所需的 attention_mask。
    """
    backbone = get_text_backbone(model)
    inps, layer_kwargs = capture_first_block_inputs(
        model=model,
        backbone=backbone,
        calibration_batches=dataloader,
        device=device,
    )
    outs = torch.zeros_like(inps)
    return inps, outs, layer_kwargs


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(args, model, tokenizer, device=None, prune_n=0, prune_m=0, dataloader=None):
    """Wanda 剪枝主函数。"""
    backbone = get_text_backbone(model)
    decoder_config = backbone.decoder_config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False

    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(model, dataloader, device)

    layers = backbone.layers
    for i in range(len(layers)):
        layer = layers[i]
        target_device = get_layer_device(backbone, i)
        inps = inps.to(target_device)
        outs = outs.to(target_device)
        layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
        subset = find_prunable_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                out = unwrap_layer_output(out)
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = unwrap_layer_output(layer(inps[j].unsqueeze(0), **layer_kwargs))
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            if wrapped_layers[name].nsamples == 0:
                print(f"skip pruning layer {i} name {name}: no calibration samples routed to this layer")
                continue
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            if not torch.isfinite(W_metric).all() or W_metric.sum() == 0:
                print(f"skip pruning layer {i} name {name}: invalid or empty Wanda metric")
                continue

            W_mask = (torch.zeros_like(W_metric) == 1)  ## 初始化全 False 的 mask
            if prune_n != 0:
                # n:m 半结构化剪枝
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda 变体
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # 非结构化剪枝
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## 将权重置零

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = unwrap_layer_output(layer(inps[j].unsqueeze(0), **layer_kwargs))
        inps, outs = outs, inps

    decoder_config.use_cache = use_cache
# Migrate pruning to device_map loading for future multi-GPU support.
