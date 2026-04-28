import torch
import torch.nn as nn
from algorithm.common.modeling import get_text_backbone
from algorithm.common.modeling import get_layer_device
from algorithm.common.modeling import move_tensors_to_device
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
        subset = find_layers(layer)

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

    通过 Catcher 模式直接捕获模型传给 decoder layer 的全部参数，
    包括 position_embeddings 等，无需针对不同模型手动构建 kwargs。
    """
    backbone = get_text_backbone(model)
    decoder_config = backbone.decoder_config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False
    layers = backbone.layers

    # device_map 模式下，直接从模型参数获取 embedding 所在设备
    capture_device = next(model.parameters()).device

    dtype = next(iter(model.parameters())).dtype
    sample_count = len(dataloader)
    inps = torch.zeros((sample_count, model.seqlen, decoder_config.hidden_size), dtype=dtype, device=capture_device)
    inps.requires_grad = False
    cache = {'i': 0, 'layer_kwargs': {}}

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
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = dict(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(capture_device), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    decoder_config.use_cache = use_cache

    return inps, outs, dict(cache['layer_kwargs'])


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
        target_device = next(layer.parameters()).device
        inps = inps.to(target_device)
        outs = outs.to(target_device)
        layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

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
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

    decoder_config.use_cache = use_cache
# Migrate pruning to device_map loading for future multi-GPU support.
