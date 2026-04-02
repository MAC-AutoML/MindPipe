import time 
import heapq 
import torch 
import torch.nn as nn 
from algorithm.common.device import empty_cache
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 
from .backend import move_optional_tensor
from .backend import resolve_runtime_device
from .backend import sparsity_threshold


def _get_decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported decoder backbone: {type(model)}")


def _get_hf_device_map(model):
    return getattr(model, "hf_device_map", {}) or {}


def _resolve_layer_device(hf_device_map, candidate_keys, default_device):
    for candidate_key in candidate_keys:
        if candidate_key in hf_device_map:
            return resolve_runtime_device(hf_device_map[candidate_key])
    return resolve_runtime_device(default_device)


def _module_device(module):
    if module is None:
        return None
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return None


def _build_qwen2_5_vl_position_ids(hidden_states, position_ids, cache_position):
    if position_ids is not None:
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            return position_ids[1:]
        if position_ids.ndim == 3:
            return position_ids
        if position_ids.ndim == 2:
            return position_ids.unsqueeze(0).expand(3, position_ids.shape[0], -1)
    if cache_position is None:
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    if cache_position.ndim == 1:
        return cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
    if cache_position.ndim == 2:
        return cache_position.unsqueeze(0).expand(3, -1, -1)
    raise ValueError(f"Unsupported cache_position shape for Qwen2.5-VL: {tuple(cache_position.shape)}")


def _build_layer_forward_kwargs(model, layer, hidden_states, attention_mask, position_ids, cache_position=None):
    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    model_type = getattr(model.config, "model_type", None)
    if model_type == "minicpmv":
        kwargs.pop("cache_position", None)
        kwargs["use_cache"] = False
        return kwargs
    decoder_root = _get_decoder_root(model)
    layer_rotary_embedding = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
    decoder_rotary_embedding = getattr(decoder_root, "rotary_emb", None)
    if model_type == "qwen2_5_vl":
        rotary_embedding = layer_rotary_embedding or decoder_rotary_embedding
    else:
        rotary_embedding = decoder_rotary_embedding or layer_rotary_embedding
    rotary_device = _module_device(rotary_embedding)
    if rotary_embedding is not None and rotary_device is not None and rotary_device != hidden_states.device:
        rotary_embedding = rotary_embedding.to(hidden_states.device)
    rotary_position_ids = position_ids
    if model_type == "qwen2_5_vl":
        rotary_position_ids = _build_qwen2_5_vl_position_ids(hidden_states, position_ids, cache_position)
        if position_ids is None or (position_ids.ndim == 3 and position_ids.shape[0] == 3):
            kwargs["position_ids"] = None
    if rotary_embedding is not None and rotary_position_ids is not None:
        kwargs["position_embeddings"] = rotary_embedding(hidden_states, rotary_position_ids)
    return kwargs

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = _get_decoder_root(model).layers
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

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    device = resolve_runtime_device(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    decoder_root = _get_decoder_root(model)
    layers = decoder_root.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    hf_device_map = _get_hf_device_map(model)
    for candidate_key in ("model.embed_tokens", "language_model.embed_tokens", "model.language_model.embed_tokens"):
        if candidate_key in hf_device_map:
            device = resolve_runtime_device(hf_device_map[candidate_key])
            break

    dtype = next(iter(model.parameters())).dtype
    if hasattr(decoder_root, "embed_tokens"):
        decoder_root.embed_tokens = decoder_root.embed_tokens.to(device)
    if hasattr(decoder_root, "rotary_emb"):
        decoder_root.rotary_emb = decoder_root.rotary_emb.to(device)
    layers[0] = layers[0].to(device)
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None}

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
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['cache_position'] = kwargs.get('cache_position')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device), use_cache=False)
        except ValueError:
            pass 
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if hasattr(decoder_root, "embed_tokens"):
        decoder_root.embed_tokens = decoder_root.embed_tokens.cpu()
    if hasattr(decoder_root, "rotary_emb"):
        decoder_root.rotary_emb = decoder_root.rotary_emb.cpu()
    empty_cache(device)

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, cache['cache_position'] 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=None, prune_n=0, prune_m=0):
    device = resolve_runtime_device(device)
    layers = _get_decoder_root(model).layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = sparsity_threshold(W_metric, args.sparsity_ratio, device)
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=None, prune_n=0, prune_m=0):
    device = resolve_runtime_device(device)
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, cache_position = prepare_calibration_input(model, dataloader, device)

    layers = _get_decoder_root(model).layers
    hf_device_map = _get_hf_device_map(model)
    for i in range(len(layers)):
        dev = _resolve_layer_device(
            hf_device_map,
            (
                f"model.layers.{i}",
                f"language_model.layers.{i}",
                f"model.language_model.layers.{i}",
            ),
            device,
        )
        inps = inps.to(dev)
        outs = outs.to(dev)
        attention_mask = move_optional_tensor(attention_mask, dev)
        position_ids = move_optional_tensor(position_ids, dev)
        cache_position = move_optional_tensor(cache_position, dev)
        layer = layers[i].to(dev)
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
                layer_kwargs = _build_layer_forward_kwargs(
                    model,
                    layer,
                    inps[j].unsqueeze(0),
                    attention_mask,
                    position_ids,
                    cache_position=cache_position,
                )
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
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
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                layer_kwargs = _build_layer_forward_kwargs(
                    model,
                    layer,
                    inps[j].unsqueeze(0),
                    attention_mask,
                    position_ids,
                    cache_position=cache_position,
                )
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps
        layers[i] = layer.cpu()
        del layer
        empty_cache(dev)

    model.config.use_cache = use_cache 
    empty_cache(device)


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = _get_decoder_root(model).layers

    dev = resolve_runtime_device(dev)
    hf_device_map = _get_hf_device_map(model)
    for candidate_key in ("model.embed_tokens", "language_model.embed_tokens", "model.language_model.embed_tokens"):
        if candidate_key in hf_device_map:
            dev = resolve_runtime_device(hf_device_map[candidate_key])
            break

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None}

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
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['cache_position'] = kwargs.get('cache_position')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].module
    empty_cache(dev)

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    cache_position = cache['cache_position']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        dev = _resolve_layer_device(
            hf_device_map,
            (
                f"model.layers.{i}",
                f"language_model.layers.{i}",
                f"model.language_model.layers.{i}",
            ),
            dev,
        )
        print(f"layer {i} device {dev}")
        inps = inps.to(dev)
        outs = outs.to(dev)
        attention_mask = move_optional_tensor(attention_mask, dev)
        position_ids = move_optional_tensor(position_ids, dev)
        cache_position = move_optional_tensor(cache_position, dev)
        layer = layer.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            layer_kwargs = _build_layer_forward_kwargs(
                model,
                layer,
                inps[j].unsqueeze(0),
                attention_mask,
                position_ids,
                cache_position=cache_position,
            )
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            layer_kwargs = _build_layer_forward_kwargs(
                model,
                layer,
                inps[j].unsqueeze(0),
                attention_mask,
                position_ids,
                cache_position=cache_position,
            )
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        empty_cache(dev)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    empty_cache(dev)



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = _get_decoder_root(model).layers

    dev = resolve_runtime_device(dev)
    hf_device_map = _get_hf_device_map(model)
    for candidate_key in ("model.embed_tokens", "language_model.embed_tokens", "model.language_model.embed_tokens"):
        if candidate_key in hf_device_map:
            dev = resolve_runtime_device(hf_device_map[candidate_key])
            break

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None}

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
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['cache_position'] = kwargs.get('cache_position')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].module
    empty_cache(dev)

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    cache_position = cache['cache_position']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        dev = _resolve_layer_device(
            hf_device_map,
            (
                f"model.layers.{i}",
                f"language_model.layers.{i}",
                f"model.language_model.layers.{i}",
            ),
            dev,
        )
        print(f"layer {i} device {dev}")
        inps = inps.to(dev)
        outs = outs.to(dev)
        attention_mask = move_optional_tensor(attention_mask, dev)
        position_ids = move_optional_tensor(position_ids, dev)
        cache_position = move_optional_tensor(cache_position, dev)
        layer = layer.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            layer_kwargs = _build_layer_forward_kwargs(
                model,
                layer,
                inps[j].unsqueeze(0),
                attention_mask,
                position_ids,
                cache_position=cache_position,
            )
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            layer_kwargs = _build_layer_forward_kwargs(
                model,
                layer,
                inps[j].unsqueeze(0),
                attention_mask,
                position_ids,
                cache_position=cache_position,
            )
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        empty_cache(dev)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    empty_cache(dev)
