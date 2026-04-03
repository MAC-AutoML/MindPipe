import torch
import torch.nn as nn

from algorithm.common.device import empty_cache
from algorithm.common.device import resolve_device
from .data import get_loaders
from .layerwrapper import WrappedGPT


def resolve_linear_module(module, layers=(nn.Linear,)):
    layer_types = tuple(layers)
    if isinstance(module, layer_types):
        return module
    for attr_name in ("linear", "module"):
        child = getattr(module, attr_name, None)
        if isinstance(child, layer_types):
            return child
    return None


def find_layers(module, layers=[nn.Linear], name=""):
    resolved_module = resolve_linear_module(module, layers)
    if resolved_module is not None:
        return {name: resolved_module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def get_attention_projections(layer):
    return (
        resolve_linear_module(layer.self_attn.q_proj),
        resolve_linear_module(layer.self_attn.k_proj),
        resolve_linear_module(layer.self_attn.v_proj),
        resolve_linear_module(layer.self_attn.o_proj),
    )


def get_mlp_projections(layer):
    return (
        resolve_linear_module(layer.mlp.up_proj),
        resolve_linear_module(layer.mlp.gate_proj),
        resolve_linear_module(layer.mlp.down_proj),
    )


def get_projection_subset(layer):
    o_proj = resolve_linear_module(layer.self_attn.o_proj)
    down_proj = resolve_linear_module(layer.mlp.down_proj)
    if o_proj is None or down_proj is None:
        available = ", ".join(sorted(find_layers(layer).keys()))
        raise KeyError(
            "Failed to resolve structured Wanda target projections. "
            f"available_layers=[{available}]"
        )
    return {
        "self_attn.o_proj": o_proj,
        "mlp.down_proj": down_proj,
    }


def get_decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported decoder backbone: {type(model)}")


def get_decoder_layers(model):
    return get_decoder_root(model).layers


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_decoder_layers(model)

    if "model.embed_tokens" in getattr(model, "hf_device_map", {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    sample_count = len(dataloader)
    inps = torch.zeros((sample_count, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0, "layer_kwargs": {}}

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
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    return inps, outs, dict(cache["layer_kwargs"])


def move_layer_kwargs(layer_kwargs, device):
    moved_kwargs = {}
    for name, value in layer_kwargs.items():
        if torch.is_tensor(value):
            moved_kwargs[name] = value.to(device)
        elif isinstance(value, tuple):
            moved_kwargs[name] = tuple(
                item.to(device) if torch.is_tensor(item) else item for item in value
            )
        else:
            moved_kwargs[name] = value
    return moved_kwargs


def get_attention_head_geometry(layer):
    attn = layer.self_attn
    config = getattr(attn, "config", None)
    num_heads = int(getattr(attn, "num_heads", getattr(config, "num_attention_heads")))
    num_kv_heads = int(
        getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads", num_heads))
    )
    hidden_size = int(getattr(attn, "hidden_size", getattr(config, "hidden_size")))
    head_dim = int(getattr(attn, "head_dim", hidden_size // num_heads))
    num_kv_groups = num_heads // num_kv_heads
    return num_heads, num_kv_heads, num_kv_groups, head_dim


def aggregate_attention_metric(layer, metric):
    num_heads, _, _, head_dim = get_attention_head_geometry(layer)
    return metric.reshape(num_heads, head_dim).sum(dim=1)


def expand_attention_masks(layer, attn_mask, device):
    num_heads, num_kv_heads, num_kv_groups, head_dim = get_attention_head_geometry(layer)
    if attn_mask.numel() != num_heads:
        raise ValueError(
            f"Expected {num_heads} attention heads in structured Wanda mask, got {attn_mask.numel()}."
        )
    q_mask = attn_mask.repeat_interleave(head_dim)
    if num_kv_heads == num_heads:
        kv_mask = q_mask
    else:
        kv_head_mask = attn_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
        kv_mask = kv_head_mask.repeat_interleave(head_dim)
    return q_mask.to(device), kv_mask.to(device)


def compute_output_bias(mean_input, mask, output_weight):
    baseline_input = mean_input.to(device=output_weight.device, dtype=output_weight.dtype)
    inactive_mask = (~mask).to(device=output_weight.device, dtype=output_weight.dtype)
    return (baseline_input * inactive_mask) @ output_weight.T


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    q_proj, k_proj, v_proj, o_proj = get_attention_projections(layer)
    up_proj, gate_proj, down_proj = get_mlp_projections(layer)
    if unstr:
        if attn_mask is not None:
            q_mask, kv_mask = expand_attention_masks(layer, attn_mask, device)
            q_proj.weight.data *= q_mask.unsqueeze(-1)
            k_proj.weight.data *= kv_mask.unsqueeze(-1)
            v_proj.weight.data *= kv_mask.unsqueeze(-1)

            output_weight = o_proj.weight.data
            if bias:
                output_bias = compute_output_bias(attn_mean_inp, q_mask, output_weight)
                if o_proj.bias is None:
                    o_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
            o_proj.weight.data *= q_mask.unsqueeze(0)
            if bias:
                o_proj.bias.data = output_bias

        if mlp_mask is not None:
            up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            output_weight = down_proj.weight.data
            if bias:
                output_bias = compute_output_bias(mlp_mean_inp, mlp_mask.to(device), output_weight)
                if down_proj.bias is None:
                    down_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
            down_proj.weight.data *= mlp_mask.unsqueeze(0).to(device)
            if bias:
                down_proj.bias.data = output_bias
    else:
        if attn_mask is not None:
            num_heads, _, _, head_dim = get_attention_head_geometry(layer)
            retain_heads = torch.count_nonzero(attn_mask)
            q_mask, kv_mask = expand_attention_masks(layer, attn_mask, device)
            q_indices = torch.where(q_mask)[0]
            kv_indices = torch.where(kv_mask)[0]
            q_proj.weight.data = q_proj.weight.data[q_indices]
            k_proj.weight.data = k_proj.weight.data[kv_indices]
            v_proj.weight.data = v_proj.weight.data[kv_indices]
            q_proj.out_features = q_indices.numel()
            k_proj.out_features = kv_indices.numel()
            v_proj.out_features = kv_indices.numel()

            output_weight = o_proj.weight.data
            if bias:
                output_bias = compute_output_bias(attn_mean_inp, q_mask.to(device), output_weight)
            output_weight = o_proj.weight.data[:, q_indices]
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * head_dim
            if bias:
                o_proj.in_features = q_indices.numel()
                o_proj.bias.data = output_bias
            o_proj.weight.data = output_weight

        if mlp_mask is not None:
            kept_indices = torch.where(mlp_mask)[0]
            up_proj.weight.data = up_proj.weight.data[kept_indices]
            gate_proj.weight.data = gate_proj.weight.data[kept_indices]
            up_proj.out_features = mlp_mask.sum().item()
            gate_proj.out_features = mlp_mask.sum().item()

            output_weight = down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                output_bias = compute_output_bias(mlp_mean_inp, mlp_mask.to(device), output_weight)
            output_weight = down_proj.weight.data[:, kept_indices]
            if bias:
                down_proj.in_features = mlp_mask.sum().item()
                down_proj.bias.data = output_bias
            down_proj.weight.data = output_weight

    empty_cache(device)


def prune_wanda_sp(args, model, tokenizer, device=None):
    device = resolve_device(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(model, dataloader, device)

    layers = get_decoder_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = get_projection_subset(layer)

        if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            layer_kwargs = move_layer_kwargs(layer_kwargs, dev)

        wrapped_layers = {name: WrappedGPT(module) for name, module in subset.items()}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for handle in handles:
            handle.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            if name == "self_attn.o_proj":
                W_metric = aggregate_attention_metric(layer, W_metric.mean(axis=0))
                thresh = torch.sort(W_metric)[0][int(args.pruning_ratio * W_metric.numel())]
                W_mask = W_metric >= thresh
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr)
            else:
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric)[0][int(W_metric.numel() * args.pruning_ratio)]
                W_mask = W_metric >= thresh
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr)

            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps
        empty_cache(next(iter(subset.values())).weight.device if subset else device)

    model.config.use_cache = use_cache
    empty_cache(device)
