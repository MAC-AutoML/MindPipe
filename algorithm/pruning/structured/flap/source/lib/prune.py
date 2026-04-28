import torch
import torch.nn as nn
from algorithm.common.device import empty_cache
from algorithm.common.device import resolve_device
from algorithm.common.modeling import (
    get_attn_output_proj,
    get_attn_projections as _get_attn_projections,
    get_head_geometry as _get_head_geometry,
    get_mlp_projections as _get_mlp_projections,
    get_q_stride,
    supports_head_pruning,
)
from .layerwrapper import BiasGPT
import math
from tqdm import tqdm

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def resolve_linear_module(module, layers=(nn.Linear,)):
    layer_types = tuple(layers)
    if isinstance(module, layer_types):
        return module
    for attr_name in ("linear", "module"):
        child = getattr(module, attr_name, None)
        if isinstance(child, layer_types):
            return child
    return None


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
    resolved_module = resolve_linear_module(module, layers)
    if resolved_module is not None:
        return {name: resolved_module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_attention_projections(layer):
    """获取注意力投影层，linear_attention 层返回 None。"""
    result = _get_attn_projections(layer)
    if result is None:
        return None
    q, k, v, o = result
    return (
        resolve_linear_module(q),
        resolve_linear_module(k),
        resolve_linear_module(v),
        resolve_linear_module(o),
    )


def get_mlp_projections(layer):
    """获取 MLP 投影层。"""
    up, gate, down = _get_mlp_projections(layer)
    return (
        resolve_linear_module(up),
        resolve_linear_module(gate),
        resolve_linear_module(down),
    )


def get_projection_subset(layer):
    """获取 FLAP 剪枝目标投影子集。

    full_attention 层返回 attn_output + mlp.down_proj，
    linear_attention 层仅返回 mlp.down_proj（不剪 attention）。
    """
    down_proj = resolve_linear_module(layer.mlp.down_proj)
    if not supports_head_pruning(layer):
        # linear_attention 层：只剪 MLP
        if down_proj is None:
            raise KeyError("Failed to resolve mlp.down_proj")
        return {"mlp.down_proj": down_proj}
    o_proj = resolve_linear_module(layer.self_attn.o_proj)
    if o_proj is None or down_proj is None:
        available = ", ".join(sorted(find_layers(layer).keys()))
        raise KeyError(
            "Failed to resolve FLAP target projections. "
            f"available_layers=[{available}]"
        )
    return {
        "attn_output": o_proj,
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


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.

    Args:
        model (nn.Module): The model to check.

    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    decoder_config = get_decoder_root(model).config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False

    layers = get_decoder_layers(model)
    intermediate_size = decoder_config.intermediate_size
    hidden_size = decoder_config.hidden_size

    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    decoder_config.use_cache = use_cache
    return float(count)/total_params


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        layer_kwargs (dict): Cached decoder-layer kwargs.
    """
    layers = get_decoder_layers(model)
    decoder_config = get_decoder_root(model).config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False

    # device_map 模式下直接从模型参数获取设备
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


def move_layer_kwargs(layer_kwargs, device):
    moved_kwargs = {}
    for name, value in layer_kwargs.items():
        if torch.is_tensor(value):
            moved_kwargs[name] = value.to(device)
        elif isinstance(value, tuple):
            moved_kwargs[name] = tuple(
                item.to(device) if torch.is_tensor(item) else item
                for item in value
            )
        else:
            moved_kwargs[name] = value
    return moved_kwargs


def get_attention_head_geometry(layer):
    """获取 head 几何信息，linear_attention 层返回 None。"""
    return _get_head_geometry(layer)


def aggregate_attention_metric(layer, metric):
    """将逐通道 metric 聚合为逐 head metric。linear_attention 层不聚合。"""
    geo = get_attention_head_geometry(layer)
    if geo is None:
        return metric
    num_heads, num_kv_heads, num_kv_groups, head_dim = geo
    return metric.reshape(num_heads, head_dim).sum(dim=1)


def expand_attention_masks(layer, attn_mask, device):
    """将 per-head mask 展开为 per-channel mask。

    返回 (q_row_mask, kv_mask, o_col_mask)：
    - q_row_mask: q_proj 的行掩码（考虑 query+gate 绑定的 q_stride）
    - kv_mask: k_proj/v_proj 的行掩码
    - o_col_mask: o_proj 的列掩码（标准 head_dim 步长）
    """
    num_heads, num_kv_heads, num_kv_groups, head_dim = get_attention_head_geometry(layer)
    q_stride = get_q_stride(layer)

    if attn_mask.numel() != num_heads:
        raise ValueError(
            f"Expected per-query-head FLAP mask with {num_heads} elements, got {attn_mask.numel()}."
        )
    # q_proj 行掩码：每个 head 占 q_stride 行（标准模型=head_dim，Qwen3.5=head_dim*2）
    q_row_mask = attn_mask.repeat_interleave(q_stride)

    # KV 掩码：每个 kv_head 占 head_dim 行
    if num_kv_heads == num_heads:
        kv_mask = attn_mask.repeat_interleave(head_dim)
    else:
        kv_head_mask = attn_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
        kv_mask = kv_head_mask.repeat_interleave(head_dim)

    # o_proj 列掩码：每个 head 占 head_dim 列
    o_col_mask = attn_mask.repeat_interleave(head_dim)

    return q_row_mask.to(device), kv_mask.to(device), o_col_mask.to(device)


def get_attention_compression_weight(layer):
    """获取 attention 压缩权重（用于 AL-AM 阈值搜索）。"""
    geo = get_attention_head_geometry(layer)
    if geo is None:
        return 1.0
    _, _, num_kv_groups, head_dim = geo
    return head_dim * (2.0 + 2.0 / num_kv_groups) / 3.0


def compute_output_bias(mean_input, mask, output_weight):
    baseline_input = mean_input.to(device=output_weight.device, dtype=output_weight.dtype)
    inactive_mask = (~mask).to(device=output_weight.device, dtype=output_weight.dtype)
    return (baseline_input * inactive_mask) @ output_weight.T


def get_layer_weight_count(layer):
    return sum(module.weight.numel() for module in find_layers(layer).values())


def estimate_attention_zero_count(layer, head_keep_mask):
    """估算 attention 部分被剪枝后变为零的权重数。linear_attention 层返回 0。"""
    if not supports_head_pruning(layer):
        return 0
    geo = get_attention_head_geometry(layer)
    if geo is None:
        return 0
    num_heads, num_kv_heads, num_kv_groups, head_dim = geo
    if head_keep_mask.numel() != num_heads:
        raise ValueError(f"Expected {num_heads} attention heads, got {head_keep_mask.numel()}.")
    projs = get_attention_projections(layer)
    if projs is None:
        return 0
    q_proj, k_proj, v_proj, o_proj = projs
    q_stride = get_q_stride(layer)
    removed_query_heads = int((~head_keep_mask).sum().item())
    removed_query_rows = removed_query_heads * q_stride
    zero_count = 0
    zero_count += removed_query_rows * q_proj.weight.shape[1]
    zero_count += removed_query_heads * head_dim * o_proj.weight.shape[0]

    kv_keep_mask = head_keep_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
    removed_kv_rows = int((~kv_keep_mask).sum().item()) * head_dim
    zero_count += removed_kv_rows * k_proj.weight.shape[1]
    zero_count += removed_kv_rows * v_proj.weight.shape[1]
    return zero_count


def estimate_mlp_zero_count(layer, neuron_keep_mask):
    up_proj, gate_proj, down_proj = get_mlp_projections(layer)
    removed_neurons = int((~neuron_keep_mask).sum().item())
    zero_count = 0
    zero_count += removed_neurons * up_proj.weight.shape[1]
    zero_count += removed_neurons * gate_proj.weight.shape[1]
    zero_count += removed_neurons * down_proj.weight.shape[0]
    return zero_count


def estimate_unstructured_sparsity(layers, attn_masks, mlp_masks):
    zero_count = 0
    total_count = 0
    for layer, attn_mask, mlp_mask in zip(layers, attn_masks, mlp_masks):
        total_count += get_layer_weight_count(layer)
        if attn_mask is not None:
            zero_count += estimate_attention_zero_count(layer, attn_mask)
        zero_count += estimate_mlp_zero_count(layer, mlp_mask)
    return zero_count / max(total_count, 1)


def _build_full_attn_masks(attn_metric_heads, threshold, layer_has_attn, n_layers):
    """根据阈值构建 full-size 的 attn_mask 列表，linear_attention 层填 None。"""
    attn_mask_full = attn_metric_heads > threshold
    result = [None] * n_layers
    full_idx = 0
    for i in range(n_layers):
        if layer_has_attn[i]:
            result[i] = attn_mask_full[full_idx]
            full_idx += 1
    return result


def find_threshold_for_target_unstructured_sparsity(layers, attn_metric, mlp_metric, target_ratio, layer_has_attn=None):
    candidates = torch.sort(torch.cat([attn_metric.reshape(-1), mlp_metric.reshape(-1)])).values
    low, high = 0, candidates.numel() - 1
    best_threshold = candidates[0]
    best_ratio = 0.0
    best_diff = float("inf")

    while low <= high:
        mid = (low + high) // 2
        threshold = candidates[mid]
        attn_mask = attn_metric > threshold
        mlp_mask = mlp_metric > threshold

        if layer_has_attn is not None:
            # 混合架构：构建 full-size mask，linear_attention 层为 None
            full_attn_masks = _build_full_attn_masks(attn_metric, threshold, layer_has_attn, len(layers))
            estimated_ratio = estimate_unstructured_sparsity(layers, full_attn_masks, mlp_mask)
        else:
            estimated_ratio = estimate_unstructured_sparsity(layers, attn_mask, mlp_mask)

        diff = abs(estimated_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_ratio = estimated_ratio
            best_threshold = threshold
        if estimated_ratio < target_ratio:
            low = mid + 1
        else:
            high = mid - 1

    return best_threshold, best_ratio


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.

    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    up_proj, gate_proj, down_proj = get_mlp_projections(layer)

    # linear_attention 层没有标准 head 结构，跳过 attention 剪枝
    can_prune_attn = supports_head_pruning(layer) and attn_mask is not None

    if can_prune_attn:
        projs = get_attention_projections(layer)
        q_proj, k_proj, v_proj, o_proj = projs

    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if can_prune_attn:
            q_row_mask, kv_mask, o_col_mask = expand_attention_masks(layer, attn_mask, device)
            # Apply the mask to the query, key and value projection weights
            q_proj.weight.data *= q_row_mask.unsqueeze(-1)
            k_proj.weight.data *= kv_mask.unsqueeze(-1)
            v_proj.weight.data *= kv_mask.unsqueeze(-1)

            output_weight = o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                # 注意：用 o_col_mask（与 o_proj 输入维度一致），不用 q_row_mask
                output_bias = compute_output_bias(attn_mean_inp, o_col_mask, output_weight)
                if o_proj.bias is None:
                    o_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
            # In GQA, masking q_proj alone does not zero the pruned head output because
            # the head can still read shared K/V states. Zero the corresponding o_proj
            # input columns to suppress the removed query-head contribution explicitly.
            o_proj.weight.data *= o_col_mask.unsqueeze(0)
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                o_proj.bias.data = output_bias

        # MLP Weight Masking
        if mlp_mask is not None:
            # Apply the mask to the up and gate projection weights
            up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            output_weight = down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = compute_output_bias(mlp_mean_inp, mlp_mask.to(device), output_weight)
                if down_proj.bias is None:
                    down_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )

            # Mirror attention pseudo-pruning: zero the inactive down_proj input columns
            # so observed weight sparsity matches the effective removed MLP channels.
            down_proj.weight.data *= mlp_mask.unsqueeze(0).to(device)

            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                down_proj.bias.data = output_bias

    else:
        # Real Pruning
        # Attention Weight Pruning
        if can_prune_attn:
            num_heads, num_kv_heads, num_kv_groups, head_dim = get_attention_head_geometry(layer)
            retain_heads = int(torch.count_nonzero(attn_mask).item())
            q_row_mask, kv_mask, o_col_mask = expand_attention_masks(layer, attn_mask, device)
            q_indices = torch.where(q_row_mask)[0]
            kv_indices = torch.where(kv_mask)[0]
            o_indices = torch.where(o_col_mask)[0]

            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            q_proj.weight.data = q_proj.weight.data[q_indices]
            k_proj.weight.data = k_proj.weight.data[kv_indices]
            v_proj.weight.data = v_proj.weight.data[kv_indices]

            # Update output dimensions of q, k, v projections based on remaining heads
            q_proj.out_features = q_indices.numel()
            k_proj.out_features = kv_indices.numel()
            v_proj.out_features = kv_indices.numel()

            output_weight = o_proj.weight.data

            if bias:
                # Add the additional bias to compensate for the loss
                # 注意：用 o_col_mask（与 o_proj 输入维度一致），不用 q_row_mask
                output_bias = compute_output_bias(attn_mean_inp, o_col_mask.to(device), output_weight)

            # Prune the output projection weight
            output_weight = o_proj.weight.data[:, o_indices]
            # 更新 attention 属性（包括 GQA 相关属性）
            new_num_kv_heads = kv_indices.numel() // head_dim
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * head_dim
            for attr, val in [
                ("num_attention_heads", retain_heads),
                ("num_key_value_heads", new_num_kv_heads),
                ("num_key_value_groups", num_kv_groups),
            ]:
                try:
                    setattr(layer.self_attn, attr, val)
                except AttributeError:
                    pass

            if bias:
                # Re-initialize the Linear layer with new shape and bias
                o_proj.in_features = o_indices.numel()
                if o_proj.bias is None:
                    o_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
                o_proj.bias.data = output_bias

            # Assign the pruned weights
            o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            mlp_indices = torch.where(mlp_mask)[0]
            # Prune the up and gate projection weights
            up_proj.weight.data = up_proj.weight.data[mlp_indices]
            gate_proj.weight.data = gate_proj.weight.data[mlp_indices]

            # Update output dimensions of up and gate projections based on the mlp mask
            up_proj.out_features = mlp_indices.numel()
            gate_proj.out_features = mlp_indices.numel()

            output_weight = down_proj.weight.data
            layer.mlp.intermediate_size = mlp_indices.numel()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = compute_output_bias(mlp_mean_inp, mlp_mask.to(device), output_weight)

            # Prune the down projection weight
            output_weight = down_proj.weight.data[:, mlp_indices]

            if bias:
                # Re-initialize the Linear layer with new shape and bias
                down_proj.in_features = mlp_indices.numel()
                if down_proj.bias is None:
                    down_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
                down_proj.bias.data = output_bias

            # Assign the pruned weights
            down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    empty_cache(device)


def cal_remove_neuron(args, model):
    decoder_config = get_decoder_root(model).config
    intermediate_size = decoder_config.intermediate_size
    hidden_size = decoder_config.hidden_size
    num_layers = decoder_config.num_hidden_layers
    head_dim = hidden_size // decoder_config.num_attention_heads
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * head_dim
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * head_dim
        return int((remove_params - remove_head_params) / (hidden_size * 3))


def prune_flap(args, model, tokenizer, device=None, dataloader=None):
    """
    Our FLAP Pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
        dataloader (list, optional): Pre-loaded calibration batches from method.py.
    """
    device = resolve_device(device)
    decoder_config = get_decoder_root(model).config
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False

    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(model, dataloader, device)
    layers = get_decoder_layers(model)

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    # 记录哪些层有 attn_output（用于后续索引对齐）
    layer_has_attn = []

    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = get_projection_subset(layer)
        has_attn = "attn_output" in subset
        layer_has_attn.append(has_attn)

        # 获取当前层所在设备，只搬输入数据
        target_dev = next(layer.parameters()).device
        inps, outs = inps.to(target_dev), outs.to(target_dev)
        layer_kwargs = move_layer_kwargs(layer_kwargs, target_dev)

        wrapped_layers = {}
        for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

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
            if name == 'attn_output':
                raw_attn_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                if args.structure == "UL-UM":
                    W_metric = aggregate_attention_metric(layer, raw_attn_metric)
                    attn_count = W_metric.numel()
                    thresh = torch.sort(W_metric)[0][int(args.pruning_ratio * attn_count)]
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    W_metric = aggregate_attention_metric(layer, raw_attn_metric)
                    thresh = torch.sort(W_metric)[0][args.remove_heads // len(layers)]
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(raw_attn_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                if args.structure == "UL-UM":
                    thresh = torch.sort(W_metric)[0][int(W_metric.numel()*args.pruning_ratio)]
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    thresh = torch.sort(W_metric)[0][cal_remove_neuron(args, model)]
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    if args.structure in ["AL-MM", "AL-AM"]:
        if not attn_metric_list:
            raise ValueError("没有可剪枝的 attention 层（所有层都是 linear_attention）")

        # 只对 full_attention 层做 head 级聚合
        full_attn_indices = [i for i, h in enumerate(layer_has_attn) if h]
        attn_metric_full = torch.stack(attn_metric_list)  # [n_full_attn, columns]
        attn_metric_full = standarlization(attn_metric_full)

        # 从第一个 full_attention 层获取 head_dim
        first_full_attn = layers[full_attn_indices[0]]
        attn_head_dim = get_attention_head_geometry(first_full_attn)[3]
        attn_metric_heads = attn_metric_full.reshape(len(full_attn_indices), -1, attn_head_dim).mean(dim=2)

        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)

        if args.structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric_heads.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask_full = (attn_metric_heads > attn_thres)

            sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
            mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
            mlp_mask_all = (mlp_metric > mlp_thres)
        else:
            # AL-AM
            prune_metric = torch.cat([attn_metric_heads.view(-1), mlp_metric.view(-1)])
            if args.unstr:
                threshold, estimated_ratio = find_threshold_for_target_unstructured_sparsity(
                    layers,
                    attn_metric_heads,
                    mlp_metric,
                    args.pruning_ratio,
                    layer_has_attn=layer_has_attn,
                )
                print(
                    f"AL-AM exact sparsity targeting: requested={args.pruning_ratio:.4f}, "
                    f"estimated={estimated_ratio:.4f}"
                )
            else:
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(sorted_prune, dtype=torch.float32)
                attn_weights = []
                for layer_index in range(len(full_attn_indices)):
                    layer_weight = get_attention_compression_weight(layers[full_attn_indices[layer_index]])
                    attn_weights.append(
                        torch.full_like(attn_metric_heads[layer_index], fill_value=layer_weight, dtype=torch.float32)
                    )
                attn_weights = torch.stack(attn_weights).view(-1).to(sorted_prune.device)
                attn_positions = indices < attn_metric_heads.numel()
                compression_weight[attn_positions] = attn_weights[indices[attn_positions]]
                threshold = sorted_prune[
                    torch.argmin(
                        torch.abs(
                            torch.cumsum(compression_weight, 0)
                            - torch.sum(compression_weight) * (1 - args.pruning_ratio)
                        )
                    )
                ]
            attn_mask_full = (attn_metric_heads > threshold)
            mlp_mask_all = (mlp_metric > threshold)

        # 将 attn_mask 对齐到所有层：linear_attention 层为 None
        attn_mask = [None] * len(layers)
        full_idx = 0
        for i in range(len(layers)):
            if layer_has_attn[i]:
                attn_mask[i] = attn_mask_full[full_idx]
                full_idx += 1

        mlp_mask = mlp_mask_all
    else:
        # UL-UM / UL-MM 模式：补齐 linear_attention 层的 attn_mask 为 None
        if any(not h for h in layer_has_attn):
            aligned_attn_mask = []
            attn_idx = 0
            for i in range(len(layers)):
                if layer_has_attn[i]:
                    aligned_attn_mask.append(attn_mask[attn_idx])
                    attn_idx += 1
                else:
                    aligned_attn_mask.append(None)
            attn_mask = aligned_attn_mask
        else:
            attn_mask = torch.stack(attn_mask)
        mlp_mask = torch.stack(mlp_mask)

    # 应用剪枝：attn_baseline_inp_list 只有 full_attention 层的条目
    attn_inp_idx = 0
    for idx in range(len(layers)):
        target_dev = next(layers[idx].parameters()).device

        if layer_has_attn[idx]:
            compress(layers[idx], attn_mask[idx], None, attn_baseline_inp_list[attn_inp_idx], None, target_dev, unstr=args.unstr)
            attn_inp_idx += 1
        compress(layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], target_dev, unstr=args.unstr)

    decoder_config.use_cache = use_cache


def prune_magnitude_sp(args, model, tokenizer, device=None):
    """
    Magnitude Pruning on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    device = resolve_device(device)
    layers = get_decoder_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = get_projection_subset(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0)

            if name == 'attn_output':
                head_dim = get_attention_head_geometry(layer)[3]
                W_metric = W_metric.reshape(-1, head_dim).sum(dim=1) # importance score of each head
                thresh = torch.sort(W_metric)[0][int(args.pruning_ratio*layer.self_attn.num_heads)]
                W_mask = (W_metric>=thresh)
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr)
            else:
                thresh = torch.sort(W_metric)[0][int(W_metric.numel()*args.pruning_ratio)]
                W_mask = (W_metric>=thresh)
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr)

# Add pruning support for Qwen3.5.
