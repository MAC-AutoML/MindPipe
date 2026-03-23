import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT, BiasGPT
from .data import get_loaders 
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
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = get_decoder_layers(model)
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
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

    model.config.use_cache = use_cache 
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
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_decoder_layers(model)

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    sample_count = len(dataloader)
    inps = torch.zeros((sample_count, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
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
            input_ids = batch[0].to(device)
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            model(input_ids=input_ids, attention_mask=attention_mask)
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache

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
    attn = layer.self_attn
    config = getattr(attn, "config", None)
    num_heads = int(getattr(attn, "num_heads", getattr(config, "num_attention_heads")))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads", num_heads)))
    hidden_size = int(getattr(attn, "hidden_size", getattr(config, "hidden_size")))
    head_dim = int(getattr(attn, "head_dim", hidden_size // num_heads))
    num_kv_groups = num_heads // num_kv_heads
    return num_heads, num_kv_heads, num_kv_groups, head_dim


def aggregate_attention_metric(layer, metric):
    num_heads, num_kv_heads, num_kv_groups, head_dim = get_attention_head_geometry(layer)
    head_metric = metric.reshape(num_heads, head_dim).sum(dim=1)
    return head_metric


def expand_attention_masks(layer, attn_mask, device):
    num_heads, num_kv_heads, num_kv_groups, head_dim = get_attention_head_geometry(layer)
    if attn_mask.numel() != num_heads:
        raise ValueError(
            f"Expected per-query-head FLAP mask with {num_heads} elements, got {attn_mask.numel()}."
        )
    q_mask = attn_mask.repeat_interleave(head_dim)
    if num_kv_heads == num_heads:
        kv_mask = q_mask
    else:
        kv_head_mask = attn_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
        kv_mask = kv_head_mask.repeat_interleave(head_dim)
    return q_mask.to(device), kv_mask.to(device)


def get_attention_compression_weight(layer):
    _, _, num_kv_groups, head_dim = get_attention_head_geometry(layer)
    return head_dim * (2.0 + 2.0 / num_kv_groups) / 3.0


def compute_output_bias(mean_input, mask, output_weight):
    baseline_input = mean_input.to(device=output_weight.device, dtype=output_weight.dtype)
    inactive_mask = (~mask).to(device=output_weight.device, dtype=output_weight.dtype)
    return (baseline_input * inactive_mask) @ output_weight.T


def get_layer_weight_count(layer):
    return sum(module.weight.numel() for module in find_layers(layer).values())


def estimate_attention_zero_count(layer, head_keep_mask):
    num_heads, num_kv_heads, num_kv_groups, head_dim = get_attention_head_geometry(layer)
    if head_keep_mask.numel() != num_heads:
        raise ValueError(f"Expected {num_heads} attention heads, got {head_keep_mask.numel()}.")
    removed_query_heads = int((~head_keep_mask).sum().item())
    removed_query_rows = removed_query_heads * head_dim
    zero_count = 0
    zero_count += removed_query_rows * layer.self_attn.q_proj.weight.shape[1]
    zero_count += removed_query_rows * layer.self_attn.o_proj.weight.shape[0]

    kv_keep_mask = head_keep_mask.reshape(num_kv_heads, num_kv_groups).any(dim=1)
    removed_kv_rows = int((~kv_keep_mask).sum().item()) * head_dim
    zero_count += removed_kv_rows * layer.self_attn.k_proj.weight.shape[1]
    zero_count += removed_kv_rows * layer.self_attn.v_proj.weight.shape[1]
    return zero_count


def estimate_mlp_zero_count(layer, neuron_keep_mask):
    removed_neurons = int((~neuron_keep_mask).sum().item())
    zero_count = 0
    zero_count += removed_neurons * layer.mlp.up_proj.weight.shape[1]
    zero_count += removed_neurons * layer.mlp.gate_proj.weight.shape[1]
    zero_count += removed_neurons * layer.mlp.down_proj.weight.shape[0]
    return zero_count


def estimate_unstructured_sparsity(layers, attn_masks, mlp_masks):
    zero_count = 0
    total_count = 0
    for layer, attn_mask, mlp_mask in zip(layers, attn_masks, mlp_masks):
        total_count += get_layer_weight_count(layer)
        zero_count += estimate_attention_zero_count(layer, attn_mask)
        zero_count += estimate_mlp_zero_count(layer, mlp_mask)
    return zero_count / max(total_count, 1)


def find_threshold_for_target_unstructured_sparsity(layers, attn_metric, mlp_metric, target_ratio):
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
    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            q_mask, kv_mask = expand_attention_masks(layer, attn_mask, device)
            # Apply the mask to the query, key and value projection weights
            layer.self_attn.q_proj.weight.data *= q_mask.unsqueeze(-1)
            layer.self_attn.k_proj.weight.data *= kv_mask.unsqueeze(-1)
            layer.self_attn.v_proj.weight.data *= kv_mask.unsqueeze(-1)
            
            output_weight = layer.self_attn.o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = compute_output_bias(attn_mean_inp, q_mask, output_weight)
                if layer.self_attn.o_proj.bias is None:
                    layer.self_attn.o_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
            # In GQA, masking q_proj alone does not zero the pruned head output because
            # the head can still read shared K/V states. Zero the corresponding o_proj
            # input columns to suppress the removed query-head contribution explicitly.
            layer.self_attn.o_proj.weight.data *= q_mask.unsqueeze(0)
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.self_attn.o_proj.bias.data = output_bias

        # MLP Weight Masking
        if mlp_mask is not None:
            # Apply the mask to the up and gate projection weights
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.mlp.down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = compute_output_bias(mlp_mean_inp, mlp_mask.to(device), output_weight)
                if layer.mlp.down_proj.bias is None:
                    layer.mlp.down_proj.bias = nn.Parameter(
                        torch.zeros(output_weight.shape[0], device=device, dtype=output_weight.dtype)
                    )
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
            
            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            
            output_weight = layer.self_attn.o_proj.weight.data
            
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = compute_output_bias(attn_mean_inp, attn_mask.to(device), output_weight)
                
            # Prune the output projection weight
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * 128
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.self_attn.o_proj.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.self_attn.o_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = compute_output_bias(mlp_mean_inp, mlp_mask.to(device), output_weight)
              
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.mlp.down_proj.in_features = mlp_mask.sum().item()
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.mlp.down_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
        
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()
    
    
def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))


def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(model, dataloader, device)
    layers = get_decoder_layers(model)

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []
        
    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            layer_kwargs = move_layer_kwargs(layer_kwargs, dev)

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
            if name == 'self_attn.o_proj':
                raw_attn_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                if args.structure == "UL-UM":
                    W_metric = aggregate_attention_metric(layer, raw_attn_metric)
                    attn_count = W_metric.numel()
                    thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio * attn_count)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    W_metric = aggregate_attention_metric(layer, raw_attn_metric)
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(raw_attn_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                if args.structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][cal_remove_neuron(args, model)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    if args.structure in ["AL-MM", "AL-AM"]:
        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)
        attn_head_dim = get_attention_head_geometry(layers[0])[3]
        attn_metric = attn_metric.reshape(len(layers), -1, attn_head_dim).mean(dim=2)
        
        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)
        
        if args.structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask = (attn_metric > attn_thres)  # 1 means retain
            
            sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
            mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
            mlp_mask = (mlp_metric > mlp_thres)
        else:
            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            if args.unstr:
                threshold, estimated_ratio = find_threshold_for_target_unstructured_sparsity(
                    layers,
                    attn_metric,
                    mlp_metric,
                    args.pruning_ratio,
                )
                print(
                    f"AL-AM exact sparsity targeting: requested={args.pruning_ratio:.4f}, "
                    f"estimated={estimated_ratio:.4f}"
                )
            else:
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(sorted_prune, dtype=torch.float32)
                attn_weights = []
                for layer_index in range(len(layers)):
                    layer_weight = get_attention_compression_weight(layers[layer_index])
                    attn_weights.append(
                        torch.full_like(attn_metric[layer_index], fill_value=layer_weight, dtype=torch.float32)
                    )
                attn_weights = torch.stack(attn_weights).view(-1).to(sorted_prune.device)
                attn_positions = indices < attn_metric.numel()
                compression_weight[attn_positions] = attn_weights[indices[attn_positions]]
                threshold = sorted_prune[
                    torch.argmin(
                        torch.abs(
                            torch.cumsum(compression_weight, 0)
                            - torch.sum(compression_weight) * (1 - args.pruning_ratio)
                        )
                    )
                ]
            attn_mask = (attn_metric > threshold)
            mlp_mask = (mlp_metric > threshold)
    else:
        attn_mask = torch.stack(attn_mask) 
        mlp_mask = torch.stack(mlp_mask)
    
    for idx in range(len(layers)):
        if f"model.layers.{idx}" in getattr(model, 'hf_device_map', {}): 
            compress(layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
        else:
            compress(layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device, unstr=args.unstr)
                
        if f"model.layers.{idx}" in getattr(model, 'hf_device_map', {}): 
            compress(layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
        else:
            compress(layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, unstr=args.unstr)
                
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
   
   
def prune_wanda_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Wanda on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=128,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(model, dataloader, device)

    layers = get_decoder_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            layer_kwargs = move_layer_kwargs(layer_kwargs, dev)

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
            
            if name == 'self_attn.o_proj':
                W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1)    # importance score of each head
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr)
            else:
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr)
          
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    
def prune_magnitude_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Magnitude Pruning on structured pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    layers = get_decoder_layers(model) 

    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0)

            if name == 'self_attn.o_proj':
                W_metric = W_metric.reshape(-1, 128).sum(dim=1) # importance score of each head
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr)
            else:
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr)
            
