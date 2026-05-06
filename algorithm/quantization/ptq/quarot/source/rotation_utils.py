import model_utils
import torch
import typing
import utils
import transformers
import tqdm, math
import quant_utils
from algorithm.common.device import empty_cache
from algorithm.common.device import preferred_rotation_dtype
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from algorithm.common.hadamard import hadamard_transform

def _resolve_text_root(model):
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported QuaRot backbone root: {type(model)}")


def _resolve_decoder_config(model, root):
    for config in (getattr(root, "config", None), getattr(model, "config", None)):
        if config is None:
            continue
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return text_config
        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            return config
    raise AttributeError(f"Cannot resolve decoder config for model={type(model)} root={type(root)}")


def _resolve_attention_head_dim(decoder_config) -> int:
    head_dim = getattr(decoder_config, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    hidden_size = int(getattr(decoder_config, "hidden_size"))
    num_heads = int(getattr(decoder_config, "num_attention_heads"))
    return hidden_size // num_heads


def _get_token_mixer(layer):
    mixer = getattr(layer, "self_attn", None)
    if mixer is not None:
        return mixer
    mixer = getattr(layer, "linear_attn", None)
    if mixer is not None:
        return mixer
    raise AttributeError(f"Unsupported decoder layer without token mixer: {type(layer)}")


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        norm_gain = layernorm.weight.double()
        if layernorm.__class__.__name__.startswith("Qwen3_5RMSNorm"):
            norm_gain = 1.0 + norm_gain
        linear.weight.data = (W_ * norm_gain).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_merger_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """Fuse Qwen2.5-VL merger RMSNorm into the adjacent grouped linear."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        weight = linear.weight.data.double()
        out_features, in_features = weight.shape
        norm_size = layernorm.weight.shape[0]
        linear.weight.data = (
            (weight.view(out_features, -1, norm_size) * layernorm.weight.double())
            .to(linear_dtype)
            .view(out_features, in_features)
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                weight.view(out_features, -1, norm_size), layernorm.bias.double()
            ).sum(dim=-1)
            linear.bias.data = linear.bias.data.to(linear_dtype)
            
def bake_mean_into_conv(conv: torch.nn.Module) -> None:
    """
    Subtract the output-channel mean so the downstream module receives zero-mean
    features after layer-norm fusion.
    """
    conv_dtype = conv.weight.dtype
    weight = conv.weight.data.double()
    conv.weight.data = (weight - weight.mean(dim=0, keepdim=True)).to(conv_dtype)
    if conv.bias is not None:
        bias = conv.bias.data.double()
        conv.bias.data = (bias - bias.mean()).to(conv_dtype)

def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def rotate_conv(layer, q_visual: torch.Tensor, embed_dims: int):
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = layer.weight.dtype
    weight_shape = layer.weight.data.shape
    weight = layer.weight.data.to(dtype=rotation_dtype).view(embed_dims, -1)
    layer.weight.data = (
        torch.matmul(q_visual.T.to(weight.device, dtype=rotation_dtype), weight)
        .to(dtype=dtype)
        .view(weight_shape)
    )
    if layer.bias is not None:
        bias = layer.bias.data.to(dtype=rotation_dtype)
        layer.bias.data = torch.matmul(bias, q_visual.to(bias.device, dtype=rotation_dtype)).to(
            dtype=dtype,
        )

         
            
def fuse_layer_norms(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            token_mixer = _get_token_mixer(layer)
            if hasattr(token_mixer, "q_proj"):
                input_linears = [token_mixer.q_proj, token_mixer.k_proj, token_mixer.v_proj]
            else:
                input_linears = [
                    token_mixer.in_proj_qkv,
                    token_mixer.in_proj_z,
                    token_mixer.in_proj_b,
                    token_mixer.in_proj_a,
                ]
            fuse_ln_linear(layer.input_layernorm, input_linears)
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    hidden_size = int(getattr(_resolve_decoder_config(model, _resolve_text_root(model)), "hidden_size"))
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(hidden_size),
        replace_layers=False,
    )
    

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    empty_cache(device)
    random_matrix = torch.randn(size, size, dtype=preferred_rotation_dtype(device)).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(dtype=rotation_dtype)
        W.weight.data = torch.matmul(W_, Q.to(W_.device)).to(dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the token-mixer input weights.
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    token_mixer = _get_token_mixer(layer)
    if hasattr(token_mixer, "q_proj"):
        linears = [token_mixer.q_proj, token_mixer.k_proj, token_mixer.v_proj]
    else:
        linears = [
            token_mixer.in_proj_qkv,
            token_mixer.in_proj_z,
            token_mixer.in_proj_b,
            token_mixer.in_proj_a,
        ]
    for W in linears:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=rotation_dtype)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate token-mixer output matrix.
    token_mixer = _get_token_mixer(layer)
    W = token_mixer.o_proj if hasattr(token_mixer, "o_proj") else token_mixer.out_proj

    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=rotation_dtype)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=rotation_dtype)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=rotation_dtype)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)
    
def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=rotation_dtype)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    try:
        apply_exact_had_to_linear(W, had_dim=-1, output=False, device=W.weight.device) #apply exact (inverse) hadamard on the weights of mlp output
    except (AssertionError, ValueError):
        pass
    if W.bias is not None:
        b = W.bias.data.to(dtype=rotation_dtype)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from algorithm.common.hadamard import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().to(utils.DEV).view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    from algorithm.common.hadamard import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().to(utils.DEV), hardK)
    W.weight.data = W.weight.data.to(dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    if hasattr(W, "weight") and W.weight is not None:
        model_type = model_utils.model_type_extractor(model)
        for embedding in model_utils.get_embeddings(model, model_type):
            if (
                hasattr(embedding, "weight")
                and embedding.weight is not None
                and embedding.weight.data_ptr() == W.weight.data_ptr()
            ):
                return
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=rotation_dtype)
    W.weight.data = torch.matmul(W_, Q.to(W_.device)).to(dtype=dtype)


def _get_qwen2_5_vl_visual_root(model):
    multimodal_root = getattr(model, "model", model)
    return getattr(multimodal_root, "visual", None)


def _rotate_qwen2_5_vl_visual_attention_inputs(layer, q_visual: torch.Tensor) -> None:
    qkv = layer.attn.qkv
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = qkv.weight.dtype
    weight = qkv.weight.data.to(dtype=rotation_dtype)
    qkv.weight.data = torch.matmul(weight, q_visual.to(weight.device, dtype=rotation_dtype)).to(
        dtype=dtype,
    )


def _rotate_qwen2_5_vl_visual_attention_output(layer, q_visual: torch.Tensor) -> None:
    proj = layer.attn.proj
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = proj.weight.dtype
    weight = proj.weight.data.to(dtype=rotation_dtype)
    proj.weight.data = torch.matmul(q_visual.T.to(weight.device, dtype=rotation_dtype), weight).to(
        dtype=dtype,
    )
    if proj.bias is not None:
        bias = proj.bias.data.to(dtype=rotation_dtype)
        proj.bias.data = torch.matmul(
            q_visual.T.to(bias.device, dtype=rotation_dtype), bias
        ).to(dtype=dtype)


def _rotate_qwen2_5_vl_visual_mlp_input(layer, q_visual: torch.Tensor) -> None:
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    for linear in (layer.mlp.gate_proj, layer.mlp.up_proj):
        dtype = linear.weight.dtype
        weight = linear.weight.data.to(dtype=rotation_dtype)
        linear.weight.data = torch.matmul(weight, q_visual.to(weight.device, dtype=rotation_dtype)).to(
            dtype=dtype,
        )


def _rotate_qwen2_5_vl_visual_mlp_output(layer, q_visual: torch.Tensor) -> None:
    out_layer = layer.mlp.down_proj
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = out_layer.weight.dtype
    weight = out_layer.weight.data.to(dtype=rotation_dtype)
    out_layer.weight.data = torch.matmul(
        q_visual.T.to(weight.device, dtype=rotation_dtype), weight
    ).to(dtype=dtype)
    if out_layer.bias is not None:
        bias = out_layer.bias.data.to(dtype=rotation_dtype)
        out_layer.bias.data = torch.matmul(
            q_visual.T.to(bias.device, dtype=rotation_dtype), bias
        ).to(dtype=dtype)


def _rotate_qwen2_5_vl_visual_ov_proj(layer) -> None:
    qkv = layer.attn.qkv
    proj = layer.attn.proj
    num_heads = layer.attn.num_heads
    head_dim = qkv.in_features // num_heads
    q_head = get_orthogonal_matrix(head_dim, "hadamard")
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    q_head = q_head.to(dtype=rotation_dtype)

    q_weight, k_weight, v_weight = qkv.weight.data.chunk(3, dim=0)
    qkv_dtype = qkv.weight.dtype
    v_weight_t = v_weight.to(dtype=rotation_dtype).T.reshape(-1, num_heads, head_dim)
    v_weight_rotated = (
        torch.matmul(v_weight_t, q_head).reshape(-1, num_heads * head_dim).T.to(dtype=qkv_dtype)
    )
    qkv.weight.data = torch.cat([q_weight, k_weight, v_weight_rotated], dim=0).contiguous()

    if qkv.bias is not None:
        q_bias, k_bias, v_bias = qkv.bias.data.chunk(3, dim=0)
        v_bias_rotated = (
            torch.matmul(
                v_bias.to(dtype=rotation_dtype).reshape(num_heads, head_dim),
                q_head,
            )
            .reshape(-1)
            .to(dtype=qkv.bias.dtype)
        )
        qkv.bias.data = torch.cat([q_bias, k_bias, v_bias_rotated], dim=0).contiguous()

    proj_dtype = proj.weight.dtype
    proj_weight = proj.weight.data.to(dtype=rotation_dtype).reshape(-1, num_heads, head_dim)
    proj.weight.data = (
        torch.matmul(proj_weight, q_head).reshape(-1, num_heads * head_dim).to(dtype=proj_dtype)
    )


def _rotate_qwen2_5_vl_visual_merger_input(merger, q_visual: torch.Tensor) -> None:
    if not hasattr(merger, "mlp") or len(merger.mlp) == 0:
        return
    first_linear = merger.mlp[0]
    if not isinstance(first_linear, torch.nn.Linear):
        return
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    dtype = first_linear.weight.dtype
    out_features, in_features = first_linear.weight.shape
    q_dim = q_visual.shape[0]
    weight = first_linear.weight.data.to(dtype=rotation_dtype).view(out_features, -1, q_dim)
    first_linear.weight.data = torch.matmul(
        weight, q_visual.to(weight.device, dtype=rotation_dtype)
    ).to(dtype=dtype).view(out_features, in_features)


def rotate_qwen2_5_vl_visual_branch(model, rotate_mode: str) -> None:
    visual = _get_qwen2_5_vl_visual_root(model)
    if visual is None or not hasattr(visual, "blocks") or len(visual.blocks) == 0:
        return

    hidden_size = visual.blocks[0].attn.qkv.in_features
    q_visual = get_orthogonal_matrix(hidden_size, rotate_mode)

    if hasattr(visual, "patch_embed") and hasattr(visual.patch_embed, "proj"):
        rotate_conv(visual.patch_embed.proj, q_visual, hidden_size)

    for layer in visual.blocks:
        _rotate_qwen2_5_vl_visual_attention_inputs(layer, q_visual)
        _rotate_qwen2_5_vl_visual_attention_output(layer, q_visual)
        _rotate_qwen2_5_vl_visual_mlp_input(layer, q_visual)
        _rotate_qwen2_5_vl_visual_mlp_output(layer, q_visual)
        _rotate_qwen2_5_vl_visual_ov_proj(layer)

    if hasattr(visual, "merger"):
        _rotate_qwen2_5_vl_visual_merger_input(visual.merger, q_visual)
    utils.cleanup_memory()


def _get_extra_rotation_modules(model):
    """Return bridge modules that must follow the rotated text hidden basis."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    multimodal_root = getattr(model, "model", model)
    visual = getattr(multimodal_root, "visual", None)
    if visual is None:
        return []
    merger = getattr(visual, "merger", None)
    if (
        model_type == "qwen2_5_vl"
        and merger is not None
        and hasattr(merger, "mlp")
        and len(merger.mlp) > 0
        and isinstance(merger.mlp[-1], torch.nn.Linear)
    ):
        return [merger.mlp[-1]]
    if model_type in {"qwen3_vl", "qwen3_5"}:
        modules = []
        if merger is not None and hasattr(merger, "linear_fc2") and isinstance(merger.linear_fc2, torch.nn.Linear):
            modules.append(merger.linear_fc2)
        deepstack_mergers = getattr(visual, "deepstack_merger_list", None)
        if deepstack_mergers is not None:
            for deepstack_merger in deepstack_mergers:
                if hasattr(deepstack_merger, "linear_fc2") and isinstance(
                    deepstack_merger.linear_fc2, torch.nn.Linear
                ):
                    modules.append(deepstack_merger.linear_fc2)
        return modules
    return []

def rotate_extra_modules(modules, Q: torch.Tensor) -> None:
    """Rotate bridge outputs into the same basis as the text backbone."""
    rotation_dtype = preferred_rotation_dtype(utils.DEV)
    for module in modules:
        dtype = module.weight.data.dtype
        weight = module.weight.data.to(dtype=rotation_dtype)
        module.weight.data = torch.matmul(Q.to(weight.device).T, weight).to(dtype=dtype)
        if module.bias is not None:
            bias = module.bias.data.to(dtype=rotation_dtype)
            module.bias.data = torch.matmul(Q.to(bias.device).T, bias).to(dtype=dtype)

def rotate_ov_proj(layer, model_type, head_num, head_dim):
    if not hasattr(layer, "self_attn"):
        return
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, device=v_proj.weight.device)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False, device=o_proj.weight.device)


@torch.inference_mode()
def rotate_model(model, args):
    text_root = _resolve_text_root(model)
    decoder_config = _resolve_decoder_config(model, text_root)
    hidden_size = int(getattr(decoder_config, "hidden_size"))
    num_heads = int(getattr(decoder_config, "num_attention_heads"))
    head_dim = _resolve_attention_head_dim(decoder_config)
    Q = get_orthogonal_matrix(hidden_size, args.rotate_mode)
    model_type = model_utils.model_type_extractor(model)
    if getattr(getattr(model, "config", None), "model_type", None) == "qwen2_5_vl":
        if bool(getattr(args, "quarot_qwen2_5_vl_rotate_visual_branch", True)):
            rotate_qwen2_5_vl_visual_branch(model, args.rotate_mode)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    if bool(getattr(args, "quarot_qwen2_5_vl_rotate_merger_output", True)):
        rotate_extra_modules(_get_extra_rotation_modules(model), Q)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model,
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        # device_map 模式下将 Q 移动到当前层所在设备
        layer_device = next(layer.parameters()).device
        Q_layer = Q.to(device=layer_device)
        rotate_attention_inputs(layers[idx], Q_layer, model_type)
        rotate_attention_output(layers[idx], Q_layer, model_type)
        rotate_mlp_input(layers[idx], Q_layer, model_type)
        rotate_mlp_output(layers[idx], Q_layer, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        override_num_heads = None if kwargs is None else kwargs.pop("qk_num_heads", None)
        override_head_dim = None if kwargs is None else kwargs.pop("qk_head_dim", None)
        num_heads = int(
            override_num_heads if override_num_heads is not None else config.num_attention_heads
        )
        model_dim = config.hidden_size
        head_dim = int(
            override_head_dim
            if override_head_dim is not None
            else getattr(config, "head_dim", model_dim // num_heads)
        )
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.qk_num_heads = num_heads
        self.qk_head_dim = head_dim
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        self.k_pre_rope = False
        self.k_hadamard = True
        self.k_tokenwise_per_head = False
        self.k_per_head_channel = False
        self.k_equalize = False
        self.k_equalize_alpha = 1.0
        self.k_equalize_max_scale = 8.0
        self.k_equalize_with_q = False
        self.k_equalize_q_power = 1.0
        self._k_equalize_collect = False
        self.k_equalize_scale = None
        self.k_absmax = None
        self.q_absmax = None
        if kwargs is not None:
            valid_groupsize = kwargs['k_groupsize'] == -1 or (
                kwargs['k_groupsize'] > 0 and head_dim % kwargs['k_groupsize'] == 0
            )
            assert valid_groupsize, (
                f'Only token-wise or per-head group sizes dividing head_dim={head_dim} are supported for K-cache'
            )
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_pre_rope = bool(kwargs.get('k_pre_rope', False))
            self.k_hadamard = bool(kwargs.get('k_hadamard', True))
            self.k_tokenwise_per_head = bool(kwargs.get('k_tokenwise_per_head', False))
            self.k_per_head_channel = bool(kwargs.get('k_per_head_channel', False))
            self.k_equalize = bool(kwargs.get('k_equalize', False))
            self.k_equalize_alpha = float(kwargs.get('k_equalize_alpha', 1.0))
            self.k_equalize_max_scale = float(kwargs.get('k_equalize_max_scale', 8.0))
            self.k_equalize_with_q = bool(kwargs.get('k_equalize_with_q', False))
            self.k_equalize_q_power = float(kwargs.get('k_equalize_q_power', 1.0))
            if self.k_per_head_channel and self.k_groupsize != head_dim:
                raise ValueError(
                    f'Per-head-channel K quantization currently requires k_groupsize=head_dim={head_dim}, '
                    f'but got {self.k_groupsize}.'
                )
            if self.k_equalize and self.k_pre_rope:
                raise ValueError('K equalization currently supports only post-RoPE K quantization.')
            quant_groupsize = -1 if self.k_groupsize == -1 else self.k_groupsize
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=quant_groupsize,
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def start_k_equalize_calibration(self) -> None:
        if not self.k_equalize:
            return
        self._k_equalize_collect = True
        self.k_equalize_scale = None
        self.k_absmax = None
        self.q_absmax = None

    def finish_k_equalize_calibration(self) -> None:
        if not self.k_equalize:
            return
        self._k_equalize_collect = False
        if self.k_absmax is None:
            self.k_equalize_scale = None
            return
        eps = torch.tensor(1e-5, device=self.k_absmax.device, dtype=self.k_absmax.dtype)
        k_stats = self.k_absmax.clamp_min(eps)
        if self.k_equalize_with_q and self.q_absmax is not None:
            q_stats = self.q_absmax.to(device=k_stats.device, dtype=k_stats.dtype).clamp_min(eps)
            raw_stats = k_stats / torch.pow(q_stats, self.k_equalize_q_power)
        else:
            raw_stats = k_stats
        reference = torch.exp(torch.mean(torch.log(raw_stats), dim=-1, keepdim=True))
        scale = torch.pow(raw_stats / reference, self.k_equalize_alpha)
        if self.k_equalize_max_scale > 0:
            max_scale = float(self.k_equalize_max_scale)
            scale = torch.clamp(scale, min=1.0 / max_scale, max=max_scale)
        self.k_equalize_scale = scale

    @staticmethod
    def _reduce_q_stats_to_kv_heads(q_stats: torch.Tensor, kv_heads: int) -> torch.Tensor:
        if q_stats.shape[0] == kv_heads:
            return q_stats
        if q_stats.shape[0] % kv_heads != 0:
            raise ValueError(
                f'Cannot reduce Q stats from q_heads={q_stats.shape[0]} to kv_heads={kv_heads}.'
            )
        group_size = q_stats.shape[0] // kv_heads
        return q_stats.view(kv_heads, group_size, q_stats.shape[-1]).amax(dim=1)

    def _update_k_equalize_stats(self, q_bhsd: torch.Tensor, k_bhsd: torch.Tensor) -> None:
        current_k = k_bhsd.detach().abs().amax(dim=(0, 2))
        if self.k_absmax is None:
            self.k_absmax = current_k
        else:
            self.k_absmax = torch.maximum(self.k_absmax.to(current_k.device), current_k)

        if not self.k_equalize_with_q:
            return

        current_q = q_bhsd.detach().abs().amax(dim=(0, 2))
        current_q = self._reduce_q_stats_to_kv_heads(current_q, current_k.shape[0])
        if self.q_absmax is None:
            self.q_absmax = current_q
        else:
            self.q_absmax = torch.maximum(self.q_absmax.to(current_q.device), current_q)

    def _apply_k_equalize(self, q_bhsd: torch.Tensor, k_bhsd: torch.Tensor):
        if not self.k_equalize or self.k_equalize_scale is None:
            return q_bhsd, k_bhsd
        k_scale = self.k_equalize_scale.to(device=q_bhsd.device, dtype=q_bhsd.dtype).unsqueeze(0).unsqueeze(2)
        if q_bhsd.shape[1] == k_bhsd.shape[1]:
            q_scale = k_scale
        else:
            if q_bhsd.shape[1] % k_bhsd.shape[1] != 0:
                raise ValueError(
                    f'Cannot expand K equalization scale from kv_heads={k_bhsd.shape[1]} '
                    f'to query_heads={q_bhsd.shape[1]}.'
                )
            q_scale = k_scale.repeat_interleave(q_bhsd.shape[1] // k_bhsd.shape[1], dim=1)
        return q_bhsd * q_scale, k_bhsd / k_scale

    @staticmethod
    def _rotate_half_along_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        half = x.shape[dim] // 2
        x1, x2 = torch.split(x, half, dim=dim)
        return torch.cat((-x2, x1), dim=dim)

    @staticmethod
    def _infer_seq_rot_dims(x: torch.Tensor, cos: torch.Tensor) -> tuple[int, int]:
        if x.dim() != 4:
            raise ValueError(f'Expected 4D q/k tensor, but got shape={tuple(x.shape)}')

        seq_len = cos.shape[-2]
        rot_dim = cos.shape[-1]

        seq_candidates = [d for d in range(1, x.dim()) if x.shape[d] == seq_len]
        rot_candidates = [d for d in range(1, x.dim()) if x.shape[d] == rot_dim]
        if not seq_candidates or not rot_candidates:
            raise ValueError(
                f'Cannot infer sequence/head dims from x={tuple(x.shape)}, cos={tuple(cos.shape)}'
            )

        seq_dim = 2 if 2 in seq_candidates else seq_candidates[0]
        rot_dim_idx = (x.dim() - 1) if (x.dim() - 1) in rot_candidates and (x.dim() - 1) != seq_dim else None
        if rot_dim_idx is None:
            rot_dim_idx = next(d for d in rot_candidates if d != seq_dim)

        return seq_dim, rot_dim_idx

    @staticmethod
    def _to_canonical_bhsd(x: torch.Tensor, seq_dim: int, rot_dim: int) -> tuple[torch.Tensor, list[int]]:
        # Convert to [B, H, S, D] for Hadamard/K-quantization then restore later.
        head_dim = next(d for d in (1, 2, 3) if d not in (seq_dim, rot_dim))
        perm = [0, head_dim, seq_dim, rot_dim]
        x_bhsd = x.permute(perm).contiguous()
        inv_perm = [0] * 4
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return x_bhsd, inv_perm

    def _apply_rotary_fallback(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if cos.shape[0] == 1 and q.shape[0] != 1:
            cos = cos.expand(q.shape[0], -1, -1)
            sin = sin.expand(q.shape[0], -1, -1)

        seq_dim, rot_dim = self._infer_seq_rot_dims(q, cos)

        view_shape = [1] * q.dim()
        view_shape[0] = cos.shape[0]
        view_shape[seq_dim] = cos.shape[1]
        view_shape[rot_dim] = cos.shape[2]
        cos_b = cos.view(view_shape)
        sin_b = sin.view(view_shape)

        q_embed = (q * cos_b) + (self._rotate_half_along_dim(q, rot_dim) * sin_b)
        k_embed = (k * cos_b) + (self._rotate_half_along_dim(k, rot_dim) * sin_b)
        return q_embed, k_embed

    @staticmethod
    def _get_cos_for_layout(args, tensor_hint: torch.Tensor) -> torch.Tensor:
        if len(args) >= 3 and torch.is_tensor(args[2]):
            cos = args[2]
            if cos.dim() == 2:
                cos = cos.unsqueeze(0)
            return cos
        return torch.empty(
            tensor_hint.shape[0],
            tensor_hint.shape[-2],
            tensor_hint.shape[-1],
            device=tensor_hint.device,
            dtype=tensor_hint.dtype,
        )

    def _call_rotary(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except RuntimeError as err:
            err_text = str(err)
            if (
                len(args) >= 4
                and all(torch.is_tensor(t) for t in args[:4])
                and ("must match the size" in err_text or "size of tensor" in err_text)
            ):
                return self._apply_rotary_fallback(args[0], args[1], args[2], args[3])
            raise err

    def _to_quant_space(self, q: torch.Tensor, k: torch.Tensor, cos_for_layout: torch.Tensor):
        seq_dim, rot_dim = self._infer_seq_rot_dims(q, cos_for_layout)
        q_bhsd, q_inv_perm = self._to_canonical_bhsd(q, seq_dim, rot_dim)
        k_bhsd, k_inv_perm = self._to_canonical_bhsd(k, seq_dim, rot_dim)

        if self.k_hadamard:
            dtype = q_bhsd.dtype
            q_bhsd = hadamard_transform(q_bhsd.float(), scale=1/math.sqrt(q_bhsd.shape[-1])).to(dtype)
            k_bhsd = hadamard_transform(k_bhsd.float(), scale=1/math.sqrt(k_bhsd.shape[-1])).to(dtype)
        return q_bhsd, k_bhsd, q_inv_perm, k_inv_perm

    @staticmethod
    def _from_quant_space(q_bhsd, k_bhsd, q_inv_perm, k_inv_perm):
        q_out = q_bhsd.permute(q_inv_perm).contiguous()
        k_out = k_bhsd.permute(k_inv_perm).contiguous()
        return q_out, k_out

    def _quantize_k_per_head_channel_bhsd(self, k_bhsd: torch.Tensor, q_bhsd: torch.Tensor) -> torch.Tensor:
        maxq = self.k_quantizer.maxq.to(k_bhsd.device)
        x_dtype = k_bhsd.dtype

        xmax = torch.amax(k_bhsd, dim=2, keepdim=True) * self.k_clip_ratio
        xmin = torch.amin(k_bhsd, dim=2, keepdim=True) * self.k_clip_ratio
        if self.k_sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            zeros = xmax == 0
            scale = xmax / maxq
            scale[zeros] = 1
            k_bhsd = quant_utils.sym_quant_dequant(k_bhsd, scale, maxq).to(x_dtype)
        else:
            zeros = (xmin == 0) & (xmax == 0)
            xmin = xmin.clone()
            xmax = xmax.clone()
            xmin[zeros] = -1
            xmax[zeros] = 1
            scale = (xmax - xmin).clamp(min=1e-5) / maxq
            zero = torch.round(-xmin / scale)
            k_bhsd = quant_utils.asym_quant_dequant(k_bhsd, scale, zero, maxq).to(x_dtype)
        return k_bhsd.to(q_bhsd)

    def _quantize_k_bhsd(self, k_bhsd: torch.Tensor, q_bhsd: torch.Tensor) -> torch.Tensor:
        if self.k_bits == 16:
            return k_bhsd

        (bsz, num_heads, seq_len, head_dim) = k_bhsd.shape
        kv_hidden_size = num_heads * head_dim

        if self.k_per_head_channel:
            return self._quantize_k_per_head_channel_bhsd(k_bhsd, q_bhsd)

        if self.k_groupsize == -1 and not self.k_tokenwise_per_head: #token-wise quantization across KV heads
            token_wise_k = k_bhsd.transpose(1, 2).reshape(-1, kv_hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k_bhsd = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q_bhsd)
        else: #strict per-KV-head grouped quantization
            per_head_k = k_bhsd.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k_bhsd = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q_bhsd)

        self.k_quantizer.free()
        return k_bhsd

    def forward(self, *args, **kwargs):
        if len(args) < 2 or not all(torch.is_tensor(t) for t in args[:2]):
            return self._call_rotary(*args, **kwargs)

        if self.k_pre_rope:
            cos_for_layout = self._get_cos_for_layout(args, args[0])
            q_bhsd, k_bhsd, q_inv_perm, k_inv_perm = self._to_quant_space(args[0], args[1], cos_for_layout)
            k_bhsd = self._quantize_k_bhsd(k_bhsd, q_bhsd)
            q_in, k_in = self._from_quant_space(q_bhsd, k_bhsd, q_inv_perm, k_inv_perm)
            return self._call_rotary(q_in, k_in, *args[2:], **kwargs)

        q, k = self._call_rotary(*args, **kwargs)
        if not self.k_hadamard and self.k_bits == 16:
            return q, k

        cos_for_layout = self._get_cos_for_layout(args, q)
        q_bhsd, k_bhsd, q_inv_perm, k_inv_perm = self._to_quant_space(q, k, cos_for_layout)
        if self._k_equalize_collect:
            self._update_k_equalize_stats(q_bhsd, k_bhsd)
            return self._from_quant_space(q_bhsd, k_bhsd, q_inv_perm, k_inv_perm)
        q_bhsd, k_bhsd = self._apply_k_equalize(q_bhsd, k_bhsd)
        k_bhsd = self._quantize_k_bhsd(k_bhsd, q_bhsd)
        return self._from_quant_space(q_bhsd, k_bhsd, q_inv_perm, k_inv_perm)



class LinearAttnQKRotationWrapper(QKRotationWrapper):

    def forward(self, query: torch.Tensor, key: torch.Tensor, *args, **kwargs):
        if not (torch.is_tensor(query) and torch.is_tensor(key)):
            return self.func(query, key, *args, **kwargs)

        q_bhsd = query.permute(0, 2, 1, 3).contiguous()
        k_bhsd = key.permute(0, 2, 1, 3).contiguous()
        if self.k_hadamard:
            dtype = q_bhsd.dtype
            q_bhsd = hadamard_transform(q_bhsd.float(), scale=1 / math.sqrt(q_bhsd.shape[-1])).to(dtype)
            k_bhsd = hadamard_transform(k_bhsd.float(), scale=1 / math.sqrt(k_bhsd.shape[-1])).to(dtype)

        if self._k_equalize_collect:
            self._update_k_equalize_stats(q_bhsd, k_bhsd)
        else:
            q_bhsd, k_bhsd = self._apply_k_equalize(q_bhsd, k_bhsd)
            k_bhsd = self._quantize_k_bhsd(k_bhsd, q_bhsd)

        query = q_bhsd.permute(0, 2, 1, 3).contiguous()
        key = k_bhsd.permute(0, 2, 1, 3).contiguous()
        return self.func(query, key, *args, **kwargs)


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)
    return wrapper


def add_qk_rotation_wrapper_to_linear_attn(module, *args, **kwargs):
    wrappers = {}
    for attr_name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        if not hasattr(module, attr_name):
            continue
        original = getattr(module, attr_name)
        wrapper = LinearAttnQKRotationWrapper(original, *args, **kwargs)
        setattr(module, attr_name, wrapper)
        wrappers[attr_name] = wrapper

    if not wrappers:
        return None

    primary_wrapper = wrappers.get("chunk_gated_delta_rule", next(iter(wrappers.values())))
    setattr(module, "linear_attn_qk_rotation_wrapper", primary_wrapper)
    if "recurrent_gated_delta_rule" in wrappers:
        setattr(module, "linear_attn_recurrent_qk_rotation_wrapper", wrappers["recurrent_gated_delta_rule"])
    return primary_wrapper
# Synchronize quantization device_map support for multi-GPU execution.
