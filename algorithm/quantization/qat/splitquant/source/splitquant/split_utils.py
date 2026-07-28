import os
import re
import torch
from collections import OrderedDict
from splitquant.backbone_utils import get_decoder_layers
from splitquant.function_utils import get_paras_dict_by_name
import logging


_LEGACY_GROUP_PARAM_RE = re.compile(
    r"^(?P<prefix>.+)\.linear_(?P<kind>[uv])_list\.(?P<index>\d+)\.parametrizations\.weight\.original$"
)
_LEGACY_GROUP_DIAG_RE = re.compile(r"^(?P<prefix>.+)\.linear_diag_list\.(?P<index>\d+)$")

def kronecker_matmul(x, hadL, hadR):
    """equivalent to
    
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


def reparameterize_ln(ln, trans):
    # assert isinstance(ln, (LlamaRMSNorm, Qwen2RMSNorm))
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def reparameterize_splitquant_model(model):
    layers = get_decoder_layers(model)
    for idx in range(len(layers)):
        layer = layers[idx]
        layer.self_attn.reparameterize()
        layer.mlp.reparameterize()
        # fuse per-channel scaling to layernorm
        if getattr(layer.self_attn, "ln_trans", None) is not None and layer.self_attn.ln_trans.add_diag:
            reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
        if getattr(layer.mlp, "up_gate_trans", None) is not None and layer.mlp.up_gate_trans.add_diag:
            reparameterize_ln(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
    return model


def save_parametrized_checkpoint(model, args):
    quanted_parameters = {}
    layers = get_decoder_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        quanted_parameters[i] = layer.state_dict()
    torch.save(quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"parametrized_paras.pth")))


def convert_legacy_splitquant_group_parameters(state_dict):
    converted = OrderedDict()
    grouped = {}
    changed = False

    for key, value in state_dict.items():
        match = _LEGACY_GROUP_PARAM_RE.match(key)
        if match:
            prefix = match.group("prefix")
            kind = match.group("kind")
            index = int(match.group("index"))
            grouped.setdefault(prefix, {"u": {}, "v": {}, "diag": {}})[kind][index] = value
            changed = True
            continue

        match = _LEGACY_GROUP_DIAG_RE.match(key)
        if match:
            prefix = match.group("prefix")
            index = int(match.group("index"))
            grouped.setdefault(prefix, {"u": {}, "v": {}, "diag": {}})["diag"][index] = value
            changed = True
            continue

        converted[key] = value

    if not changed:
        return state_dict

    for prefix, parts in grouped.items():
        indices = sorted(parts["u"])
        if not indices or indices != sorted(parts["v"]) or indices != sorted(parts["diag"]):
            raise ValueError(f"Incomplete legacy SplitQuant group parameters for {prefix}")
        expected = list(range(indices[-1] + 1))
        if indices != expected:
            raise ValueError(f"Non-contiguous legacy SplitQuant group indices for {prefix}: {indices[:3]}...{indices[-3:]}")
        converted[f"{prefix}.linear_u_raw"] = torch.stack([parts["u"][idx] for idx in indices], dim=0)
        converted[f"{prefix}.linear_v_raw"] = torch.stack([parts["v"][idx] for idx in indices], dim=0)
        converted[f"{prefix}.linear_diag"] = torch.stack([parts["diag"][idx] for idx in indices], dim=0)

    return converted


def convert_legacy_splitquant_checkpoint(splitquant_parameters):
    converted = {}
    changed = False
    for layer_idx, layer_params in splitquant_parameters.items():
        converted_layer = convert_legacy_splitquant_group_parameters(layer_params)
        converted[layer_idx] = converted_layer
        changed = changed or converted_layer is not layer_params
    return converted if changed else splitquant_parameters


def load_splitquant_parameters(args, model, path=None):
    checkpoint_dir = args.exp_dir if path is None else path
    checkpoint_path = os.path.join(checkpoint_dir, "splitquant_parameters.pth")
    # Checkpoints may have been saved from a different CUDA topology. Load on
    # CPU first, then let load_state_dict copy tensors to each layer's device.
    splitquant_parameters = torch.load(checkpoint_path, map_location="cpu")
    splitquant_parameters = convert_legacy_splitquant_checkpoint(splitquant_parameters)
    layers = get_decoder_layers(model)
    
    for i in range(len(splitquant_parameters.keys())):
        splitquant_param = splitquant_parameters[i]
        layers[i].load_state_dict(splitquant_param, strict=False)
    return model


def save_splitquant_matrices(args, model, rank=None):
    splitquant_matrices = {}
    layers = get_decoder_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        layer.self_attn.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        paras_name = ["trans.matrix", "trans.diag_scale", "matrix_list", "matrix_inv_t_list", "clip_factor_w", "clip_factor_a"]
        splitquant_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    if rank is not None:
        matrices_path = os.path.join(args.exp_dir, f"splitquant_matrices_{rank}.pth")
    else:
        matrices_path = os.path.join(args.exp_dir, "splitquant_matrices.pth")
    torch.save(splitquant_matrices, matrices_path)
    logging.info("saved parameters at %s", matrices_path)


def load_splitquant_matrices(args, model, path=None):
    checkpoint_dir = args.exp_dir if path is None else path
    checkpoint_path = os.path.join(checkpoint_dir, "splitquant_matrices.pth")
    splitquant_matrices = torch.load(checkpoint_path, map_location="cpu")
    layers = get_decoder_layers(model)
    
    for i in range(len(splitquant_matrices.keys())):
        splitquant_param = splitquant_matrices[i]
        layers[i].self_attn.rep_matrix_only()
        layers[i].mlp.rep_matrix_only()
        layers[i].load_state_dict(splitquant_param, strict=False)
    return model
# Replace legacy quantization shell scripts with a config-based GPU/NPU runner covering AWQ, GPTQ, FlatQuant, SplitQuant, SmoothQuant, and OmniQuant.
