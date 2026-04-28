import os
import torch
from splitquant.backbone_utils import get_decoder_layers
from splitquant.function_utils import get_paras_dict_by_name
import logging

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
        if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
            reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
        if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
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


def load_splitquant_parameters(args, model, path=None):
    if path is None:
        splitquant_parameters = torch.load(os.path.join(args.exp_dir, "splitquant_parameters.pth"))
    else:
        splitquant_parameters = torch.load(os.path.join(path, "splitquant_parameters.pth"))
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
        paras_name = ["trans.matrix", "trans.diag_scale", "clip_factor_w", "clip_factor_a"]
        splitquant_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    if rank is not None:
        matrices_path = os.path.join(args.exp_dir, f"splitquant_matrices_{rank}.pth")
    else:
        matrices_path = os.path.join(args.exp_dir, "splitquant_matrices.pth")
    torch.save(splitquant_matrices, matrices_path)
    logging.info("saved parameters at %s", matrices_path)


def load_splitquant_matrices(args, model, path=None):
    if path is None:
        splitquant_matrices = torch.load(os.path.join(args.exp_dir, "splitquant_matrices.pth"))
    else:
        splitquant_matrices = torch.load(os.path.join(path, "splitquant_matrices.pth"))
    layers = get_decoder_layers(model)
    
    for i in range(len(splitquant_matrices.keys())):
        splitquant_param = splitquant_matrices[i]
        layers[i].self_attn.rep_matrix_only()
        layers[i].mlp.rep_matrix_only()
        layers[i].load_state_dict(splitquant_param, strict=False)
    return model
# Maintenance touch for repository metadata refresh.
