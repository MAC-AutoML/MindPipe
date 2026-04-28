import torch
from collections import OrderedDict

def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_init_weight(dim, ):
    # SplitQuant transforms should start as exact no-ops so wrapper insertion
    # does not perturb the baseline model before calibration/training.
    return torch.eye(dim, dtype=torch.float32)


def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params


def get_paras_dict_by_name(model, required_names, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                destination[prefix + name] = param.detach()
    return destination


def check_params_grad(model):
    for name, param in model.named_parameters():
        print(name, ':{}'.format(param.requires_grad))
    return
def set_require_grad_all(model, requires_grad):
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return
# Adapt FlatQuant to new models and address SplitQuant degradation on Qwen3.5.
