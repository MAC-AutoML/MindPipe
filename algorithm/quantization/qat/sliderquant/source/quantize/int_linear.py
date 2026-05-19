import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if hasattr(org_module,"bias") and  org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None

        if hasattr(org_module,"in_features"):
            self.in_features = org_module.in_features
            self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.quant_rate = 1.0
        # import ipdb;ipdb.set_trace()
        # initialize quantizer
        if weight_quant_params["n_bits"] > 1:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,is_weight_quant=True)
            if not disable_input_quant:
                self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
            else:
                self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.weight_quantizer(self.temp_weight,self.quant_rate)
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight,self.quant_rate)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input,self.quant_rate)


        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False, quant_rate:float = 1.0):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        self.quant_rate = quant_rate
