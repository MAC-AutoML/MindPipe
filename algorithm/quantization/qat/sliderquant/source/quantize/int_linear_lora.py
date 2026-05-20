import math
from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn

from quantize.int_linear import QuantLinear


def identity(x):
    return x


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRAQuantLinear(QuantLinear, LoRALayer):
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
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        lora_attr={
            "lora_iter_num" : 1,
            "lora_quant" : False,
            "lora_r":4,
            "lora_only":False,
        },
    ):
        super().__init__(
            org_module, weight_quant_params, act_quant_params, disable_input_quant
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.lora_iter_num = lora_attr["lora_iter_num"]
        self.lora_quant = lora_attr["lora_quant"]
        self.lora_only = lora_attr["lora_only"]

        if "lora_r" in lora_attr:
            r = lora_attr["lora_r"]
            self.r = lora_attr["lora_r"]

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        if self.r >0 :  #sign_lora is in weight_quantizer
            out_features, in_features = self.weight.shape
            self.lora_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r, in_features))) for _ in range(self.lora_iter_num)])
            self.lora_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, r))) for _ in range(self.lora_iter_num)])
            self.scaling = self.lora_alpha / r


        self.reset_lora_parameters()
        self.rms_norm = None


    def update_quant_parms(self,weight_quant_params):
        for k,v in weight_quant_params.items():
            self.weight_quantizer.__setattr__(k,v)


    def reset_lora_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.lora_iter_num):
                nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i])

    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias

        if weight.device != input.device:
            weight = weight.to(input.device)
            if bias is not None:
                bias = bias.to(input.device)

        if self.merged:
            weight = weight
        else:
            if self.use_temporary_parameter or self.use_weight_quant:
                if self.r > 0:
                    weight = self.weight_quantizer(weight + self.lora_B[0] @ self.lora_A[0] * self.scaling, self.quant_rate)
                else:
                    weight = self.weight_quantizer(weight, self.quant_rate)
            else:
                weight = weight


        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input,self.quant_rate)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def extra_repr(self):
        s = super().extra_repr()
        s += ", use_temporary_parameter={}".format(self.use_temporary_parameter)
        s += ", use_act_quant={}".format(self.use_act_quant)
        s += ", use_weight_quant={}".format(self.use_weight_quant)
        s += ", disable_input_quant={}".format(self.disable_input_quant)
        # s += ", lora_quant"
        return s
