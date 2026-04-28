from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import UniformAffineQuantizer


class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
        disable_input_quant: bool = False,
    ):
        super().__init__()
        if weight_quant_params is None:
            weight_quant_params = {}
        if act_quant_params is None:
            act_quant_params = {}
        self.fwd_kwargs = {}
        self.fwd_func = F.linear
        self.register_buffer("weight", org_module.weight.detach().clone())
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias.detach().clone())
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.use_weight_quant = False
        self.use_act_quant = False
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None
        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant and self.act_quantizer is not None:
            input = self.act_quantizer(input)
        return self.fwd_func(input, weight, bias, **self.fwd_kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

# Maintenance touch for repository metadata refresh.
