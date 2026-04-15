from __future__ import annotations

import torch
import torch.nn as nn

from .quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict | None = None,
        x2_quant_params: dict | None = None,
        disable_act_quant: bool = False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        if x1_quant_params is None:
            x1_quant_params = {}
        if x2_quant_params is None:
            x2_quant_params = {}
        self.use_act_quant = False
        self.use_weight_quant = False
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func
        self.disable_act_quant = disable_act_quant

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1: torch.Tensor) -> torch.Tensor:
        if self.use_act_quant and not self.disable_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2: torch.Tensor) -> torch.Tensor:
        if self.use_act_quant and not self.disable_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.matmul_func(x1, x2)

