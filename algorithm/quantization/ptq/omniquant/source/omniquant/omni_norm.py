from __future__ import annotations

import torch
import torch.nn as nn


class OmniLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", ori_norm.weight.detach().clone())
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias
        output = weight * hidden_states
        if bias is not None:
            output = output + bias
        return output.to(input_dtype)

# Maintenance touch for repository metadata refresh.
