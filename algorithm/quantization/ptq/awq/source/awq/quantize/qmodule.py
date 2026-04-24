import torch
import torch.nn as nn


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        view_shape = [1] * x.ndim
        view_shape[-1] = -1
        return self.act(x) / self.scales.view(*view_shape).to(device=x.device, dtype=x.dtype)
