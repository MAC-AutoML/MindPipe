from __future__ import annotations

import math

import torch
import torch.nn as nn


CLIP_MIN = 1e-5


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=None,
        metric: str = "minmax",
        dynamic: bool = False,
        dynamic_method: str = "per_cluster",
        group_size: int | None = None,
        shape=None,
        lwc: bool = False,
        disable_zero_point: bool = False,
    ):
        super().__init__()
        if per_channel_axes is None:
            per_channel_axes = []
        if not 2 <= int(n_bits) <= 16:
            raise AssertionError("bitwidth not supported")
        self.symmetric = bool(symmetric)
        self.disable_zero_point = bool(disable_zero_point)
        self.n_bits = int(n_bits)
        if self.disable_zero_point:
            self.qmin = -(2 ** (self.n_bits - 1))
            self.qmax = 2 ** (self.n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** self.n_bits - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = bool(lwc)
        self.group_size = group_size

        init_value = 4.0
        if self.lwc:
            if shape is None:
                raise ValueError("shape is required when learnable weight clipping is enabled")
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    if not self.symmetric:
                        raise AssertionError("learnable grouped clipping with padding expects symmetric quantization")
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True

    def change_n_bits(self, n_bits: int) -> None:
        self.n_bits = int(n_bits)
        if self.disable_zero_point:
            self.qmin = -(2 ** (self.n_bits - 1))
            self.qmax = 2 ** (self.n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** self.n_bits - 1

    def fake_quant(self, x: torch.Tensor, scale: torch.Tensor, round_zero_point: torch.Tensor | None) -> torch.Tensor:
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            if len(x.shape) != 2:
                raise AssertionError("only support linear layer now")
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, :-self.deficiency]
        return x_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        if self.dynamic_method in {"per_token", "per_channel"}:
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError(f"Unsupported dynamic calibration method: {self.dynamic_method}")
        return self.fake_quant(x, self.scale, self.round_zero_point)

    def per_token_dynamic_calibration(self, x: torch.Tensor) -> None:
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        xmin = x.amin([-1], keepdim=True)
        xmax = x.amax([-1], keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIP_MIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            value_range = xmax - xmin
            scale = value_range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIP_MIN, max=1e4)
            zero_point = -(xmin) / self.scale
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

    def register_scales_and_zeros(self) -> None:
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point

