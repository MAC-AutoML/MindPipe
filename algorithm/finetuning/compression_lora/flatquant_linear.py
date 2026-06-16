"""FlatQuant-specific linear wrapper for compression-aware LoRA."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float
    init: str = "lora"
    weight_checkpointing: bool = False
    adapter_type: str = "lora"
    dora_simple: bool = True
    dora_eps: float = 1e-6


class CompressionLoRAFlatQuantLinear(nn.Module):
    """Wrap a FlatQuantizedLinear and train LoRA inside its quantization path."""

    def __init__(self, base: nn.Module, mask: torch.Tensor, config: LoRAConfig) -> None:
        super().__init__()
        if int(config.rank) <= 0:
            raise ValueError("compression LoRA rank must be positive.")
        self.base = base
        self.rank = int(config.rank)
        self.alpha = float(config.alpha)
        self.scaling = self.alpha / float(self.rank)
        self.weight_checkpointing = bool(config.weight_checkpointing)
        self.adapter_type = str(config.adapter_type).lower()
        if self.adapter_type not in {"lora", "dora"}:
            raise ValueError(f"Unsupported compression adapter type: {config.adapter_type!r}.")
        self.dora_simple = bool(config.dora_simple)
        self.dora_eps = float(config.dora_eps)
        # Dropout is recorded for config compatibility. Weight-merged LoRA uses
        # a deterministic delta W, so applying dropout to hidden_states would
        # corrupt the frozen base branch.
        self.dropout = nn.Identity()
        weight = base.linear.weight
        out_features, in_features = weight.shape
        # Keep trainable LoRA parameters in FP32. AMP GradScaler cannot unscale
        # FP16 trainable gradients, while the merged delta is cast to base dtype
        # before entering the FlatQuant path.
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, device=weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank, device=weight.device, dtype=torch.float32))
        if self.adapter_type == "dora":
            dora_magnitude = weight.detach().to(torch.float32).norm(dim=1, keepdim=True)
            self.dora_magnitude = nn.Parameter(dora_magnitude.to(device=weight.device, dtype=torch.float32))
        self.register_buffer("pruning_mask", mask.detach().to(device=weight.device).bool(), persistent=True)
        self.reset_lora_parameters(config.init)

        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def reset_lora_parameters(self, init: str) -> None:
        if init == "lora":
            # Use deterministic finite initialization. In the full
            # quantization->pruning->LoRA workflow, device random init/copy can
            # produce NaNs on some runs after earlier CUDA kernels. A=0 keeps the
            # initial delta exactly zero; nonzero B lets A receive gradients.
            self.lora_A.data.zero_()
            rows = torch.arange(self.lora_B.shape[0], device=self.lora_B.device).unsqueeze(1)
            cols = torch.arange(self.lora_B.shape[1], device=self.lora_B.device).unsqueeze(0)
            signs = ((rows + cols) % 2).mul(2).sub(1).to(dtype=self.lora_B.dtype)
            scale = 1.0 / math.sqrt(float(self.lora_B.shape[0] + self.lora_B.shape[1]))
            self.lora_B.data.copy_(signs * scale)
            return
        if init == "pissa":
            with torch.no_grad():
                weight = self.base.linear.weight.detach().to(torch.float32)
                # For DoRA, dora_magnitude intentionally keeps the original pre-PiSSA row norm.
                u, s, v = torch.svd_lowrank(weight, q=self.rank, niter=16)
                s = s / self.scaling
                sqrt_s = torch.sqrt(s)
                self.lora_A.copy_((torch.diag(sqrt_s) @ v.T).to(self.lora_A))
                self.lora_B.copy_((u @ torch.diag(sqrt_s)).to(self.lora_B))
                delta = self._lora_delta().detach().to(torch.float32)
                self.base.linear.weight.data.copy_((weight - delta).to(self.base.linear.weight))
            return
        raise ValueError(f"Unsupported compression LoRA init: {init!r}.")

    def _lora_delta(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def _adapted_weight(self, base_weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (lora_B @ lora_A)
        direction = base_weight + delta.to(base_weight.dtype)
        if self.adapter_type == "lora":
            return direction

        row_norm = direction.to(torch.float32).norm(dim=1, keepdim=True).clamp_min(self.dora_eps)
        if self.dora_simple:
            row_norm = row_norm.detach()
        norm_scale = self.dora_magnitude.to(torch.float32) / row_norm
        return direction * norm_scale.to(device=direction.device, dtype=direction.dtype)

    def _ori_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self._adapted_weight(self.base.linear.weight.detach(), self.lora_A, self.lora_B)
        return F.linear(hidden_states, weight, self.base.linear.bias)

    def _compressed_weight_from_lora(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        qa_trans=None,
        out_trans=None,
    ) -> torch.Tensor:
        # Match FlatQuantizedLinear._train_forward as closely as possible while
        # keeping gradients only through the LoRA delta. Some transform modules
        # are numerically sensitive to dtype changes, so do not upcast this path
        # to FP32 here.
        base_weight = self.base.linear.weight.detach()
        weight = self._adapted_weight(base_weight, lora_A, lora_B)
        if qa_trans is not None:
            weight = self.base.apply_trans(weight, qa_trans)
        if getattr(self.base, "lwc", False):
            weight = self.base.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        self.base.weight_quantizer.find_params(weight)
        weight = self.base.weight_quantizer(weight)
        return weight * self.pruning_mask.to(device=weight.device, dtype=weight.dtype)

    def _compressed_weight(self, qa_trans=None, out_trans=None) -> torch.Tensor:
        if self.weight_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                lambda lora_A, lora_B: self._compressed_weight_from_lora(
                    lora_A,
                    lora_B,
                    qa_trans=qa_trans,
                    out_trans=out_trans,
                ),
                self.lora_A,
                self.lora_B,
                use_reentrant=False,
            )
        return self._compressed_weight_from_lora(self.lora_A, self.lora_B, qa_trans=qa_trans, out_trans=out_trans)

    def forward(self, hidden_states: torch.Tensor, qa_trans=None, out_trans=None) -> torch.Tensor:
        weight_device = self.base.linear.weight.device
        if hidden_states.device != weight_device:
            hidden_states = hidden_states.to(weight_device)
        x_dtype = hidden_states.dtype
        hidden_states = self.base.act_quantizer(hidden_states).to(x_dtype)
        weight = self._compressed_weight(qa_trans=qa_trans, out_trans=out_trans)
        bias = self.base.linear.bias
        if out_trans is not None and bias is not None:
            bias = out_trans(bias)
        if bias is not None:
            bias = bias.to(x_dtype)
        return F.linear(hidden_states, weight.to(x_dtype), bias)

    @torch.no_grad()
    def merge_and_apply_fixed_compression(self, qa_trans=None, out_trans=None) -> nn.Module:
        merged = self._adapted_weight(self.base.linear.weight.detach(), self.lora_A, self.lora_B)
        original_weight = self.base.linear.weight.data.clone()
        try:
            self.base.linear.weight.data.copy_(merged)
            final_weight = self._compressed_weight_from_lora(
                torch.zeros_like(self.lora_A),
                torch.zeros_like(self.lora_B),
                qa_trans=qa_trans,
                out_trans=out_trans,
            )
        finally:
            self.base.linear.weight.data.copy_(original_weight)
        self.base.linear.weight.data.copy_(final_weight.to(self.base.linear.weight.dtype))
        self.base._eval_mode = True
        return self.base

    @torch.no_grad()
    def merge_into_base(self) -> nn.Module:
        merged = self._adapted_weight(self.base.linear.weight.detach(), self.lora_A, self.lora_B)
        self.base.linear.weight.data.copy_(merged)
        return self.base

    def reparameterize(self, qa_trans=None, out_trans=None) -> None:
        self.merge_into_base()
        self.base.reparameterize(qa_trans=qa_trans, out_trans=out_trans)


def is_compression_lora_flatquant_linear(module: nn.Module) -> bool:
    return isinstance(module, CompressionLoRAFlatQuantLinear)
