"""SplitQuant-specific linear wrapper for compression-aware LoRA."""

from __future__ import annotations

import torch

from .flatquant_linear import CompressionLoRAFlatQuantLinear
from .mask_utils import materialize_mask


class CompressionLoRASplitQuantLinear(CompressionLoRAFlatQuantLinear):
    """Train LoRA inside SplitQuant's grouped fake-quantization path."""

    def _compressed_weight_from_lora(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        qa_trans=None,
        out_trans=None,
    ) -> torch.Tensor:
        base_weight = self.base.linear.weight.detach()
        weight = self._adapted_weight(base_weight, lora_A, lora_B)
        if qa_trans is not None:
            weight = self.base.apply_trans(weight, qa_trans)
        if getattr(self.base, "lwc", False):
            weight = self.base.group_weight(weight)
            weight = self.base.apply_wclip(weight)
            weight = self.base.degroup_weight(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        weight = self.base.group_weight(weight)
        self.base.weight_quantizer.find_params(weight)
        weight = self.base.weight_quantizer(weight)
        weight = self.base.degroup_weight(weight)
        return weight * materialize_mask(self.pruning_mask, device=weight.device).to(dtype=weight.dtype)


def is_compression_lora_splitquant_linear(module) -> bool:
    return isinstance(module, CompressionLoRASplitQuantLinear)
