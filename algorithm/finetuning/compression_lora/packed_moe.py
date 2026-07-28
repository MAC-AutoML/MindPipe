"""Exact compression-aware LoRA for packed Qwen3-MoE experts."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .flatquant_linear import LoRAConfig, deterministic_svd_lowrank
from .mask_utils import PackedMask, materialize_mask


def _grouped_mm(inputs, expert_ids, weights, alignment=8):
    grouped_mm = getattr(torch, "_grouped_mm", None)
    if grouped_mm is None or inputs.device.type != "cuda":
        return None
    if inputs.dtype not in (torch.float16, torch.bfloat16) or weights.shape[1] % 16 or weights.shape[2] % 16:
        return None
    unique, counts = torch.unique_consecutive(expert_ids, return_counts=True)
    chunks, raw_counts, padded_counts = [], [], []
    start = 0
    for value in counts.tolist():
        count = int(value)
        padded = (count + alignment - 1) // alignment * alignment
        chunk = inputs[start : start + count]
        if padded != count:
            chunk = torch.cat((chunk, torch.zeros((padded - count, chunk.shape[-1]), device=chunk.device, dtype=chunk.dtype)))
        chunks.append(chunk)
        raw_counts.append(count)
        padded_counts.append(padded)
        start += count
    packed = torch.cat(chunks, dim=0).contiguous()
    offsets = torch.tensor(padded_counts, device=inputs.device, dtype=torch.int32).cumsum(0)
    try:
        output = grouped_mm(packed, weights[unique].contiguous(), offsets)
    except (RuntimeError, NotImplementedError):
        return None
    result, start = [], 0
    for count, padded in zip(raw_counts, padded_counts):
        result.append(output[start : start + count])
        start += padded
    return torch.cat(result, dim=0)


class PackedCompressionLoRAExperts(nn.Module):
    """Packed Qwen3 expert LoRA with the exact quantize-after-delta objective."""

    def __init__(self, base: nn.Module, masks: dict[str, object], prefix: str, config: LoRAConfig, kind: str):
        super().__init__()
        self.base = base
        self.kind = kind
        self.prefix = prefix
        self.rank = int(config.rank)
        self.alpha = float(config.alpha)
        self.scaling = self.alpha / self.rank
        self.weight_checkpointing = bool(config.weight_checkpointing)
        self.adapter_type = str(config.adapter_type).lower()
        if self.adapter_type != "lora":
            raise ValueError("Packed Qwen3-MoE compression training currently supports LoRA only.")
        gate_up = base.gate_up_proj
        down = base.down_proj
        self.num_experts = int(gate_up.shape[0])
        self.intermediate_size = int(gate_up.shape[1] // 2)
        self.hidden_size = int(gate_up.shape[2])
        self.gate_A = nn.Parameter(torch.empty(self.num_experts, self.rank, self.hidden_size, device=gate_up.device, dtype=torch.float32))
        self.gate_B = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.rank, device=gate_up.device, dtype=torch.float32))
        self.up_A = nn.Parameter(torch.empty_like(self.gate_A))
        self.up_B = nn.Parameter(torch.empty_like(self.gate_B))
        self.down_A = nn.Parameter(torch.empty(self.num_experts, self.rank, self.intermediate_size, device=down.device, dtype=torch.float32))
        self.down_B = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.rank, device=down.device, dtype=torch.float32))
        self._masks = masks
        self._packed_masks = self._build_packed_mask_storage()
        self.args = getattr(base, "args", None)
        self.num_experts = int(gate_up.shape[0])
        self._ori_mode = False
        self._eval_mode = False
        self._reset(config.init)
        for param in self.base.parameters():
            param.requires_grad = False

    def _reset(self, init: str):
        if init == "pissa":
            # Per-expert PiSSA is intentionally initialized one expert at a time
            # to avoid allocating a full extra [E, out, in] FP32 tensor.
            with torch.no_grad():
                for index in range(self.num_experts):
                    self._reset_one_pissa(index, self.base.gate_up_proj[index, : self.intermediate_size], self.gate_A, self.gate_B)
                    self._reset_one_pissa(index, self.base.gate_up_proj[index, self.intermediate_size :], self.up_A, self.up_B)
                    self._reset_one_pissa(index, self.base.down_proj[index], self.down_A, self.down_B)
                    self.base.gate_up_proj.data[index, : self.intermediate_size] -= self.scaling * (self.gate_B[index] @ self.gate_A[index]).to(self.base.gate_up_proj.dtype)
                    self.base.gate_up_proj.data[index, self.intermediate_size :] -= self.scaling * (self.up_B[index] @ self.up_A[index]).to(self.base.gate_up_proj.dtype)
                    self.base.down_proj.data[index] -= self.scaling * (self.down_B[index] @ self.down_A[index]).to(self.base.down_proj.dtype)
            return
        self.gate_A.data.zero_(); self.up_A.data.zero_(); self.down_A.data.zero_()
        for tensor in (self.gate_B, self.up_B, self.down_B):
            rows = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(1)
            cols = torch.arange(tensor.shape[2], device=tensor.device).unsqueeze(0)
            tensor.data.copy_((((rows + cols) % 2).mul(2).sub(1)).to(tensor.dtype) / math.sqrt(float(tensor.shape[1] + tensor.shape[2])))

    def _reset_one_pissa(self, index, weight, A, B):
        u, s, v = deterministic_svd_lowrank(weight.to(torch.float32), self.rank, niter=4)
        root = torch.sqrt(s / self.scaling)
        A[index].copy_(torch.diag(root) @ v.T)
        B[index].copy_(u @ torch.diag(root))

    def adapter_parameters(self):
        return (self.gate_A, self.gate_B, self.up_A, self.up_B, self.down_A, self.down_B)

    def adapter_parameter_names(self):
        return ("gate_A", "gate_B", "up_A", "up_B", "down_A", "down_B")

    def _build_packed_mask_storage(self):
        storage = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            expert_masks = [
                self._masks[f"{self.prefix}.experts.{expert}.{name}"]
                for expert in range(self.num_experts)
            ]
            if not all(isinstance(mask, PackedMask) for mask in expert_masks):
                continue
            first = expert_masks[0]
            if not all(mask.shape == first.shape and mask.numel == first.numel for mask in expert_masks):
                raise ValueError(f"Packed expert masks for {name} do not share one shape.")
            storage[name] = {
                "packed": torch.stack([mask.packed for mask in expert_masks]),
                "shape": first.shape,
                "numel": first.numel,
            }
        return storage

    def _delta(self, A, B, indices):
        return self.scaling * torch.bmm(B.index_select(0, indices), A.index_select(0, indices))

    def _mask(self, name, indices, shape, device, dtype):
        packed = self._packed_masks.get(name)
        if packed is not None:
            cpu_indices = indices.detach().to(device="cpu")
            values = packed["packed"].index_select(0, cpu_indices).to(device=device, non_blocking=True)
            shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
            values = values.unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and_(1).flatten(1)
            values = values[:, : packed["numel"]].reshape(shape)
            return values.to(dtype=dtype)
        values = []
        for expert in indices.tolist():
            values.append(materialize_mask(self._masks[f"{self.prefix}.experts.{expert}.{name}"], device=device))
        return torch.stack(values).reshape(shape).to(dtype=dtype)

    @staticmethod
    def _linear_fallback(inputs, expert_ids, weights):
        _, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        pieces = []
        start = 0
        for local_index, count in enumerate(counts.tolist()):
            pieces.append(F.linear(inputs[start : start + count], weights[local_index]))
            start += count
        return torch.cat(pieces, dim=0)

    def _quantize_weight(self, weight):
        return self.base._quantize_weight(weight)

    def _gate_up_weights(self, indices, input_trans, gate_A, gate_B, up_A, up_B):
        # Both quantizers derive scales independently per output row (or per
        # weight group), so selecting routed experts before quantization is
        # exactly equivalent for those experts and avoids materializing all E.
        gate = self.base.gate_up_proj[:, : self.intermediate_size].index_select(0, indices)
        up = self.base.gate_up_proj[:, self.intermediate_size :].index_select(0, indices)
        gate = gate + self._delta(gate_A, gate_B, indices).to(gate.dtype)
        up = up + self._delta(up_A, up_B, indices).to(up.dtype)
        if input_trans is not None:
            shape = gate.shape
            gate = input_trans(gate.reshape(-1, shape[-1]), inv_t=True).reshape(shape)
            up = input_trans(up.reshape(-1, shape[-1]), inv_t=True).reshape(shape)
        gate_up = self._quantize_weight(torch.cat((gate, up), dim=1))
        gate, up = gate_up.split(self.intermediate_size, dim=1)
        gate = gate * self._mask("gate_proj", indices, gate.shape, gate.device, gate.dtype)
        up = up * self._mask("up_proj", indices, up.shape, up.device, up.dtype)
        return gate, up

    def _down_weights(self, indices, down_A, down_B):
        down = self.base.down_proj.index_select(0, indices)
        down = down + self._delta(down_A, down_B, indices).to(down.dtype)
        down_trans = getattr(self.base, "_down_trans", None)
        if down_trans is not None:
            shape = down.shape
            down = down_trans(down.reshape(-1, shape[-1]), inv_t=True).reshape(shape)
        down = self._quantize_weight(down)
        down = down * self._mask("down_proj", indices, down.shape, down.device, down.dtype)
        return down

    def _forward_from_lora(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_A,
        gate_B,
        up_A,
        up_B,
        down_A,
        down_B,
        input_trans,
    ):
        tokens, top_k = top_k_index.shape
        token_ids = torch.arange(tokens, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        top_positions = torch.arange(top_k, device=hidden_states.device).unsqueeze(0).expand(tokens, -1).reshape(-1)
        expert_ids = top_k_index.reshape(-1)
        order = torch.argsort(expert_ids, stable=True)
        expert_ids = expert_ids.index_select(0, order)
        token_ids = token_ids.index_select(0, order)
        top_positions = top_positions.index_select(0, order)
        active = torch.unique(expert_ids, sorted=True)
        local_expert_ids = torch.searchsorted(active, expert_ids)
        gate, up = self._gate_up_weights(active, input_trans, gate_A, gate_B, up_A, up_B)
        current = hidden_states.index_select(0, token_ids)
        if self.kind == "flatquant":
            current = self.base.act_quantizer(current)
        else:
            current = self.base.act_quantizer(current)
        gate_out = _grouped_mm(current, local_expert_ids, gate.transpose(1, 2).contiguous())
        if gate_out is None:
            gate_out = self._linear_fallback(current, local_expert_ids, gate)
        up_out = _grouped_mm(current, local_expert_ids, up.transpose(1, 2).contiguous())
        if up_out is None:
            up_out = self._linear_fallback(current, local_expert_ids, up)
        intermediate = self.base.act_fn(gate_out) * up_out
        if self.kind == "splitquant":
            intermediate = self.base.hidden_act_quantizer(intermediate)
        else:
            intermediate = self.base.act_quantizer(intermediate)
        down_trans = getattr(self.base, "_down_trans", None)
        if down_trans is not None:
            intermediate = down_trans(intermediate)
        down = self._down_weights(active, down_A, down_B)
        down_out = _grouped_mm(intermediate, local_expert_ids, down.transpose(1, 2).contiguous())
        if down_out is None:
            down_out = self._linear_fallback(intermediate, local_expert_ids, down)
        route = top_k_weights.index_select(0, token_ids).gather(1, top_positions[:, None]).to(down_out.dtype)
        output = torch.zeros_like(hidden_states)
        output.index_add_(0, token_ids, down_out * route)
        return output

    def forward(self, hidden_states, top_k_index, top_k_weights, input_trans=None):
        if self.weight_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                lambda hidden, gate_A, gate_B, up_A, up_B, down_A, down_B: self._forward_from_lora(
                    hidden,
                    top_k_index,
                    top_k_weights,
                    gate_A,
                    gate_B,
                    up_A,
                    up_B,
                    down_A,
                    down_B,
                    input_trans,
                ),
                hidden_states,
                self.gate_A,
                self.gate_B,
                self.up_A,
                self.up_B,
                self.down_A,
                self.down_B,
                use_reentrant=False,
            )
        return self._forward_from_lora(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.gate_A,
            self.gate_B,
            self.up_A,
            self.up_B,
            self.down_A,
            self.down_B,
            input_trans,
        )

    @torch.no_grad()
    def merge_into_base(self):
        gate_delta = self.scaling * torch.bmm(self.gate_B, self.gate_A).to(self.base.gate_up_proj.dtype)
        self.base.gate_up_proj.data[:, : self.intermediate_size] += gate_delta
        del gate_delta
        up_delta = self.scaling * torch.bmm(self.up_B, self.up_A).to(self.base.gate_up_proj.dtype)
        self.base.gate_up_proj.data[:, self.intermediate_size :] += up_delta
        del up_delta
        down_delta = self.scaling * torch.bmm(self.down_B, self.down_A).to(self.base.down_proj.dtype)
        self.base.down_proj.data += down_delta
        del down_delta
        return self.base

    def reparameterize(self, *args, **kwargs):
        return self.base.reparameterize(*args, **kwargs)

    def rep_matrix_only(self):
        return self.base.rep_matrix_only()


@torch.no_grad()
def apply_packed_moe_masks(model: nn.Module, masks: dict[str, object]) -> list[str]:
    applied = []
    for name, block in model.named_modules():
        if block.__class__.__name__ not in {
            "FlatQuantQwen3MoeSparseMoeBlock",
            "SplitQuantQwen3MoeSparseMoeBlock",
        }:
            continue
        experts = getattr(block, "experts", None)
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        intermediate = experts.gate_up_proj.shape[1] // 2
        for expert_index in range(experts.gate_up_proj.shape[0]):
            gate_key = f"{name}.experts.{expert_index}.gate_proj"
            up_key = f"{name}.experts.{expert_index}.up_proj"
            down_key = f"{name}.experts.{expert_index}.down_proj"
            if gate_key not in masks:
                continue
            gate = materialize_mask(masks[gate_key], device=experts.gate_up_proj.device)
            up = materialize_mask(masks[up_key], device=experts.gate_up_proj.device)
            down = materialize_mask(masks[down_key], device=experts.down_proj.device)
            experts.gate_up_proj.data[expert_index, :intermediate].mul_(gate.to(experts.gate_up_proj.dtype))
            experts.gate_up_proj.data[expert_index, intermediate:].mul_(up.to(experts.gate_up_proj.dtype))
            experts.down_proj.data[expert_index].mul_(down.to(experts.down_proj.dtype))
        applied.append(name)
    return applied
