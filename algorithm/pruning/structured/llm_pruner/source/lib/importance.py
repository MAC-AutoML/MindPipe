"""Importance metrics for LLM-Pruner: Magnitude and Taylor expansion.

MindPipe prunes one GQA attention group at a time: all query heads tied to one
KV head, plus that KV head itself. Importance is therefore aggregated over:
  - q_proj output rows of the group's query heads
  - o_proj input columns of the group's query heads
  - k_proj output rows of the group's KV head
  - v_proj output rows of the group's KV head

对于 Qwen3.5 等混合架构模型，linear_attention 层没有标准 head 概念，
attention 重要性返回全零张量（不做 attention 剪枝）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from algorithm.common.modeling import (
    get_attn_projections,
    get_head_geometry,
    get_mlp_projections,
    get_q_stride,
    supports_head_pruning,
)


def _aggregate_by_head(per_channel_imp: torch.Tensor, num_heads: int, stride: int) -> torch.Tensor:
    """Reshape [num_heads*stride] → [num_heads, stride] → sum → [num_heads]。"""
    return per_channel_imp.reshape(num_heads, stride).sum(dim=1)


def _reduce_q_heads_to_kv_groups(q_head_imp: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Reduce per-query-head importance to per-KV-group importance."""
    return q_head_imp.reshape(num_kv_heads, -1).sum(dim=1)


class MagnitudeImportance:
    """Lp-norm based importance: |W|^p aggregated per KV group / per neuron."""

    def __init__(self, p: int = 2):
        self.p = p

    @torch.no_grad()
    def compute_attention_importance(self, layer) -> torch.Tensor:
        """Return importance score per KV group: shape [num_kv_heads].

        linear_attention 层返回全零张量。
        """
        if not supports_head_pruning(layer):
            return torch.tensor([])

        projs = get_attn_projections(layer)
        q_proj, k_proj, v_proj, o_proj = projs
        geo = get_head_geometry(layer)
        num_heads, num_kv_heads, num_kv_groups, head_dim = geo
        q_stride = get_q_stride(layer)  # head_dim 或 head_dim*2（Qwen3.5 query+gate）

        # q_proj: 每个 head 占 q_stride 行（标准模型=head_dim，Qwen3.5=head_dim*2）
        q_head_imp = _aggregate_by_head(
            q_proj.weight.data.abs().pow(self.p).sum(dim=1),
            num_heads,
            q_stride,
        )
        # o_proj: 列方向聚合，每个 head 占 head_dim 列
        o_head_imp = _aggregate_by_head(
            o_proj.weight.data.abs().pow(self.p).sum(dim=0),
            num_heads,
            head_dim,
        )
        k_group_imp = _aggregate_by_head(
            k_proj.weight.data.abs().pow(self.p).sum(dim=1),
            num_kv_heads,
            head_dim,
        )
        v_group_imp = _aggregate_by_head(
            v_proj.weight.data.abs().pow(self.p).sum(dim=1),
            num_kv_heads,
            head_dim,
        )

        q_group_imp = _reduce_q_heads_to_kv_groups(q_head_imp, num_kv_heads)
        o_group_imp = _reduce_q_heads_to_kv_groups(o_head_imp, num_kv_heads)
        return q_group_imp + o_group_imp + k_group_imp + v_group_imp

    @torch.no_grad()
    def compute_mlp_importance(self, layer) -> torch.Tensor:
        """Return importance score per intermediate neuron: shape [intermediate_size]."""
        up_proj, gate_proj, down_proj = get_mlp_projections(layer)

        # up_proj [intermediate_size, hidden_size]: output channel (row) importance
        up_imp = up_proj.weight.data.abs().pow(self.p).sum(dim=1)   # [intermediate_size]
        # gate_proj: same
        gate_imp = gate_proj.weight.data.abs().pow(self.p).sum(dim=1)  # [intermediate_size]
        # down_proj [hidden_size, intermediate_size]: input channel (column) importance
        down_imp = down_proj.weight.data.abs().pow(self.p).sum(dim=0)  # [intermediate_size]

        return up_imp + gate_imp + down_imp  # [intermediate_size]


class TaylorImportance:
    """Taylor-expansion based importance: |W * grad| aggregated per KV group."""

    def __init__(self, taylor: str = "param_first"):
        """
        Args:
            taylor: One of 'param_first' (|W*grad|), 'param_second' (|W*acc_grad*W|),
                    'param_mix' (|W*grad - 0.5*W*acc_grad*W|).
        """
        if taylor in ("param_second", "param_mix"):
            raise ValueError(
                f"Taylor mode '{taylor}' requires acc_grad accumulation which is not "
                "yet implemented. Use 'param_first' for now."
            )
        self.taylor = taylor

    @torch.no_grad()
    def _compute_salience(self, linear: nn.Linear) -> torch.Tensor:
        """Compute element-wise salience = weight * gradient (param_first)."""
        w = linear.weight.data
        g = linear.weight.grad
        if g is None:
            return torch.zeros_like(w)
        g = g.to(device=w.device, dtype=w.dtype)
        return w * g

    @torch.no_grad()
    def compute_attention_importance(self, layer) -> torch.Tensor:
        """Return Taylor importance per KV group: shape [num_kv_heads].

        linear_attention 层返回全零张量。
        """
        if not supports_head_pruning(layer):
            return torch.tensor([])

        projs = get_attn_projections(layer)
        q_proj, k_proj, v_proj, o_proj = projs
        geo = get_head_geometry(layer)
        num_heads, num_kv_heads, num_kv_groups, head_dim = geo
        q_stride = get_q_stride(layer)

        q_head_imp = _aggregate_by_head(
            self._compute_salience(q_proj).abs().sum(dim=1),
            num_heads, q_stride,
        )
        o_head_imp = _aggregate_by_head(
            self._compute_salience(o_proj).abs().sum(dim=0),
            num_heads, head_dim,
        )
        k_group_imp = _aggregate_by_head(
            self._compute_salience(k_proj).abs().sum(dim=1),
            num_kv_heads, head_dim,
        )
        v_group_imp = _aggregate_by_head(
            self._compute_salience(v_proj).abs().sum(dim=1),
            num_kv_heads, head_dim,
        )

        q_group_imp = _reduce_q_heads_to_kv_groups(q_head_imp, num_kv_heads)
        o_group_imp = _reduce_q_heads_to_kv_groups(o_head_imp, num_kv_heads)
        return q_group_imp + o_group_imp + k_group_imp + v_group_imp

    @torch.no_grad()
    def compute_mlp_importance(self, layer) -> torch.Tensor:
        """Return Taylor importance per intermediate neuron: shape [intermediate_size]."""
        up_proj, gate_proj, down_proj = get_mlp_projections(layer)

        # up_proj: row-wise (output channel)
        up_imp = self._compute_salience(up_proj).abs().sum(dim=1)    # [intermediate_size]
        # gate_proj: row-wise
        gate_imp = self._compute_salience(gate_proj).abs().sum(dim=1)  # [intermediate_size]
        # down_proj: column-wise (input channel)
        down_imp = self._compute_salience(down_proj).abs().sum(dim=0)  # [intermediate_size]

        return up_imp + gate_imp + down_imp  # [intermediate_size]
# Add pruning support for Qwen3.5.
