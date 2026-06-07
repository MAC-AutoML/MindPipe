from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS as QWEN3_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as qwen3_eager_attention_forward
from transformers.models.qwen3_5.modeling_qwen3_5 import ALL_ATTENTION_FUNCTIONS as QWEN3_5_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb
from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward as qwen3_5_eager_attention_forward
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import ALL_ATTENTION_FUNCTIONS as QWEN3_5_MOE_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_mask_to_padding_states as qwen3_5_moe_apply_mask_to_padding_states
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb as qwen3_5_moe_apply_rotary_pos_emb
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import eager_attention_forward as qwen3_5_moe_eager_attention_forward
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS as QWEN3_VL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward as qwen3_vl_eager_attention_forward

from omniquant.int_linear import QuantLinear
from omniquant.int_matmul import QuantMatMul
from omniquant.model_tools.qwen_utils import QuantQwenMLP
from omniquant.model_tools.qwen_utils import _resolve_attention_interface
from omniquant.model_tools.qwen_utils import initialize_omni_parameters as initialize_qwen3_omni_parameters
from omniquant.omni_norm import OmniLlamaRMSNorm
from omniquant.quantizer import UniformAffineQuantizer


def _move_optional_tensor(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_optional_tensor(item, device) for item in value)
    if isinstance(value, list):
        return [_move_optional_tensor(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_optional_tensor(item, device) for key, item in value.items()}
    return value


def _resolve_rms_eps(norm_module: nn.Module, default: float = 1e-6) -> float:
    return float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", default)))


def _quantizer_shape_for_packed_weight(weight: torch.Tensor) -> tuple[int, int]:
    return (int(weight.reshape(-1, weight.shape[-1]).shape[0]), int(weight.shape[-1]))


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    for buffer in module.buffers():
        return buffer.device
    return fallback


class _QuantQwen3AttentionMixin:
    attention_functions = QWEN3_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_apply_rotary_pos_emb)

    def _init_quant_attention(self, org_module: nn.Module, args) -> None:
        self.config = org_module.config
        self.layer_idx = getattr(org_module, "layer_idx", None)
        self.hidden_size = getattr(org_module, "hidden_size", org_module.config.hidden_size)
        self.num_heads = getattr(org_module, "num_heads", org_module.config.num_attention_heads)
        self.num_key_value_heads = getattr(org_module, "num_key_value_heads", org_module.config.num_key_value_heads)
        self.num_key_value_groups = getattr(
            org_module,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.head_dim = getattr(
            org_module,
            "head_dim",
            getattr(org_module.config, "head_dim", self.hidden_size // self.num_heads),
        )
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim
        self.full_attention_size = self.num_heads * self.head_dim
        self.is_gqa = self.num_heads != self.num_key_value_heads
        self.scaling = getattr(org_module, "scaling", self.head_dim**-0.5)
        self.attention_dropout = getattr(org_module, "attention_dropout", org_module.config.attention_dropout)
        self.sliding_window = getattr(org_module, "sliding_window", None)
        self.is_causal = getattr(org_module, "is_causal", True)

        self.q_proj = QuantLinear(org_module.q_proj, args.weight_quant_params, args.act_quant_params)
        self.k_proj = QuantLinear(org_module.k_proj, args.weight_quant_params, args.act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj, args.weight_quant_params, args.act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj, args.weight_quant_params, args.act_quant_params)
        self.q_norm = OmniLlamaRMSNorm(org_module.q_norm, eps=_resolve_rms_eps(org_module.q_norm))
        self.k_norm = OmniLlamaRMSNorm(org_module.k_norm, eps=_resolve_rms_eps(org_module.k_norm))
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul)
        self.use_weight_quant = False
        self.use_act_quant = False

    def expand_kv_to_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_gqa:
            return tensor
        return (
            tensor.reshape(self.num_key_value_heads, self.head_dim)
            .unsqueeze(1)
            .expand(self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            .reshape(self.num_heads, self.head_dim)
            .reshape(-1)
        )

    def reduce_attention_to_kv(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        if not self.is_gqa:
            return tensor
        grouped = tensor.reshape(self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
        if reduction == "amax":
            return grouped.amax(dim=1).reshape(-1)
        if reduction == "mean":
            return grouped.mean(dim=1).reshape(-1)
        raise ValueError(f"Unsupported GQA reduction mode: {reduction}")

    def _attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
        **kwargs,
    ):
        attention_interface = _resolve_attention_interface(
            self.attention_functions,
            self.config._attn_implementation,
            self.eager_attention_fn,
        )
        return attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            output_attentions=output_attentions,
            **kwargs,
        )


class QuantQwen3Attention(_QuantQwen3AttentionMixin, Qwen3Attention):
    def __init__(self, org_module: Qwen3Attention, args):
        super().__init__(org_module.config, org_module.layer_idx)
        self._init_quant_attention(org_module, args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del use_cache, cache_position, position_ids
        if position_embeddings is None:
            raise AttributeError("QuantQwen3Attention requires `position_embeddings` from the parent decoder layer.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        position_embeddings = _move_optional_tensor(position_embeddings, query_states.device)
        attention_mask = _move_optional_tensor(attention_mask, query_states.device)
        past_key_values = _move_optional_tensor(past_key_values, query_states.device)
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)

        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class QuantQwen3VLTextAttention(_QuantQwen3AttentionMixin, Qwen3VLTextAttention):
    attention_functions = QWEN3_VL_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_vl_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_vl_apply_rotary_pos_emb)

    def __init__(self, org_module: Qwen3VLTextAttention, args):
        super().__init__(org_module.config, org_module.layer_idx)
        self._init_quant_attention(org_module, args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del use_cache, cache_position, position_ids
        if position_embeddings is None:
            raise AttributeError("QuantQwen3VLTextAttention requires `position_embeddings` from the parent decoder layer.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        position_embeddings = _move_optional_tensor(position_embeddings, query_states.device)
        attention_mask = _move_optional_tensor(attention_mask, query_states.device)
        past_key_values = _move_optional_tensor(past_key_values, query_states.device)
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)

        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class QuantQwen3DecoderLayer(nn.Module):
    def __init__(self, ori_layer, args):
        super().__init__()
        self.hidden_size = getattr(
            ori_layer,
            "hidden_size",
            getattr(getattr(ori_layer.self_attn, "config", None), "hidden_size", None),
        )
        if isinstance(ori_layer.self_attn, Qwen3VLTextAttention):
            attention_cls = QuantQwen3VLTextAttention
        else:
            attention_cls = QuantQwen3Attention
        self.self_attn = attention_cls(ori_layer.self_attn, args)
        self.mlp = QuantQwenMLP(ori_layer.mlp, args)
        self.input_layernorm = OmniLlamaRMSNorm(
            ori_layer.input_layernorm,
            eps=_resolve_rms_eps(ori_layer.input_layernorm),
        )
        self.post_attention_layernorm = OmniLlamaRMSNorm(
            ori_layer.post_attention_layernorm,
            eps=_resolve_rms_eps(ori_layer.post_attention_layernorm),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        layer_device = _module_device(self, hidden_states.device)
        hidden_states = hidden_states.to(layer_device)
        attention_mask = _move_optional_tensor(attention_mask, layer_device)
        position_ids = _move_optional_tensor(position_ids, layer_device)
        position_embeddings = _move_optional_tensor(position_embeddings, layer_device)
        kwargs = _move_optional_tensor(kwargs, layer_device)
        output_attentions = bool(kwargs.pop("output_attentions", False))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        return hidden_states


class QuantQwen3_5Attention(Qwen3_5Attention):
    def __init__(self, org_module: Qwen3_5Attention, args):
        super().__init__(org_module.config, org_module.layer_idx)
        self.config = org_module.config
        self.layer_idx = org_module.layer_idx
        self.head_dim = org_module.head_dim
        self.num_key_value_groups = org_module.num_key_value_groups
        self.scaling = org_module.scaling
        self.attention_dropout = org_module.attention_dropout
        self.is_causal = getattr(org_module, "is_causal", True)
        self.q_proj = QuantLinear(org_module.q_proj, args.weight_quant_params, args.act_quant_params)
        self.k_proj = QuantLinear(org_module.k_proj, args.weight_quant_params, args.act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj, args.weight_quant_params, args.act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj, args.weight_quant_params, args.act_quant_params)
        self.q_norm = org_module.q_norm
        self.k_norm = org_module.k_norm
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul)
        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        position_embeddings = _move_optional_tensor(position_embeddings, query_states.device)
        attention_mask = _move_optional_tensor(attention_mask, query_states.device)
        past_key_values = _move_optional_tensor(past_key_values, query_states.device)
        cos, sin = position_embeddings
        query_states, key_states = qwen3_5_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = _resolve_attention_interface(
            QWEN3_5_ATTENTION_FUNCTIONS,
            self.config._attn_implementation,
            qwen3_5_eager_attention_forward,
        )
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QuantQwen3_5MoeAttention(Qwen3_5MoeAttention):
    def __init__(self, org_module: Qwen3_5MoeAttention, args):
        super().__init__(org_module.config, org_module.layer_idx)
        self.config = org_module.config
        self.layer_idx = org_module.layer_idx
        self.head_dim = org_module.head_dim
        self.num_key_value_groups = org_module.num_key_value_groups
        self.scaling = org_module.scaling
        self.attention_dropout = org_module.attention_dropout
        self.is_causal = getattr(org_module, "is_causal", True)
        self.q_proj = QuantLinear(org_module.q_proj, args.weight_quant_params, args.act_quant_params)
        self.k_proj = QuantLinear(org_module.k_proj, args.weight_quant_params, args.act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj, args.weight_quant_params, args.act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj, args.weight_quant_params, args.act_quant_params)
        self.q_norm = org_module.q_norm
        self.k_norm = org_module.k_norm
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul)
        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        position_embeddings = _move_optional_tensor(position_embeddings, query_states.device)
        attention_mask = _move_optional_tensor(attention_mask, query_states.device)
        past_key_values = _move_optional_tensor(past_key_values, query_states.device)
        cos, sin = position_embeddings
        query_states, key_states = qwen3_5_moe_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = _resolve_attention_interface(
            QWEN3_5_MOE_ATTENTION_FUNCTIONS,
            self.config._attn_implementation,
            qwen3_5_moe_eager_attention_forward,
        )
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QuantQwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.hidden_size = org_module.hidden_size
        self.num_v_heads = org_module.num_v_heads
        self.num_k_heads = org_module.num_k_heads
        self.head_k_dim = org_module.head_k_dim
        self.head_v_dim = org_module.head_v_dim
        self.key_dim = org_module.key_dim
        self.value_dim = org_module.value_dim
        self.conv_kernel_size = org_module.conv_kernel_size
        self.layer_idx = org_module.layer_idx
        self.activation = org_module.activation
        self.act = org_module.act
        self.layer_norm_epsilon = org_module.layer_norm_epsilon
        self.conv_dim = org_module.conv_dim
        self.conv1d = org_module.conv1d
        self.dt_bias = org_module.dt_bias
        self.A_log = org_module.A_log
        self.norm = org_module.norm
        self.out_proj = QuantLinear(org_module.out_proj, args.weight_quant_params, args.act_quant_params)
        self.causal_conv1d_fn = org_module.causal_conv1d_fn
        self.causal_conv1d_update = org_module.causal_conv1d_update
        self.chunk_gated_delta_rule = org_module.chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = org_module.recurrent_gated_delta_rule
        self.in_proj_qkv = QuantLinear(org_module.in_proj_qkv, args.weight_quant_params, args.act_quant_params)
        self.in_proj_z = QuantLinear(org_module.in_proj_z, args.weight_quant_params, args.act_quant_params)
        self.in_proj_b = QuantLinear(org_module.in_proj_b, args.weight_quant_params, args.act_quant_params)
        self.in_proj_a = QuantLinear(org_module.in_proj_a, args.weight_quant_params, args.act_quant_params)
        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
        )
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class QuantQwen3_5MoeGatedDeltaNet(QuantQwen3_5GatedDeltaNet):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = qwen3_5_moe_apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
        )
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class QuantQwen3_5MoeMLP(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.gate_proj = QuantLinear(org_module.gate_proj, args.weight_quant_params, args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj, args.weight_quant_params, args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj, args.weight_quant_params, args.act_quant_params)
        self.act_fn = org_module.act_fn
        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden_states)


class QuantQwen3_5MoePackedExperts(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.num_experts = org_module.num_experts
        self.hidden_dim = org_module.hidden_dim
        self.intermediate_dim = org_module.intermediate_dim
        self.act_fn = org_module.act_fn
        self.register_buffer("gate_up_proj", org_module.gate_up_proj.detach().clone())
        self.register_buffer("down_proj", org_module.down_proj.detach().clone())
        gate_up_shape = _quantizer_shape_for_packed_weight(self.gate_up_proj)
        down_shape = _quantizer_shape_for_packed_weight(self.down_proj)
        self.gate_up_quantizer = UniformAffineQuantizer(**args.weight_quant_params, shape=gate_up_shape)
        self.down_quantizer = UniformAffineQuantizer(**args.weight_quant_params, shape=down_shape)
        self.act_quantizer = UniformAffineQuantizer(**args.act_quant_params)
        self.use_weight_quant = False
        self.use_act_quant = False
        self.use_temporary_parameter = False

    def _quantize_weight(self, weight: torch.Tensor, quantizer: UniformAffineQuantizer) -> torch.Tensor:
        original_shape = weight.shape
        quantizer.to(device=weight.device)
        quantized = quantizer(weight.reshape(-1, original_shape[-1]))
        return quantized.reshape(original_shape).to(dtype=weight.dtype)

    def _gate_up_weight(self) -> torch.Tensor:
        if self.use_temporary_parameter:
            return self.temp_gate_up_proj
        if self.use_weight_quant:
            return self._quantize_weight(self.gate_up_proj, self.gate_up_quantizer)
        return self.gate_up_proj

    def _down_weight(self) -> torch.Tensor:
        if self.use_temporary_parameter:
            return self.temp_down_proj
        if self.use_weight_quant:
            return self._quantize_weight(self.down_proj, self.down_quantizer)
        return self.down_proj

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def prepare_temporary_quant(self) -> None:
        self.temp_gate_up_proj = self._quantize_weight(self.gate_up_proj, self.gate_up_quantizer)
        self.temp_down_proj = self._quantize_weight(self.down_proj, self.down_quantizer)
        self.use_temporary_parameter = True

    def clear_temporary_quant(self) -> None:
        if hasattr(self, "temp_gate_up_proj"):
            del self.temp_gate_up_proj
        if hasattr(self, "temp_down_proj"):
            del self.temp_down_proj
        self.use_temporary_parameter = False

    def quantize_inplace(self) -> None:
        self.gate_up_proj = self._quantize_weight(self.gate_up_proj, self.gate_up_quantizer)
        self.down_proj = self._quantize_weight(self.down_proj, self.down_quantizer)
        self.use_temporary_parameter = False

    def register_scales_and_zeros(self) -> None:
        self.gate_up_quantizer.register_scales_and_zeros()
        self.down_quantizer.register_scales_and_zeros()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        gate_up_proj = self._gate_up_weight()
        down_proj = self._down_weight()
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if self.use_act_quant:
                current_state = self.act_quantizer(current_state)
            gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if self.use_act_quant:
                current_hidden_states = self.act_quantizer(current_hidden_states)
            current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class QuantQwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.gate = org_module.gate
        self.experts = QuantQwen3_5MoePackedExperts(org_module.experts, args)
        self.shared_expert = QuantQwen3_5MoeMLP(org_module.shared_expert, args)
        self.shared_expert_gate = QuantLinear(org_module.shared_expert_gate, args.weight_quant_params, args.act_quant_params)
        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class QuantQwen3_5DecoderLayer(nn.Module):
    def __init__(self, ori_layer, args):
        super().__init__()
        self.hidden_size = ori_layer.hidden_size
        self.layer_type = ori_layer.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = QuantQwen3_5GatedDeltaNet(ori_layer.linear_attn, args)
        elif self.layer_type == "full_attention":
            self.self_attn = QuantQwen3_5Attention(ori_layer.self_attn, args)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {self.layer_type}")
        self.mlp = QuantQwenMLP(ori_layer.mlp, args)
        self.input_layernorm = ori_layer.input_layernorm
        self.post_attention_layernorm = ori_layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        layer_device = _module_device(self, hidden_states.device)
        hidden_states = hidden_states.to(layer_device)
        attention_mask = _move_optional_tensor(attention_mask, layer_device)
        position_ids = _move_optional_tensor(position_ids, layer_device)
        position_embeddings = _move_optional_tensor(position_embeddings, layer_device)
        kwargs = _move_optional_tensor(kwargs, layer_device)
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            output_attentions = bool(kwargs.pop("output_attentions", False))
            hidden_states, _attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )

        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        return hidden_states


class QuantQwen3_5MoeDecoderLayer(nn.Module):
    def __init__(self, ori_layer, args):
        super().__init__()
        self.hidden_size = ori_layer.hidden_size
        self.layer_type = ori_layer.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = QuantQwen3_5MoeGatedDeltaNet(ori_layer.linear_attn, args)
        elif self.layer_type == "full_attention":
            self.self_attn = QuantQwen3_5MoeAttention(ori_layer.self_attn, args)
        else:
            raise ValueError(f"Unsupported Qwen3.5-MoE layer_type: {self.layer_type}")
        self.mlp = QuantQwen3_5MoeSparseMoeBlock(ori_layer.mlp, args)
        self.input_layernorm = ori_layer.input_layernorm
        self.post_attention_layernorm = ori_layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        layer_device = _module_device(self, hidden_states.device)
        hidden_states = hidden_states.to(layer_device)
        attention_mask = _move_optional_tensor(attention_mask, layer_device)
        position_ids = _move_optional_tensor(position_ids, layer_device)
        position_embeddings = _move_optional_tensor(position_embeddings, layer_device)
        kwargs = _move_optional_tensor(kwargs, layer_device)
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            hidden_states, _attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _router_logits = hidden_states
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        return hidden_states


def initialize_qwen3_5_omni_parameters(qlayer, layer_prefix: str, args, act_scales, act_shifts, use_shift: bool = False) -> None:
    if not args.let:
        return
    raise NotImplementedError(
        "Qwen3.5 OmniQuant support in MindPipe currently follows the conservative LWC-only path; set --omniquant_let false."
    )


def initialize_qwen3_5_moe_omni_parameters(qlayer, layer_prefix: str, args, act_scales, act_shifts, use_shift: bool = False) -> None:
    if not args.let:
        return
    raise NotImplementedError(
        "Qwen3.5-MoE OmniQuant support in MindPipe currently follows the LWC-only path; set --omniquant_let false."
    )

# Adapt OmniQuant and SplitQuant to new models.
