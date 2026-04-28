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


def _resolve_rms_eps(norm_module: nn.Module, default: float = 1e-6) -> float:
    return float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", default)))


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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
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

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def initialize_qwen3_5_omni_parameters(qlayer, layer_prefix: str, args, act_scales, act_shifts, use_shift: bool = False) -> None:
    if not args.let:
        return
    raise NotImplementedError(
        "Qwen3.5 OmniQuant support in MindPipe currently follows the conservative LWC-only path; set --omniquant_let false."
    )

# Maintenance touch for repository metadata refresh.
