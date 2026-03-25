"""QuaRot runtime wrappers for Qwen text backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLMLP
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward as vl_eager_attention_forward

from quarot.nn.hadamard import OnlineHadamard
from quarot.nn.normalization import RMSNorm


class QuaRotFP16Qwen2Attention(Qwen2Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.o_proj_hadamard = OnlineHadamard(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(self.o_proj_hadamard(attn_output))
        return attn_output, attn_weights


class QuaRotFP16Qwen2MLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__(config)
        self.down_proj_hadamard = OnlineHadamard(config.intermediate_size)

    def forward(self, x):
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.down_proj_hadamard(hidden))


class QuaRotFP16Qwen2VLAttention(Qwen2_5_VLAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.o_proj_hadamard = OnlineHadamard(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = vl_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(self.o_proj_hadamard(attn_output))
        return attn_output, attn_weights


class QuaRotFP16Qwen2VLMLP(Qwen2_5_VLMLP):
    def __init__(self, config, bias: bool = False):
        super().__init__(config, bias=bias)
        self.down_proj_hadamard = OnlineHadamard(config.intermediate_size)

    def forward(self, hidden_state):
        hidden = self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        return self.down_proj(self.down_proj_hadamard(hidden))


def _load_module_state(new_module: nn.Module, old_module: nn.Module) -> None:
    reference_tensor = None
    for tensor in list(old_module.parameters()) + list(old_module.buffers()):
        reference_tensor = tensor
        break
    if reference_tensor is not None:
        new_module = new_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    new_module.load_state_dict(old_module.state_dict(), strict=False)


def install_runtime_quarot_layers(model) -> None:
    model_type = getattr(model.config, "model_type", None)

    if model_type == "qwen2":
        root = model.model
        config = model.config
        root.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(root.layers):
            new_attn = QuaRotFP16Qwen2Attention(config=config, layer_idx=layer_idx)
            _load_module_state(new_attn, layer.self_attn)
            layer.self_attn = new_attn

            new_mlp = QuaRotFP16Qwen2MLP(config)
            _load_module_state(new_mlp, layer.mlp)
            layer.mlp = new_mlp

            layer.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        return

    if model_type == "qwen2_5_vl":
        root = model.language_model
        config = root.config
        root.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(root.layers):
            new_attn = QuaRotFP16Qwen2VLAttention(config=config, layer_idx=layer_idx)
            _load_module_state(new_attn, layer.self_attn)
            layer.self_attn = new_attn

            mlp_bias = layer.mlp.gate_proj.bias is not None
            new_mlp = QuaRotFP16Qwen2VLMLP(config, bias=mlp_bias)
            _load_module_state(new_mlp, layer.mlp)
            layer.mlp = new_mlp

            layer.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        return

    raise NotImplementedError(f"Unsupported QuaRot runtime model type: {model_type}")
