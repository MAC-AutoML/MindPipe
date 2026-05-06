from __future__ import annotations

import torch
import torch.nn as nn

from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS as QWEN2_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward as qwen2_eager_attention_forward
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS as QWEN2_VL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward as qwen2_vl_eager_attention_forward

from omniquant.int_linear import QuantLinear
from omniquant.int_matmul import QuantMatMul
from omniquant.omni_norm import OmniLlamaRMSNorm


def _resolve_attention_interface(attention_functions, implementation: str, eager_attention_fn):
    if hasattr(attention_functions, "get_interface"):
        return attention_functions.get_interface(implementation, eager_attention_fn)
    if implementation == "eager":
        return eager_attention_fn
    return attention_functions[implementation]


def _resolve_mrope_section(config) -> list[int]:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "mrope_section" in rope_parameters:
        return rope_parameters["mrope_section"]
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
        return rope_scaling["mrope_section"]
    raise KeyError("Qwen2.5-VL config is missing mrope_section in rope_parameters/rope_scaling.")


class QuantQwenMLP(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.config = org_module.config
        self.gate_proj = QuantLinear(org_module.gate_proj, args.weight_quant_params, args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj, args.weight_quant_params, args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj, args.weight_quant_params, args.act_quant_params)
        self.act_fn = org_module.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _QuantQwenAttentionMixin:
    attention_functions = QWEN2_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen2_eager_attention_forward)

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
        self.mrope_section = _resolve_mrope_section(org_module.config) if isinstance(org_module, Qwen2_5_VLAttention) else None
        if hasattr(org_module, "rotary_emb"):
            self.rotary_emb = org_module.rotary_emb

        self.q_proj = QuantLinear(org_module.q_proj, args.weight_quant_params, args.act_quant_params)
        self.k_proj = QuantLinear(org_module.k_proj, args.weight_quant_params, args.act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj, args.weight_quant_params, args.act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj, args.weight_quant_params, args.act_quant_params)
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
        position_ids: torch.LongTensor | None = None,
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
            position_ids=position_ids,
            **kwargs,
        )


class QuantQwen2Attention(_QuantQwenAttentionMixin, Qwen2Attention):
    def __init__(self, org_module: Qwen2Attention, args):
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
        del use_cache
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            if not hasattr(self, "rotary_emb"):
                raise AttributeError(
                    "QuantQwen2Attention requires `position_embeddings` or a `rotary_emb` module for Qwen2 models."
                )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            try:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            except TypeError:
                cache_kwargs = {"sin": sin, "cos": cos}
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)

        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_ids=position_ids,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class QuantQwen2_5_VLAttention(_QuantQwenAttentionMixin, Qwen2_5_VLAttention):
    attention_functions = QWEN2_VL_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen2_vl_eager_attention_forward)

    def __init__(self, org_module: Qwen2_5_VLAttention, args):
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
        del use_cache
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            if not hasattr(self, "rotary_emb"):
                raise AttributeError(
                    "QuantQwen2_5_VLAttention requires `position_embeddings` or a `rotary_emb` module for Qwen2.5-VL models."
                )
            rotary_position_ids = position_ids
            if rotary_position_ids is not None and rotary_position_ids.ndim == 2:
                rotary_position_ids = rotary_position_ids.unsqueeze(0).expand(3, rotary_position_ids.shape[0], -1)
            elif rotary_position_ids is None:
                rotary_position_ids = cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
            position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.mrope_section,
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)

        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_ids=position_ids,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class QuantQwenDecoderLayer(nn.Module):
    def __init__(self, ori_layer, args):
        super().__init__()
        self.hidden_size = getattr(ori_layer, "hidden_size", ori_layer.self_attn.config.hidden_size)
        if isinstance(ori_layer.self_attn, Qwen2_5_VLAttention):
            attention_cls = QuantQwen2_5_VLAttention
        else:
            attention_cls = QuantQwen2Attention
        self.self_attn = attention_cls(ori_layer.self_attn, args)
        self.mlp = QuantQwenMLP(ori_layer.mlp, args)
        self.input_layernorm = OmniLlamaRMSNorm(
            ori_layer.input_layernorm,
            eps=ori_layer.input_layernorm.variance_epsilon,
        )
        self.post_attention_layernorm = OmniLlamaRMSNorm(
            ori_layer.post_attention_layernorm,
            eps=ori_layer.post_attention_layernorm.variance_epsilon,
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


def initialize_omni_parameters(qlayer, layer_prefix: str, args, act_scales, act_shifts, use_shift: bool = False) -> None:
    if not args.let:
        return

    attention_module = qlayer.self_attn
    parameter_specs = (
        ("self_attn.q_proj", "qkv"),
        ("self_attn.o_proj", "out"),
        ("mlp.up_proj", "fc1"),
    )

    qlayer.register_parameter(
        "qkt_smooth_scale",
        nn.Parameter(
            torch.ones(
                attention_module.kv_hidden_size if attention_module.is_gqa else attention_module.q_proj.out_features,
                device=attention_module.q_proj.weight.device,
            )
        ),
    )
    named_modules = dict(qlayer.named_modules())
    for module_name, target_name in parameter_specs:
        module = named_modules[module_name]
        act = act_scales[f"{layer_prefix}.{module_name}"].to(device=module.weight.device, dtype=module.weight.dtype).clamp(min=1e-5)
        weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
        if target_name == "out" and attention_module.is_gqa:
            act = attention_module.reduce_attention_to_kv(act, reduction="amax")
            weight = attention_module.reduce_attention_to_kv(weight, reduction="amax")
        scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
        if act_shifts is not None:
            shift = act_shifts[f"{layer_prefix}.{module_name}"].to(device=module.weight.device, dtype=module.weight.dtype)
            if target_name == "out" and attention_module.is_gqa:
                shift = attention_module.reduce_attention_to_kv(shift, reduction="mean")
        else:
            shift = torch.zeros_like(scale)
        shift_init = shift if use_shift else torch.zeros_like(shift)
        qlayer.register_parameter(f"{target_name}_smooth_shift", nn.Parameter(shift_init))
        qlayer.register_parameter(f"{target_name}_smooth_scale", nn.Parameter(scale))
# Adapt OmniQuant to Qwen2.5, Qwen2.5-VL, LLaMA-family, and MiniCPM models.
