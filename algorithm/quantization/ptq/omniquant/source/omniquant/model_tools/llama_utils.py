from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv

from omniquant.int_linear import QuantLinear
from omniquant.int_matmul import QuantMatMul
from omniquant.omni_norm import OmniLlamaRMSNorm


_LLAMA_APPLY_ROTARY_HAS_POSITION_IDS = "position_ids" in inspect.signature(apply_rotary_pos_emb).parameters


def _apply_llama_rotary_pos_emb(query_states, key_states, cos, sin, position_ids):
    if _LLAMA_APPLY_ROTARY_HAS_POSITION_IDS:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


class QuantLlamaMLP(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.config = org_module.config
        self.gate_proj = QuantLinear(org_module.gate_proj, args.weight_quant_params, args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj, args.weight_quant_params, args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj, args.weight_quant_params, args.act_quant_params)
        self.act_fn = org_module.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantLlamaAttention(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
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
        self.is_causal = True
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
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
                    "QuantLlamaAttention requires `position_embeddings` or a `rotary_emb` module for LLaMA models."
                )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = _apply_llama_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        cache_obj = past_key_values if past_key_values is not None else kwargs.get("past_key_value")
        if cache_obj is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            try:
                key_states, value_states = cache_obj.update(key_states, value_states, self.layer_idx, cache_kwargs)
            except TypeError:
                key_states, value_states = cache_obj.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            min_value = torch.tensor(
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            attn_weights = torch.maximum(attn_weights, min_value)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self, ori_layer, args):
        super().__init__()
        self.hidden_size = getattr(ori_layer, "hidden_size", ori_layer.self_attn.config.hidden_size)
        self.self_attn = QuantLlamaAttention(ori_layer.self_attn, args)
        self.mlp = QuantLlamaMLP(ori_layer.mlp, args)
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


def initialize_omni_parameters(qlayer, layer_prefix: str, args, act_scales, act_shifts) -> None:
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
        qlayer.register_parameter(f"{target_name}_smooth_shift", nn.Parameter(torch.zeros_like(shift)))
        qlayer.register_parameter(f"{target_name}_smooth_scale", nn.Parameter(scale))
