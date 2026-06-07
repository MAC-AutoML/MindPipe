from __future__ import annotations

import math

import torch
import torch.nn as nn

from omniquant.int_linear import QuantLinear
from omniquant.int_matmul import QuantMatMul
from omniquant.omni_norm import OmniLlamaRMSNorm


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


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    for buffer in module.buffers():
        return buffer.device
    return fallback


class QuantMiniCPMMLP(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.config = org_module.config
        self.hidden_size = org_module.hidden_size
        self.intermediate_size = org_module.intermediate_size
        self.gate_proj = QuantLinear(org_module.gate_proj, args.weight_quant_params, args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj, args.weight_quant_params, args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj, args.weight_quant_params, args.act_quant_params)
        self.act_fn = org_module.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantMiniCPMAttention(nn.Module):
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
        self.head_dim = getattr(org_module, "head_dim", self.hidden_size // self.num_heads)
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim
        self.full_attention_size = self.num_heads * self.head_dim
        self.is_gqa = self.num_heads != self.num_key_value_heads
        self.attention_dropout = getattr(org_module, "attention_dropout", org_module.config.attention_dropout)
        self.is_causal = getattr(org_module, "is_causal", True)
        self.rotary_emb = org_module.rotary_emb
        self.apply_rotary_pos_emb = org_module.forward.__globals__["apply_rotary_pos_emb"]
        self.repeat_kv = org_module.forward.__globals__["repeat_kv"]

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

    def _manual_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs and attention_mask is None:
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        attention_mask = _move_optional_tensor(attention_mask, query_states.device)
        position_ids = _move_optional_tensor(position_ids, query_states.device)
        past_key_value = _move_optional_tensor(past_key_value, query_states.device)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("MiniCPM attention requires layer_idx when KV cache is enabled.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        if self.config._attn_implementation == "sdpa" and not output_attentions:
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal and attention_mask is None and q_len > 1,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._manual_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        if not use_cache:
            past_key_value = None
        return attn_output, attn_weights, past_key_value


class QuantMiniCPMDecoderLayer(nn.Module):
    def __init__(self, ori_layer, args):
        super().__init__()
        self.hidden_size = getattr(ori_layer, "hidden_size", ori_layer.self_attn.config.hidden_size)
        self.self_attn = QuantMiniCPMAttention(ori_layer.self_attn, args)
        self.mlp = QuantMiniCPMMLP(ori_layer.mlp, args)
        self.input_layernorm = OmniLlamaRMSNorm(
            ori_layer.input_layernorm,
            eps=ori_layer.input_layernorm.variance_epsilon,
        )
        self.post_attention_layernorm = OmniLlamaRMSNorm(
            ori_layer.post_attention_layernorm,
            eps=ori_layer.post_attention_layernorm.variance_epsilon,
        )
        self.scale_depth = ori_layer.scale_depth
        self.num_hidden_layers = ori_layer.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        layer_device = _module_device(self, hidden_states.device)
        hidden_states = hidden_states.to(layer_device)
        attention_mask = _move_optional_tensor(attention_mask, layer_device)
        position_ids = _move_optional_tensor(position_ids, layer_device)
        kwargs = _move_optional_tensor(kwargs, layer_device)
        cache_obj = past_key_value if past_key_value is not None else past_key_values
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache_obj,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


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
