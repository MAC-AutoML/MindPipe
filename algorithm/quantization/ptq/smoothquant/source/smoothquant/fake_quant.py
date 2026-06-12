from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from algorithm.common.modeling import get_text_backbone


@torch.no_grad()
def quantize_weight_per_channel_absmax(weight, n_bits=8):
    if int(n_bits) >= 16:
        return weight.detach().clone()
    quantized = weight.detach().clone()
    scales = quantized.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    quantized.div_(scales).round_().mul_(scales)
    return quantized


@torch.no_grad()
def quantize_activation_per_token_absmax(tensor, n_bits=8):
    if int(n_bits) >= 16:
        return tensor
    quantized = tensor.clone()
    scales = quantized.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    quantized.div_(scales).round_().mul_(scales)
    return quantized


def _identity(tensor):
    return tensor


class SmoothQuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_bits=8,
        activation_bits=8,
        act_quant="per_token",
        quantize_output=False,
        dtype=torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=dtype, requires_grad=False),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features,), dtype=dtype, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant != "per_token":
            raise ValueError(f"Invalid act_quant: {act_quant}")
        self.weight_quant_name = "per_channel" if self.weight_bits < 16 else "none"
        self.act_quant_name = act_quant if self.activation_bits < 16 else "none"
        self.act_quant = (
            partial(quantize_activation_per_token_absmax, n_bits=self.activation_bits)
            if self.activation_bits < 16
            else _identity
        )

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "none"
            self.output_quant = lambda x: x

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        return self.output_quant(y)

    @staticmethod
    def from_float(
        module,
        weight_bits=8,
        activation_bits=8,
        act_quant="per_token",
        quantize_output=False,
    ):
        assert isinstance(module, nn.Linear)
        new_module = SmoothQuantLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            act_quant=act_quant,
            quantize_output=quantize_output,
            dtype=module.weight.dtype,
        )
        new_module = new_module.to(device=module.weight.device, dtype=module.weight.dtype)
        new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=weight_bits)
        if module.bias is not None:
            new_module.bias = module.bias.detach().clone()
        return new_module

    def __repr__(self):
        return (
            f"SmoothQuantLinear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, weight_bits={self.weight_bits}, "
            f"activation_bits={self.activation_bits}, weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
        )


class SmoothQuantPackedMoeExperts(nn.Module):
    def __init__(
        self,
        gate_up_proj,
        down_proj,
        act_fn,
        *,
        weight_bits=8,
        activation_bits=8,
    ):
        super().__init__()
        self.num_experts = int(gate_up_proj.shape[0])
        self.hidden_dim = int(gate_up_proj.shape[2])
        self.intermediate_dim = int(down_proj.shape[2])
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.act_fn = act_fn
        self.register_buffer(
            "gate_up_proj",
            quantize_weight_per_channel_absmax(gate_up_proj, n_bits=weight_bits),
        )
        self.register_buffer(
            "down_proj",
            quantize_weight_per_channel_absmax(down_proj, n_bits=weight_bits),
        )
        self.act_quant = (
            partial(quantize_activation_per_token_absmax, n_bits=self.activation_bits)
            if self.activation_bits < 16
            else _identity
        )

    @staticmethod
    def from_float(module, weight_bits=8, activation_bits=8):
        return SmoothQuantPackedMoeExperts(
            module.gate_up_proj.detach(),
            module.down_proj.detach(),
            module.act_fn,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = self.act_quant(hidden_states[token_idx])
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.act_quant(current_hidden_states)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def __repr__(self):
        return (
            f"SmoothQuantPackedMoeExperts(num_experts={self.num_experts}, "
            f"hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}, "
            f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits})"
        )


def _replace_linear_with_smoothquant(
    module,
    proj_name,
    quantized_linear_names,
    qualified_name,
    *,
    weight_bits,
    activation_bits,
    act_quant,
    quantize_output=False,
):
    setattr(
        module,
        proj_name,
        SmoothQuantLinear.from_float(
            getattr(module, proj_name),
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            act_quant=act_quant,
            quantize_output=quantize_output,
        ),
    )
    quantized_linear_names.append(qualified_name)


def quantize_llama_like(model, weight_bits=8, activation_bits=8, act_quant="per_token", quantize_bmm_input=True):
    supported_model_types = {
        "llama",
        "qwen2",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_moe",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "minicpm",
        "minicpmv",
    }
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in supported_model_types:
        raise ValueError(f"Unsupported model type: {model_type!r}")

    backbone = get_text_backbone(model)
    quantized_linear_names = []
    for layer_index, layer in enumerate(backbone.layers):
        layer_prefix = f"{backbone.prefix}.layers.{layer_index}"

        if model_type == "qwen3_5":
            if getattr(layer, "layer_type", None) == "linear_attention":
                # These projections feed Qwen3.5's token-mixer directly, so keep
                # their output fake-quant enabled to cover the custom linear-attn core.
                for proj_name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                    _replace_linear_with_smoothquant(
                        layer.linear_attn,
                        proj_name,
                        quantized_linear_names,
                        f"{layer_prefix}.linear_attn.{proj_name}",
                        weight_bits=weight_bits,
                        activation_bits=activation_bits,
                        act_quant=act_quant,
                        quantize_output=quantize_bmm_input,
                    )
                _replace_linear_with_smoothquant(
                    layer.linear_attn,
                    "out_proj",
                    quantized_linear_names,
                    f"{layer_prefix}.linear_attn.out_proj",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                )
            else:
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    _replace_linear_with_smoothquant(
                        layer.self_attn,
                        proj_name,
                        quantized_linear_names,
                        f"{layer_prefix}.self_attn.{proj_name}",
                        weight_bits=weight_bits,
                        activation_bits=activation_bits,
                        act_quant=act_quant,
                        quantize_output=quantize_bmm_input,
                    )
                _replace_linear_with_smoothquant(
                    layer.self_attn,
                    "o_proj",
                    quantized_linear_names,
                    f"{layer_prefix}.self_attn.o_proj",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                )
        elif model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
            if getattr(layer, "layer_type", None) == "linear_attention":
                for proj_name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                    _replace_linear_with_smoothquant(
                        layer.linear_attn,
                        proj_name,
                        quantized_linear_names,
                        f"{layer_prefix}.linear_attn.{proj_name}",
                        weight_bits=weight_bits,
                        activation_bits=activation_bits,
                        act_quant=act_quant,
                        quantize_output=quantize_bmm_input,
                    )
                _replace_linear_with_smoothquant(
                    layer.linear_attn,
                    "out_proj",
                    quantized_linear_names,
                    f"{layer_prefix}.linear_attn.out_proj",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                )
            else:
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    _replace_linear_with_smoothquant(
                        layer.self_attn,
                        proj_name,
                        quantized_linear_names,
                        f"{layer_prefix}.self_attn.{proj_name}",
                        weight_bits=weight_bits,
                        activation_bits=activation_bits,
                        act_quant=act_quant,
                        quantize_output=quantize_bmm_input,
                    )
                _replace_linear_with_smoothquant(
                    layer.self_attn,
                    "o_proj",
                    quantized_linear_names,
                    f"{layer_prefix}.self_attn.o_proj",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                )

            shared_expert = layer.mlp.shared_expert
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                _replace_linear_with_smoothquant(
                    shared_expert,
                    proj_name,
                    quantized_linear_names,
                    f"{layer_prefix}.mlp.shared_expert.{proj_name}",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                )
            _replace_linear_with_smoothquant(
                layer.mlp,
                "shared_expert_gate",
                quantized_linear_names,
                f"{layer_prefix}.mlp.shared_expert_gate",
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                act_quant=act_quant,
            )
            layer.mlp.experts = SmoothQuantPackedMoeExperts.from_float(
                layer.mlp.experts,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
            )
            quantized_linear_names.append(f"{layer_prefix}.mlp.experts.gate_up_proj")
            quantized_linear_names.append(f"{layer_prefix}.mlp.experts.down_proj")
        elif model_type == "qwen3_moe" and hasattr(layer.mlp, "experts"):
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                _replace_linear_with_smoothquant(
                    layer.self_attn,
                    proj_name,
                    quantized_linear_names,
                    f"{layer_prefix}.self_attn.{proj_name}",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
            _replace_linear_with_smoothquant(
                layer.self_attn,
                "o_proj",
                quantized_linear_names,
                f"{layer_prefix}.self_attn.o_proj",
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                act_quant=act_quant,
            )

            if isinstance(layer.mlp.experts, nn.ModuleList):
                for expert_index, expert in enumerate(layer.mlp.experts):
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        _replace_linear_with_smoothquant(
                            expert,
                            proj_name,
                            quantized_linear_names,
                            f"{layer_prefix}.mlp.experts.{expert_index}.{proj_name}",
                            weight_bits=weight_bits,
                            activation_bits=activation_bits,
                            act_quant=act_quant,
                        )
            else:
                layer.mlp.experts = SmoothQuantPackedMoeExperts.from_float(
                    layer.mlp.experts,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                )
                quantized_linear_names.append(f"{layer_prefix}.mlp.experts.gate_up_proj")
                quantized_linear_names.append(f"{layer_prefix}.mlp.experts.down_proj")
        else:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                _replace_linear_with_smoothquant(
                    layer.self_attn,
                    proj_name,
                    quantized_linear_names,
                    f"{layer_prefix}.self_attn.{proj_name}",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
            _replace_linear_with_smoothquant(
                layer.self_attn,
                "o_proj",
                quantized_linear_names,
                f"{layer_prefix}.self_attn.o_proj",
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                act_quant=act_quant,
            )
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                _replace_linear_with_smoothquant(
                    layer.mlp,
                    proj_name,
                    quantized_linear_names,
                    f"{layer_prefix}.mlp.{proj_name}",
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    act_quant=act_quant,
                )
    return quantized_linear_names


def quantize_model(
    model,
    weight_bits=8,
    activation_bits=8,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=True,
):
    if weight_quant != "per_channel":
        raise ValueError(f"Invalid weight_quant: {weight_quant}")
    return quantize_llama_like(
        model,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        act_quant=act_quant,
        quantize_bmm_input=quantize_bmm_input,
    )
# Adapt SmoothQuant to Qwen3, Qwen3-VL, and Qwen3.5 models.
