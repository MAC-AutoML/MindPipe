"""Qwen3.5/3.6 MoE expert 拆分工具。"""

from __future__ import annotations

import logging
import os
import gc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
_WARNED_NPU_MOE_GROUPED = False
_FALSE_ENV_VALUES = {"0", "false", "off", "no"}


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _empty_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu"):
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def _load_torch_npu_grouped_matmul():
    try:
        import torch_npu
    except Exception:
        return None
    return getattr(torch_npu, "npu_grouped_matmul", None)


def _get_first_attr(*items):
    for obj, attr_name in items:
        if obj is not None and hasattr(obj, attr_name):
            return getattr(obj, attr_name)
    raise AttributeError(f"Missing required attribute among {[name for _, name in items]}")


def _linear_from_weight_view(weight: torch.Tensor) -> nn.Linear:
    """用原始权重视图创建 Linear，避免拆 expert 时额外复制大权重。"""
    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device="meta",
    )
    linear.weight = nn.Parameter(weight.detach(), requires_grad=weight.requires_grad)
    return linear


def _copy_or_pad_2d(weight: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """复制 2D 权重；目标更大时用 0 补齐，目标更小时裁剪。"""
    result = weight.new_zeros((rows, cols))
    copy_rows = min(rows, weight.shape[0])
    copy_cols = min(cols, weight.shape[1])
    result[:copy_rows, :copy_cols].copy_(weight[:copy_rows, :copy_cols])
    return result


def _resize_linear_weight(linear: nn.Linear, rows: int, cols: int) -> None:
    """原地调整 Linear 权重形状，保留已有通道并用 0 补齐新增通道。"""
    old_weight = linear.weight.data
    linear.weight = nn.Parameter(
        _copy_or_pad_2d(old_weight, rows, cols),
        requires_grad=linear.weight.requires_grad,
    )
    linear.out_features = rows
    linear.in_features = cols
    if linear.bias is not None:
        old_bias = linear.bias.data
        new_bias = old_bias.new_zeros(rows)
        copy_rows = min(rows, old_bias.numel())
        new_bias[:copy_rows].copy_(old_bias[:copy_rows])
        linear.bias = nn.Parameter(new_bias, requires_grad=linear.bias.requires_grad)


def _resize_mlp_to_intermediate_size(mlp: nn.Module, intermediate_size: int) -> None:
    """把 gate/up/down MLP 调整到统一 intermediate size，便于 HF 原生加载。"""
    gate_proj = getattr(mlp, "gate_proj")
    up_proj = getattr(mlp, "up_proj")
    down_proj = getattr(mlp, "down_proj")
    hidden_size = int(gate_proj.weight.shape[1])
    _resize_linear_weight(gate_proj, intermediate_size, hidden_size)
    _resize_linear_weight(up_proj, intermediate_size, hidden_size)
    _resize_linear_weight(down_proj, int(down_proj.weight.shape[0]), intermediate_size)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(intermediate_size)


def _get_text_root(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            return language_model
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            return language_model.model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return None


def _get_text_layers(model: nn.Module):
    root = _get_text_root(model)
    if root is None or not hasattr(root, "layers"):
        return []
    return list(root.layers)


def _is_full_attention_layer(layer: nn.Module) -> bool:
    attn = getattr(layer, "self_attn", None)
    return all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))


def _record_qwen3_5_attention_export_shapes(model: nn.Module) -> None:
    if hasattr(model, "_mindpipe_qwen35_attention_export_shapes"):
        return
    text_config = _get_text_config(model)
    if text_config is None:
        return

    for layer in _get_text_layers(model):
        if not _is_full_attention_layer(layer):
            continue
        attn = layer.self_attn
        hidden_size = int(getattr(text_config, "hidden_size"))
        num_attention_heads = int(getattr(text_config, "num_attention_heads"))
        num_key_value_heads = int(getattr(text_config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(text_config, "head_dim", hidden_size // num_attention_heads))
        model._mindpipe_qwen35_attention_export_shapes = {
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "num_key_value_groups": int(
                getattr(
                    text_config,
                    "num_key_value_groups",
                    num_attention_heads // num_key_value_heads,
                )
            ),
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "q_proj": tuple(attn.q_proj.weight.shape),
            "k_proj": tuple(attn.k_proj.weight.shape),
            "v_proj": tuple(attn.v_proj.weight.shape),
            "o_proj": tuple(attn.o_proj.weight.shape),
        }
        return


def _restore_qwen3_5_attention_for_hf_save(model: nn.Module) -> int:
    shapes = getattr(model, "_mindpipe_qwen35_attention_export_shapes", None)
    if not isinstance(shapes, dict):
        _record_qwen3_5_attention_export_shapes(model)
        shapes = getattr(model, "_mindpipe_qwen35_attention_export_shapes", None)
    if not isinstance(shapes, dict):
        return 0

    text_config = _get_text_config(model)
    if text_config is not None:
        _set_config_attr_if_present(text_config, "num_attention_heads", int(shapes["num_attention_heads"]))
        _set_config_attr_if_present(text_config, "num_key_value_heads", int(shapes["num_key_value_heads"]))
        _set_config_attr_if_present(text_config, "num_key_value_groups", int(shapes["num_key_value_groups"]))
        _set_config_attr_if_present(text_config, "head_dim", int(shapes["head_dim"]))
        _set_config_attr_if_present(text_config, "hidden_size", int(shapes["hidden_size"]))

    restored = 0
    for layer in _get_text_layers(model):
        if not _is_full_attention_layer(layer):
            continue
        attn = layer.self_attn
        for attr_name, value in (
            ("num_heads", shapes["num_attention_heads"]),
            ("num_attention_heads", shapes["num_attention_heads"]),
            ("num_key_value_heads", shapes["num_key_value_heads"]),
            ("num_key_value_groups", shapes["num_key_value_groups"]),
            ("head_dim", shapes["head_dim"]),
            ("hidden_size", shapes["hidden_size"]),
        ):
            try:
                setattr(attn, attr_name, int(value))
            except AttributeError:
                pass
        _resize_linear_weight(attn.q_proj, int(shapes["q_proj"][0]), int(shapes["q_proj"][1]))
        _resize_linear_weight(attn.k_proj, int(shapes["k_proj"][0]), int(shapes["k_proj"][1]))
        _resize_linear_weight(attn.v_proj, int(shapes["v_proj"][0]), int(shapes["v_proj"][1]))
        _resize_linear_weight(attn.o_proj, int(shapes["o_proj"][0]), int(shapes["o_proj"][1]))
        restored += 1
    if restored:
        LOGGER.info(
            "Restored %d Qwen3.5/3.6 full-attention block(s) for HF save "
            "(num_attention_heads=%d, num_key_value_heads=%d, head_dim=%d).",
            restored,
            int(shapes["num_attention_heads"]),
            int(shapes["num_key_value_heads"]),
            int(shapes["head_dim"]),
        )
    return restored


def _route_tokens(gate: nn.Module, hidden_states: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """兼容 Qwen3.5 原生 router 和 llm-compressor Linear router。"""
    gate_output = gate(hidden_states)
    if isinstance(gate_output, tuple):
        _, routing_weights, selected_experts = gate_output
        return routing_weights.to(hidden_states.dtype), selected_experts

    router_probs = F.softmax(gate_output, dtype=torch.float, dim=-1)
    routing_weights, selected_experts = torch.topk(
        router_probs,
        top_k,
        dim=-1,
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.to(hidden_states.dtype), selected_experts


class UnfusedQwen3_5MoeMLP(nn.Module):
    """单个 routed expert，暴露为 gate/up/down 三个 Linear。"""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        act_fn,
    ):
        super().__init__()
        self.gate_proj = _linear_from_weight_view(gate_weight)
        self.up_proj = _linear_from_weight_view(up_weight)
        self.down_proj = _linear_from_weight_view(down_weight)
        self.act_fn = act_fn
        self.intermediate_size = int(gate_weight.shape[0])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class SequentialQwen3_5MoeExperts(nn.ModuleList):
    """把 Qwen3.5/3.6 fused 3D expert 参数拆成 per-expert Linear。"""

    def __init__(self, config: Any, original: nn.Module):
        gate_up_data = original.gate_up_proj
        down_data = original.down_proj
        self.num_experts = int(gate_up_data.shape[0])
        intermediate_size = int(getattr(config, "moe_intermediate_size", down_data.shape[-1]))
        act_fn = getattr(original, "act_fn", None)
        if act_fn is None:
            from transformers.activations import ACT2FN

            act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

        experts = []
        for expert_index in range(self.num_experts):
            gate_up = gate_up_data[expert_index]
            down = down_data[expert_index]
            experts.append(
                UnfusedQwen3_5MoeMLP(
                    gate_weight=gate_up[:intermediate_size, :],
                    up_weight=gate_up[intermediate_size:, :],
                    down_weight=down,
                    act_fn=act_fn,
                )
            )
        super().__init__(experts)


class RefusedQwen3_5MoeExperts(nn.Module):
    """HF 原生 3D Parameter 形态的 routed experts。"""

    def __init__(
        self,
        experts: nn.ModuleList,
        intermediate_size: int,
        hidden_size: int,
        act_fn,
        parameter_device: str | torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.hidden_dim = int(hidden_size)
        self.intermediate_dim = int(intermediate_size)
        self.act_fn = act_fn

        first_expert = experts[0]
        first_gate = first_expert.gate_proj.weight
        first_down = first_expert.down_proj.weight
        requires_grad = any(parameter.requires_grad for expert in experts for parameter in expert.parameters())
        target_device = torch.device(parameter_device) if parameter_device is not None else first_gate.device

        gate_up = torch.zeros(
            (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim),
            dtype=first_gate.dtype,
            device=target_device,
        )
        down = torch.zeros(
            (self.num_experts, self.hidden_dim, self.intermediate_dim),
            dtype=first_down.dtype,
            device=target_device,
        )

        for expert_index, expert in enumerate(experts):
            gate_weight = expert.gate_proj.weight.data
            up_weight = expert.up_proj.weight.data
            down_weight = expert.down_proj.weight.data

            if gate_weight.shape[0] != up_weight.shape[0]:
                raise ValueError(
                    "Qwen3.5/3.6 MoE expert gate_proj/up_proj width mismatch: "
                    f"expert={expert_index}, gate={tuple(gate_weight.shape)}, up={tuple(up_weight.shape)}"
                )
            if down_weight.shape[1] != gate_weight.shape[0]:
                raise ValueError(
                    "Qwen3.5/3.6 MoE expert down_proj input width mismatch: "
                    f"expert={expert_index}, down={tuple(down_weight.shape)}, gate={tuple(gate_weight.shape)}"
                )

            width = int(gate_weight.shape[0])
            if width > self.intermediate_dim:
                raise ValueError(
                    "Qwen3.5/3.6 MoE expert width exceeds export target: "
                    f"expert={expert_index}, width={width}, target={self.intermediate_dim}"
                )
            gate_up[expert_index, :width, :].copy_(gate_weight)
            gate_up[expert_index, self.intermediate_dim : self.intermediate_dim + width, :].copy_(up_weight)
            down[expert_index, :, :width].copy_(down_weight)

        self.gate_up_proj = nn.Parameter(gate_up, requires_grad=requires_grad)
        self.down_proj = nn.Parameter(down, requires_grad=requires_grad)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_index in expert_hit:
            expert_index = int(expert_index[0])
            top_k_pos, token_index = torch.where(expert_mask[expert_index])
            current_state = hidden_states[token_index]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_index]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_index])
            current_hidden_states = current_hidden_states * top_k_weights[token_index, top_k_pos, None]
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return final_hidden_states


class UnfusedQwen3_5MoeSparseMoeBlock(nn.Module):
    """Qwen3_5MoeSparseMoeBlock 的本地 unfused 版本。"""

    def __init__(
        self,
        original: nn.Module,
        config: Any,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = getattr(config, "text_config", config)
        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.top_k = int(_get_first_attr((text_config, "num_experts_per_tok"), (original, "top_k")))
        self.num_experts = int(_get_first_attr((text_config, "num_experts"), (original, "num_experts")))
        self.hidden_dim = int(_get_first_attr((text_config, "hidden_size"), (original, "hidden_dim")))
        self.hidden_size = self.hidden_dim
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.experts = SequentialQwen3_5MoeExperts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        routing_weights, selected_experts = _route_tokens(
            self.gate,
            hidden_states_reshaped,
            self.top_k,
        )

        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if (
            not self.calibrate_all_experts
            and _env_enabled("MINDPIPE_QWEN35_NPU_UNFUSED_MOE_GROUPED_MATMUL", False)
            and hidden_states.device.type == "npu"
            and hidden_states.dtype in {torch.float16, torch.bfloat16, torch.float32}
        ):
            grouped_output = self._forward_routed_experts_npu_grouped(
                hidden_states_reshaped,
                selected_experts,
                routing_weights,
                expert_mask,
                final_hidden_states,
            )
            if grouped_output is not None:
                final_hidden_states = grouped_output
                shared_expert_output = self.shared_expert(hidden_states_reshaped)
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
                    * shared_expert_output
                )
                final_hidden_states = final_hidden_states + shared_expert_output
                return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        for expert_index, expert_layer in enumerate(self.experts):
            top_k_pos, token_index = torch.where(expert_mask[expert_index])
            if not self.calibrate_all_experts and token_index.numel() == 0:
                continue
            if self.calibrate_all_experts:
                expert_output = expert_layer(hidden_states_reshaped)[token_index]
            else:
                expert_output = expert_layer(hidden_states_reshaped[token_index])

            if len(token_index) > 0:
                current_hidden_states = (
                    expert_output * routing_weights[token_index, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_index,
                    current_hidden_states.to(hidden_states.dtype),
                )

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def _forward_routed_experts_npu_grouped(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_mask: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        del selected_experts
        grouped_matmul = _load_torch_npu_grouped_matmul()
        if grouped_matmul is None:
            return None

        global _WARNED_NPU_MOE_GROUPED
        try:
            current_states: list[torch.Tensor] = []
            gate_weights: list[torch.Tensor] = []
            up_weights: list[torch.Tensor] = []
            down_weights: list[torch.Tensor] = []
            token_indices: list[torch.Tensor] = []
            top_k_positions: list[torch.Tensor] = []

            for expert_index, expert_layer in enumerate(self.experts):
                top_k_pos, token_index = torch.where(expert_mask[expert_index])
                if token_index.numel() == 0:
                    continue
                gate_proj = expert_layer.gate_proj
                up_proj = expert_layer.up_proj
                down_proj = expert_layer.down_proj
                current_states.append(hidden_states[token_index])
                gate_weights.append(gate_proj.weight.transpose(0, 1))
                up_weights.append(up_proj.weight.transpose(0, 1))
                down_weights.append(down_proj.weight.transpose(0, 1))
                token_indices.append(token_index)
                top_k_positions.append(top_k_pos)

            if not current_states:
                return final_hidden_states

            gate_outputs = grouped_matmul(current_states, gate_weights, group_type=-1)
            up_outputs = grouped_matmul(current_states, up_weights, group_type=-1)
            intermediate_states = [
                self.experts[0].act_fn(gate) * up
                for gate, up in zip(gate_outputs, up_outputs)
            ]
            down_outputs = grouped_matmul(intermediate_states, down_weights, group_type=-1)

            for token_index, top_k_pos, down_output in zip(token_indices, top_k_positions, down_outputs):
                current_hidden_states = down_output * routing_weights[token_index, top_k_pos, None]
                final_hidden_states.index_add_(
                    0,
                    token_index,
                    current_hidden_states.to(final_hidden_states.dtype),
                )
            return final_hidden_states
        except Exception as exc:
            if not _WARNED_NPU_MOE_GROUPED:
                LOGGER.warning("Qwen3.5/3.6 NPU grouped unfused experts failed; using per-expert path: %r", exc)
                _WARNED_NPU_MOE_GROUPED = True
            return None


class RefusedQwen3_5MoeSparseMoeBlock(nn.Module):
    """保存前合回 HF 原生 Qwen3_5MoeSparseMoeBlock 参数布局。"""

    def __init__(
        self,
        original: UnfusedQwen3_5MoeSparseMoeBlock,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        expert_parameter_device: str | torch.device | None = None,
    ):
        super().__init__()
        self.gate = original.gate
        self.top_k = original.top_k
        self.num_experts = original.num_experts
        self.hidden_dim = original.hidden_dim
        self.hidden_size = original.hidden_size

        self.shared_expert = original.shared_expert
        _resize_mlp_to_intermediate_size(
            self.shared_expert,
            int(shared_expert_intermediate_size),
        )
        self.shared_expert_gate = original.shared_expert_gate

        act_fn = getattr(original.experts[0], "act_fn", None)
        self.experts = RefusedQwen3_5MoeExperts(
            original.experts,
            intermediate_size=int(intermediate_size),
            hidden_size=int(self.hidden_dim),
            act_fn=act_fn,
            parameter_device=expert_parameter_device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        routing_weights, selected_experts = _route_tokens(
            self.gate,
            hidden_states_reshaped,
            self.top_k,
        )

        expert_output = self.experts(
            hidden_states_reshaped,
            selected_experts,
            routing_weights,
        )
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def _is_qwen3_5_moe_block(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock"
        and hasattr(module, "experts")
        and hasattr(module.experts, "gate_up_proj")
        and hasattr(module.experts, "down_proj")
    )


def unfuse_qwen3_5_moe_experts(
    model: nn.Module,
    calibrate_all_experts: bool = True,
) -> int:
    """原地拆分 Qwen3.5/3.6 MoE routed experts，返回替换模块数量。"""
    replacements = []
    unfused_count = 0
    for name, module in model.named_modules():
        if isinstance(module, UnfusedQwen3_5MoeSparseMoeBlock):
            module.calibrate_all_experts = calibrate_all_experts
            unfused_count += 1
        elif _is_qwen3_5_moe_block(module):
            replacements.append(name)

    if not replacements and not unfused_count:
        return 0

    _record_qwen3_5_attention_export_shapes(model)
    for name in replacements:
        original = model.get_submodule(name)
        replacement = UnfusedQwen3_5MoeSparseMoeBlock(
            original=original,
            config=getattr(model, "config", getattr(original, "config", None)),
            calibrate_all_experts=calibrate_all_experts,
        )
        model.set_submodule(name, replacement)
        del original

    if replacements:
        LOGGER.info("Unfused %d Qwen3.5/3.6 MoE block(s) for pruning.", len(replacements))
    return len(replacements)


def _get_text_config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    return getattr(config, "text_config", config)


def _get_mlp_width(mlp: nn.Module) -> int:
    gate_proj = getattr(mlp, "gate_proj")
    up_proj = getattr(mlp, "up_proj")
    down_proj = getattr(mlp, "down_proj")
    if gate_proj.weight.shape[0] != up_proj.weight.shape[0]:
        raise ValueError(
            "Qwen3.5/3.6 MoE MLP gate_proj/up_proj width mismatch: "
            f"gate={tuple(gate_proj.weight.shape)}, up={tuple(up_proj.weight.shape)}"
        )
    if down_proj.weight.shape[1] != gate_proj.weight.shape[0]:
        raise ValueError(
            "Qwen3.5/3.6 MoE MLP down_proj input width mismatch: "
            f"down={tuple(down_proj.weight.shape)}, gate={tuple(gate_proj.weight.shape)}"
        )
    return int(gate_proj.weight.shape[0])


def _set_config_attr_if_present(config: Any, name: str, value: int) -> None:
    if config is not None and hasattr(config, name):
        setattr(config, name, int(value))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((int(value) + multiple - 1) // multiple) * multiple


def refuse_qwen3_5_moe_experts_for_hf_save(model: nn.Module) -> int:
    """保存前把 routed experts 合回 HF 原生 3D Parameter，返回替换模块数量。

    结构化真剪枝可能让不同 expert/layer 的中间维度不同；HF 原生 Qwen3.5/3.6
    配置只能表达一个全局 moe_intermediate_size。这里采用“全局最大宽度 + 0 补齐”，
    并对齐到 grouped MoE kernel 友好的宽度，保持 forward 等价，同时让
    `from_pretrained` 可以直接加载和生成。
    """
    target_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, UnfusedQwen3_5MoeSparseMoeBlock)
    ]
    if not target_names:
        return 0

    _restore_qwen3_5_attention_for_hf_save(model)
    routed_widths: list[int] = []
    shared_widths: list[int] = []
    for name in target_names:
        block = model.get_submodule(name)
        for expert in block.experts:
            routed_widths.append(_get_mlp_width(expert))
        shared_widths.append(_get_mlp_width(block.shared_expert))

    # HF/torch grouped_mm requires bf16/fp16 expert strides to be 16-byte aligned.
    target_routed_width = _ceil_to_multiple(max(routed_widths), 8)
    target_shared_width = _ceil_to_multiple(max(shared_widths), 8)

    text_config = _get_text_config(model)
    _set_config_attr_if_present(text_config, "moe_intermediate_size", target_routed_width)
    _set_config_attr_if_present(text_config, "shared_expert_intermediate_size", target_shared_width)

    expert_parameter_device = (
        "cpu" if _env_enabled("MINDPIPE_QWEN35_HF_SAVE_EXPERTS_ON_CPU", True) else None
    )
    if expert_parameter_device == "cpu":
        LOGGER.info("Building refused Qwen3.5/3.6 routed expert tensors on CPU for HF save.")

    for name in target_names:
        block = model.get_submodule(name)
        replacement = RefusedQwen3_5MoeSparseMoeBlock(
            original=block,
            intermediate_size=target_routed_width,
            shared_expert_intermediate_size=target_shared_width,
            expert_parameter_device=expert_parameter_device,
        )
        model.set_submodule(name, replacement)
        del block
        _empty_accelerator_cache()

    LOGGER.info(
        "Refused %d Qwen3.5/3.6 MoE block(s) for HF save "
        "(moe_intermediate_size=%d, shared_expert_intermediate_size=%d).",
        len(target_names),
        target_routed_width,
        target_shared_width,
    )
    return len(target_names)


def set_qwen3_5_moe_calibrate_all_experts(
    model: nn.Module,
    enabled: bool,
) -> int:
    """切换 unfused MoE forward 是否在校准时跑所有 routed experts。"""
    updated = 0
    for module in model.modules():
        if isinstance(module, UnfusedQwen3_5MoeSparseMoeBlock):
            module.calibrate_all_experts = enabled
            updated += 1
    return updated
