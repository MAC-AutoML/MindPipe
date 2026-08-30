"""SplitQuant wrappers for Hugging Face Mixtral models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from splitquant.model_tools.device_utils import align_attention_auxiliary_tensors
from splitquant.model_tools.device_utils import get_module_device
from splitquant.model_tools.device_utils import move_tensor_tree_to_device
from splitquant.model_tools.llama_utils import SplitQuantLlamaAttention
from splitquant.model_tools.llama_utils import _build_group_trans
from splitquant.model_tools.llama_utils import _resolve_split_group_size
from splitquant.quant_utils import ActivationQuantizer
from splitquant.split_linear import SplitQuantizedLinear
from splitquant.split_utils import reparameterize_ln
from splitquant.utils import skip_initialization

from transformers.models.mixtral.modeling_mixtral import ALL_ATTENTION_FUNCTIONS
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers.models.mixtral.modeling_mixtral import apply_rotary_pos_emb
from transformers.models.mixtral.modeling_mixtral import eager_attention_forward


def _decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported Mixtral backbone: {type(model)}")


def _resolve_attention_interface(implementation):
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        return ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
    if implementation == "eager":
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[implementation]


def _apply_trans_to_weight(weight, trans):
    if trans is None:
        return weight
    return trans(weight, inv_t=True)


def _unpack_mixtral_experts(module):
    if isinstance(module.experts, nn.ModuleList):
        return module.experts
    if not hasattr(module.experts, "gate_up_proj") or not hasattr(module.experts, "down_proj"):
        raise TypeError(f"Unsupported Mixtral expert container: {type(module.experts)}")

    packed = module.experts
    experts = []
    for expert_index in range(packed.num_experts):
        gate_up = packed.gate_up_proj[expert_index]
        down = packed.down_proj[expert_index]
        intermediate_size = gate_up.shape[0] // 2

        source = nn.Module()
        source.w1 = nn.Linear(gate_up.shape[1], intermediate_size, bias=False, device="meta")
        source.w1.weight = nn.Parameter(gate_up[:intermediate_size], requires_grad=gate_up.requires_grad)
        source.w3 = nn.Linear(gate_up.shape[1], intermediate_size, bias=False, device="meta")
        source.w3.weight = nn.Parameter(gate_up[intermediate_size:], requires_grad=gate_up.requires_grad)
        source.w2 = nn.Linear(down.shape[1], down.shape[0], bias=False, device="meta")
        source.w2.weight = nn.Parameter(down, requires_grad=down.requires_grad)
        source.act_fn = packed.act_fn
        experts.append(source)
    return nn.ModuleList(experts)


class SplitQuantMixtralAttention(MixtralAttention):
    """LLaMA-style SplitQuant attention with Mixtral's attention API."""

    add_fq_trans = SplitQuantLlamaAttention.add_fq_trans
    _trans_forward_after_ln = SplitQuantLlamaAttention._trans_forward_after_ln
    _ori_forward_after_ln = SplitQuantLlamaAttention._ori_forward_after_ln
    quant_vcache = SplitQuantLlamaAttention.quant_vcache
    quant_kcache = SplitQuantLlamaAttention.quant_kcache
    _project_attn_output = SplitQuantLlamaAttention._project_attn_output
    reparameterize = SplitQuantLlamaAttention.reparameterize
    init_diag_scale = SplitQuantLlamaAttention.init_diag_scale
    rep_matrix_only = SplitQuantLlamaAttention.rep_matrix_only

    def __init__(self, args, module):
        nn.Module.__init__(self)
        self.args = args
        self.config = module.config
        self.layer_idx = getattr(module, "layer_idx", None)
        self.hidden_size = getattr(module, "hidden_size", module.config.hidden_size)
        self.num_heads = getattr(module, "num_heads", module.config.num_attention_heads)
        self.num_key_value_heads = getattr(module, "num_key_value_heads", module.config.num_key_value_heads)
        self.num_key_value_groups = getattr(
            module,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.head_dim = getattr(module, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = getattr(module, "scaling", self.head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", module.config.attention_dropout)
        self.is_causal = getattr(module, "is_causal", True)
        if hasattr(module, "rotary_emb"):
            self.rotary_emb = module.rotary_emb

        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1
        self.q_proj = SplitQuantizedLinear(args, module.q_proj)
        self.k_proj = SplitQuantizedLinear(args, module.k_proj)
        self.v_proj = SplitQuantizedLinear(args, module.v_proj)
        self.o_proj = SplitQuantizedLinear(args, module.o_proj)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(args.q_bits, sym=not args.q_asym, lac=args.lac, groupsize=-1)
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(args.k_bits, sym=not args.k_asym, lac=args.lac, groupsize=-1)
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(args.v_bits, sym=not args.v_asym, lac=args.lac, groupsize=-1)

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = self.q_proj.linear.weight.device
            self.register_buffer(
                "ln_smax",
                torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0], device=stat_device) * 1e-5,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        del use_cache
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        attn_dtype = query_states.dtype

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        attention_mask, position_ids, cache_position, position_embeddings = align_attention_auxiliary_tensors(
            query_states.device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if position_embeddings is None:
            if not hasattr(self, "rotary_emb"):
                raise AttributeError("SplitQuantMixtralAttention requires position embeddings or rotary_emb.")
            try:
                cos, sin = self.rotary_emb(value_states, position_ids)
            except TypeError:
                cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
        else:
            cos, sin = position_embeddings
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            try:
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx, cache_kwargs)
            except TypeError:
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx)

        attention_interface = _resolve_attention_interface(self.config._attn_implementation)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).to(attn_dtype)
        attn_output = self._project_attn_output(attn_output)
        if not output_attentions:
            attn_weights = None
        elif attn_weights is not None:
            attn_weights = attn_weights.to(attn_dtype)
        return attn_output, attn_weights


class SplitQuantMixtralExpert(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.w1 = SplitQuantizedLinear(args, module.w1)
        self.w2 = SplitQuantizedLinear(args, module.w2)
        self.w3 = SplitQuantizedLinear(args, module.w3)
        self.act_fn = module.act_fn
        self._ori_mode = False
        group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1
        self._down_trans = (
            _build_group_trans(self.w2.linear.weight.shape[1], group_size, args.add_diag, "Mixtral expert down transform")
            if group_size > 0
            else None
        )

    def forward(self, hidden_states, input_trans=None):
        if self._ori_mode:
            gate = self.w1._ori_forward(hidden_states)
            up = self.w3._ori_forward(hidden_states)
            return self.w2._ori_forward(self.act_fn(gate) * up)
        gate = self.w1(hidden_states, qa_trans=input_trans)
        up = self.w3(hidden_states, qa_trans=input_trans)
        hidden_states = self.act_fn(gate) * up
        if self._down_trans is not None:
            hidden_states = self._down_trans(hidden_states)
        return self.w2(hidden_states, qa_trans=self._down_trans)

    def reparameterize(self, input_trans=None):
        if self._down_trans is not None:
            self._down_trans.to_eval_mode()
        self.w1.reparameterize(qa_trans=input_trans)
        self.w3.reparameterize(qa_trans=input_trans)
        # The down projection must absorb the inverse diagonal before the
        # matching positive diagonal is fused into the up projection.
        self.w2.reparameterize(qa_trans=self._down_trans)
        if self._down_trans is not None and self._down_trans.add_diag:
            up_weight = self.w3.linear.weight
            scaled = up_weight.to(torch.float64).T
            scaled.mul_(self._down_trans.diag_scale.to(device=scaled.device, dtype=torch.float64))
            self.w3.linear.weight.data = scaled.T.to(up_weight.dtype)
            self._down_trans.use_diag = False

    def rep_matrix_only(self):
        if self._down_trans is not None:
            self._down_trans.to_eval_mode()


class SplitQuantMixtralSparseMoeBlock(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.gate = module.gate
        self.top_k = int(getattr(module, "top_k", getattr(module.gate, "top_k", 2)))
        self.jitter_noise = float(getattr(module, "jitter_noise", 0.0))
        unpacked_experts = _unpack_mixtral_experts(module)
        self.experts = nn.ModuleList([SplitQuantMixtralExpert(args, expert) for expert in unpacked_experts])
        self.num_experts = len(self.experts)
        self._ori_mode = False
        self._eval_mode = False
        self._parent_post_attention_layernorm = None
        group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1
        self._moe_in_trans = (
            _build_group_trans(self.gate.weight.shape[1], group_size, args.add_diag, "Mixtral MoE input transform")
            if group_size > 0
            else None
        )

    def _route(self, hidden_states, input_trans=None):
        if self._ori_mode or self._eval_mode or input_trans is None:
            gate_output = self.gate(hidden_states)
        else:
            gate_output = F.linear(hidden_states, _apply_trans_to_weight(self.gate.weight, input_trans))
        if isinstance(gate_output, tuple):
            return gate_output
        router_logits = gate_output
        router_probs = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return router_logits, routing_weights.to(hidden_states.dtype), selected_experts

    def forward(self, hidden_states):
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(1 - self.jitter_noise, 1 + self.jitter_noise)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        input_trans = None
        if not self._ori_mode and self._moe_in_trans is not None:
            hidden_states = self._moe_in_trans(hidden_states)
            input_trans = self._moe_in_trans
        _, routing_weights, selected_experts = self._route(hidden_states, input_trans=input_trans)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            expert._ori_mode = self._ori_mode
            top_k_pos, token_index = torch.where(expert_mask[expert_index])
            if token_index.numel() == 0:
                continue
            expert_output = expert(hidden_states[token_index], input_trans=input_trans)
            expert_output = expert_output * routing_weights[token_index, top_k_pos, None]
            final_hidden_states.index_add_(0, token_index, expert_output.to(final_hidden_states.dtype))
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def reparameterize(self):
        if self._moe_in_trans is not None:
            self._moe_in_trans.to_eval_mode()
            self.gate.weight.data = _apply_trans_to_weight(self.gate.weight.data, self._moe_in_trans)
        for expert in self.experts:
            expert.reparameterize(input_trans=self._moe_in_trans)
        if self._moe_in_trans is not None and self._moe_in_trans.add_diag:
            if self._parent_post_attention_layernorm is None:
                raise RuntimeError("SplitQuant Mixtral MoE is missing its parent post-attention layernorm.")
            reparameterize_ln(self._parent_post_attention_layernorm, self._moe_in_trans)
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        del alpha

    def rep_matrix_only(self):
        if self._moe_in_trans is not None:
            self._moe_in_trans.to_eval_mode()
        for expert in self.experts:
            expert.rep_matrix_only()


class SplitQuantMixtralDecoderLayer(nn.Module):
    def __init__(self, args, original_layer):
        super().__init__()
        self.hidden_size = original_layer.hidden_size
        self.self_attn = SplitQuantMixtralAttention(args, original_layer.self_attn)
        original_moe = getattr(original_layer, "mlp", None)
        if original_moe is None:
            original_moe = original_layer.block_sparse_moe
        self.mlp = SplitQuantMixtralSparseMoeBlock(args, original_moe)
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        # This is a back-reference used only while folding the MoE input
        # transform.  Registering it as a child module duplicates the RMSNorm
        # in state_dict and makes save_pretrained reject the shared tensor.
        object.__setattr__(self.mlp, "_parent_post_attention_layernorm", self.post_attention_layernorm)

    @property
    def block_sparse_moe(self):
        return self.mlp

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        **kwargs,
    ):
        layer_device = get_module_device(self)
        hidden_states = move_tensor_tree_to_device(hidden_states, layer_device)
        attention_mask, position_ids, _, position_embeddings = align_attention_auxiliary_tensors(
            layer_device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states.to(residual.device)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states.to(residual.device)


def apply_splitquant_to_mixtral(args, model):
    skip_initialization()
    decoder_root = _decoder_root(model)
    for layer_index, original_layer in enumerate(decoder_root.layers):
        decoder_root.layers[layer_index] = SplitQuantMixtralDecoderLayer(args, original_layer)
    return model
