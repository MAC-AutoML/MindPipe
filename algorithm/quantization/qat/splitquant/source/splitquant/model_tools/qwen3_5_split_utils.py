import torch
import torch.nn as nn
import torch.nn.functional as F

from splitquant.function_utils import get_init_scale
from splitquant.quant_utils import set_quantizer_state
from splitquant.split_linear import SplitQuantizedLinear
from splitquant.utils import skip_initialization

from splitquant.model_tools.qwen_split_utils import SplitQuantQwenMLP
from splitquant.model_tools.qwen_split_utils import _build_group_trans
from splitquant.model_tools.qwen_split_utils import _resolve_split_group_size
from splitquant.model_tools.qwen_split_utils import _weight_device

from transformers.models.qwen3_5.modeling_qwen3_5 import ALL_ATTENTION_FUNCTIONS as QWEN3_5_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb
from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward as qwen3_5_eager_attention_forward


def _decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported Qwen3.5 backbone: {type(model)}")


def _resolve_attention_interface(attention_functions, implementation, eager_attention_fn):
    if hasattr(attention_functions, "get_interface"):
        return attention_functions.get_interface(implementation, eager_attention_fn)
    if implementation == "eager":
        return eager_attention_fn
    return attention_functions[implementation]


class SplitQuantQwen3_5Attention(Qwen3_5Attention):
    def __init__(self, args, module: Qwen3_5Attention):
        super().__init__(module.config, module.layer_idx)
        self.args = args
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = getattr(module, "is_causal", True)
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1

        self.q_proj = SplitQuantizedLinear(args, module.q_proj)
        self.k_proj = SplitQuantizedLinear(args, module.k_proj)
        self.v_proj = SplitQuantizedLinear(args, module.v_proj)
        self.o_proj = SplitQuantizedLinear(args, module.o_proj)
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = _weight_device(self.q_proj.linear)
            self.register_buffer(
                "ln_smax",
                torch.ones_like(
                    self.q_proj.linear.weight.abs().max(dim=0)[0],
                    device=stat_device,
                ) * 1e-5,
            )
            self.register_buffer(
                "o_smax",
                torch.ones_like(
                    self.o_proj.linear.weight.abs().max(dim=0)[0],
                    device=stat_device,
                ) * 1e-5,
            )

    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self.ln_trans = _build_group_trans(
                self.q_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "Qwen3.5 attention input transform",
            )
            self.o_trans = _build_group_trans(
                self.o_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "Qwen3.5 attention output transform",
            )
        else:
            self.ln_trans, self.o_trans = None, None
        self.kcache_trans = None
        self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style":
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def _project_attn_output(self, attn_output):
        if self._ori_mode:
            return self.o_proj._ori_forward(attn_output)
        if self.o_trans is not None:
            attn_output = self.o_trans(attn_output)
        return self.o_proj(attn_output, qa_trans=self.o_trans)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        del position_ids, use_cache, cache_position
        if position_embeddings is None:
            raise AttributeError("SplitQuantQwen3_5Attention requires `position_embeddings` from the parent decoder layer.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        attn_dtype = query_states.dtype

        query_states, gate = torch.chunk(
            query_states.view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen3_5_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = _resolve_attention_interface(
            QWEN3_5_ATTENTION_FUNCTIONS,
            self.config._attn_implementation,
            qwen3_5_eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).to(attn_dtype)
        attn_output = attn_output * torch.sigmoid(gate)

        if self._ori_mode and self.diag_init == "sq_style":
            self.o_smax = torch.maximum(
                self.o_smax,
                attn_output.reshape(-1, attn_output.shape[-1]).abs().max(0)[0].clone().detach(),
            )

        attn_output = self._project_attn_output(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = attn_weights.to(attn_dtype)
        return attn_output, attn_weights

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans)
        self.o_proj.reparameterize(qa_trans=self.o_trans)

    def init_diag_scale(self, alpha=0.5):
        if self.diag_init != "sq_style":
            return
        qkvw_smax = torch.cat(
            [self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight],
            dim=0,
        ).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        if self.o_trans is not None:
            ow_smax = self.o_proj.linear.weight.abs().max(dim=0)[0]
            self.o_trans.diag_scale.data = get_init_scale(ow_smax, self.o_smax, alpha)
        del self.ln_smax, self.o_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


class SplitQuantQwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, args, module: nn.Module):
        super().__init__()
        self.args = args
        self.hidden_size = module.hidden_size
        self.num_v_heads = module.num_v_heads
        self.num_k_heads = module.num_k_heads
        self.head_k_dim = module.head_k_dim
        self.head_v_dim = module.head_v_dim
        self.key_dim = module.key_dim
        self.value_dim = module.value_dim
        self.conv_kernel_size = module.conv_kernel_size
        self.layer_idx = module.layer_idx
        self.activation = module.activation
        self.act = module.act
        self.layer_norm_epsilon = module.layer_norm_epsilon
        self.conv_dim = module.conv_dim
        self.conv1d = module.conv1d
        self.dt_bias = module.dt_bias
        self.A_log = module.A_log
        self.norm = module.norm
        self.causal_conv1d_fn = module.causal_conv1d_fn
        self.causal_conv1d_update = module.causal_conv1d_update
        self.chunk_gated_delta_rule = module.chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = module.recurrent_gated_delta_rule
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1

        self.in_proj_qkv = SplitQuantizedLinear(args, module.in_proj_qkv)
        self.in_proj_z = SplitQuantizedLinear(args, module.in_proj_z)
        self.in_proj_b = SplitQuantizedLinear(args, module.in_proj_b)
        self.in_proj_a = SplitQuantizedLinear(args, module.in_proj_a)
        self.out_proj = SplitQuantizedLinear(args, module.out_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = _weight_device(self.in_proj_qkv.linear)
            self.register_buffer(
                "ln_smax",
                torch.ones_like(
                    self.in_proj_qkv.linear.weight.abs().max(dim=0)[0],
                    device=stat_device,
                ) * 1e-5,
            )
            self.register_buffer(
                "o_smax",
                torch.ones_like(
                    self.out_proj.linear.weight.abs().max(dim=0)[0],
                    device=stat_device,
                ) * 1e-5,
            )

    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self.ln_trans = _build_group_trans(
                self.in_proj_qkv.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "Qwen3.5 linear attention input transform",
            )
            self.o_trans = _build_group_trans(
                self.out_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "Qwen3.5 linear attention output transform",
            )
        else:
            self.ln_trans, self.o_trans = None, None
        self.kcache_trans = None
        self.vcache_trans = None

    def _trans_forward_inputs(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        mixed_qkv = self.in_proj_qkv(hidden_states, qa_trans=self.ln_trans).transpose(1, 2)
        z = self.in_proj_z(hidden_states, qa_trans=self.ln_trans)
        b = self.in_proj_b(hidden_states, qa_trans=self.ln_trans)
        a = self.in_proj_a(hidden_states, qa_trans=self.ln_trans)
        return mixed_qkv, z, b, a

    def _ori_forward_inputs(self, hidden_states):
        if self.diag_init == "sq_style":
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        mixed_qkv = self.in_proj_qkv._ori_forward(hidden_states).transpose(1, 2)
        z = self.in_proj_z._ori_forward(hidden_states)
        b = self.in_proj_b._ori_forward(hidden_states)
        a = self.in_proj_a._ori_forward(hidden_states)
        return mixed_qkv, z, b, a

    def _project_output(self, hidden_states):
        if self._ori_mode:
            return self.out_proj._ori_forward(hidden_states)
        if self.o_trans is not None:
            hidden_states = self.o_trans(hidden_states)
        return self.out_proj(hidden_states, qa_trans=self.o_trans)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        del position_ids, output_attentions, use_cache, cache_position, position_embeddings, kwargs
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            past_key_values is not None
            and past_key_values.has_previous_state(self.layer_idx)
            and seq_len == 1
        )
        if use_precomputed_states:
            conv_state = past_key_values.layers[self.layer_idx].conv_states
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states

        if self._ori_mode:
            mixed_qkv, z, b, a = self._ori_forward_inputs(hidden_states)
        else:
            mixed_qkv, z, b, a = self._trans_forward_inputs(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if past_key_values is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = past_key_values.update_conv_state(conv_state, self.layer_idx)
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
                output_final_state=past_key_values is not None,
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
                output_final_state=past_key_values is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if past_key_values is not None:
            past_key_values.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        if self._ori_mode and self.diag_init == "sq_style":
            self.o_smax = torch.maximum(
                self.o_smax,
                core_attn_out.reshape(-1, core_attn_out.shape[-1]).abs().max(0)[0].clone().detach(),
            )

        return self._project_output(core_attn_out)

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.in_proj_qkv.reparameterize(qa_trans=self.ln_trans)
        self.in_proj_z.reparameterize(qa_trans=self.ln_trans)
        self.in_proj_b.reparameterize(qa_trans=self.ln_trans)
        self.in_proj_a.reparameterize(qa_trans=self.ln_trans)
        self.out_proj.reparameterize(qa_trans=self.o_trans)

    def init_diag_scale(self, alpha=0.5):
        if self.diag_init != "sq_style":
            return
        in_proj_smax = torch.cat(
            [
                self.in_proj_qkv.linear.weight,
                self.in_proj_z.linear.weight,
                self.in_proj_b.linear.weight,
                self.in_proj_a.linear.weight,
            ],
            dim=0,
        ).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(in_proj_smax, self.ln_smax, alpha)
        if self.o_trans is not None:
            out_proj_smax = self.out_proj.linear.weight.abs().max(dim=0)[0]
            self.o_trans.diag_scale.data = get_init_scale(out_proj_smax, self.o_smax, alpha)
        del self.ln_smax, self.o_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


class SplitQuantQwen3_5DecoderLayer(nn.Module):
    def __init__(self, args, ori_layer):
        super().__init__()
        self.hidden_size = ori_layer.hidden_size
        self.layer_type = ori_layer.layer_type
        if self.layer_type == "linear_attention":
            self.self_attn = SplitQuantQwen3_5GatedDeltaNet(args, ori_layer.linear_attn)
        elif self.layer_type == "full_attention":
            self.self_attn = SplitQuantQwen3_5Attention(args, ori_layer.self_attn)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {self.layer_type}")
        self.mlp = SplitQuantQwenMLP(args, ori_layer.mlp)
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            output_attentions = bool(kwargs.pop("output_attentions", False))
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


def apply_splitquant_to_qwen3_5(args, model):
    skip_initialization()
    decoder_root = _decoder_root(model)
    for layer_index in range(len(decoder_root.layers)):
        decoder_root.layers[layer_index] = SplitQuantQwen3_5DecoderLayer(
            args,
            decoder_root.layers[layer_index],
        )
    set_quantizer_state(model, enable=True)
    return model
