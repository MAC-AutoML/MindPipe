import math

import torch
import torch.nn as nn

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.function_utils import get_decompose_dim
from flatquant.function_utils import get_init_scale
from flatquant.quant_utils import ActivationQuantizer
from flatquant.trans_utils import InvDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix
from flatquant.trans_utils import SVDDecomposeTransMatrix
from flatquant.trans_utils import SVDSingleTransMatrix
from flatquant.utils import skip_initialization

from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS as QWEN2_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward as qwen2_eager_attention_forward
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS as QWEN2_VL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward as qwen2_vl_eager_attention_forward
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv as repeat_kv_vl


def _decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported Qwen backbone: {type(model)}")


def _is_vl_model(model):
    return getattr(model.config, "model_type", None) == "qwen2_5_vl"


def _weight_device(module):
    return module.weight.device


class FlatQuantQwen2MLP(torch.nn.Module):
    def __init__(self, args, module: Qwen2MLP):
        super().__init__()
        self.args = args
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.act_fn = module.act_fn
        self.up_proj = FlatQuantizedLinear(args, module.up_proj)
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = _weight_device(self.up_proj.linear)
            self.register_buffer(
                "up_smax",
                torch.ones_like(
                    self.up_proj.linear.weight.abs().max(dim=0)[0],
                    device=stat_device,
                ) * 1e-5,
            )
            self.register_buffer(
                "down_smax",
                torch.ones_like(
                    self.down_proj.linear.weight.abs().max(dim=0)[0],
                    device=stat_device,
                ) * 1e-5,
            )

    def add_fq_trans(self):
        if self.args.direct_inv:
            decompose_trans_matrix = InvDecomposeTransMatrix
        else:
            decompose_trans_matrix = SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            self.up_gate_trans = decompose_trans_matrix(
                up_dim_left,
                up_dim_right,
                add_diag=self.args.add_diag,
            )
            down_dim_left, down_dim_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
            self.down_trans = decompose_trans_matrix(
                down_dim_left,
                down_dim_right,
                add_diag=self.args.add_diag,
            )
        else:
            self.up_gate_trans, self.down_trans = None, None

    def _trans_forward(self, x):
        if self.up_gate_trans is not None:
            x_ts = self.up_gate_trans(x)
        else:
            x_ts = x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)

        x_act_fn = self.act_fn(gate_states) * up_states
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.down_proj(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(
                self.up_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(
                self.down_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.diag_scale.to(torch.float64)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = torch.cat(
            [self.up_proj.linear.weight, self.gate_proj.linear.weight],
            dim=0,
        ).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax, alpha)
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax, alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class _FlatQuantQwenAttentionMixin:
    repeat_kv_fn = staticmethod(repeat_kv)
    attention_functions = QWEN2_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen2_eager_attention_forward)

    def _init_flatquant_attention(self, args, module):
        self.args = args
        self.hidden_size = getattr(module, "hidden_size", module.config.hidden_size)
        self.num_heads = getattr(module, "num_heads", module.config.num_attention_heads)
        self.num_key_value_heads = getattr(module, "num_key_value_heads", module.config.num_key_value_heads)
        self.num_key_value_groups = getattr(
            module,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.head_dim = getattr(
            module,
            "head_dim",
            getattr(module.config, "head_dim", self.hidden_size // self.num_heads),
        )
        self.attention_dropout = getattr(module, "attention_dropout", module.config.attention_dropout)
        self.scaling = getattr(module, "scaling", self.head_dim**-0.5)
        self.rope_scaling = getattr(module, "rope_scaling", getattr(module.config, "rope_scaling", None))
        self.sliding_window = getattr(module, "sliding_window", None)
        self.is_causal = getattr(module, "is_causal", True)
        if hasattr(module, "rotary_emb"):
            self.rotary_emb = module.rotary_emb

        self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(
                bits=args.q_bits,
                sym=not args.q_asym,
                lac=args.lac,
                groupsize=-1,
            )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(
                bits=args.k_bits,
                sym=not args.k_asym,
                lac=args.lac,
                groupsize=-1,
            )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(
                bits=args.v_bits,
                sym=not args.v_asym,
                lac=args.lac,
                groupsize=-1,
            )

        self._ori_mode = False
        self._eval_mode = False
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

    def add_fq_trans(self):
        if self.args.direct_inv:
            single_trans_matrix, decompose_trans_matrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            single_trans_matrix, decompose_trans_matrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = decompose_trans_matrix(
                ln_dim_left,
                ln_dim_right,
                add_diag=self.args.add_diag,
            )
            self.o_trans = single_trans_matrix(self.config.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = single_trans_matrix(head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = single_trans_matrix(head_dim)
        else:
            self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(
                hidden_states,
                qa_trans=self.ln_trans,
                out_trans=self.vcache_trans,
            )
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans and self.vcache_trans is not None:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def _project_attn_output(self, attn_output):
        if self._ori_mode:
            return self.o_proj._ori_forward(attn_output)
        if self.o_trans is None and self.vcache_trans is None:
            return self.o_proj(attn_output)
        if self.o_trans is None and self.vcache_trans is not None:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(
                -1,
                self.config.num_attention_heads,
                self.config.hidden_size // self.config.num_attention_heads,
            )
            attn_output = torch.matmul(
                attn_output,
                self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output),
            ).reshape(init_shape)
            return self.o_proj(attn_output)

        init_shape = attn_output.shape
        attn_output = attn_output.reshape(
            -1,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        attn_output = torch.matmul(
            self.o_trans.get_matrix().T.to(attn_output),
            attn_output,
        ).reshape(init_shape)
        if not self._eval_mode:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            return self.o_proj(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
        return self.o_proj(attn_output)

    def _attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        output_attentions,
        position_ids=None,
    ):
        attention_interface = self.eager_attention_fn
        if self.config._attn_implementation != "eager":
            attention_interface = self.attention_functions[self.config._attn_implementation]
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
        )

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.ln_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = torch.cat(
            [self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight],
            dim=0,
        ).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


class FlatQuantQwen2Attention(_FlatQuantQwenAttentionMixin, Qwen2Attention):
    def __init__(self, args, module: Qwen2Attention):
        super().__init__(module.config, module.layer_idx)
        self._init_flatquant_attention(args, module)

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
        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        attn_dtype = query_states.dtype

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_ids=position_ids,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).to(attn_dtype)
        attn_output = self._project_attn_output(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = attn_weights.to(attn_dtype)
        return attn_output, attn_weights


class FlatQuantQwen2_5_VLAttention(_FlatQuantQwenAttentionMixin, Qwen2_5_VLAttention):
    repeat_kv_fn = staticmethod(repeat_kv_vl)
    attention_functions = QWEN2_VL_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen2_vl_eager_attention_forward)

    def __init__(self, args, module: Qwen2_5_VLAttention):
        super().__init__(module.config, module.layer_idx)
        self._init_flatquant_attention(args, module)

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
        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        attn_dtype = query_states.dtype

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
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
            self.rope_scaling["mrope_section"],
        )

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_ids=position_ids,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).to(attn_dtype)
        attn_output = self._project_attn_output(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = attn_weights.to(attn_dtype)
        return attn_output, attn_weights


def apply_flatquant_to_qwen(args, model):
    skip_initialization()
    decoder_root = _decoder_root(model)
    attention_wrapper_cls = FlatQuantQwen2_5_VLAttention if _is_vl_model(model) else FlatQuantQwen2Attention
    for layer_index in range(len(decoder_root.layers)):
        decoder_root.layers[layer_index].self_attn = attention_wrapper_cls(
            args,
            decoder_root.layers[layer_index].self_attn,
        )
        decoder_root.layers[layer_index].mlp = FlatQuantQwen2MLP(
            args,
            decoder_root.layers[layer_index].mlp,
        )
    return model
