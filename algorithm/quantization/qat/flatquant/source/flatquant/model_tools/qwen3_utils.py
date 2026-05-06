import torch

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.function_utils import get_decompose_dim
from flatquant.function_utils import get_init_scale
from flatquant.quant_utils import ActivationQuantizer
from flatquant.trans_utils import InvDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix
from flatquant.trans_utils import SVDDecomposeTransMatrix
from flatquant.trans_utils import SVDSingleTransMatrix
from flatquant.utils import skip_initialization

from flatquant.model_tools.qwen_utils import FlatQuantQwen2MLP
from flatquant.model_tools.qwen_utils import _weight_device

from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS as QWEN3_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as qwen3_eager_attention_forward
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS as QWEN3_VL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward as qwen3_vl_eager_attention_forward


def _decoder_root(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise NotImplementedError(f"Unsupported Qwen3 backbone: {type(model)}")


def _resolve_attention_interface(attention_functions, implementation, eager_attention_fn):
    if hasattr(attention_functions, "get_interface"):
        return attention_functions.get_interface(implementation, eager_attention_fn)
    if implementation == "eager":
        return eager_attention_fn
    return attention_functions[implementation]


class _FlatQuantQwen3AttentionMixin:
    attention_functions = QWEN3_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_apply_rotary_pos_emb)

    def _init_flatquant_attention(self, args, module):
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
        self.head_dim = getattr(
            module,
            "head_dim",
            getattr(module.config, "head_dim", self.hidden_size // self.num_heads),
        )
        self.attention_dropout = getattr(module, "attention_dropout", module.config.attention_dropout)
        self.scaling = getattr(module, "scaling", self.head_dim**-0.5)
        self.sliding_window = getattr(module, "sliding_window", None)
        self.is_causal = getattr(module, "is_causal", True)

        self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj)
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm
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
            self.o_trans = single_trans_matrix(self.num_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = single_trans_matrix(self.head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = single_trans_matrix(self.head_dim)
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
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            attn_output = torch.matmul(
                attn_output,
                self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output),
            ).reshape(init_shape)
            return self.o_proj(attn_output)

        init_shape = attn_output.shape
        attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
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
        **kwargs,
    ):
        attention_interface = _resolve_attention_interface(
            self.attention_functions,
            self.config._attn_implementation,
            self.eager_attention_fn,
        )
        attention_kwargs = {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            "output_attentions": output_attentions,
            **kwargs,
        }
        if self.sliding_window is not None:
            attention_kwargs["sliding_window"] = self.sliding_window
        return attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **attention_kwargs,
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


class FlatQuantQwen3Attention(_FlatQuantQwen3AttentionMixin, Qwen3Attention):
    def __init__(self, args, module: Qwen3Attention):
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
        del position_ids, use_cache, cache_position
        if position_embeddings is None:
            raise AttributeError("FlatQuantQwen3Attention requires `position_embeddings` from the parent decoder layer.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        attn_dtype = query_states.dtype

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).to(attn_dtype)
        attn_output = self._project_attn_output(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = attn_weights.to(attn_dtype)
        return attn_output, attn_weights


class FlatQuantQwen3VLTextAttention(_FlatQuantQwen3AttentionMixin, Qwen3VLTextAttention):
    attention_functions = QWEN3_VL_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_vl_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_vl_apply_rotary_pos_emb)

    def __init__(self, args, module: Qwen3VLTextAttention):
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
        del position_ids, use_cache, cache_position
        if position_embeddings is None:
            raise AttributeError(
                "FlatQuantQwen3VLTextAttention requires `position_embeddings` from the parent decoder layer."
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        attn_dtype = query_states.dtype

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attn_output, attn_weights = self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).to(attn_dtype)
        attn_output = self._project_attn_output(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = attn_weights.to(attn_dtype)
        return attn_output, attn_weights


def apply_flatquant_to_qwen3(args, model):
    skip_initialization()
    decoder_root = _decoder_root(model)
    attention_wrapper_cls = (
        FlatQuantQwen3VLTextAttention if getattr(model.config, "model_type", None) == "qwen3_vl" else FlatQuantQwen3Attention
    )
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
# Adapt FlatQuant to new models and address SplitQuant degradation on Qwen3.5.
