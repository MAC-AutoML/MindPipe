import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from splitquant.function_utils import get_init_scale
from splitquant.quant_utils import ActivationQuantizer
from splitquant.quant_utils import WeightQuantizer
from splitquant.quant_utils import set_quantizer_state
from splitquant.split_linear import SplitQuantizedLinear
from splitquant.split_utils import reparameterize_ln
from splitquant.trans_utils import SVDSingleTransMatrix
from splitquant.utils import skip_initialization

from splitquant.model_tools.qwen_split_utils import SplitQuantQwenMLP
from splitquant.model_tools.qwen_split_utils import _build_group_trans
from splitquant.model_tools.qwen_split_utils import _resolve_split_group_size
from splitquant.model_tools.qwen_split_utils import _weight_device
from splitquant.model_tools.device_utils import align_attention_auxiliary_tensors
from splitquant.model_tools.device_utils import get_module_device
from splitquant.model_tools.device_utils import move_tensor_tree_to_device

from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS as QWEN3_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as qwen3_eager_attention_forward
from transformers.models.qwen3_moe.modeling_qwen3_moe import ALL_ATTENTION_FUNCTIONS as QWEN3_MOE_ATTENTION_FUNCTIONS
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb as qwen3_moe_apply_rotary_pos_emb
from transformers.models.qwen3_moe.modeling_qwen3_moe import eager_attention_forward as qwen3_moe_eager_attention_forward
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


def _apply_trans_to_weight(weight: torch.Tensor, trans: nn.Module) -> torch.Tensor:
    original_shape = weight.shape
    return trans(weight.reshape(-1, original_shape[-1]), inv_t=True).reshape(original_shape)


def _apply_trans_to_packed_weight(weight: torch.Tensor, trans: nn.Module) -> torch.Tensor:
    original_shape = weight.shape
    return _apply_trans_to_weight(weight.reshape(-1, original_shape[-1]), trans).reshape(original_shape)


def _grouped_expert_mm(inputs: torch.Tensor, expert_indices: torch.Tensor, weights: torch.Tensor, alignment: int = 8):
    grouped_mm = getattr(torch, "_grouped_mm", None)
    if grouped_mm is None or inputs.device.type != "cuda" or inputs.dtype not in (torch.float16, torch.bfloat16):
        return None
    if weights.ndim != 3 or weights.shape[1] % 16 or weights.shape[2] % 16:
        return None
    unique, counts = torch.unique_consecutive(expert_indices, return_counts=True)
    if unique.numel() == 0:
        return None
    chunks, raw_counts, padded_counts = [], [], []
    start = 0
    for count_value in counts.tolist():
        count = int(count_value)
        padded = (count + alignment - 1) // alignment * alignment
        chunk = inputs[start : start + count]
        if padded != count:
            chunk = torch.cat((chunk, torch.zeros((padded - count, chunk.shape[-1]), device=chunk.device, dtype=chunk.dtype)))
        chunks.append(chunk)
        raw_counts.append(count)
        padded_counts.append(padded)
        start += count
    packed_inputs = torch.cat(chunks, dim=0).contiguous()
    offsets = torch.tensor(padded_counts, device=inputs.device, dtype=torch.int32).cumsum(0)
    try:
        output = grouped_mm(packed_inputs, weights[unique].contiguous(), offsets)
    except (RuntimeError, NotImplementedError):
        return None
    outputs, start = [], 0
    for count, padded in zip(raw_counts, padded_counts):
        outputs.append(output[start : start + count])
        start += padded
    return torch.cat(outputs, dim=0)


def _route_index_tensors(top_k_index: torch.Tensor):
    token_count, top_k = top_k_index.shape
    token_indices = torch.arange(token_count, device=top_k_index.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    top_k_positions = torch.arange(top_k, device=top_k_index.device).unsqueeze(0).expand(token_count, -1).reshape(-1)
    expert_indices = top_k_index.reshape(-1)
    order = torch.argsort(expert_indices, stable=True)
    return expert_indices.index_select(0, order), token_indices.index_select(0, order), top_k_positions.index_select(0, order)


class _SplitQuantQwen3AttentionMixin:
    attention_functions = QWEN3_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_apply_rotary_pos_emb)

    def _init_splitquant_attention(self, args, module):
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
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1

        self.q_proj = SplitQuantizedLinear(args, module.q_proj)
        self.k_proj = SplitQuantizedLinear(args, module.k_proj)
        self.v_proj = SplitQuantizedLinear(args, module.v_proj)
        self.o_proj = SplitQuantizedLinear(args, module.o_proj)
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
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self.ln_trans = _build_group_trans(
                self.q_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "Qwen3 attention input transform",
            )
            self.o_trans = SVDSingleTransMatrix(self.num_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SVDSingleTransMatrix(self.head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SVDSingleTransMatrix(self.head_dim)
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


class SplitQuantQwen3Attention(_SplitQuantQwen3AttentionMixin, Qwen3Attention):
    def __init__(self, args, module: Qwen3Attention):
        super().__init__(module.config, module.layer_idx)
        self._init_splitquant_attention(args, module)

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
        del use_cache
        if position_embeddings is None:
            raise AttributeError("SplitQuantQwen3Attention requires `position_embeddings` from the parent decoder layer.")

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

        attention_mask, position_ids, cache_position, position_embeddings = align_attention_auxiliary_tensors(
            query_states.device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
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


class SplitQuantQwen3VLTextAttention(_SplitQuantQwen3AttentionMixin, Qwen3VLTextAttention):
    attention_functions = QWEN3_VL_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_vl_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_vl_apply_rotary_pos_emb)

    def __init__(self, args, module: Qwen3VLTextAttention):
        super().__init__(module.config, module.layer_idx)
        self._init_splitquant_attention(args, module)

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
        del use_cache
        if position_embeddings is None:
            raise AttributeError(
                "SplitQuantQwen3VLTextAttention requires `position_embeddings` from the parent decoder layer."
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

        attention_mask, position_ids, cache_position, position_embeddings = align_attention_auxiliary_tensors(
            query_states.device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
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


class SplitQuantQwen3MoeAttention(_SplitQuantQwen3AttentionMixin, Qwen3MoeAttention):
    attention_functions = QWEN3_MOE_ATTENTION_FUNCTIONS
    eager_attention_fn = staticmethod(qwen3_moe_eager_attention_forward)
    apply_rotary = staticmethod(qwen3_moe_apply_rotary_pos_emb)

    def __init__(self, args, module: Qwen3MoeAttention):
        super().__init__(module.config, module.layer_idx)
        self._init_splitquant_attention(args, module)

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
        del use_cache
        if past_key_values is None:
            past_key_values = kwargs.pop("past_key_value", None)
        else:
            kwargs.pop("past_key_value", None)
        if position_embeddings is None:
            raise AttributeError("SplitQuantQwen3MoeAttention requires `position_embeddings` from the parent decoder layer.")

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

        attention_mask, position_ids, cache_position, position_embeddings = align_attention_auxiliary_tensors(
            query_states.device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            try:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
            except TypeError:
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


class SplitQuantQwen3MoeMLP(nn.Module):
    def __init__(self, args, module: nn.Module):
        super().__init__()
        self.args = args
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1
        self.gate_proj = SplitQuantizedLinear(args, module.gate_proj)
        self.up_proj = SplitQuantizedLinear(args, module.up_proj)
        self.down_proj = SplitQuantizedLinear(args, module.down_proj)
        self.act_fn = module.act_fn
        self._ori_mode = False
        self.add_fq_trans()

    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self._down_trans = _build_group_trans(
                self.down_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "Qwen3-MoE expert down transform",
            )
        else:
            self._down_trans = None

    def forward(self, hidden_states: torch.Tensor, input_trans=None) -> torch.Tensor:
        if self._ori_mode:
            hidden_states = self.act_fn(self.gate_proj._ori_forward(hidden_states)) * self.up_proj._ori_forward(hidden_states)
            return self.down_proj._ori_forward(hidden_states)
        hidden_states = self.act_fn(self.gate_proj(hidden_states, qa_trans=input_trans)) * self.up_proj(hidden_states, qa_trans=input_trans)
        if self._down_trans is not None:
            hidden_states = self._down_trans(hidden_states)
        return self.down_proj(hidden_states, qa_trans=self._down_trans)

    def reparameterize(self, input_trans=None):
        if self._down_trans is not None:
            self._down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=input_trans)
        self.up_proj.reparameterize(qa_trans=input_trans)
        self.down_proj.reparameterize(qa_trans=self._down_trans)

    def init_diag_scale(self, alpha=0.5):
        del alpha

    def rep_matrix_only(self):
        if self._down_trans is not None:
            self._down_trans.to_eval_mode()


class SplitQuantQwen3MoePackedExperts(nn.Module):
    def __init__(self, args, module: nn.Module):
        super().__init__()
        self.args = args
        self.num_experts = module.num_experts
        self.hidden_dim = module.hidden_dim
        self.intermediate_dim = module.intermediate_dim
        self.act_fn = module.act_fn
        self.gate_up_proj = module.gate_up_proj
        self.down_proj = module.down_proj
        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
        self.group_size = args.w_groupsize if args.w_groupsize > 0 else -1
        self.act_quantizer = ActivationQuantizer(
            bits=args.a_bits,
            sym=not(args.a_asym),
            lac=args.lac,
            groupsize=args.a_groupsize,
            in_channels=self.hidden_dim,
        )
        self.hidden_act_quantizer = ActivationQuantizer(
            bits=args.a_bits,
            sym=not(args.a_asym),
            lac=args.lac,
            groupsize=args.a_groupsize,
            in_channels=self.intermediate_dim,
        )
        self._ori_mode = False
        self._eval_mode = False
        self.add_fq_trans()

    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self._down_trans = _build_group_trans(
                self.down_proj.shape[-1],
                _resolve_split_group_size(self.args),
                self.args.add_diag,
                "Qwen3-MoE packed expert down transform",
            )
        else:
            self._down_trans = None

    def _group_weight(self, weight):
        if self.group_size > 0:
            return weight.reshape(-1, self.group_size)
        return weight

    def _degroup_weight(self, weight, original_shape):
        if self.group_size > 0:
            return weight.reshape(original_shape)
        return weight

    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.args.w_bits >= 16:
            return weight
        original_shape = weight.shape
        grouped_weight = self._group_weight(weight)
        self.weight_quantizer.find_params(grouped_weight)
        quantized = self.weight_quantizer(grouped_weight)
        return self._degroup_weight(quantized, original_shape).to(weight.dtype)

    def _weights(self, input_trans=None):
        if self._ori_mode or self._eval_mode:
            return self.gate_up_proj, self.down_proj
        gate_up_weight = self.gate_up_proj
        down_weight = self.down_proj
        if input_trans is not None:
            gate_up_weight = _apply_trans_to_packed_weight(gate_up_weight, input_trans)
        if self._down_trans is not None:
            down_weight = _apply_trans_to_packed_weight(down_weight, self._down_trans)
        return self._quantize_weight(gate_up_weight), self._quantize_weight(down_weight)

    def _fuse_down_diag_into_gate_up(self, gate_up_weight: torch.Tensor) -> torch.Tensor:
        if self._down_trans is None or not self._down_trans.add_diag:
            return gate_up_weight
        gate_size = gate_up_weight.shape[1] // 2
        scale = self._down_trans.diag_scale.to(dtype=torch.float64, device=gate_up_weight.device)
        up_weight = gate_up_weight[:, gate_size:, :]
        up_weight_fp64 = up_weight.to(torch.float64)
        up_weight_fp64.mul_(scale.view(1, -1, 1))
        up_weight.copy_(up_weight_fp64.to(up_weight.dtype))
        self._down_trans.use_diag = False
        return gate_up_weight

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, input_trans=None) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        gate_up_proj, down_proj = self._weights(input_trans=input_trans)
        route_experts, token_indices, top_k_positions = _route_index_tensors(top_k_index)
        grouped_input = hidden_states.index_select(0, token_indices)
        if not self._ori_mode:
            grouped_input = self.act_quantizer(grouped_input)
        gate_up_output = _grouped_expert_mm(grouped_input, route_experts, gate_up_proj)
        if gate_up_output is not None:
            gate_output, up_output = gate_up_output.chunk(2, dim=-1)
            intermediate = self.act_fn(gate_output) * up_output
            if not self._ori_mode:
                intermediate = self.hidden_act_quantizer(intermediate)
            if not self._ori_mode and self._down_trans is not None:
                intermediate = self._down_trans(intermediate)
            down_output = _grouped_expert_mm(
                intermediate,
                route_experts,
                down_proj.transpose(1, 2).contiguous(),
            )
            if down_output is not None:
                weights = top_k_weights.index_select(0, token_indices).gather(1, top_k_positions.unsqueeze(1))
                final_hidden_states.index_add_(0, token_indices, (down_output * weights).to(final_hidden_states.dtype))
                return final_hidden_states
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if not self._ori_mode:
                current_state = self.act_quantizer(current_state)
            gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if not self._ori_mode:
                current_hidden_states = self.hidden_act_quantizer(current_hidden_states)
            if not self._ori_mode and self._down_trans is not None:
                current_hidden_states = self._down_trans(current_hidden_states)
            current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    def reparameterize(self, input_trans=None):
        gate_up_weight = self.gate_up_proj.data
        if input_trans is not None:
            gate_up_weight = _apply_trans_to_packed_weight(gate_up_weight, input_trans)
        if self._down_trans is not None:
            self._down_trans.to_eval_mode()
            gate_up_weight = self._fuse_down_diag_into_gate_up(gate_up_weight)
        self.gate_up_proj.data = self._quantize_weight(gate_up_weight)
        del gate_up_weight

        down_weight = self.down_proj.data
        if self._down_trans is not None:
            down_weight = _apply_trans_to_packed_weight(down_weight, self._down_trans)
        self.down_proj.data = self._quantize_weight(down_weight)
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        del alpha

    def rep_matrix_only(self):
        if self._down_trans is not None:
            self._down_trans.to_eval_mode()


class SplitQuantQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, args, module: nn.Module):
        super().__init__()
        self.gate = module.gate
        self.top_k = int(getattr(module, "top_k", getattr(module.gate, "top_k", 1)))
        self.num_experts = int(getattr(module, "num_experts", getattr(module.gate, "num_experts", len(module.experts) if isinstance(module.experts, nn.ModuleList) else module.experts.num_experts)))
        self.norm_topk_prob = bool(getattr(module, "norm_topk_prob", getattr(module.gate, "norm_topk_prob", True)))
        self.experts_are_packed = hasattr(module.experts, "gate_up_proj")
        if self.experts_are_packed:
            self.experts = SplitQuantQwen3MoePackedExperts(args, module.experts)
        else:
            self.experts = nn.ModuleList([SplitQuantQwen3MoeMLP(args, expert) for expert in module.experts])
        self._ori_mode = False
        self._eval_mode = False
        self.calibrate_all_experts = False
        self._parent_post_attention_layernorm = None
        self.add_fq_trans()

    def add_fq_trans(self):
        sample_weight = self.gate.weight
        args = self.experts.args if self.experts_are_packed else self.experts[0].args
        if args.w_bits < 16 or args.a_bits < 16:
            self._moe_in_trans = _build_group_trans(
                sample_weight.shape[1],
                _resolve_split_group_size(args),
                args.add_diag,
                "Qwen3-MoE input transform",
            )
        else:
            self._moe_in_trans = None

    def _route(self, hidden_states: torch.Tensor, input_trans=None):
        if self._ori_mode or self._eval_mode or input_trans is None:
            gate_output = self.gate(hidden_states)
        else:
            router_weight = _apply_trans_to_weight(self.gate.weight, input_trans)
            router_bias = getattr(self.gate, "bias", None)
            gate_output = F.linear(hidden_states, router_weight, router_bias)
        if isinstance(gate_output, tuple):
            return gate_output
        router_logits = gate_output
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return router_logits, routing_weights.to(hidden_states.dtype), selected_experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        input_trans = None
        if not self._ori_mode and self._moe_in_trans is not None:
            hidden_states_reshaped = self._moe_in_trans(hidden_states_reshaped)
            input_trans = self._moe_in_trans
        router_logits, routing_weights, selected_experts = self._route(hidden_states_reshaped, input_trans=input_trans)
        if self.experts_are_packed:
            expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights, input_trans=input_trans)
        else:
            for expert in self.experts:
                expert._ori_mode = self._ori_mode
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_output = torch.zeros_like(hidden_states_reshaped)
            for expert_index, expert_layer in enumerate(self.experts):
                top_k_pos, token_index = torch.where(expert_mask[expert_index])
                if not self.calibrate_all_experts and token_index.numel() == 0:
                    continue
                if self.calibrate_all_experts:
                    current_hidden_states = expert_layer(hidden_states_reshaped, input_trans=input_trans)[token_index]
                else:
                    current_hidden_states = expert_layer(hidden_states_reshaped[token_index], input_trans=input_trans)
                if token_index.numel() > 0:
                    current_hidden_states = current_hidden_states * routing_weights[token_index, top_k_pos, None]
                    expert_output.index_add_(0, token_index, current_hidden_states.to(expert_output.dtype))
        return expert_output.reshape(batch_size, sequence_length, hidden_dim), router_logits

    def reparameterize(self):
        if self._moe_in_trans is not None:
            self._moe_in_trans.to_eval_mode()
            self.gate.weight.data = _apply_trans_to_weight(self.gate.weight.data, self._moe_in_trans)
        if self.experts_are_packed:
            self.experts.reparameterize(input_trans=self._moe_in_trans)
        else:
            for expert in self.experts:
                expert.reparameterize(input_trans=self._moe_in_trans)
        if self._moe_in_trans is not None and self._moe_in_trans.add_diag:
            if self._parent_post_attention_layernorm is None:
                raise RuntimeError("SplitQuant Qwen3-MoE MLP is missing parent post_attention_layernorm for diag fusion.")
            reparameterize_ln(self._parent_post_attention_layernorm, self._moe_in_trans)
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        del alpha

    def rep_matrix_only(self):
        if self._moe_in_trans is not None:
            self._moe_in_trans.to_eval_mode()
        if self.experts_are_packed:
            self.experts.rep_matrix_only()
        else:
            for expert in self.experts:
                expert.rep_matrix_only()

    def unfuse_experts(self, calibrate_all_experts: bool = False) -> bool:
        if not self.experts_are_packed:
            self.calibrate_all_experts = calibrate_all_experts
            return False

        packed = self.experts
        experts = []
        for expert_index in range(packed.num_experts):
            gate_up = packed.gate_up_proj[expert_index]
            down = packed.down_proj[expert_index]
            intermediate = gate_up.shape[0] // 2

            gate_linear = nn.Linear(gate_up.shape[1], intermediate, bias=False, device="meta")
            gate_linear.weight = nn.Parameter(gate_up[:intermediate], requires_grad=gate_up.requires_grad)
            up_linear = nn.Linear(gate_up.shape[1], intermediate, bias=False, device="meta")
            up_linear.weight = nn.Parameter(gate_up[intermediate:], requires_grad=gate_up.requires_grad)
            down_linear = nn.Linear(down.shape[1], down.shape[0], bias=False, device="meta")
            down_linear.weight = nn.Parameter(down, requires_grad=down.requires_grad)

            source = nn.Module()
            source.gate_proj = gate_linear
            source.up_proj = up_linear
            source.down_proj = down_linear
            source.act_fn = packed.act_fn
            expert = SplitQuantQwen3MoeMLP(packed.args, source)
            expert._down_trans = packed._down_trans

            # The packed weights are already transformed and quantized. During
            # Wanda only activation quantization must remain active.
            expert.gate_proj._eval_mode = packed._eval_mode
            expert.up_proj._eval_mode = packed._eval_mode
            expert.down_proj._eval_mode = packed._eval_mode
            expert.gate_proj.act_quantizer = copy.deepcopy(packed.act_quantizer)
            expert.up_proj.act_quantizer = copy.deepcopy(packed.act_quantizer)
            expert.down_proj.act_quantizer = copy.deepcopy(packed.hidden_act_quantizer)
            experts.append(expert)

        self.experts = nn.ModuleList(experts)
        self.experts_are_packed = False
        self.calibrate_all_experts = calibrate_all_experts
        return True

    def refuse_experts(self) -> bool:
        if self.experts_are_packed:
            return False
        experts = list(self.experts)
        if not experts:
            return False

        gate_up = torch.stack(
            [torch.cat((expert.gate_proj.linear.weight, expert.up_proj.linear.weight), dim=0) for expert in experts],
            dim=0,
        )
        down = torch.stack([expert.down_proj.linear.weight for expert in experts], dim=0)
        source = nn.Module()
        source.num_experts = len(experts)
        source.hidden_dim = gate_up.shape[-1]
        source.intermediate_dim = gate_up.shape[1] // 2
        source.act_fn = experts[0].act_fn
        source.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
        source.down_proj = nn.Parameter(down, requires_grad=False)
        packed = SplitQuantQwen3MoePackedExperts(experts[0].args, source)
        packed._down_trans = getattr(experts[0], "_down_trans", None)
        packed._eval_mode = self._eval_mode
        packed.act_quantizer = copy.deepcopy(experts[0].gate_proj.act_quantizer)
        packed.hidden_act_quantizer = copy.deepcopy(experts[0].down_proj.act_quantizer)
        self.experts = packed
        self.experts_are_packed = True
        return True


def unfuse_splitquant_qwen3_moe_experts(model: nn.Module, calibrate_all_experts: bool = False) -> int:
    """Expose packed Qwen3-MoE experts as SplitQuant Linear modules.

    This is used only while collecting/applying fixed Wanda masks and training
    compression LoRA. The packed representation is restored before export.
    The existing packed transforms are reused, so expanding the module tree
    does not replace the calibrated SplitQuant parameters.
    """
    replaced = 0
    blocks = [module for module in model.modules() if isinstance(module, SplitQuantQwen3MoeSparseMoeBlock)]
    for block in blocks:
        if not isinstance(block, SplitQuantQwen3MoeSparseMoeBlock):
            continue
        replaced += int(block.unfuse_experts(calibrate_all_experts=calibrate_all_experts))
    return replaced


def refuse_splitquant_qwen3_moe_experts(model: nn.Module) -> int:
    """Pack temporary per-expert SplitQuant modules back into fused tensors."""
    replaced = 0
    blocks = [module for module in model.modules() if isinstance(module, SplitQuantQwen3MoeSparseMoeBlock)]
    for block in blocks:
        if isinstance(block, SplitQuantQwen3MoeSparseMoeBlock):
            replaced += int(block.refuse_experts())
    return replaced


class SplitQuantQwen3MoeDecoderLayer(nn.Module):
    def __init__(self, args, ori_layer):
        super().__init__()
        self.hidden_size = ori_layer.hidden_size
        self.self_attn = SplitQuantQwen3MoeAttention(args, ori_layer.self_attn)
        if hasattr(ori_layer.mlp, "experts"):
            self.mlp = SplitQuantQwen3MoeSparseMoeBlock(args, ori_layer.mlp)
        else:
            self.mlp = SplitQuantQwenMLP(args, ori_layer.mlp)
        self.input_layernorm = ori_layer.input_layernorm
        self.post_attention_layernorm = ori_layer.post_attention_layernorm
        if hasattr(self.mlp, "_parent_post_attention_layernorm"):
            self.mlp._parent_post_attention_layernorm = self.post_attention_layernorm

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
        layer_device = get_module_device(self)
        hidden_states = move_tensor_tree_to_device(hidden_states, layer_device)
        attention_mask, position_ids, _cache_position, position_embeddings = align_attention_auxiliary_tensors(
            layer_device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output_attentions = bool(kwargs.pop("output_attentions", False))
        kwargs.pop("output_router_logits", None)
        past_key_values = kwargs.pop("past_key_value", past_key_values)
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
        hidden_states = residual + hidden_states.to(residual.device)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _router_logits = hidden_states
        hidden_states = residual + hidden_states.to(residual.device)
        return hidden_states


def apply_splitquant_to_qwen3(args, model):
    skip_initialization()
    decoder_root = _decoder_root(model)
    attention_wrapper_cls = (
        SplitQuantQwen3VLTextAttention if getattr(model.config, "model_type", None) == "qwen3_vl" else SplitQuantQwen3Attention
    )
    for layer_index in range(len(decoder_root.layers)):
        decoder_root.layers[layer_index].self_attn = attention_wrapper_cls(
            args,
            decoder_root.layers[layer_index].self_attn,
        )
        decoder_root.layers[layer_index].mlp = SplitQuantQwenMLP(
            args,
            decoder_root.layers[layer_index].mlp,
        )
    return model


def apply_splitquant_to_qwen3_moe(args, model):
    skip_initialization()
    decoder_root = _decoder_root(model)
    for layer_index in range(len(decoder_root.layers)):
        decoder_root.layers[layer_index] = SplitQuantQwen3MoeDecoderLayer(
            args,
            decoder_root.layers[layer_index],
        )
    set_quantizer_state(model, enable=True)
    return model
# Adapt OmniQuant and SplitQuant to new models.
