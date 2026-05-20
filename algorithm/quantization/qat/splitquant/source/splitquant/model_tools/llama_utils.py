import inspect
import math

import torch
import torch.nn as nn

from splitquant.quant_utils import ActivationQuantizer
from splitquant.utils import skip_initialization
from splitquant.function_utils import get_init_scale
from splitquant.trans_utils import SVDSingleGroupTransMatrix
from splitquant.trans_utils import SVDSingleTransMatrix
from splitquant.split_linear import SplitQuantizedLinear
from splitquant.model_tools.device_utils import align_attention_auxiliary_tensors
from splitquant.model_tools.device_utils import build_cache_kwargs_on_device

from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, \
                                                     apply_rotary_pos_emb, eager_attention_forward


from tqdm import tqdm


_LLAMA_APPLY_ROTARY_HAS_POSITION_IDS = "position_ids" in inspect.signature(apply_rotary_pos_emb).parameters


def _apply_llama_rotary_pos_emb(query_states, key_states, cos, sin, position_ids):
    if _LLAMA_APPLY_ROTARY_HAS_POSITION_IDS:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _resolve_split_group_size(args) -> int:
    group_sizes = []
    if args.w_bits < 16:
        group_sizes.append(int(args.w_groupsize))
    if args.a_bits < 16:
        group_sizes.append(int(args.a_groupsize))
    if not group_sizes:
        return -1
    if any(size <= 0 for size in group_sizes):
        raise ValueError("SplitQuant requires positive group sizes for quantized weights/activations.")
    first = group_sizes[0]
    if any(size != first for size in group_sizes[1:]):
        raise ValueError("SplitQuant requires activation and weight group sizes to match.")
    return first


def _build_group_trans(in_features: int, group_size: int, add_diag: bool, trans_name: str):
    if in_features % group_size != 0:
        raise ValueError(
            f"SplitQuant requires {trans_name} in_features={in_features} divisible by split group size={group_size}."
        )
    return SVDSingleGroupTransMatrix(in_features, group_size, add_diag=add_diag)


class SplitQuantLlamaMLP(LlamaMLP):
    def __init__(self, args, module: LlamaMLP):
        super().__init__(module.config)
        self.args = args
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1
        self.up_proj = SplitQuantizedLinear(args, module.up_proj)
        self.gate_proj = SplitQuantizedLinear(args, module.gate_proj)
        self.down_proj = SplitQuantizedLinear(args, module.down_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = self.up_proj.linear.weight.device
            self.register_buffer(
                "up_smax",
                torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0], device=stat_device) * 1e-5,
            )
            self.register_buffer(
                "down_smax",
                torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0], device=stat_device) * 1e-5,
            )
        
    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self.up_gate_trans = _build_group_trans(
                self.up_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "LLaMA MLP up/gate transform",
            )
            self.down_trans = _build_group_trans(
                self.down_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "LLaMA MLP down transform",
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
        '''origin implement: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(self.down_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        # merge trans's diag scale
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.diag_scale.to(torch.float64)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax, alpha)
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax, alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class SplitQuantLlamaAttention(LlamaAttention):
    def __init__(self, args, module: LlamaAttention):
        super().__init__(module.config, module.layer_idx)
        self.args = args
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1
        self.hidden_size = getattr(module, "hidden_size", module.config.hidden_size)
        self.num_heads = getattr(module, "num_heads", module.config.num_attention_heads)
        self.num_key_value_heads = getattr(module, "num_key_value_heads", module.config.num_key_value_heads)
        self.num_key_value_groups = getattr(
            module,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.head_dim = getattr(module, "head_dim", self.hidden_size // self.num_heads)
        if hasattr(module, "rotary_emb"):
            self.rotary_emb = module.rotary_emb
        
        self.q_proj = SplitQuantizedLinear(args, module.q_proj)
        self.k_proj = SplitQuantizedLinear(args, module.k_proj)
        self.v_proj = SplitQuantizedLinear(args, module.v_proj)
        self.o_proj = SplitQuantizedLinear(args, module.o_proj)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(bits=args.q_bits, \
                                        sym=not(args.q_asym), lac=args.lac, groupsize=-1, )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=args.k_bits, \
                                        sym=not(args.k_asym), lac=args.lac, groupsize=-1, )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=args.v_bits, \
                                        sym=not(args.v_asym), lac=args.lac, groupsize=-1, )

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = self.q_proj.linear.weight.device
            self.register_buffer(
                "ln_smax",
                torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0], device=stat_device) * 1e-5,
            )

    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self.ln_trans = _build_group_trans(
                self.q_proj.linear.weight.shape[1],
                self.group_size,
                self.args.add_diag,
                "LLaMA attention input transform",
            )
            self.o_trans = SVDSingleTransMatrix(self.config.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SVDSingleTransMatrix(head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SVDSingleTransMatrix(head_dim)
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
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(self.ln_smax, \
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach())
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

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        # Q/K transform
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)

        hidden_shape = (*input_shape, -1, self.head_dim)
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
                raise AttributeError(
                    "SplitQuantLlamaAttention requires `position_embeddings` for modern transformers Llama models."
                )
            try:
                cos, sin = self.rotary_emb(value_states, position_ids)
            except TypeError:
                kv_seq_len = key_states.shape[-2]
                cache_obj = past_key_value if past_key_value is not None else kwargs.get("past_key_values")
                if cache_obj is not None:
                    if self.layer_idx is None:
                        raise ValueError(
                            f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                            "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                            "with a layer index."
                        )
                    kv_seq_len += cache_obj.get_usable_length(kv_seq_len, self.layer_idx)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings
        query_states, key_states = _apply_llama_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )
        # ---- here do the quantization ----
        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        cache_obj = past_key_value if past_key_value is not None else kwargs.get("past_key_values")
        if cache_obj is not None:
            cache_kwargs = build_cache_kwargs_on_device(query_states.device, sin=sin, cos=cos, cache_position=cache_position)
            key_states, value_states = cache_obj.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self._project_attn_output(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

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
        qkvw_smax = torch.cat([self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


def apply_splitquant_to_llama(args, model):
    skip_initialization()
    # Replace module with SplitQuant version
    for layer in tqdm(range(model.config.num_hidden_layers), desc="Applying SplitQuant to model"):
        # attn
        model.model.layers[layer].self_attn = SplitQuantLlamaAttention(args, model.model.layers[layer].self_attn)
        # mlp
        model.model.layers[layer].mlp = SplitQuantLlamaMLP(args, model.model.layers[layer].mlp)
    return model
# Adapt SplitQuant to Qwen2.5, LLaMA-2, LLaMA-3, Qwen2.5-VL, and MiniCPM models.
