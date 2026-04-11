import torch
import torch.nn as nn
import torch.nn.functional as F

from splitquant.backbone_utils import get_decoder_layers
from splitquant.split_linear import SplitQuantizedLinear
from splitquant.function_utils import get_init_scale
from splitquant.quant_utils import ActivationQuantizer
from splitquant.trans_utils import SVDSingleGroupTransMatrix
from splitquant.trans_utils import SVDSingleTransMatrix
from splitquant.utils import skip_initialization

from tqdm import tqdm


def _weight_device(module):
    return module.weight.device


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


class SplitQuantMiniCPMMLP(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        object.__setattr__(self, "_original_module", module)
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.act_fn = module.act_fn
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1

        self.up_proj = SplitQuantizedLinear(args, module.up_proj)
        self.gate_proj = SplitQuantizedLinear(args, module.gate_proj)
        self.down_proj = SplitQuantizedLinear(args, module.down_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = _weight_device(self.up_proj.linear)
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
            self.up_gate_trans = SVDSingleGroupTransMatrix(
                self.up_proj.linear.weight.shape[1],
                self.group_size,
                add_diag=self.args.add_diag,
            )
            self.down_trans = SVDSingleGroupTransMatrix(
                self.down_proj.linear.weight.shape[1],
                self.group_size,
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
        hidden = self.act_fn(gate_states) * up_states
        if self.down_trans is not None:
            hidden = self.down_trans(hidden)
        return self.down_proj(hidden, qa_trans=self.down_trans)

    def _ori_forward(self, x):
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
            original_module = self._original_module
            if getattr(original_module.config, "pretraining_tp", 1) > 1:
                slice_size = original_module.intermediate_size // original_module.config.pretraining_tp
                gate_proj = torch.cat(
                    [
                        F.linear(x, gate_proj_slice)
                        for gate_proj_slice in original_module.gate_proj.weight.split(slice_size, dim=0)
                    ],
                    dim=-1,
                )
                up_proj = torch.cat(
                    [
                        F.linear(x, up_proj_slice)
                        for up_proj_slice in original_module.up_proj.weight.split(slice_size, dim=0)
                    ],
                    dim=-1,
                )
                hidden = original_module.act_fn(gate_proj) * up_proj
            else:
                hidden = original_module.act_fn(original_module.gate_proj(x)) * original_module.up_proj(x)
            self.down_smax = torch.maximum(
                self.down_smax,
                hidden.reshape(-1, hidden.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        return self._original_module(x)

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
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
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


class SplitQuantMiniCPMAttention(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        object.__setattr__(self, "_original_module", module)
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
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        self.is_causal = getattr(module, "is_causal", True)
        self.scale_depth = getattr(module, "scale_depth", None)
        self.rotary_emb = module.rotary_emb
        self.apply_rotary_pos_emb = module.forward.__globals__["apply_rotary_pos_emb"]
        self.repeat_kv = module.forward.__globals__["repeat_kv"]
        self.group_size = _resolve_split_group_size(args) if (args.w_bits < 16 or args.a_bits < 16) else -1

        self.q_proj = SplitQuantizedLinear(args, module.q_proj)
        self.k_proj = SplitQuantizedLinear(args, module.k_proj)
        self.v_proj = SplitQuantizedLinear(args, module.v_proj)
        self.o_proj = SplitQuantizedLinear(args, module.o_proj)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(bits=args.q_bits, sym=not args.q_asym, lac=args.lac, groupsize=-1)
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=args.k_bits, sym=not args.k_asym, lac=args.lac, groupsize=-1)
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=args.v_bits, sym=not args.v_asym, lac=args.lac, groupsize=-1)

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            stat_device = _weight_device(self.q_proj.linear)
            self.register_buffer(
                "ln_smax",
                torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0], device=stat_device) * 1e-5,
            )

    def add_fq_trans(self):
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            self.ln_trans = SVDSingleGroupTransMatrix(
                self.q_proj.linear.weight.shape[1],
                self.group_size,
                add_diag=self.args.add_diag,
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
        attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
        if not self._eval_mode:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            return self.o_proj(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        if self._ori_mode:
            if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
                self.ln_smax = torch.maximum(
                    self.ln_smax,
                    hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach(),
                )
            return self._original_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states, key_states = self.quant_kcache(query_states, key_states)
        value_states = self.quant_vcache(value_states)

        present_key_value = None
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            present_key_value = (key_states, value_states)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)
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
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self._project_attn_output(attn_output)

        attn_weights = None
        if use_cache and present_key_value is None:
            present_key_value = (key_states, value_states)
        return attn_output, attn_weights, present_key_value

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

    def rep_matrix_only(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


def apply_splitquant_to_minicpm(args, model):
    skip_initialization()
    layers = get_decoder_layers(model)
    for layer_index in tqdm(range(len(layers)), desc="Applying SplitQuant to model"):
        layers[layer_index].self_attn =SplitQuantMiniCPMAttention(
            args,
            layers[layer_index].self_attn,
        )
        layers[layer_index].mlp =SplitQuantMiniCPMMLP(
            args,
            layers[layer_index].mlp,
        )
    return model
