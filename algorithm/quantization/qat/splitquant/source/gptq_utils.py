import math
import time
import tqdm
import torch
import torch.nn as nn
import logging
from algorithm.common.device import empty_cache
from algorithm.common.device import synchronize
from algorithm.common.modeling import move_tensors_to_device
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask as create_qwen3_5_causal_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask as create_qwen3_5_moe_causal_mask

from splitquant.backbone_utils import get_decoder_config
from splitquant.backbone_utils import get_decoder_layers
from splitquant.backbone_utils import move_front_modules
from splitquant.backbone_utils import unwrap_layer_output
from splitquant.utils import cleanup_memory
from splitquant.quant_utils import WeightQuantizer


def _build_calibration_forward_kwargs(model, sample):
    model_type = getattr(model.config, "model_type", None)
    if model_type in {"qwen2_5_vl", "qwen3_vl", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return {"attention_mask": torch.ones_like(sample, dtype=torch.long, device=sample.device)}
    return {}


def _build_qwen3_5_layer_kwargs(decoder_config, inputs, layer_kwargs, model_type):
    position_ids = layer_kwargs.get("position_ids")
    if position_ids is None:
        seq_len = inputs.shape[1]
        position_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)

    causal_mask_fn = (
        create_qwen3_5_moe_causal_mask
        if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}
        else create_qwen3_5_causal_mask
    )
    full_attention_kwargs = dict(layer_kwargs)
    full_attention_kwargs["attention_mask"] = causal_mask_fn(
        config=decoder_config,
        inputs_embeds=inputs[:1],
        attention_mask=torch.ones((1, inputs.shape[1]), dtype=torch.long, device=inputs.device),
        cache_position=full_attention_kwargs.get("cache_position"),
        past_key_values=full_attention_kwargs.get("past_key_values"),
        position_ids=position_ids,
    )

    linear_attention_kwargs = dict(layer_kwargs)
    linear_attention_kwargs["attention_mask"] = None
    return {
        "full_attention": full_attention_kwargs,
        "linear_attention": linear_attention_kwargs,
    }


def _select_layer_kwargs(layer, layer_kwargs, layer_kwargs_by_type):
    if layer_kwargs_by_type is None:
        return layer_kwargs
    return layer_kwargs_by_type.get(getattr(layer, "layer_type", None), layer_kwargs)


def _sequential_groups_for_layer(layer):
    if hasattr(layer.mlp, "shared_expert"):
        if getattr(layer, "layer_type", None) == "linear_attention":
            return [
                ["self_attn.in_proj_qkv.linear"],
                ["self_attn.in_proj_z.linear", "self_attn.in_proj_a.linear", "self_attn.in_proj_b.linear"],
                ["self_attn.out_proj.linear"],
                ["mlp.shared_expert.up_proj.linear", "mlp.shared_expert.gate_proj.linear"],
                ["mlp.shared_expert.down_proj.linear"],
                ["mlp.shared_expert_gate.linear"],
            ]
        return [
            ["self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.q_proj.linear"],
            ["self_attn.o_proj.linear"],
            ["mlp.shared_expert.up_proj.linear", "mlp.shared_expert.gate_proj.linear"],
            ["mlp.shared_expert.down_proj.linear"],
            ["mlp.shared_expert_gate.linear"],
        ]
    if isinstance(getattr(layer.mlp, "experts", None), nn.ModuleList):
        groups = [
            ["self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.q_proj.linear"],
            ["self_attn.o_proj.linear"],
        ]
        for expert_index in range(len(layer.mlp.experts)):
            groups.append(
                [
                    f"mlp.experts.{expert_index}.up_proj.linear",
                    f"mlp.experts.{expert_index}.gate_proj.linear",
                ]
            )
            groups.append([f"mlp.experts.{expert_index}.down_proj.linear"])
        return groups
    if getattr(layer, "layer_type", None) == "linear_attention":
        return [
            ["self_attn.in_proj_qkv.linear"],
            ["self_attn.in_proj_z.linear", "self_attn.in_proj_a.linear", "self_attn.in_proj_b.linear"],
            ["self_attn.out_proj.linear"],
            ["mlp.up_proj.linear", "mlp.gate_proj.linear"],
            ["mlp.down_proj.linear"],
        ]
    return [
        ["self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.q_proj.linear"],
        ["self_attn.o_proj.linear"],
        ["mlp.up_proj.linear", "mlp.gate_proj.linear"],
        ["mlp.down_proj.linear"],
    ]


def _quantizable_names_for_layer(layer):
    if hasattr(layer.mlp, "shared_expert"):
        if getattr(layer, "layer_type", None) == "linear_attention":
            return {
                "self_attn.in_proj_qkv.linear",
                "self_attn.in_proj_z.linear",
                "self_attn.in_proj_a.linear",
                "self_attn.in_proj_b.linear",
                "self_attn.out_proj.linear",
                "mlp.shared_expert.up_proj.linear",
                "mlp.shared_expert.gate_proj.linear",
                "mlp.shared_expert.down_proj.linear",
                "mlp.shared_expert_gate.linear",
            }
        return {
            "self_attn.q_proj.linear",
            "self_attn.k_proj.linear",
            "self_attn.v_proj.linear",
            "self_attn.o_proj.linear",
            "mlp.shared_expert.up_proj.linear",
            "mlp.shared_expert.gate_proj.linear",
            "mlp.shared_expert.down_proj.linear",
            "mlp.shared_expert_gate.linear",
        }
    if isinstance(getattr(layer.mlp, "experts", None), nn.ModuleList):
        names = {
            "self_attn.q_proj.linear",
            "self_attn.k_proj.linear",
            "self_attn.v_proj.linear",
            "self_attn.o_proj.linear",
        }
        for expert_index in range(len(layer.mlp.experts)):
            names.update(
                {
                    f"mlp.experts.{expert_index}.up_proj.linear",
                    f"mlp.experts.{expert_index}.gate_proj.linear",
                    f"mlp.experts.{expert_index}.down_proj.linear",
                }
            )
        return names
    if getattr(layer, "layer_type", None) == "linear_attention":
        return {
            "self_attn.in_proj_qkv.linear",
            "self_attn.in_proj_z.linear",
            "self_attn.in_proj_a.linear",
            "self_attn.in_proj_b.linear",
            "self_attn.out_proj.linear",
            "mlp.up_proj.linear",
            "mlp.gate_proj.linear",
            "mlp.down_proj.linear",
        }
    return {
        "self_attn.q_proj.linear",
        "self_attn.k_proj.linear",
        "self_attn.v_proj.linear",
        "self_attn.o_proj.linear",
        "mlp.up_proj.linear",
        "mlp.gate_proj.linear",
        "mlp.down_proj.linear",
    }


def find_qlayers(module, layers=[torch.nn.Linear, ], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        if dead.any():
            W[:, dead.to(W.device)] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=H.device)
        Hinv = None
        current_damp = damp
        last_error = None
        for _ in range(6):
            H_work = H.clone()
            H_work[diag, diag] += current_damp
            try:
                H_work = torch.linalg.cholesky(H_work)
                H_work = torch.cholesky_inverse(H_work)
                H_work = torch.linalg.cholesky(H_work, upper=True)
                Hinv = H_work
                break
            except RuntimeError as error:
                last_error = error
                if "cholesky" not in str(error).lower():
                    raise
                current_damp = current_damp * 10 if current_damp > 0 else torch.tensor(1e-4, device=H.device)
        if Hinv is None:
            raise last_error

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2].to(self.dev)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:].to(self.dev))

        if self.dev.type in {"cuda", "npu"}:
            synchronize(self.dev)

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        empty_cache(self.dev)
        cleanup_memory(verbose=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    decoder_config = get_decoder_config(model)
    use_cache = decoder_config.use_cache
    decoder_config.use_cache = False
    layers = get_decoder_layers(model)

    # device_map 模式下不手动移动 front modules 和 layers[0]

    dtype = next(iter(model.parameters())).dtype
    layer0_device = next(layers[0].parameters()).device
    inps = torch.zeros(
        (args.nsamples, model.seqlen, decoder_config.hidden_size), dtype=dtype, device=layer0_device
    )
    cache = {'i': 0, 'layer_kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = dict(kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            sample = batch[0].to(layer0_device)
            model(sample, **_build_calibration_forward_kwargs(model, sample))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # device_map 模式下不手动移动到 cpu
    empty_cache(dev)

    outs = torch.zeros_like(inps)
    layer_kwargs = dict(cache['layer_kwargs'])
    layer_kwargs_by_type = None
    model_type = getattr(model.config, "model_type", None)
    if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"}:
        layer_kwargs_by_type = _build_qwen3_5_layer_kwargs(decoder_config, inps, layer_kwargs, model_type)

    quantizers = {}
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        # device_map 模式下不手动移动 layer
        layer = layers[i]
        # 将所有数据移到当前层设备
        layer_dev = next(layer.parameters()).device
        inps = inps.to(layer_dev)
        outs = outs.to(layer_dev)
        layer_kwargs = move_tensors_to_device(layer_kwargs, layer_dev)
        if layer_kwargs_by_type is not None:
            layer_kwargs_by_type = {
                lt: move_tensors_to_device(kw, layer_dev)
                for lt, kw in layer_kwargs_by_type.items()
            }
        active_layer_kwargs = _select_layer_kwargs(layer, layer_kwargs, layer_kwargs_by_type)
        sequential = _sequential_groups_for_layer(layer)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.gptq_mse
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = unwrap_layer_output(layer(inps[j].unsqueeze(0), **active_layer_kwargs))
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = unwrap_layer_output(layer(inps[j].unsqueeze(0), **active_layer_kwargs))

        # device_map 模式下不手动移动到 cpu
        del gptq

        inps, outs = outs, inps

    decoder_config.use_cache = use_cache
    cleanup_memory(verbose=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    layers = get_decoder_layers(model)
    empty_cache(dev)

    quantizers = {}
    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        # device_map 模式下不手动移动 layer
        layer = layers[i]
        quantizable_names = _quantizable_names_for_layer(layer)

        subset = find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            if name not in quantizable_names:
                continue
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.gptq_mse
            )
            W = subset[name].weight.data
            w_dtype = W.dtype
            orig_shape = W.shape
            if args.w_groupsize != -1:
                assert orig_shape[1] % args.w_groupsize == 0
                W = W.reshape(-1, args.w_groupsize)
            quantizer.find_params(W)
            quantized_weight = quantizer.quantize(W).to(w_dtype)
            if args.w_groupsize != -1:
                quantized_weight = quantized_weight.reshape(orig_shape)
            subset[name].weight.data = quantized_weight
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        # device_map 模式下不手动移动到 cpu
            
    cleanup_memory(verbose=True)
    return quantizers
# Synchronize quantization device_map support for multi-GPU execution.
