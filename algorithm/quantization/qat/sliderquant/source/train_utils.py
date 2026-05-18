from collections import OrderedDict

import torch

from quantize.utils import register_scales_and_zeros
from quantize.int_linear_lora import LoRALayer, LoRAQuantLinear
from quantize.int_linear import QuantLinear
from tqdm import tqdm
from copy import deepcopy

def get_lws_parameters(sub_layers, round_idx):
    normal_params = []
    normal_params_names = []

    scale_params = []
    scale_params_names = []

    for sub_layer_idx in range(len(sub_layers)):
        for n, p in sub_layers[sub_layer_idx].named_parameters():
            if not p.requires_grad:
                continue
            if "scale" in n:
                scale_params.append(p)
                scale_params_names.append(
                    "round{}_sub{}_{}".format(round_idx, sub_layer_idx, n)
                )
            else:
                normal_params.append(p)
                normal_params_names.append(
                    "round{}_sub{}_{}".format(round_idx, sub_layer_idx, n)
                )
    return normal_params, scale_params, normal_params_names, scale_params_names





@torch.no_grad()
def init_model(config,layers,args,DecoderLayer,model_attr,logger,dev,layer_id_list=None):
    is_llama = model_attr["is_llama"]
    pairs = model_attr["pairs"]
    slider_parameters = model_attr["slider_parameters"]
    dtype = model_attr["dtype"]

    bits = args.weight_quant_params["n_bits"]

    if layer_id_list is None:
        layer_id_list = list(range(len(layers)))

    for layer_id in layer_id_list:

        args.weight_quant_params["n_bits"] = bits

        use_lora = True if layer_id in args.lora_layer_list else False
        lora_iter_num = args.lora_iter_num_list[layer_id]
        lora_r = args.lora_r_list[layer_id]
        lora_quant = args.lora_quant
        lora_attr = dict(
            lora_iter_num=lora_iter_num,
            lora_quant=lora_quant,
            lora_r=lora_r,
            lora_only = bool(args.quant_mode == "lora_only"),
        )


        layer_config = config

        qlayer = DecoderLayer(
            config=layer_config, ori_layer=layers[layer_id],args=args,layer_id=layer_id,use_lora=use_lora,lora_attr=lora_attr
        )

        qlayer.let = args.let
        use_shift = True
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let and args.quant_mode_layer_list[layer_id] not in ["fp16","lora_only"]:
            # init channel-wise scaling and shift
            if args.gqa_scales == "mean":
                model_dim = qlayer.self_attn.q_proj.out_features
            else:
                model_dim = qlayer.self_attn.k_proj.out_features
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(model_dim,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear) or isinstance(module, LoRAQuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            scale = torch.ones(module.in_features,device=dev, dtype=dtype)
                            shift =  torch.zeros(module.in_features,device=dev, dtype=dtype)
                            logger.info(f"init slider_parameters from in layer_{layer_id} {pairs[key]} with ones!")
                            # import ipdb; ipdb.set_trace()
                            if pairs[key] == "out" and args.gqa_scales == "copy":  # modify scale  for GQA
                                scale = scale.view(-1,qlayer.self_attn.num_key_value_groups,qlayer.self_attn.head_dim).mean(dim=1).view(-1)
                                shift = shift.view(-1,qlayer.self_attn.num_key_value_groups,qlayer.self_attn.head_dim).mean(dim=1).view(-1)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))

        if args.resume and (layer_id < args.resume_layers_num or args.test_mode):
            try:
                layer_slider_parameters = slider_parameters[layer_id]
                if args.wo_lwc:
                    # import ipdb;ipdb.set_trace()
                    layer_slider_parameters = { key:layer_slider_parameters[key] for key in layer_slider_parameters.keys() if "bound_factor" not in key}
                qlayer.load_state_dict(layer_slider_parameters, strict=False)
                logger.info(f"load slider_parameters from {args.resume} in layer{layer_id} successfully!")
            except Exception as e:
                import ipdb;ipdb.set_trace()
                logger.info(f"load state occurs {e}, skip!")
        if args.test_mode is True:
            qlayer.float()
        layers[layer_id] = qlayer
    return layers

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}



@torch.no_grad()
def model_to_inference_mode(layers, args,dtype,dev="cpu",layer_id_list=None):
    if layer_id_list is None:
        layer_id_list = range(len(layers))
    for layer_id in tqdm(layer_id_list):
        qlayer = layers[layer_id].to(dev)
        # qlayer.to(dtype)
        qlayer.clear_temp_variable()

        qlayer.eval_mode = False if qlayer.quant_mode == "fp16" else True

            # uptate to quant mode
        if args.test_mode is True:
            if args.weight_merge is True and args.quant_mode_layer_list[layer_id] not in ["fp16","direct"]:
                qlayer.update_quant_mode("weight_merge",args=args)
            else:
                qlayer.update_quant_mode(args.quant_mode_layer_list[layer_id],args=args)
        else:
            qlayer.update_quant_mode(args.quant_mode_layer_list[layer_id],args=args)
        layers[layer_id] = qlayer.to("cpu")

    return layers





def obtain_teacher_output(sub_layers, inp, attention_mask, position_ids,position_embeddings,args,devs=None):
    # if args.low_memory is True:
    #     inp = inp.to(devs[0])
    for sub_layer_idx in range(len(sub_layers)):
        # import ipdb;ipdb.set_trace()
        if args.teach_model is None:  # 独立加载fp16模型
            sub_layers[sub_layer_idx].update_quant_mode("fp16",args=args)
        if sub_layer_idx == 0:
            if inp.device != devs[sub_layer_idx]:
                inp = inp.to(devs[sub_layer_idx])
                if attention_mask is not None:
                    attention_mask = attention_mask.to(devs[sub_layer_idx])
                if position_ids is not None:
                    position_ids = position_ids.to(devs[sub_layer_idx])
                    position_embeddings = tuple([position_embeddings[0].to(devs[sub_layer_idx]),position_embeddings[1].to(devs[sub_layer_idx])])
            out = sub_layers[sub_layer_idx](
                inp, attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings,
            )[0]
        else:
            if out.device != devs[sub_layer_idx]:
                out = out.to(devs[sub_layer_idx])
                if attention_mask is not None:
                    attention_mask = attention_mask.to(devs[sub_layer_idx])
                if position_ids is not None:
                    position_ids = position_ids.to(devs[sub_layer_idx])
                    position_embeddings = tuple([position_embeddings[0].to(devs[sub_layer_idx]),position_embeddings[1].to(devs[sub_layer_idx])])
            out = sub_layers[sub_layer_idx](
                out,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
    if args.low_memory is True:
        out = out.to("cpu")
    return out

class SubLayer(torch.nn.Module):
    def __init__(self,sub_layers,quant_mode_sub_layer_list,attention_mask,position_ids,position_embeddings,args):
        super().__init__()
        self.module = sub_layers
        self.quant_mode_sub_layer_list = quant_mode_sub_layer_list
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.position_embeddings = position_embeddings
        self.args = args

    def forward(self,x):
        dev = x.device
        for sub_layer_idx in range(len(self.module)):
            self.module[sub_layer_idx].update_quant_mode(self.quant_mode_sub_layer_list[sub_layer_idx],args=self.args)
            if self.attention_mask is not None:
                attention_mask = self.attention_mask.to(dev)[:len(x)]
            if self.position_ids is not None:
                position_ids = self.position_ids.to(dev)
                position_embeddings = tuple([self.position_embeddings[0].to(dev),self.position_embeddings[1].to(dev)])
            if sub_layer_idx == 0:
                out = self.module[sub_layer_idx](
                    x, attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings,
                )[0]
            else:
                out = self.module[sub_layer_idx](
                    out,
                    attention_mask=attention_mask,
                    position_ids=position_ids,position_embeddings=position_embeddings,
                )[0]
        return out




def obtain_studnet_output(sub_layers,quant_mode_sub_layer_list, inp, attention_mask, position_ids,position_embeddings, args,devs=None,return_gpu=False):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layers[sub_layer_idx].update_quant_mode(quant_mode_sub_layer_list[sub_layer_idx],args=args)
        if sub_layer_idx == 0:
            if inp.device != devs[sub_layer_idx]:
                inp = inp.to(devs[sub_layer_idx])
                if attention_mask is not None:
                    attention_mask = attention_mask.to(devs[sub_layer_idx])
                if position_ids is not None:
                    position_ids = position_ids.to(devs[sub_layer_idx])
                    position_embeddings = tuple([position_embeddings[0].to(devs[sub_layer_idx]),position_embeddings[1].to(devs[sub_layer_idx])])
            out = sub_layers[sub_layer_idx](
                inp, attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings,
            )[0]
        else:
            # if sub_layer_idx == 2:
            #     print("debuf!")
            if out.device != devs[sub_layer_idx]:
                out = out.to(devs[sub_layer_idx])
                if attention_mask is not None:
                    attention_mask = attention_mask.to(devs[sub_layer_idx])
                if position_ids is not None:
                    position_ids = position_ids.to(devs[sub_layer_idx])
                    position_embeddings = tuple([position_embeddings[0].to(devs[sub_layer_idx]),position_embeddings[1].to(devs[sub_layer_idx])])
            out = sub_layers[sub_layer_idx](
                out,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            # print("end debug")
    if args.low_memory is True and return_gpu is False:
        out = out.to("cpu")
    return out


def replace_qlayer(config, sub_layers, args, layer_id_list,DecoderLayer):
    start_layer_id = args.num_layer - args.sliding_layer if layer_id_list[0] >0 and args.test_mode is False else 0
    for sub_layer_idx in range(start_layer_id,len(sub_layers)):
        layer_id = layer_id_list[sub_layer_idx]
        use_lora = True if layer_id in args.lora_layer_list else False
        lora_iter_num = args.lora_iter_num_list[layer_id]
        lora_r = args.lora_r_list[layer_id]
        lora_quant = args.lora_quant
        lora_attr = dict(
            lora_iter_num=lora_iter_num,
            lora_quant=lora_quant,
            lora_r=lora_r,
        )
        sub_layers[sub_layer_idx] = DecoderLayer(
            config=config, ori_layer=sub_layers[sub_layer_idx],args=args,layer_id=layer_id,use_lora=use_lora,lora_attr=lora_attr
        )
    return sub_layers


def replace_ori_layer(layers, sub_layers, layer_id_list, args):
    for sub_layer_idx,layer_idx in enumerate(layer_id_list):
        layers[layer_idx] = sub_layers[sub_layer_idx]


def to_dev(sub_layers, devs):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(devs[sub_layer_idx])
    return sub_layers

def weight_to_cpu(sub_layers):
    for sub_layer_idx in range(len(sub_layers)):
        for name, module in sub_layers[sub_layer_idx].named_modules():
            if isinstance(module, (LoRAQuantLinear,QuantLinear)):
                module.weight = module.weight.cpu()
    return sub_layers


def to_float(sub_layers,dtype):
    with torch.no_grad():
        for sub_layer_idx in range(len(sub_layers)):
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(dtype)
    return sub_layers


def to_half(sub_layers,dtype):
    with torch.no_grad():
        for sub_layer_idx in range(len(sub_layers)):
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(dtype)
    return sub_layers


def load_qlayer_lora_state_dict(sub_layers, state_dict):
    for idx, sub_layer in enumerate(sub_layers):
        sub_layer.load_state_dict(state_dict[idx], strict=False)




def get_qlayer_lora_state_dict(sub_layers):
    return_dict = OrderedDict()
    for idx, sub_layer in enumerate(sub_layers):
        return_dict[idx] = sub_layer.qllm_lora_state_dict()
    return return_dict


def get_qlayer_cr_state_dict(sub_layers):
    return_dict = OrderedDict()
    for idx, sub_layer in enumerate(sub_layers):
        return_dict[idx] = sub_layer.qllm_sm_state_dict()
    return return_dict


def lora_merge(sub_layers, logger, round_idx, args):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for name, module in sub_layer.named_modules():
            if isinstance(module, (LoRAQuantLinear)):
                logger.info(
                    "Merging weight for layer {}: {}".format(
                        round_idx * args.num_layer + sub_layer_idx, name
                    )
                )
                weight_diff = (
                    module.lora_B.float() @ module.lora_A.float() * module.scaling
                )
                after_training_weight = (module.weight.float() + weight_diff).to(
                    module.weight.dtype
                )
                module.weight.data = after_training_weight.data
                module.merged = True
