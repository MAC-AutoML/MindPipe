import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
import copy
import contextlib
import math
import os
from tqdm import tqdm
from train_utils import to_float,to_half
from quantize.utils import get_slider_parameters, get_lwc_parameters, slider_state_dict

from train_utils import to_dev,obtain_teacher_output,obtain_studnet_output,replace_ori_layer,init_model,model_to_inference_mode,SubLayer
import time
from transformers import get_scheduler
from quantize.utils import cleanup_memory
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from quantize.utils import evaluate


def _build_dense_causal_mask(seqlen, dtype, device):
    mask = torch.full((seqlen, seqlen), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seqlen, seqlen)


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 使用 torchrun 自动设置的环境变量
    )

def cleanup_ddp():
    dist.destroy_process_group()


class Quant_dataset(Dataset):
    def __init__(self,aug_quant_inps=None,aug_fp_inps=None, aug_quant_targets=None,aug_fp_targets=None,samples_num=512,windows_num=1,args=None):
        """
        In i-th round
        quant_inps: the output from (i-1)th quant model using quant_inps.
        aug_fp_inps: the output from (i-1)th fp16 model using fp_inps.
        fp_target: the output from i-th fp16 model using fp_inps.
        aug_quant_target: the output from i-th fp16 model using quant_inps.

        fp_inps --------->  [ fp16 model] ------------> fp_target
        quant_inps --------->  [ fp16 model] ------------> quant_target

        quant_inps ----->   [quant model] -------------> out1 <-> [fp_target,quant_target]
        fp_inps ----->   [quant model] -------------> out2 <->fp_target
        """
        # self.quant_inps = quant_inps
        self.samples_num = samples_num
        self.windows_num = windows_num
        assert self.windows_num == aug_quant_inps.shape[1]
        assert self.samples_num == len(aug_quant_inps)

        self.aug_quant_inps = aug_quant_inps
        self.aug_fp_targets = aug_fp_targets

        if aug_fp_inps is not None:
            self.aug_fp_inps = aug_fp_inps
        else:
            self.aug_fp_inps = torch.ones(self.samples_num,self.windows_num)

        if aug_quant_targets is not None:
            self.aug_quant_targets = aug_quant_targets
        else:
            self.aug_quant_targets = torch.ones(self.samples_num,self.windows_num)

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        return self.aug_quant_inps[idx],self.aug_fp_inps[idx],self.aug_quant_targets[idx],self.aug_fp_targets[idx]



MB = 1024.0 * 1024.0


def _autocast_context(args, dtype):
    if getattr(args, "deactive_amp", False):
        return contextlib.nullcontext()
    return torch.cuda.amp.autocast(dtype=dtype)


def _maybe_slice_batch_mask(attention_mask, batch_size):
    if attention_mask is None:
        return None
    return attention_mask[:batch_size]


def train_one_round(r,epochs,sub_layers,layer_id_list,qdataset,cur_epochs,optimizer,lr_scheduler,attention_mask_batch,position_ids,position_embeddings,devs,args,logger,max_train_steps,init_quant_rate,fp16_type,global_start_time,total_epochs,loss_func,acts_round_idx):
    if args.use_ddp is True:
        rank = dist.get_rank()
        sub_layers = DDP(sub_layers,device_ids=[rank])
        sampler = DistributedSampler(qdataset,shuffle=True)
        shuffle = None
    else:
        rank = 0
        sampler = None
        shuffle = True
    qdataloader = DataLoader(qdataset,batch_size=args.batch_size,shuffle=shuffle,num_workers=0,pin_memory=True,sampler=sampler)



    if args.loss_type == "mean":
        if args.use_base_loss == "none":
            base_loss_num = 0
        elif args.use_base_loss == "last":
            base_loss_num = 1
        elif args.use_base_loss == "all":
            base_loss_num =  args.last_round_inp_num
        else:
            raise NotImplementedError()
        Accumulated_loss_num = (int(args.use_fp_inp_loss) + int(args.use_quant_tar_loss) ) *  args.last_round_inp_num + base_loss_num
    elif args.loss_type == "add":
        Accumulated_loss_num = 1
    else:
        raise NotImplementedError("noly support mean and add!")

    logger.info(f"Accumulated_loss_num is {Accumulated_loss_num}")

    trainable_layers = sub_layers.module.module if args.use_ddp else sub_layers
    trainable_layers.train() # 确保开启训练模型

    for e in range(epochs):
        if args.use_ddp is True:
            sampler.set_epoch(e)
        start_time = time.time()
        epoch_losses = []
        epoch_norms = []
        epoch_losses_fp = []
        epoch_losses_quant = []
        epoch_losses_base = []
        # import ipdb;ipdb.set_trace()


        for quant_input_list,fp_input_list,quant_tar_list,fp_tar_list in qdataloader:
            batch_loss = [0.0 for _ in range(quant_input_list.shape[1])]
            batch_base_loss = [0.0 for _ in range(quant_input_list.shape[1])]
            batch_fp_loss = [0.0 for _ in range(quant_input_list.shape[1])]
            batch_quant_loss = [0.0 for _ in range(quant_input_list.shape[1])]

            optimizer.zero_grad()
            # optimizer.zero_grad(set_to_none=True) # MOE节省显存

            for w_idx in range(quant_input_list.shape[1]):
                quant_input,fp_input,quant_tar,fp_tar = quant_input_list[:,w_idx],fp_input_list[:,w_idx],quant_tar_list[:,w_idx],fp_tar_list[:,w_idx]
                inp_list = [quant_input] if args.use_fp_inp_loss is False else [quant_input,fp_input]
                for inp_idx,inp in enumerate(inp_list):
                    if inp_idx == 0 and (args.use_base_loss == "none" or args.use_base_loss == "last" and w_idx != quant_input_list.shape[1] -1) and args.use_quant_tar_loss is False:
                        continue
                    loss_list = []

                    train_context = _autocast_context(args, torch.bfloat16)

                    with train_context:
                        if args.use_ddp:
                            out = sub_layers(inp)
                        else:
                            out = obtain_studnet_output(
                                sub_layers,[args.quant_mode_layer_list[i] for i in layer_id_list],
                                inp, _maybe_slice_batch_mask(attention_mask_batch, len(inp)), position_ids,position_embeddings, args,
                                devs=devs,return_gpu=True
                            )
                        if inp_idx == 0:
                            # out is quant_out
                            if args.use_base_loss == "all" or  (w_idx == quant_input_list.shape[1] -1 and args.use_base_loss == "last"):
                                loss_base = loss_func(out, fp_tar.to(out.device))
                                loss_list.append(loss_base)
                                batch_base_loss[w_idx] += loss_base.item()
                            else:
                                loss_base = torch.tensor(0.0)

                            if args.use_quant_tar_loss is True:
                                loss_quant = loss_func(out, quant_tar.to(out.device))
                                loss_list.append(loss_quant)
                                batch_quant_loss[w_idx] += loss_quant.item()
                            else:
                                loss_quant = torch.tensor(0.0)

                        else:
                            # out is fp_out
                            if args.use_fp_inp_loss is True:
                                loss_fp = loss_func(out, fp_tar.to(out.device))
                                loss_list.append(loss_fp)
                                batch_fp_loss[w_idx] += loss_fp.item()
                            else:
                                loss_fp = torch.tensor(0.0)
                        loss = sum(loss_list) / Accumulated_loss_num

                    loss.backward()
                    batch_loss[w_idx] += loss.detach().item()


                if args.debug is True and not math.isfinite(loss.detach().item()):
                    logger.info("Loss is NAN, stopping training")
                    import ipdb;ipdb.set_trace()
                assert math.isfinite(loss.item()),"Loss is NAN, stopping training!"
            if args.grad_clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_layers.parameters(), max_norm=args.grad_clip)
                # logger.info(f"Gradient norm: {total_norm:.4f} Max norm: {args.grad_clip}")

            optimizer.step()
            epoch_norms.append(0.0)
            epoch_losses.append(batch_loss)
            epoch_losses_base.append(batch_base_loss)
            epoch_losses_fp.append(batch_fp_loss)
            epoch_losses_quant.append(batch_quant_loss)

            if args.use_lr_scheduler is True:
                lr_scheduler.step()

            current_memory = torch.cuda.memory_allocated() / MB
            max_memory = torch.cuda.max_memory_allocated() / MB


        cur_epochs += 1
        epoch_mean_loss = torch.tensor(epoch_losses).mean(dim=0)
        # epoch_mean_norms = sum(epoch_norms) / len(epoch_norms)
        epoch_mean_loss_base = torch.tensor(epoch_losses_base).mean(dim=0)
        epoch_mean_loss_quant = torch.tensor(epoch_losses_quant).mean(dim=0)
        epoch_mean_loss_fp = torch.tensor(epoch_losses_fp).mean(dim=0)
        loss_str = ""
        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss r{acts_round_idx[r_idx]}:{epoch_mean_loss[r_idx]} "

        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss base r{acts_round_idx[r_idx]}:{epoch_mean_loss_base[r_idx]} "

        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss quant r{acts_round_idx[r_idx]}:{epoch_mean_loss_quant[r_idx]} "

        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss fp r{acts_round_idx[r_idx]}:{epoch_mean_loss_fp[r_idx]} "

        if rank == 0:
            logger.info(
                f"Round {r} epoch {e} {loss_str} max memory_allocated: {max_memory}MB current memory_allocated: {current_memory}MB epoch_time: {time.time() - start_time:.2f}s use_time:{(time.time() - global_start_time)/3600:.2f}h ETA: {(time.time() - global_start_time) / cur_epochs * (total_epochs-cur_epochs) / 3600:.2f}h"
            )


    return sub_layers,cur_epochs


def sliderquant(
    lm,
    args,
    dataloader,
    logger=None,
    teach_lm=None,
):
    logger.info("Starting ...")

    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False

    if args.use_ddp:
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0


    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        total_layers = len(layers)
        max_layers = getattr(args, "max_layers", None)
        if max_layers is not None:
            max_layers = max(1, min(int(max_layers), total_layers))
            logger.info(f"SliderQuant debug max_layers={max_layers}; only layers 0-{max_layers - 1} will be quantized.")
        active_layer_count = max_layers or total_layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
        }
        if args.use_down_scale is True:
            pairs["down_proj"] = "fc2"
        layer_name_prefix = "model.layers"
    else:
        raise NotImplementedError("Only llama/qwen/vicuna are kept in this open-source snapshot.")



    # import ipdb;ipdb.set_trace()
    layers[0] = layers[0].to(dev)
    fp32_type = torch.float
    fp16_type =  torch.bfloat16 if args.use_bfloat16 is True else torch.float16
    act_dtype =  fp16_type if args.fp16_act is True else fp32_type
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=act_dtype, device="cpu"
    )
    # import ipdb;ipdb.set_trace()
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise NotImplementedError("Only llama/qwen/vicuna are kept in this open-source snapshot.")


    # import ipdb;ipdb.set_trace()
    inps = inps.to("cpu")
    # same input of first layer for fp model and quant model

    cleanup_memory(logger=logger)

    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask[:,:,:,:args.seqlen]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1)
        infer_attention_mask = attention_mask.repeat(args.inference_batch_size,1,1,1)
    else:
        logger.info(
            "No attention mask caught from the first layer."
        )
        mask_dtype = inps.dtype if torch.is_floating_point(inps) else fp16_type
        attention_mask = _build_dense_causal_mask(args.seqlen, mask_dtype, "cpu")
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1)
        infer_attention_mask = attention_mask.repeat(args.inference_batch_size,1,1,1)
        logger.info("Built dense causal attention mask for SliderQuant calibration.")


    # import ipdb;ipdb.set_trace()
    if is_llama:
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]
    else:
        position_ids = None
        position_embeddings = None


    if  args.quant_mode in ["fp16"]:
        args.resume = None
    if args.resume:
        slider_parameters = torch.load(args.resume)
    else:
        slider_parameters = {}

    if args.train_resume is not None and args.test_mode is False:
        slider_parameters = torch.load(args.train_resume)
        args.resume = args.train_resume


    args.quant_layer_list = [int(layer_id) for layer_id in range(active_layer_count)]
    logger.info(f"these layer will quant:{args.quant_layer_list}")

    if args.use_lora is True:
        args.lora_layer_list = args.quant_layer_list  # only when quant use lora
    else:
        args.lora_layer_list = []
    logger.info(f"these layer will refine with lora:{args.lora_layer_list}")


    args.lora_iter_num_list = {layer_id:1 for layer_id in range(active_layer_count)}
    logger.info(f"each layer will refine with lora num iter:{args.lora_iter_num_list}")


    args.lora_r_list = {layer_id:args.lora_rank for layer_id in range(active_layer_count)}
    logger.info(f"each layer lora's r:{args.lora_r_list}")


    args.quant_mode_layer_list = { layer_id:(args.quant_mode if layer_id in args.quant_layer_list else "fp16") for layer_id in range(len(layers)) }



    logger.info(f"each layer quant mode:{args.quant_mode_layer_list}")

    if args.sliding_layer is None:
        args.sliding_layer = args.num_layer
    logger.info(f"sliding_layer:{args.sliding_layer}")


    init_quant_rate = args.quant_rate

    if args.quant_rate_list is None:
        args.quant_rate_list = np.linspace(0, 1, args.quant_step+1).tolist()[1:]

    logger.info(f"quant_step:{args.quant_step} quant_rate_list:{args.quant_rate_list} lora_quant:{args.lora_quant}")

    if args.test_mode is True:
        args.quant_rate = 1.0

    if  (args.resume is not None or args.train_resume is not None) and  args.resume_layers_num is None:
         args.resume_layers_num = active_layer_count


    # 模型初始化
    model_attr = dict(
        is_llama=is_llama,
        pairs=pairs,
        layer_name_prefix=layer_name_prefix,
        slider_parameters=slider_parameters,
        dtype=fp32_type,
    )

    init_model(config=lm.model.config,layers=layers,args=args,DecoderLayer=DecoderLayer,model_attr=model_attr,logger=logger,dev="cpu",layer_id_list=args.quant_layer_list)
    logger.info("Model Initialized")

    num_update_steps_per_epoch = math.ceil(args.nsamples / args.batch_size)
    global_start_time = time.time()


    cur_epochs = 0 # 已经训练的epochs

    if args.fill_window_size is not None:
        args.fill_start_window_size = args.fill_window_size
        args.fill_end_window_size = args.fill_window_size


    if args.layer_windows_scheduler is not None:
        layer_windows_scheduler = []
        for window_str in args.layer_windows_scheduler.split(","):
            layer_windows_scheduler.append([int(s) for s in window_str.split("-")])
        num_round = len(layer_windows_scheduler)
        # import ipdb;ipdb.set_trace()
        assert layer_windows_scheduler[-1][-1] == active_layer_count - 1

    elif args.fill_window_size is not None:
        total_num_layers = active_layer_count
        if args.fill_start_window_size is not None:
            start_layer_windows_scheduler = [list(range(i+1))  for i in range(args.fill_start_window_size)]
            start_len = (args.fill_start_window_size -  args.sliding_layer)
        else:
            start_layer_windows_scheduler = []
            start_len = 0

        if args.fill_end_window_size is not None:
            end_start_layer_windows_scheduler = [list(range(total_num_layers-args.fill_end_window_size +i,total_num_layers))  for i in range(args.fill_end_window_size)]
            end_len = (args.fill_end_window_size -  args.sliding_layer)
        else:
            end_start_layer_windows_scheduler = []
            end_len = 0
        mid_len = total_num_layers - start_len - end_len

        mid_round = math.ceil((mid_len - args.num_layer) / args.sliding_layer) + 1
        mid_layer_windows_scheduler = [
                [i for i in range(r * args.sliding_layer + start_len , min(r * args.sliding_layer + args.num_layer + start_len ,active_layer_count))]
            for r in range(mid_round)
        ]
        layer_windows_scheduler = start_layer_windows_scheduler + mid_layer_windows_scheduler + end_start_layer_windows_scheduler
        num_round = len(layer_windows_scheduler)
    else:
        num_round = math.ceil((active_layer_count - args.num_layer) / args.sliding_layer) + 1
        layer_windows_scheduler = [
            [i for i in range(r * args.sliding_layer, min(r * args.sliding_layer + args.num_layer,active_layer_count))]
            for r in range(num_round)
        ]

    logger.info(f"layer_windows_scheduler is  {layer_windows_scheduler}")

    if teach_lm is not None:
        teach_model = teach_lm.model
        teach_model.config.use_cache = False
        teach_layers = teach_model.model.layers
        teach_layers = teach_layers.to(dev)
        logger.info(f"Teacher model Initialized from {args.teach_model}")
    else:
        teach_layers = layers


    cleanup_memory(logger=logger)
    assert args.quant_step == len(args.quant_rate_list)


    if args.circular_aug:
        assert len(args.quant_rate_list) > 1

    if args.littlt_bs_round is not None:
        littlt_bs_round = [int(i) for i in args.littlt_bs_round.split(",")]
        littlt_bs_round = [i if i >=0 else num_round+i for i in littlt_bs_round]
    else:
        littlt_bs_round = []
    global_batch_size = args.batch_size
    for step,quant_rate in enumerate(args.quant_rate_list):
        if args.circular_aug is True and  step+1 == len(args.quant_rate_list):
            windows_quant_inps[:,-1] =    copy.deepcopy(inps)
            windows_fp_inps[:,-1]    =    copy.deepcopy(inps)
        else:
            windows_quant_inps =    copy.deepcopy(inps).unsqueeze(1).repeat(1,args.last_round_inp_num,1, 1) # if None, not need cache. if True, need but had not been cached
            windows_fp_inps    =    copy.deepcopy(windows_quant_inps) # if None, not need cache. if True, need but had not been cached

        if step+1 == args.quant_step and args.debug is False:
            inps = None
            print("delete inps!")
        cleanup_memory(logger=logger)

        if args.use_quant_tar_loss:
            windows_quant_targets = copy.deepcopy(windows_quant_inps) # if None, not need cache. if True, need but had not been cached
        else:
            windows_quant_targets = None


        windows_fp_targets =  copy.deepcopy(windows_fp_inps)

        args.quant_rate = quant_rate
        if args.low_memory is False:
            windows_quant_inps = windows_quant_inps.to(dev)
            windows_fp_inps = windows_fp_inps.to(dev)


        cleanup_memory(logger=logger)

        for r in range(num_round):

            if r in littlt_bs_round:
                args.batch_size = 1
            else:
                args.batch_size = global_batch_size
            if args.test_mode is True:
                break
            layer_id_list = layer_windows_scheduler[r]
            logger.info(f"=== Step: {step+1}/{args.quant_step}   Round: {r+1}/{num_round} ===")
            logger.info(
                f"=== Start quantize layer{layer_id_list[0]}-layer{layer_id_list[-1]} ==="
            )

            if args.loss_function == "mse":
                loss_func = torch.nn.MSELoss()
            elif args.loss_function == "huber":
                delta = 0.1 + r/num_round * args.huber_loss_max
                loss_func = torch.nn.HuberLoss(delta=delta)
            else:
                raise NotImplementedError("only support mse and huber loss function")

            # del finished layers
            if args.low_cpu_memory is True and step+1 == len(args.quant_rate_list):
                for l_idx in range(layer_id_list[0]):
                    if lm.model.model.layers[l_idx] is not None:
                        lm.model.model.layers[l_idx] = torch.nn.Identity()
                # import ipdb;ipdb.set_trace()
                logger.info(f"del layer 0-{layer_id_list[0]-1}")


            sub_layers = layers[layer_id_list[0]:layer_id_list[-1]+1]
            teach_sub_layers = teach_layers[layer_id_list[0]:layer_id_list[-1]+1]


            cleanup_memory(logger=logger)
            logger.info(f"layer_id_list: {layer_id_list}")

            sub_layers = to_float(sub_layers,dtype=fp32_type)
            sub_layers = to_dev(sub_layers, [dev] * len(sub_layers))  #single gpu

            teach_sub_layers = to_float(teach_sub_layers,dtype=fp32_type)
            teach_sub_layers = to_dev(teach_sub_layers, [dev] * len(teach_sub_layers))  #single gpu


            acts_round_idx = list(range(r+1-args.last_round_inp_num,r+1))
            logger.info(f"act_round_idx is {acts_round_idx}")


            # import ipdb;ipdb.set_trace()
            if  r >= args.start_round:
                with torch.no_grad():
                    with _autocast_context(args, fp16_type):
                        for r_idx in range(args.last_round_inp_num):
                            # get quant_target
                            if  args.use_quant_tar_loss:
                                for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                                    bs_local = min(args.inference_batch_size,args.nsamples-j)
                                    windows_quant_targets[j:j+bs_local,r_idx] = obtain_teacher_output(
                                        teach_sub_layers,
                                        windows_quant_inps[j:j+bs_local,r_idx],
                                        _maybe_slice_batch_mask(infer_attention_mask, bs_local),
                                        position_ids,
                                        position_embeddings=position_embeddings,
                                        args=args,
                                        devs=[dev] * len(teach_sub_layers),
                                    )
                                logger.info(f"finish to obtain quant_target round {acts_round_idx[r_idx]} of full-precision model!")
                            # get fp_target
                            for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                                bs_local = min(args.inference_batch_size,args.nsamples-j)
                                windows_fp_targets[j:j+bs_local,r_idx] = obtain_teacher_output(
                                    teach_sub_layers,
                                    windows_fp_inps[j:j+bs_local,r_idx],
                                    _maybe_slice_batch_mask(infer_attention_mask, bs_local),
                                    position_ids,
                                    position_embeddings=position_embeddings,
                                    args=args,
                                    devs=[dev] * len(teach_sub_layers),
                                )
                            logger.info(f"finish to obtain fp_target round {acts_round_idx[r_idx]} of full-precision model!")

            cleanup_memory(logger=logger)


            epochs = args.epochs // args.quant_step
            total_epochs = args.epochs*num_round


            if args.layers_assigned_gpu is not None:
                devs = [torch.device(f"cuda:{gpu}") for gpu in args.layers_assigned_gpu.split(",")]
                assert len(devs) == len(sub_layers), "layers_assigned_gpu number is not equal to layer number!"
                sub_layers = to_dev(sub_layers, devs)  #mutil-gpu
            else:
                devs = [dev] * len(sub_layers)

            max_train_steps = epochs * num_update_steps_per_epoch

            lr_factor = args.batch_size
            logger.info(f"auto lr scale is {args.auto_lr_scale} lora_lr is {args.lora_lr*lr_factor} scale_lr is {args.scale_lr*lr_factor} lwc_lr is {args.lwc_lr*lr_factor}")


            params = []


            if args.use_lora is True and (r not in littlt_bs_round or args.quant_mode == "lora_only"):
                params.append({"params":get_slider_parameters(sub_layers, ["lora_"]),"lr":args.lora_lr*lr_factor,"weight_decay":0.0})
            if args.scale_lr > 0:
                params.append({"params":get_slider_parameters(sub_layers, ["scale"]),"lr":args.scale_lr*lr_factor,"weight_decay":0.0})
            if args.lwc_lr > 0:
                params.append({"params":get_lwc_parameters(sub_layers),"lr":args.lwc_lr*lr_factor,"weight_decay":0.0})



            optimizer = torch.optim.AdamW(params)

            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=max_train_steps*args.warmup_ratio,
                num_training_steps=max_train_steps,
            )


            # import ipdb;ipdb.set_trace()
            if args.use_ddp and r >= args.start_round:
                sub_layers = SubLayer(sub_layers,quant_mode_sub_layer_list=[args.quant_mode_layer_list[i] for i in layer_id_list],
                            attention_mask=attention_mask_batch,position_ids=position_ids,position_embeddings=position_embeddings,args=args)


            cleanup_memory(logger=logger)


            if args.use_ddp and r >= args.start_round:
                sub_layers = sub_layers.cuda()

            if args.use_ddp:
                dist.barrier()

            # train loop
            if r < args.start_round:
                logger.info(f"round {r} skip because resume from disk!")
                qdataset = None
            else:
                qdataset = Quant_dataset(aug_quant_inps=windows_quant_inps,aug_fp_inps=windows_fp_inps,aug_quant_targets=windows_quant_targets,aug_fp_targets=windows_fp_targets,samples_num=args.nsamples,windows_num=args.last_round_inp_num,args=args)
                sub_layers,cur_epochs = train_one_round(r=r,epochs=epochs,sub_layers=sub_layers,layer_id_list=layer_id_list,attention_mask_batch=attention_mask_batch,cur_epochs=cur_epochs,
                                position_ids=position_ids,position_embeddings=position_embeddings,devs=devs,args=args,logger=logger,max_train_steps=max_train_steps,optimizer=optimizer,lr_scheduler=lr_scheduler,qdataset=qdataset,
                                init_quant_rate=init_quant_rate,fp16_type=fp16_type,global_start_time=global_start_time,total_epochs=total_epochs,loss_func=loss_func,acts_round_idx=acts_round_idx)

            if args.use_ddp and r >= args.start_round:
                sub_layers = sub_layers.module.module
                sub_layers = sub_layers.to("cpu")




            if args.use_ddp:
                dist.barrier()

            del optimizer,qdataset,lr_scheduler
            for r_idx in range(args.last_round_inp_num-1):
                windows_quant_inps[:,r_idx] = windows_quant_inps[:,r_idx+1]
                windows_fp_inps[:,r_idx] = windows_fp_inps[:,r_idx+1]



            cleanup_memory(logger=logger)


            sliding_layer = layer_windows_scheduler[min(r+1,num_round-1)][0] - layer_windows_scheduler[r][0]
            if r < num_round-1 and sliding_layer>0:
                sub_layers = to_dev(sub_layers, [dev] * len(sub_layers))  #single gpu
                with torch.no_grad():
                    with _autocast_context(args, fp16_type):
                        # get next fp_16
                        for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                            bs_local = min(args.inference_batch_size,args.nsamples-j)
                            windows_fp_inps[j:j+bs_local,-1] = obtain_teacher_output(
                                teach_sub_layers[:sliding_layer],
                                windows_fp_inps[j:j+bs_local,-1],
                                _maybe_slice_batch_mask(infer_attention_mask, bs_local),
                                position_ids,
                                position_embeddings=position_embeddings,
                                args=args,
                                devs=[dev] * len(teach_sub_layers),
                            )
                        logger.info(f"finish to obtain round {r+1} inps of full-precision model!")

                        for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                            bs_local = min(args.inference_batch_size,args.nsamples-j)
                            windows_quant_inps[j:j+bs_local,-1] = obtain_studnet_output(
                                sub_layers[:sliding_layer],
                                [args.quant_mode_layer_list[i] for i in layer_id_list[:sliding_layer]],
                                windows_quant_inps[j:j+bs_local,-1],
                                _maybe_slice_batch_mask(infer_attention_mask, bs_local),
                                position_ids,
                                position_embeddings=position_embeddings,
                                args=args,
                                devs=[dev] * len(sub_layers),
                            )
                        logger.info(f"finish to obtain round {r+1} inps of quant model!")

            with torch.no_grad():
                for idx,i in enumerate(layer_id_list):
                    qlayer = sub_layers[idx]
                    qlayer.clear_temp_variable()
                    if epochs>0:
                        sub_layers[idx] = qlayer.to("cpu")
                        slider_parameters[i] = slider_state_dict(qlayer)
                        if args.use_ddp:
                            if  args.use_ddp and dist.get_rank() == 0:
                                torch.save(slider_parameters, os.path.join(args.output_dir, f"slider_parameters.pth"))
                                logger.info(f"save slider_parameters in layer{i} successfully!")
                        else:
                            torch.save(slider_parameters, os.path.join(args.output_dir, f"slider_parameters.pth"))
                            logger.info(f"save slider_parameters in layer{i} successfully!")
                    else:
                        sub_layers[idx] = qlayer.to("cpu")

                del qlayer
                sub_layers = to_half(sub_layers,dtype=fp16_type)
                sub_layers = to_dev(sub_layers, ["cpu"] * len(sub_layers))  #single gpu
                teach_sub_layers = to_half(teach_sub_layers,dtype=fp16_type)
                teach_sub_layers = to_dev(teach_sub_layers, ["cpu"] * len(teach_sub_layers))  #single gpu
                replace_ori_layer(layers, sub_layers, layer_id_list, args)


            del sub_layers,teach_sub_layers
            cleanup_memory(logger=logger)

    if args.use_ddp:  # 保证参数保存完整
        dist.barrier()

    logger.info("Model quantization finished! start change model to inference mode!")
    args.quant_rate = 1.0
    model_to_inference_mode(layers=layers,args=args,dtype=fp16_type,dev=dev,layer_id_list=args.quant_layer_list)
    model.to(fp16_type)


    try:
        del quant_inps
        del fp_inps
        # del tmp_fp_inps
    except Exception as e:
        logger.info(f"del tensor occurs {e}, skip!")
    cleanup_memory(logger=logger)

    logger.info("Model is changed to inderence mode!")

    cleanup_memory(logger=logger)

    model.config.use_cache = use_cache

    return model,inps,infer_attention_mask,position_ids
