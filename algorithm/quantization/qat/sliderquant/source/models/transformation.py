
import torch



class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        truncated_tensor = input.clone()
        truncated_tensor = torch.where(truncated_tensor.abs() < threshold, truncated_tensor.sign() * threshold, truncated_tensor)
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        grad_input = grad_output.clone()
        grad_input[input.abs() < threshold] = 0
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)



def smooth_ln_fcs_temporary(ln, fcs, scales,shifts,weight_in_cpu=False):
    ln.use_temporary_parameter = True
    if weight_in_cpu is True:
        ln.weight = ln.weight.to(scales.device)
        for fc in fcs:
            fc.weight = fc.weight.to(scales.device)


    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts) / scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + torch.matmul(fc.weight,shifts)
        else:
            fc.temp_bias = torch.matmul(fc.weight,shifts)
        fc.temp_weight = fc.weight * scales.view(1,-1)

    if weight_in_cpu is True:
        ln.weight = ln.weight.cpu()
        for fc in fcs:
            fc.weight = fc.weight.cpu()


def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            # import ipdb;ipdb.set_trace()
            fc.register_buffer('bias',fc.weight@shifts)
        fc.weight.mul_(scales.view(1,-1))



def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None,num_key_value_groups=1,head_dim=128,weight_in_cpu=False,args=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    if weight_in_cpu is True:
        fc1.weight = fc1.weight.to(scales.device)
        fc2.weight = fc2.weight.to(scales.device)

    # import ipdb;ipdb.set_trace()
    if num_key_value_groups > 1:
        # import ipdb;ipdb.set_trace()
        if args.gqa_scales == "copy":
            kv_scales = scales
            kv_shift = shifts
            scales = scales.view(-1,head_dim).repeat_interleave(num_key_value_groups,dim=0).view(-1)
            shifts = shifts.view(-1,head_dim).repeat_interleave(num_key_value_groups,dim=0).view(-1)
        elif args.gqa_scales == "mean":
            kv_scales = scales.view(-1,num_key_value_groups,head_dim).mean(dim=1).view(-1)
            kv_shift = shifts.view(-1,num_key_value_groups,head_dim).mean(dim=1).view(-1)
        else:
            raise NotImplementedError("Only implemented copy and mean for gqa")
    else:
        kv_scales = scales
        kv_shift = shifts

    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = fc1.temp_bias - kv_shift
        fc1.temp_bias = fc1.temp_bias/kv_scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/kv_scales.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/kv_scales.view(-1)
        fc1.temp_weight = fc1.weight/kv_scales.view(-1,1)

    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@shifts
    else:
        fc2.temp_bias = fc2.weight@shifts
    fc2.temp_weight = fc2.weight * scales.view(1,-1)


    if weight_in_cpu is True:
        fc1.weight = fc1.weight.cpu()
        fc2.weight = fc2.weight.cpu()

def smooth_fc_fc_inplace(fc1, fc2, scales,shifts=None,num_key_value_groups=1,head_dim=128,args=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False

    if num_key_value_groups > 1:
        # import ipdb;ipdb.set_trace()
        if args.gqa_scales == "copy":
            kv_scales = scales
            kv_shift = shifts
            scales = scales.view(-1,head_dim).repeat_interleave(num_key_value_groups,dim=0).view(-1)
            shifts = shifts.view(-1,head_dim).repeat_interleave(num_key_value_groups,dim=0).view(-1)
        elif args.gqa_scales == "mean":
            kv_scales = scales.view(-1,num_key_value_groups,head_dim).mean(dim=1).view(-1)
            kv_shift = shifts.view(-1,num_key_value_groups,head_dim).mean(dim=1).view(-1)
        else:
           raise NotImplementedError("Only implemented copy and mean for gqa")
    else:
        kv_scales = scales
        kv_shift = shifts


    fc1.bias.sub_(kv_shift)
    fc1.bias.div_(kv_scales.view(-1))
    fc1.weight.div_(kv_scales.view(-1,1))


    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
    fc2.weight.mul_(scales.view(1,-1))



def smooth_q_k_temporary(q_proj, k_proj,scales,num_key_value_groups=1,head_dim=128,weight_in_cpu=False,args=None):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True

    if weight_in_cpu is True:
        q_proj.weight = q_proj.weight.to(scales.device)
        k_proj.weight = k_proj.weight.to(scales.device)

    if num_key_value_groups > 1:
        # import ipdb;ipdb.set_trace()
        if args.gqa_scales == "copy":
            kv_scales = scales
            scales = scales.view(-1,head_dim).repeat_interleave(num_key_value_groups,dim=0).view(-1)
        elif args.gqa_scales == "mean":
            kv_scales = scales.view(-1,num_key_value_groups,head_dim).mean(dim=1).view(-1)
        else:
           raise NotImplementedError("Only implemented copy and mean for gqa")
    else:
        kv_scales = scales

    q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)

    k_proj.temp_weight = k_proj.temp_weight*kv_scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*kv_scales.view(-1)

    if weight_in_cpu is True:
        q_proj.weight = q_proj.weight.cpu()
        k_proj.weight = k_proj.weight.cpu()


def smooth_q_k_inplace(q_proj, k_proj, scales,num_key_value_groups=1,head_dim=128,args=None):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False

    if num_key_value_groups > 1:
        # import ipdb;ipdb.set_trace()
        if args.gqa_scales == "copy":
            kv_scales = scales
            scales = scales.view(-1,head_dim).repeat_interleave(num_key_value_groups,dim=0).view(-1)
        elif args.gqa_scales == "mean":
            kv_scales = scales.view(-1,num_key_value_groups,head_dim).mean(dim=1).view(-1)
        else:
           raise NotImplementedError("Only implemented copy and mean for gqa")
    else:
        kv_scales = scales

    q_proj.weight.div_(scales.view(-1,1))
    q_proj.bias.div_(scales.view(-1))
    k_proj.weight.mul_(kv_scales.view(-1,1))
    k_proj.bias.mul_(kv_scales.view(-1))