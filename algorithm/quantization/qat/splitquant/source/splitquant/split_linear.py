import torch
import torch.nn as nn
import torch.nn.functional as F

from splitquant.quant_utils import WeightQuantizer, ActivationQuantizer
from splitquant.split_utils import kronecker_matmul

class SplitQuantizedLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear):
        super(SplitQuantizedLinear, self).__init__()
        self.args = args
        self.linear = linear
        _, in_features = linear.weight.shape

        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
        self.act_quantizer = ActivationQuantizer(
            bits=args.a_bits,
            sym=not(args.a_asym),
            lac=args.lac,
            groupsize=args.a_groupsize,
            in_channels=in_features,
        )

        self.group_size = args.w_groupsize if args.w_groupsize > 0 else -1
        self.weight_shape = self.linear.weight.shape

        self.lwc = args.lwc
        if self.lwc:
            lwc_dim = self.group_weight(self.linear.weight).shape[0] if self.lwc else -1
            init_value = 4.
            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False

    def group_weight(self, weight):
        if self.group_size > 0:
            return weight.reshape(-1, self.group_size)
        return weight

    def degroup_weight(self, weight):
        if self.group_size > 0:
            return weight.reshape(self.weight_shape)
        return weight

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            weight = kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        else:
            weight = qa_trans(weight, inv_t=True)
        return weight

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def _train_forward(self, hidden_states, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        # learnable weight clipping 
        if self.lwc:
            weight = self.group_weight(weight)
            weight = self.apply_wclip(weight)
            weight = self.degroup_weight(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T

        weight = self.group_weight(weight)
        # quantize weight
        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)
        weight = self.degroup_weight(weight)
        # quantize activation
        hidden_states = self.act_quantizer(hidden_states)

        if out_trans is not None and self.linear.bias is not None:
            bias = out_trans(self.linear.bias.data)
        else:
            bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)
        return output

    def forward(self, hidden_states, qa_trans=None, out_trans=None):
        if not self._eval_mode:
            return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans)
        else:
            return self._eval_forward(hidden_states)

    def _eval_forward(self, hidden_states):
        x_dtype = hidden_states.dtype
        hidden_states = self.act_quantizer(hidden_states).to(x_dtype)

        output = self.linear(hidden_states)
        return output

    def reparameterize(self, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.group_weight(weight)
            weight = self.apply_wclip(weight)
            weight = self.degroup_weight(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        
        self.linear.weight.data = weight.to(ori_dtype)
        self._eval_mode = True
