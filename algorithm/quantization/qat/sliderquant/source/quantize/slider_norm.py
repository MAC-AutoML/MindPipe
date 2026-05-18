import torch
import torch.nn as nn


'''
Modify normalization layer to adapt the training of learnable equivalent transformation
'''


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, ori_layer_norm,mean_dim=4096, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        # self.mean_dim = ori_layer_norm.normalized_shape[0]
        # self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('weight',ori_layer_norm.weight)
        if hasattr(ori_layer_norm, 'bias') and ori_layer_norm.bias is not None:
            self.register_buffer('bias',ori_layer_norm.bias)
        else:
            self.bias = None
        self.use_temporary_parameter = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)




class SliderLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        self.register_buffer('weight',ori_layer_norm.weight)
        if hasattr(ori_layer_norm,"bias") and ori_layer_norm.bias is not None:
            self.register_buffer('bias',ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False


    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.norm_func(x,self.normalized_shape,weight, bias,eps=self.eps)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class SliderLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False


    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, 'bias') else None

        return (weight * hidden_states+bias).to(input_dtype) if bias is not None else (weight * hidden_states).to(input_dtype)
