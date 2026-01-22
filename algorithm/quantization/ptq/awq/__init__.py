"""
AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
Paper: https://arxiv.org/abs/2306.00978
"""

from .quantize.pre_quant import run_awq, apply_awq, get_blocks, get_named_linears
from .quantize.quantizer import pseudo_quantize_tensor, pseudo_quantize_model_weight, real_quantize_model_weight
from .quantize.qmodule import WQLinear, ScaledActivation
from .utils.calib_data import get_calib_dataset
