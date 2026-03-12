# -*- coding: utf-8 -*-
"""
MindPipe 适配层

提供与第三方库的统一接口:
- bitsandbytes: NVIDIA GPU 量化库
- compressed_tensor: 压缩张量格式
- modelslim: 华为昇腾 ModelSlim
"""

from .bitsandbytes import BitsAndBytesAdapter
from .compressed_tensor import CompressedTensorAdapter
from .modelslim import ModelSlimAdapter

__all__ = [
    "BitsAndBytesAdapter",
    "CompressedTensorAdapter",
    "ModelSlimAdapter",
]

