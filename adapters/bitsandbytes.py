# -*- coding: utf-8 -*-
"""
bitsandbytes 适配器

适配 NVIDIA bitsandbytes 量化库，支持 LLM.int8() 等量化方法。
"""

from typing import Optional


class BitsAndBytesAdapter:
    """
    bitsandbytes 适配器

    提供与 bitsandbytes 库的统一接口。
    """

    def __init__(self):
        self._check_available()

    def _check_available(self) -> bool:
        """检查 bitsandbytes 是否可用"""
        try:
            import bitsandbytes as bnb
            self.bnb = bnb
            return True
        except ImportError:
            self.bnb = None
            return False

    @property
    def is_available(self) -> bool:
        """是否可用"""
        return self.bnb is not None

    def quantize_model(self, model, bits: int = 8, **kwargs):
        """
        量化模型

        Args:
            model: 待量化模型
            bits: 量化位数 (4 或 8)

        Returns:
            量化后的模型
        """
        # TODO: 实现 bitsandbytes 量化逻辑
        if not self.is_available:
            raise RuntimeError("bitsandbytes is not installed")

        raise NotImplementedError("bitsandbytes quantization not implemented yet")
