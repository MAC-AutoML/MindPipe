# -*- coding: utf-8 -*-
"""
模型集成模块

提供统一的模型加载、保存、转换接口。
"""

from typing import Optional, Union


class ModelLoader:
    """
    模型加载器

    支持从多种来源加载模型:
    - HuggingFace Hub
    - 本地路径
    - ModelScope
    """

    @staticmethod
    def load(
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = "float16",
        **kwargs
    ):
        """
        加载模型

        Args:
            model_name_or_path: 模型名称或路径
            device: 目标设备
            dtype: 数据类型

        Returns:
            加载的模型
        """
        # TODO: 实现统一的模型加载逻辑
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if dtype == "float16" else torch.float32,
            device_map="auto" if device is None else device,
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        return model, tokenizer


class ModelWrapper:
    """
    模型包装器

    为模型提供统一的接口，屏蔽不同模型架构的差异。
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def save(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)

    def to(self, device: str):
        """移动模型到指定设备"""
        self.model = self.model.to(device)
        return self

    def eval(self):
        """设置为评估模式"""
        self.model.eval()
        return self
