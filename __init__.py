# -*- coding: utf-8 -*-
"""
MindPipe 模型压缩框架

极致易用的大模型压缩框架，支持量化、剪枝等主流压缩技术，
适配昇腾NPU、CUDA GPU等多种硬件后端。

使用示例:
    # 方式1: 一键压缩
    from mindpipe import AutoCompressor

    compressor = AutoCompressor()
    result = compressor.compress(
        model="path/to/model",
        method="pruning",
        algorithm="flap",
        pruning_ratio=0.2
    )

    # 方式2: 使用工作流
    from mindpipe.workflows import PruningWorkflow

    workflow = PruningWorkflow(
        model="path/to/model",
        algorithm="flap",
        pruning_ratio=0.2
    )
    result = workflow.run()
"""

__version__ = "0.1.0"
__author__ = "MindPipe Team"

from .auto_compressor import AutoCompressor
from .core import DeviceManager, get_device, set_device
from .workflows import (
    PruningWorkflow,
    QuantizationWorkflow,
    HybridWorkflow,
    EvaluationWorkflow,
)

__all__ = [
    # 易用化工具
    "AutoCompressor",
    # 核心组件
    "DeviceManager",
    "get_device",
    "set_device",
    # 工作流
    "PruningWorkflow",
    "QuantizationWorkflow",
    "HybridWorkflow",
    "EvaluationWorkflow",
]
