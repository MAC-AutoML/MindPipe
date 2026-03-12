# -*- coding: utf-8 -*-
"""
MindPipe 基础底座层

包含:
- device: 设备适配 (CANN/NPU, CUDA/GPU, MUSA/GPU)
- distributed: 分布式优化
- model: 模型集成
- cluster: 集群优化
"""

from .device import DeviceManager, DeviceType, get_device, set_device
from .distributed import DistributedManager
from .model import ModelLoader, ModelWrapper
from .cluster import ClusterOptimizer

__all__ = [
    # 设备管理
    "DeviceManager",
    "DeviceType",
    "get_device",
    "set_device",
    # 分布式
    "DistributedManager",
    # 模型
    "ModelLoader",
    "ModelWrapper",
    # 集群
    "ClusterOptimizer",
]


