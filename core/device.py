# -*- coding: utf-8 -*-
"""
设备管理模块

支持设备:
- CANN/NPU (华为昇腾)
- CUDA/GPU (NVIDIA)
- MUSA/GPU (摩尔线程)
- CPU
"""

from enum import Enum
from typing import Optional


class DeviceType(Enum):
    """设备类型枚举"""
    CPU = "cpu"
    CUDA = "cuda"      # NVIDIA GPU
    CANN = "cann"      # 华为昇腾 NPU
    MUSA = "musa"      # 摩尔线程 GPU


class DeviceManager:
    """
    设备管理器

    负责检测可用设备、设备切换、设备信息查询等功能。
    """

    _current_device: Optional[str] = None

    @classmethod
    def get_available_devices(cls) -> list:
        """获取所有可用设备"""
        # TODO: 实现设备检测逻辑
        devices = [DeviceType.CPU]

        # 检测 CUDA
        try:
            import torch
            if torch.cuda.is_available():
                devices.append(DeviceType.CUDA)
        except ImportError:
            pass

        # TODO: 检测 CANN (昇腾)
        # TODO: 检测 MUSA (摩尔线程)

        return devices

    @classmethod
    def get_device(cls) -> str:
        """获取当前设备"""
        if cls._current_device is None:
            cls._current_device = cls._auto_select_device()
        return cls._current_device

    @classmethod
    def set_device(cls, device: str):
        """设置当前设备"""
        cls._current_device = device

    @classmethod
    def _auto_select_device(cls) -> str:
        """自动选择最优设备"""
        available = cls.get_available_devices()

        # 优先级: CUDA > CANN > MUSA > CPU
        priority = [DeviceType.CUDA, DeviceType.CANN, DeviceType.MUSA, DeviceType.CPU]
        for device in priority:
            if device in available:
                return device.value

        return DeviceType.CPU.value


# 便捷函数
def get_device() -> str:
    """获取当前设备"""
    return DeviceManager.get_device()


def set_device(device: str):
    """设置当前设备"""
    DeviceManager.set_device(device)
