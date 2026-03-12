# -*- coding: utf-8 -*-
"""
自动化压缩工具

提供一键式模型压缩入口，自动选择最优压缩策略。
"""

from typing import Optional, Dict, Any, Union


class AutoCompressor:
    """
    自动化压缩工具

    提供简单易用的一键压缩接口，自动选择合适的工作流和算法。

    使用示例:
        >>> compressor = AutoCompressor()
        >>> result = compressor.compress(
        ...     model="path/to/model",
        ...     method="pruning",
        ...     algorithm="flap",
        ...     pruning_ratio=0.2
        ... )
    """

    SUPPORTED_METHODS = ["pruning", "quantization", "hybrid"]

    def __init__(self, device: Optional[str] = None):
        """
        初始化自动压缩工具

        Args:
            device: 目标设备 (auto, cuda, cpu, npu)
        """
        self.device = device or "auto"

    def compress(
        self,
        model: str,
        method: str = "auto",
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行模型压缩

        Args:
            model: 模型路径或名称
            method: 压缩方法 (pruning, quantization, hybrid, auto)
            output_dir: 输出目录
            **kwargs: 其他参数，根据 method 不同而不同
                - pruning: algorithm, pruning_ratio, ...
                - quantization: algorithm, bits, ...
                - hybrid: pruning_config, quantization_config, ...

        Returns:
            包含压缩结果的字典
        """
        if method == "auto":
            method = self._auto_select_method(model, kwargs)

        if method == "pruning":
            return self._run_pruning(model, output_dir, **kwargs)
        elif method == "quantization":
            return self._run_quantization(model, output_dir, **kwargs)
        elif method == "hybrid":
            return self._run_hybrid(model, output_dir, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _auto_select_method(self, model: str, kwargs: dict) -> str:
        """自动选择压缩方法"""
        # TODO: 实现自动选择逻辑
        # 根据模型大小、目标设备等因素选择最优方法
        return "pruning"

    def _run_pruning(self, model: str, output_dir: Optional[str], **kwargs) -> Dict[str, Any]:
        """执行剪枝"""
        from .workflows import PruningWorkflow

        algorithm = kwargs.get("algorithm", "wanda")
        pruning_ratio = kwargs.get("pruning_ratio", 0.2)

        workflow = PruningWorkflow(
            model=model,
            algorithm=algorithm,
            pruning_ratio=pruning_ratio,
            **kwargs
        )

        result = workflow.run()

        if output_dir:
            result["output_dir"] = output_dir

        return result

    def _run_quantization(self, model: str, output_dir: Optional[str], **kwargs) -> Dict[str, Any]:
        """执行量化"""
        from .workflows import QuantizationWorkflow

        algorithm = kwargs.get("algorithm", "gptq")
        bits = kwargs.get("bits", 4)

        workflow = QuantizationWorkflow(
            model=model,
            algorithm=algorithm,
            bits=bits,
            **kwargs
        )

        result = workflow.run()

        if output_dir:
            result["output_dir"] = output_dir

        return result

    def _run_hybrid(self, model: str, output_dir: Optional[str], **kwargs) -> Dict[str, Any]:
        """执行混合压缩"""
        from .workflows import HybridWorkflow

        workflow = HybridWorkflow(
            model=model,
            pruning_config=kwargs.get("pruning_config"),
            quantization_config=kwargs.get("quantization_config"),
            **kwargs
        )

        result = workflow.run()

        if output_dir:
            result["output_dir"] = output_dir

        return result

    def evaluate(self, model: str, metrics: Optional[list] = None) -> Dict[str, Any]:
        """
        评估模型

        Args:
            model: 模型路径
            metrics: 评估指标列表

        Returns:
            评估结果
        """
        from .workflows import EvaluationWorkflow

        workflow = EvaluationWorkflow(model=model, metrics=metrics)
        return workflow.run()

