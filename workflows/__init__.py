# -*- coding: utf-8 -*-
"""
MindPipe 工作流模块

提供完整的压缩工作流:
- PruningWorkflow: 剪枝工作流
- QuantizationWorkflow: 量化工作流
- HybridWorkflow: 混合压缩工作流
- EvaluationWorkflow: 模型评估工作流
"""

from .pruning_workflow import PruningWorkflow
from .quantization_workflow import QuantizationWorkflow
from .hybrid_workflow import HybridWorkflow
from .evaluation_workflow import EvaluationWorkflow

__all__ = [
    "PruningWorkflow",
    "QuantizationWorkflow",
    "HybridWorkflow",
    "EvaluationWorkflow",
]

