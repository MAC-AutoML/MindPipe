"""Workflow validation helpers."""

from __future__ import annotations

from .schema import WorkflowConfig
from ..pruning.registry import METHOD_REGISTRY as PRUNING_METHOD_REGISTRY
from ..quantization.registry import METHOD_REGISTRY as QUANTIZATION_METHOD_REGISTRY


VALID_STAGE_TYPES = {"quantization", "pruning"}


def validate_workflow_config(config: WorkflowConfig) -> None:
    if not config.stages:
        raise ValueError("Workflow must contain at least one stage")
    if not config.flatten_single_stage and config.output_dir is None:
        raise ValueError("Multi-stage workflow requires an explicit output_dir")
    for stage in config.stages:
        if stage.stage_type not in VALID_STAGE_TYPES:
            raise ValueError(f"Unsupported stage_type: {stage.stage_type}")
        if stage.stage_type == "quantization" and stage.algorithm_name not in QUANTIZATION_METHOD_REGISTRY:
            raise ValueError(f"Unknown quantization algorithm: {stage.algorithm_name}")
        if stage.stage_type == "pruning" and stage.algorithm_name not in PRUNING_METHOD_REGISTRY:
            raise ValueError(f"Unknown pruning algorithm: {stage.algorithm_name}")
