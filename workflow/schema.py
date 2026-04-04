"""Workflow data structures."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass
class WorkflowStage:
    stage_type: str
    algorithm_name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    model_path: str
    common_args: dict[str, Any]
    stages: list[WorkflowStage]
    output_dir: Path | None = None
    result_metadata: dict[str, Any] = field(default_factory=dict)
    flatten_single_stage: bool = False
    save_model: bool = False


@dataclass
class WorkflowRunResult:
    model_path: str
    output_dir: str
    metrics_path: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
