"""Base types for pruning methods."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from ..common.io import ensure_dir
from ..common.io import model_slug


class BasePruningMethod(ABC):
    name = "base"
    npu_ready = True  # Override to False in subclasses that lack NPU support
    default_calibration_dataset: str  # Each subclass must define its own

    @abstractmethod
    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        # n:m 半结构化模式加入目录名，避免 2:4 / 4:8 撞目录
        pattern = getattr(args, 'structure_pattern', 'unstructured')
        pattern_suffix = f"_{pattern.replace(':', '-')}" if pattern != "unstructured" else ""
        run_spec = f"{self.name}_s{args.sparsity_ratio}{pattern_suffix}_seq{args.sequence_length}"
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)
