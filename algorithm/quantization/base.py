"""Base types for quantization methods."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from ..common.io import ensure_dir
from ..common.io import model_slug


class BaseQuantizationMethod(ABC):
    name = "base"
    npu_ready: bool = True  # Override to False in subclasses that lack NPU support
    default_calibration_dataset: str  # Each subclass must define its own

    @abstractmethod
    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = f"{self.name}_w{args.weight_bits}a{args.activation_bits}_seq{args.sequence_length}"
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)
