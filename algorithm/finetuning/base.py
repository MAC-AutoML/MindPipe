"""Base types for finetuning methods."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from algorithm.common.io import ensure_dir
from algorithm.common.io import model_slug


class BaseFinetuningMethod(ABC):
    name = "base"
    npu_ready = False

    @abstractmethod
    def apply_finetuning(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = f"{self.name}_seq{args.sequence_length}"
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

