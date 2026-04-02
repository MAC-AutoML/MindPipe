"""Base types for quantization methods."""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.device import enable_device_compat
from ..common.device import resolve_device_string
from ..common.io import ensure_dir
from ..common.io import model_slug
from ..common.io import write_json
from ..common.modeling import load_model_and_tokenizer
from evaluation.runner import run_evaluations


LOGGER = logging.getLogger(__name__)


@dataclass
class QuantizationRunResult:
    algorithm_name: str
    model_path: str
    output_dir: str
    metrics_path: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]


class BaseQuantizationMethod(ABC):
    name = "base"

    def load_resources(self, args):
        return load_model_and_tokenizer(args.model_path, dtype=args.dtype)

    @abstractmethod
    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = f"{self.name}_w{args.weight_bits}a{args.activation_bits}_seq{args.sequence_length}"
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def run(self, args) -> QuantizationRunResult:
        args.device = enable_device_compat(resolve_device_string(args.device))
        output_dir = self.resolve_output_dir(args)
        LOGGER.info("Loading model from %s", args.model_path)
        model, tokenizer_bundle = self.load_resources(args)
        model.seqlen = args.sequence_length
        artifacts = self.apply_fake_quantization(model, tokenizer_bundle, args)
        evaluation_args = vars(args).copy()
        evaluation_args["evaluation_output_dir"] = str(output_dir)
        metrics = run_evaluations(
            model=model,
            tokenizer_bundle=tokenizer_bundle,
            common_args=evaluation_args,
        )
        metrics.update(
            {
                "algorithm_name": self.name,
                "model_path": args.model_path,
                "device": args.device,
                "dtype": args.dtype,
            }
        )
        metrics_path = write_json(output_dir / "metrics.json", {**metrics, "artifacts": artifacts})
        if args.save_fake_model:
            model_dir = ensure_dir(output_dir / "fake_model")
            model.save_pretrained(model_dir)
            tokenizer_bundle.save_pretrained(str(model_dir))
            artifacts = {**artifacts, "fake_model_dir": str(model_dir)}
            metrics_path = write_json(output_dir / "metrics.json", {**metrics, "artifacts": artifacts})
        return QuantizationRunResult(
            algorithm_name=self.name,
            model_path=args.model_path,
            output_dir=str(output_dir),
            metrics_path=str(metrics_path),
            metrics=metrics,
            artifacts=artifacts,
        )
