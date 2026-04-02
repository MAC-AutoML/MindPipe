"""Base types for pruning methods."""

from __future__ import annotations

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


@dataclass
class PruningRunResult:
    algorithm_name: str
    model_path: str
    output_dir: str
    metrics_path: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]


class BasePruningMethod(ABC):
    name = "base"

    def load_resources(self, args):
        return load_model_and_tokenizer(args.model_path, dtype=args.dtype)

    @abstractmethod
    def apply_pruning(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = f"{self.name}_s{args.sparsity_ratio}_seq{args.sequence_length}"
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def run(self, args) -> PruningRunResult:
        args.device = enable_device_compat(resolve_device_string(args.device))
        output_dir = self.resolve_output_dir(args)
        model, tokenizer_bundle = self.load_resources(args)
        model.seqlen = args.sequence_length
        artifacts = self.apply_pruning(model, tokenizer_bundle, args)
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
                "sparsity_ratio": args.sparsity_ratio,
            }
        )
        metrics_path = write_json(output_dir / "metrics.json", {**metrics, "artifacts": artifacts})
        if args.save_pruned_model:
            model_dir = ensure_dir(output_dir / "pruned_model")
            model.save_pretrained(model_dir)
            tokenizer_bundle.save_pretrained(str(model_dir))
            artifacts = {**artifacts, "pruned_model_dir": str(model_dir)}
            metrics_path = write_json(output_dir / "metrics.json", {**metrics, "artifacts": artifacts})
        return PruningRunResult(
            algorithm_name=self.name,
            model_path=args.model_path,
            output_dir=str(output_dir),
            metrics_path=str(metrics_path),
            metrics=metrics,
            artifacts=artifacts,
        )
