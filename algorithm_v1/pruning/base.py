"""Base types for pruning methods."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.evaluation import evaluate_perplexity
from ..common.evaluation import evaluate_zero_shot
from ..common.io import ensure_dir
from ..common.io import model_slug
from ..common.io import write_json
from ..common.modeling import load_model_and_tokenizer


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
        output_dir = self.resolve_output_dir(args)
        model, tokenizer_bundle = self.load_resources(args)
        model.seqlen = args.sequence_length
        artifacts = self.apply_pruning(model, tokenizer_bundle, args)
        metrics = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.evaluation_dataset,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            max_eval_chunks=args.max_eval_chunks,
            device=args.device,
        )
        if getattr(args, "eval_zero_shot", False):
            metrics["zero_shot"] = evaluate_zero_shot(
                model=model,
                tokenizer=tokenizer_bundle.tokenizer,
                task_names=args.zero_shot_tasks,
                batch_size=int(args.zero_shot_batch_size),
                device=args.device,
                num_fewshot=int(args.zero_shot_num_fewshot),
                limit=getattr(args, "zero_shot_limit", None),
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
