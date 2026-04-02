"""Common executor for single-stage and multi-stage workflows."""

from __future__ import annotations

import argparse
import copy
import gc
import time
from pathlib import Path
from typing import Any

from algorithm.common.device import empty_cache
from algorithm.common.device import enable_device_compat
from algorithm.common.device import resolve_device_string
from algorithm.common.io import ensure_dir
from algorithm.common.io import write_json
from algorithm.common.modeling import load_model_and_tokenizer
from algorithm.pruning.registry import get_method as get_pruning_method
from algorithm.quantization.config import normalize_args as normalize_quantization_args
from algorithm.quantization.registry import get_method as get_quantization_method
from evaluation.runner import run_evaluations
from workflow.builder import validate_workflow_config
from workflow.schema import WorkflowConfig
from workflow.schema import WorkflowRunResult
from workflow.schema import WorkflowStage


def _build_stage_args(common_args: dict[str, Any], stage: WorkflowStage) -> argparse.Namespace:
    stage_args_dict = copy.deepcopy(common_args)
    stage_args_dict.update(stage.parameters)
    stage_args = argparse.Namespace(**stage_args_dict)
    if stage.stage_type == "quantization":
        normalize_quantization_args(stage_args)
    return stage_args


def _resolve_stage_method(stage: WorkflowStage):
    if stage.stage_type == "quantization":
        return get_quantization_method(stage.algorithm_name)
    return get_pruning_method(stage.algorithm_name)


def _resolve_final_output_dir(
    config: WorkflowConfig,
    stage_method,
    stage_args: argparse.Namespace,
) -> Path:
    if config.flatten_single_stage and len(config.stages) == 1:
        return ensure_dir(stage_method.resolve_output_dir(stage_args))
    if config.output_dir is None:
        raise ValueError("Workflow output_dir is required for multi-stage runs")
    return ensure_dir(config.output_dir)


def _run_stage(stage_method, stage: WorkflowStage, model, tokenizer_bundle, stage_args: argparse.Namespace):
    stage_start = time.perf_counter()
    stage_output_dir = ensure_dir(stage_method.resolve_output_dir(stage_args))
    if stage.stage_type == "quantization":
        artifacts = stage_method.apply_fake_quantization(model, tokenizer_bundle, stage_args)
    else:
        artifacts = stage_method.apply_pruning(model, tokenizer_bundle, stage_args)
    return {
        "stage_type": stage.stage_type,
        "algorithm_name": stage.algorithm_name,
        "parameters": {
            key: value
            for key, value in vars(stage_args).items()
            if key not in {"hf_token"}
        },
        "output_dir": str(stage_output_dir),
        "elapsed_seconds": time.perf_counter() - stage_start,
        "artifacts": artifacts,
    }


def run_workflow(config: WorkflowConfig) -> WorkflowRunResult:
    validate_workflow_config(config)

    common_args = copy.deepcopy(config.common_args)
    resolved_device = resolve_device_string(common_args.get("device", "auto"))
    common_args["device"] = enable_device_compat(resolved_device)

    dtype = config.common_args.get("dtype", "auto")
    model, tokenizer_bundle = load_model_and_tokenizer(config.model_path, dtype=dtype)
    sequence_length = int(common_args["sequence_length"])
    model.seqlen = sequence_length

    stage_records: list[dict[str, Any]] = []
    final_output_dir: Path | None = None
    for stage in config.stages:
        stage_method = _resolve_stage_method(stage)
        stage_args = _build_stage_args(common_args, stage)
        stage_args.model_path = config.model_path
        if final_output_dir is None:
            final_output_dir = _resolve_final_output_dir(config, stage_method, stage_args)
        stage_record = _run_stage(stage_method, stage, model, tokenizer_bundle, stage_args)
        stage_records.append(stage_record)
        gc.collect()
        empty_cache(common_args["device"])

    if final_output_dir is None:
        raise RuntimeError("Workflow produced no stages")

    common_args["evaluation_output_dir"] = str(final_output_dir)
    common_args["model_path"] = config.model_path
    metrics = run_evaluations(
        model=model,
        tokenizer_bundle=tokenizer_bundle,
        common_args=common_args,
    )
    metrics.update(config.result_metadata)
    metrics.update(
        {
            "model_path": config.model_path,
            "device": common_args["device"],
            "dtype": dtype,
        }
    )

    artifacts: dict[str, Any]
    if config.flatten_single_stage and len(stage_records) == 1:
        artifacts = stage_records[0]["artifacts"]
    else:
        artifacts = {"stages": stage_records}
    if config.save_composed_model:
        model_dir = ensure_dir(final_output_dir / "composed_model")
        model.save_pretrained(model_dir)
        tokenizer_bundle.save_pretrained(str(model_dir))
        artifacts["composed_model_dir"] = str(model_dir)

    metrics_path = write_json(final_output_dir / "metrics.json", {**metrics, "artifacts": artifacts})
    return WorkflowRunResult(
        model_path=config.model_path,
        output_dir=str(final_output_dir),
        metrics_path=str(metrics_path),
        metrics=metrics,
        artifacts=artifacts,
    )
