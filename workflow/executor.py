"""Common executor for single-stage and multi-stage workflows."""

from __future__ import annotations

import argparse
import copy
import gc
import time
from pathlib import Path
from typing import Any

from algorithm.common.device import empty_cache
from algorithm.common.device import resolve_device_string
from algorithm.common.io import ensure_dir
from algorithm.common.io import write_json
from algorithm.common.modeling import load_model_and_tokenizer
from algorithm.common.modeling import normalize_dense_qwen3_mlp_intermediate_size_for_hf_save
from algorithm.common.modeling import normalize_mixtral_expert_intermediate_size_for_hf_save
from algorithm.common.modeling import normalize_qwen3_moe_expert_intermediate_size_for_hf_save
from algorithm.finetuning.registry import get_method as get_finetuning_method
from algorithm.finetuning.compression_lora.mask_utils import extract_masks_from_pruned_model
from algorithm.finetuning.compression_lora.mask_utils import mask_sparsity
from algorithm.finetuning.compression_lora.mask_utils import restore_weights
from algorithm.finetuning.compression_lora.mask_utils import save_masks
from algorithm.finetuning.compression_lora.mask_utils import snapshot_weights
from algorithm.common.qwen3_5_moe_unfuse import refuse_qwen3_5_moe_experts_for_hf_save
from algorithm.common.qwen3_5_moe_unfuse import refuse_qwen3_moe_experts_for_hf_save
from algorithm.common.qwen3_5_moe_unfuse import refuse_flatquant_qwen3_moe_experts_for_hf_save
from algorithm.common.qwen3_5_moe_unfuse import set_qwen3_5_moe_calibrate_all_experts
from algorithm.common.qwen3_5_moe_unfuse import set_qwen3_moe_calibrate_all_experts
from algorithm.common.qwen3_5_moe_unfuse import unfuse_qwen3_5_moe_experts
from algorithm.common.qwen3_5_moe_unfuse import unfuse_qwen3_moe_experts
from algorithm.pruning.registry import get_method as get_pruning_method
from algorithm.quantization.config import normalize_args as normalize_quantization_args
from algorithm.quantization.exporters.modelslim import export_modelslim_ascend_quant
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
    if stage.stage_type == "finetuning":
        return get_finetuning_method(stage.algorithm_name)
    return get_pruning_method(stage.algorithm_name)


def _stage_reloads_model(stage: WorkflowStage) -> bool:
    return stage.stage_type == "finetuning" and stage.algorithm_name == "compression_lora"


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
    if (
        stage.stage_type == "finetuning"
        and stage.algorithm_name == "compression_lora"
        and getattr(stage_args, "_workflow_output_dir", None)
    ):
        stage_output_dir = ensure_dir(Path(stage_args._workflow_output_dir))
    else:
        stage_output_dir = ensure_dir(stage_method.resolve_output_dir(stage_args))
    if stage.stage_type == "quantization":
        stage_result = stage_method.apply_fake_quantization(model, tokenizer_bundle, stage_args)
    elif stage.stage_type == "finetuning":
        stage_result = stage_method.apply_finetuning(model, tokenizer_bundle, stage_args)
    else:
        weight_snapshot = None
        unfuse_qwen3_5_moe_experts(model, calibrate_all_experts=False)
        unfuse_qwen3_moe_experts(model, calibrate_all_experts=False)
        if getattr(stage_args, "_capture_pruning_masks", False):
            weight_snapshot = snapshot_weights(
                model,
                target_modules=getattr(stage_args, "compression_lora_target_modules", None),
            )
        try:
            stage_result = stage_method.apply_pruning(model, tokenizer_bundle, stage_args)
        finally:
            set_qwen3_5_moe_calibrate_all_experts(model, False)
            set_qwen3_moe_calibrate_all_experts(model, False)
        if weight_snapshot is not None:
            masks = extract_masks_from_pruned_model(model, weight_snapshot)
            restore_weights(model, weight_snapshot)
            workflow_output_dir = ensure_dir(Path(stage_args._workflow_output_dir))
            masks_path = save_masks(
                workflow_output_dir / "pruning_masks.pth",
                masks,
                metadata={
                    "pruning_algorithm": stage.algorithm_name,
                    "sparsity_ratio": getattr(stage_args, "sparsity_ratio", None),
                    "target_modules": getattr(stage_args, "compression_lora_target_modules", None),
                    "source_stage_output_dir": str(stage_output_dir),
                    **mask_sparsity(masks),
                },
            )
            stage_result["pruning_masks_path"] = str(masks_path)
        refused_qwen3_moe_blocks = refuse_qwen3_moe_experts_for_hf_save(model, experts_on_cpu=False)
        if refused_qwen3_moe_blocks:
            stage_result["refused_qwen3_moe_blocks_for_eval"] = refused_qwen3_moe_blocks

    if not isinstance(stage_result, dict):
        raise TypeError(
            f"Stage method {stage_method.__class__.__name__} must return a dict, got {type(stage_result)!r}."
        )

    next_model = stage_result.pop("_updated_model", model)
    next_tokenizer_bundle = stage_result.pop("_updated_tokenizer_bundle", tokenizer_bundle)
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
        "artifacts": stage_result,
    }, next_model, next_tokenizer_bundle


def _run_modelslim_export_only(
    config: WorkflowConfig,
    common_args: dict[str, Any],
) -> WorkflowRunResult:
    final_output_dir = ensure_dir(config.output_dir or Path("results/modelslim_export"))
    artifacts_path = final_output_dir / "artifacts.json"
    metrics_path = final_output_dir / "metrics.json"
    status_path = final_output_dir / "export_status.json"
    export_dir = common_args.get("export_quantized_model_dir")
    if export_dir is None:
        export_dir = final_output_dir.parent / "real_quant_modelslim_model"

    resolved_output_dir = final_output_dir.expanduser().resolve()
    resolved_export_dir = Path(export_dir).expanduser().resolve()
    started_at = time.time()

    try:
        # A failed retry must not leave a previous run's terminal metadata looking current.
        for path in (artifacts_path, metrics_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        write_json(
            status_path,
            {
                "status": "running",
                "model_path": config.model_path,
                "export_backend": common_args.get("export_backend"),
                "export_path": str(resolved_export_dir),
                "started_at_unix": started_at,
            },
        )

        validate_workflow_config(config)
        if (
            resolved_output_dir == resolved_export_dir
            or resolved_output_dir in resolved_export_dir.parents
            or resolved_export_dir in resolved_output_dir.parents
        ):
            raise ValueError(
                "ModelSlim checkpoint directory and workflow output directory must not be "
                "identical or nested: "
                f"checkpoint={resolved_export_dir}, workflow_output={resolved_output_dir}."
            )

        real_quant_export = export_modelslim_ascend_quant(
            model_path=config.model_path,
            export_dir=resolved_export_dir,
            common_args=common_args,
        )

        artifacts = {"real_quant_export": real_quant_export}
        metrics = copy.deepcopy(config.result_metadata)
        metrics.update(
            {
                "model_path": config.model_path,
                "export_backend": "modelslim",
                "export_precision": real_quant_export["precision"],
                "artifacts_path": artifacts_path.name,
                "export_status_path": status_path.name,
            }
        )
        write_json(artifacts_path, artifacts)
        metrics_path = write_json(metrics_path, metrics)
        write_json(
            status_path,
            {
                "status": "completed",
                "model_path": config.model_path,
                "export_backend": "modelslim",
                "export_path": str(resolved_export_dir),
                "artifacts_path": artifacts_path.name,
                "metrics_path": metrics_path.name,
                "started_at_unix": started_at,
                "completed_at_unix": time.time(),
            },
        )
    except BaseException as error:
        for path in (artifacts_path, metrics_path):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            write_json(
                status_path,
                {
                    "status": "failed",
                    "model_path": config.model_path,
                    "export_backend": common_args.get("export_backend"),
                    "export_path": str(resolved_export_dir),
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                    "started_at_unix": started_at,
                    "completed_at_unix": time.time(),
                },
            )
        except OSError:
            pass
        raise

    return WorkflowRunResult(
        model_path=config.model_path,
        output_dir=str(final_output_dir),
        metrics_path=str(metrics_path),
        artifacts_path=str(artifacts_path),
        metrics=metrics,
        artifacts=artifacts,
    )


def run_workflow(config: WorkflowConfig) -> WorkflowRunResult:
    common_args = copy.deepcopy(config.common_args)
    if common_args.get("export_real_quant", False):
        return _run_modelslim_export_only(config, common_args)

    validate_workflow_config(config)

    resolved_device = resolve_device_string(common_args.get("device", "auto"))
    common_args["device"] = resolved_device

    dtype = config.common_args.get("dtype", "auto")
    attn_implementation = config.common_args.get("attn_implementation")
    device_map = common_args.get("device_map")

    # 剪枝和量化都要求 device_map，因为内部已移除所有权重 .to(device)
    if config.stages and device_map is None:
        raise ValueError(
            "当使用剪枝或量化时，必须通过 --device_map 指定设备映射（推荐 auto）。"
            "例如: CUDA_VISIBLE_DEVICES=0,1 python main.py --device_map auto ..."
        )

    model, tokenizer_bundle = load_model_and_tokenizer(
        config.model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        max_memory=common_args.get("max_memory"),
        offload_folder=common_args.get("offload_folder"),
        offload_state_dict=common_args.get("offload_state_dict"),
        no_split_module_classes=common_args.get("no_split_module_classes"),
    )
    sequence_length = int(common_args["sequence_length"])
    model.seqlen = sequence_length

    # ── 压缩阶段（可为空 = 仅评测） ──
    stage_records: list[dict[str, Any]] = []
    final_output_dir: Path | None = None
    for stage in config.stages:
        stage_method = _resolve_stage_method(stage)
        stage_args = _build_stage_args(common_args, stage)
        stage_args.model_path = config.model_path
        if final_output_dir is None:
            final_output_dir = _resolve_final_output_dir(config, stage_method, stage_args)
        stage_args._workflow_output_dir = str(final_output_dir)
        if stage.stage_type == "pruning":
            stage_index = len(stage_records)
            stage_args._capture_pruning_masks = any(
                following.stage_type == "finetuning" and following.algorithm_name == "compression_lora"
                for following in config.stages[stage_index + 1:]
            )
        stage_reloads_model = _stage_reloads_model(stage)
        if stage_reloads_model:
            # compression_lora 会从 FP 权重重新加载模型。先释放前序压缩阶段模型，
            # 避免旧模型和新加载的 W_fp 在显存中短时间共存。
            model = None
            tokenizer_bundle = None
            gc.collect()
            empty_cache(common_args["device"])
        stage_record, model, tokenizer_bundle = _run_stage(stage_method, stage, model, tokenizer_bundle, stage_args)
        if stage_reloads_model and (model is None or tokenizer_bundle is None):
            raise RuntimeError(
                "compression_lora must return _updated_model and _updated_tokenizer_bundle after reloading model."
            )
        stage_records.append(stage_record)
        if stage.stage_type == "quantization" and stage.algorithm_name == "flatquant":
            flat_path = stage_record["artifacts"].get("flat_parameters_path")
            if flat_path and not common_args.get("compression_lora_flatquant_from"):
                common_args["compression_lora_flatquant_from"] = str(Path(flat_path).parent)
        if stage.stage_type == "quantization" and stage.algorithm_name == "splitquant":
            split_path = stage_record["artifacts"].get("splitquant_parameters_path")
            if split_path and not common_args.get("compression_lora_splitquant_from"):
                common_args["compression_lora_splitquant_from"] = str(Path(split_path).parent)
        if stage.stage_type == "pruning":
            masks_path = stage_record["artifacts"].get("pruning_masks_path")
            if masks_path and not common_args.get("compression_lora_masks_from"):
                common_args["compression_lora_masks_from"] = masks_path
        gc.collect()
        empty_cache(common_args["device"])

    # 仅评测模式：没有 stage 时，output_dir 由 config 或默认值决定
    if final_output_dir is None:
        if config.output_dir is not None:
            final_output_dir = ensure_dir(config.output_dir)
        else:
            final_output_dir = ensure_dir(Path("results/evaluate"))

    artifacts: dict[str, Any]
    if not stage_records:
        artifacts = {}
    elif config.flatten_single_stage and len(stage_records) == 1:
        artifacts = stage_records[0]["artifacts"]
    else:
        artifacts = {"stages": stage_records}

    metrics_path = final_output_dir / "metrics.json"
    artifacts_path = final_output_dir / "artifacts.json"
    metrics_metadata = copy.deepcopy(config.result_metadata)
    metrics_metadata.update(
        {
            "model_path": config.model_path,
            "device": common_args["device"],
            "dtype": dtype,
            "artifacts_path": artifacts_path.name,
        }
    )
    write_json(artifacts_path, artifacts)

    # ── 评测阶段（可通过 --eval_ppl false 跳过） ──
    common_args["evaluation_output_dir"] = str(final_output_dir)
    common_args["model_path"] = config.model_path
    common_args["evaluation_save_callback"] = lambda metrics: write_json(
        metrics_path,
        {**metrics, **metrics_metadata},
    )
    metrics = run_evaluations(
        model=model,
        tokenizer_bundle=tokenizer_bundle,
        common_args=common_args,
    )
    metrics.update(metrics_metadata)
    if config.save_model:
        model_dir = ensure_dir(final_output_dir / "saved_model")
        refuse_qwen3_5_moe_experts_for_hf_save(model)
        refuse_qwen3_moe_experts_for_hf_save(model)
        refuse_flatquant_qwen3_moe_experts_for_hf_save(model)
        normalize_qwen3_moe_expert_intermediate_size_for_hf_save(model)
        normalize_mixtral_expert_intermediate_size_for_hf_save(model)
        normalize_dense_qwen3_mlp_intermediate_size_for_hf_save(model)
        model.save_pretrained(model_dir)
        tokenizer_bundle.save_pretrained(str(model_dir))
        artifacts["saved_model_dir"] = str(model_dir)
        write_json(artifacts_path, artifacts)

    metrics_path = write_json(metrics_path, metrics)
    return WorkflowRunResult(
        model_path=config.model_path,
        output_dir=str(final_output_dir),
        metrics_path=str(metrics_path),
        artifacts_path=str(artifacts_path),
        metrics=metrics,
        artifacts=artifacts,
    )
