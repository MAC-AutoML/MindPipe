"""Workflow config builders and parser helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from algorithm.common.io import ensure_dir
from algorithm.common.io import model_slug
from algorithm.pruning.registry import METHOD_REGISTRY as PRUNING_METHOD_REGISTRY
from algorithm.pruning.registry import get_method as get_pruning_method
from algorithm.quantization.config import normalize_args as normalize_quantization_args
from algorithm.quantization.registry import METHOD_REGISTRY as QUANTIZATION_METHOD_REGISTRY
from algorithm.quantization.registry import get_method as get_quantization_method
from evaluation.lm_eval import DEFAULT_ZERO_SHOT_TASKS
from workflow.schema import WorkflowConfig
from workflow.schema import WorkflowStage


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
EXECUTION_ORDER_CHOICES = (
    "quantization_then_pruning",
    "pruning_then_quantization",
)
VALID_STAGE_TYPES = {"quantization", "pruning"}


def _add_zero_shot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval_zero_shot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--zero_shot_tasks", nargs="+", default=list(DEFAULT_ZERO_SHOT_TASKS))
    parser.add_argument("--zero_shot_num_fewshot", type=int, default=0)
    parser.add_argument("--zero_shot_batch_size", type=int, default=1)
    parser.add_argument("--zero_shot_limit", type=int, default=None)


def build_quantization_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified fake-quant launcher.")
    parser.add_argument("--algorithm", required=True, choices=sorted(QUANTIZATION_METHOD_REGISTRY))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=str(RESULTS_ROOT / "quantization"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--calibration_dataset", default="pileval", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    _add_zero_shot_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=16)
    parser.add_argument("--query_bits", type=int, default=16)
    parser.add_argument("--key_bits", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--weight_group_size", type=int, default=None)
    parser.add_argument("--activation_group_size", type=int, default=None)
    parser.add_argument("--kv_group_size", type=int, default=None)
    parser.add_argument("--weight_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--key_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight_method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--use_activation_order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_epochs", type=int, default=15)
    parser.add_argument("--flatquant_calibration_batch_size", type=int, default=4)
    parser.add_argument("--flatquant_lr", type=float, default=1e-5)
    parser.add_argument("--flatquant_cali_trans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_add_diag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lwc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lac", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_diag_init", default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--flatquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--flatquant_warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_deactive_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_direct_inv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_separate_vtrans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_resume_from", default=None)
    parser.add_argument("--flatquant_reload_matrix_from", default=None)
    parser.add_argument("--flatquant_save_matrix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static_groups", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--awq_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation_mode", default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_checkpoint", default=None)
    parser.add_argument("--save_fake_model", action=argparse.BooleanOptionalAction, default=False)
    return parser


def build_pruning_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified pruning launcher.")
    parser.add_argument("--algorithm", required=True, choices=sorted(PRUNING_METHOD_REGISTRY))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=str(RESULTS_ROOT / "pruning"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--calibration_dataset", default="c4", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    _add_zero_shot_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--structure_pattern", default="unstructured")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--use_variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flap_metrics", default="WIFV", choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument("--flap_remove_heads", type=int, default=8)
    parser.add_argument("--pseudo_pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_pruned_model", action=argparse.BooleanOptionalAction, default=False)
    return parser


def build_workflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified quantization-pruning workflow launcher.")
    parser.add_argument("--quantization_algorithm", required=True, choices=sorted(QUANTIZATION_METHOD_REGISTRY))
    parser.add_argument("--pruning_algorithm", required=True, choices=sorted(PRUNING_METHOD_REGISTRY))
    parser.add_argument("--execution_order", required=True, choices=EXECUTION_ORDER_CHOICES)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=str(RESULTS_ROOT / "workflow"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--quantization_calibration_dataset", default="pileval", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--pruning_calibration_dataset", default="c4", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--quantization_calibration_samples", type=int, default=4)
    parser.add_argument("--pruning_calibration_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    _add_zero_shot_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=16)
    parser.add_argument("--query_bits", type=int, default=16)
    parser.add_argument("--key_bits", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--weight_group_size", type=int, default=None)
    parser.add_argument("--activation_group_size", type=int, default=None)
    parser.add_argument("--kv_group_size", type=int, default=None)
    parser.add_argument("--weight_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--key_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight_method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--quantization_damp_percent", type=float, default=0.05)
    parser.add_argument("--pruning_damp_percent", type=float, default=0.01)
    parser.add_argument("--use_activation_order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_epochs", type=int, default=15)
    parser.add_argument("--flatquant_calibration_batch_size", type=int, default=4)
    parser.add_argument("--flatquant_lr", type=float, default=1e-5)
    parser.add_argument("--flatquant_cali_trans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_add_diag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lwc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lac", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_diag_init", default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--flatquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--flatquant_warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_deactive_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_direct_inv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_separate_vtrans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_resume_from", default=None)
    parser.add_argument("--flatquant_reload_matrix_from", default=None)
    parser.add_argument("--flatquant_save_matrix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static_groups", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--awq_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation_mode", default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_checkpoint", default=None)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--structure_pattern", default="unstructured")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--use_variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flap_metrics", default="WIFV", choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument("--flap_remove_heads", type=int, default=8)
    parser.add_argument("--pseudo_pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_composed_model", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _build_quantization_run_spec(args) -> str:
    run_spec = f"{args.quantization_algorithm}_w{args.weight_bits}a{args.activation_bits}"
    if args.quantization_algorithm == "flatquant":
        run_spec += f"_q{args.query_bits}k{args.key_bits}v{args.value_bits}"
    return run_spec


def _build_pruning_run_spec(args) -> str:
    run_spec = f"{args.pruning_algorithm}_s{args.sparsity_ratio}"
    if args.pruning_algorithm == "flap":
        run_spec += f"_{args.flap_metrics}_h{args.flap_remove_heads}"
    return run_spec


def resolve_workflow_output_dir(args) -> Path:
    model_name = model_slug(args.model_path)
    quantization_run_spec = _build_quantization_run_spec(args)
    pruning_run_spec = _build_pruning_run_spec(args)
    run_spec = f"{quantization_run_spec}__{pruning_run_spec}_seq{args.sequence_length}"
    return ensure_dir(
        Path(args.output_root)
        / model_name
        / args.execution_order
        / f"{args.quantization_algorithm}__{args.pruning_algorithm}"
        / run_spec
    )


def build_quantization_config(args) -> WorkflowConfig:
    normalize_quantization_args(args)
    method = get_quantization_method(args.algorithm)
    return WorkflowConfig(
        model_path=args.model_path,
        common_args=vars(args).copy(),
        stages=[WorkflowStage(stage_type="quantization", algorithm_name=args.algorithm)],
        output_dir=method.resolve_output_dir(args),
        result_metadata={
            "algorithm_name": args.algorithm,
        },
        flatten_single_stage=True,
        save_composed_model=args.save_fake_model,
    )


def build_pruning_config(args) -> WorkflowConfig:
    method = get_pruning_method(args.algorithm)
    return WorkflowConfig(
        model_path=args.model_path,
        common_args=vars(args).copy(),
        stages=[WorkflowStage(stage_type="pruning", algorithm_name=args.algorithm)],
        output_dir=method.resolve_output_dir(args),
        result_metadata={
            "algorithm_name": args.algorithm,
            "sparsity_ratio": args.sparsity_ratio,
        },
        flatten_single_stage=True,
        save_composed_model=args.save_pruned_model,
    )


def build_workflow_config(args) -> WorkflowConfig:
    output_dir = resolve_workflow_output_dir(args)
    base_common_args = {
        "device": args.device,
        "dtype": args.dtype,
        "log_level": args.log_level,
        "hf_token": args.hf_token,
        "evaluation_dataset": args.evaluation_dataset,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "max_eval_chunks": args.max_eval_chunks,
        "eval_zero_shot": args.eval_zero_shot,
        "zero_shot_tasks": args.zero_shot_tasks,
        "zero_shot_num_fewshot": args.zero_shot_num_fewshot,
        "zero_shot_batch_size": args.zero_shot_batch_size,
        "zero_shot_limit": args.zero_shot_limit,
        "seed": args.seed,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "query_bits": args.query_bits,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "group_size": args.group_size,
        "weight_group_size": args.weight_group_size,
        "activation_group_size": args.activation_group_size,
        "kv_group_size": args.kv_group_size,
        "weight_symmetric": args.weight_symmetric,
        "activation_symmetric": args.activation_symmetric,
        "query_symmetric": args.query_symmetric,
        "key_symmetric": args.key_symmetric,
        "value_symmetric": args.value_symmetric,
        "weight_method": args.weight_method,
        "use_activation_order": args.use_activation_order,
        "flatquant_epochs": args.flatquant_epochs,
        "flatquant_calibration_batch_size": args.flatquant_calibration_batch_size,
        "flatquant_lr": args.flatquant_lr,
        "flatquant_cali_trans": args.flatquant_cali_trans,
        "flatquant_add_diag": args.flatquant_add_diag,
        "flatquant_lwc": args.flatquant_lwc,
        "flatquant_lac": args.flatquant_lac,
        "flatquant_diag_init": args.flatquant_diag_init,
        "flatquant_diag_alpha": args.flatquant_diag_alpha,
        "flatquant_warmup": args.flatquant_warmup,
        "flatquant_deactive_amp": args.flatquant_deactive_amp,
        "flatquant_direct_inv": args.flatquant_direct_inv,
        "flatquant_separate_vtrans": args.flatquant_separate_vtrans,
        "flatquant_resume_from": args.flatquant_resume_from,
        "flatquant_reload_matrix_from": args.flatquant_reload_matrix_from,
        "flatquant_save_matrix": args.flatquant_save_matrix,
        "static_groups": args.static_groups,
        "awq_search": args.awq_search,
        "rotation_mode": args.rotation_mode,
        "rotation_checkpoint": args.rotation_checkpoint,
        "sparsity_ratio": args.sparsity_ratio,
        "structure_pattern": args.structure_pattern,
        "block_size": args.block_size,
        "use_variant": args.use_variant,
        "flap_metrics": args.flap_metrics,
        "flap_remove_heads": args.flap_remove_heads,
        "pseudo_pruning": args.pseudo_pruning,
    }
    normalized_common_args = argparse.Namespace(**base_common_args)
    normalize_quantization_args(normalized_common_args)
    base_common_args = vars(normalized_common_args)
    stage_output_root = ensure_dir(output_dir / "stages")
    quantization_stage = WorkflowStage(
        stage_type="quantization",
        algorithm_name=args.quantization_algorithm,
        parameters={
            "output_root": str(stage_output_root),
            "calibration_dataset": args.quantization_calibration_dataset,
            "calibration_samples": args.quantization_calibration_samples,
            "damp_percent": args.quantization_damp_percent,
        },
    )
    pruning_stage = WorkflowStage(
        stage_type="pruning",
        algorithm_name=args.pruning_algorithm,
        parameters={
            "output_root": str(stage_output_root),
            "calibration_dataset": args.pruning_calibration_dataset,
            "calibration_samples": args.pruning_calibration_samples,
            "damp_percent": args.pruning_damp_percent,
        },
    )
    stages = [quantization_stage, pruning_stage]
    if args.execution_order == "pruning_then_quantization":
        stages = [pruning_stage, quantization_stage]
    return WorkflowConfig(
        model_path=args.model_path,
        common_args=base_common_args,
        stages=stages,
        output_dir=output_dir,
        result_metadata={
            "execution_order": args.execution_order,
            "quantization_algorithm": args.quantization_algorithm,
            "pruning_algorithm": args.pruning_algorithm,
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "query_bits": args.query_bits,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "weight_method": args.weight_method,
            "sparsity_ratio": args.sparsity_ratio,
            "structure_pattern": args.structure_pattern,
            "quantization_calibration_dataset": args.quantization_calibration_dataset,
            "pruning_calibration_dataset": args.pruning_calibration_dataset,
            "quantization_calibration_samples": args.quantization_calibration_samples,
            "pruning_calibration_samples": args.pruning_calibration_samples,
            "flap_metrics": args.flap_metrics,
            "flap_remove_heads": args.flap_remove_heads,
            "pseudo_pruning": args.pseudo_pruning,
        },
        save_composed_model=args.save_composed_model,
    )


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
