"""Unified run command: parser + config builder."""

from __future__ import annotations

import argparse
import copy
import os
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

def _bool_flag(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use one of true/false, 1/0, yes/no."
    )

EXECUTION_ORDER_CHOICES = (
    "pruning_then_quantization",
    "quantization_then_pruning",
)
VALID_STAGE_TYPES = {"quantization", "pruning"}
DEFAULT_VLMEVALKIT_ROOT = os.environ.get(
    "VLMEVALKIT_ROOT",
    str(REPO_ROOT / "third_party" / "VLMEvalKit"),
)


# ── 参数分组 ──

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT / "run"))
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    parser.add_argument("--data_path", default=str(Path("/mnt/42_store/lcw/data2/Huawei/datasets")))


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval_ppl", type=_bool_flag, default=False)
    parser.add_argument("--eval_zero_shot", type=_bool_flag, default=False)
    parser.add_argument("--zero_shot_tasks", nargs="+", default=list(DEFAULT_ZERO_SHOT_TASKS))
    parser.add_argument("--zero_shot_num_fewshot", type=int, default=0)
    parser.add_argument("--zero_shot_batch_size", type=int, default=1)


def _add_vlm_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval_vlm", type=_bool_flag, default=False)
    parser.add_argument("--vlm_datasets", nargs="+", default=[])
    parser.add_argument("--vlm_mode", default="all", choices=["all", "infer", "eval"])
    parser.add_argument("--vlm_work_dir", default=None)
    parser.add_argument("--vlm_eval_kit_root", default=DEFAULT_VLMEVALKIT_ROOT)
    parser.add_argument("--vlm_judge", default=None)
    parser.add_argument("--vlm_api_nproc", type=int, default=4)
    parser.add_argument("--vlm_verbose", type=_bool_flag, default=False)
    parser.add_argument("--vlm_ignore_failed", type=_bool_flag, default=False)
    parser.add_argument("--vlm_pred_format", default="xlsx", choices=["xlsx", "tsv", "json"])
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of evaluation samples (applies to both zero_shot and vlm_eval)")


def _add_pruning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pruning", default=None, choices=sorted(PRUNING_METHOD_REGISTRY))
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--structure_pattern",default="unstructured",help="剪枝结构模式。当前仅对 wanda / sparsegpt / alps 生效，用于指定 n:m 半结构化剪枝；其他方法会忽略该参数。",)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--use_variant", type=_bool_flag, default=False)
    parser.add_argument("--flap_metrics", default="WIFV", choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument("--flap_remove_heads", type=int, default=8)
    parser.add_argument("--pseudo_pruning", type=_bool_flag, default=True)
    parser.add_argument("--rho", type=float, default=0.1, help="Initial rho for ALPS ADMM optimization.")


def _add_quantization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quantization", default=None, choices=sorted(QUANTIZATION_METHOD_REGISTRY))
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=16)
    parser.add_argument("--query_bits", type=int, default=16)
    parser.add_argument("--key_bits", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--weight_group_size", type=int, default=None)
    parser.add_argument("--activation_group_size", type=int, default=None)
    parser.add_argument("--kv_group_size", type=int, default=None)
    parser.add_argument("--weight_symmetric", type=_bool_flag, default=True)
    parser.add_argument("--activation_symmetric", type=_bool_flag, default=True)
    parser.add_argument("--query_symmetric", type=_bool_flag, default=True)
    parser.add_argument("--key_symmetric", type=_bool_flag, default=True)
    parser.add_argument("--value_symmetric", type=_bool_flag, default=True)
    parser.add_argument("--weight_method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--use_activation_order", type=_bool_flag, default=False)
    # FlatQuant
    parser.add_argument("--flatquant_epochs", type=int, default=15)
    parser.add_argument("--flatquant_calibration_batch_size", type=int, default=4)
    parser.add_argument("--flatquant_lr", type=float, default=1e-5)
    parser.add_argument("--flatquant_cali_trans", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_add_diag", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_lwc", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_lac", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_diag_init", default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--flatquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--flatquant_warmup", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_deactive_amp", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_direct_inv", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_separate_vtrans", type=_bool_flag, default=False)
    parser.add_argument("--flatquant_resume_from", default=None)
    parser.add_argument("--flatquant_reload_matrix_from", default=None)
    parser.add_argument("--flatquant_save_matrix", type=_bool_flag, default=False)
    parser.add_argument("--static_groups", type=_bool_flag, default=False)
    # AWQ
    parser.add_argument("--awq_search", type=_bool_flag, default=True)
    # QuaRot / SpinQuant
    parser.add_argument("--rotation_mode", default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_checkpoint", default=None)


def _add_workflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution_order", default="pruning_then_quantization", choices=EXECUTION_ORDER_CHOICES)
    # 单独指定时覆盖公共的 calibration 参数
    parser.add_argument("--pruning_calibration_dataset", default=None, choices=["wikitext2", "c4", "pileval", "pg19"])
    parser.add_argument("--quantization_calibration_dataset", default=None, choices=["wikitext2", "c4", "pileval", "pg19"])
    parser.add_argument("--pruning_calibration_samples", type=int, default=None)
    parser.add_argument("--quantization_calibration_samples", type=int, default=None)
    parser.add_argument("--pruning_damp_percent", type=float, default=None)
    parser.add_argument("--quantization_damp_percent", type=float, default=None)


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration_dataset", default=None, choices=["wikitext2", "c4", "pileval", "pg19"],
                        help="Calibration dataset. Each pruning/quantization method has its own default (e.g. shortgpt→pg19, flap→wikitext2).")
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--save_model", type=_bool_flag, default=False)


# ── 唯一 parser ──

def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MindPipe unified runner.")
    _add_common_args(parser)
    _add_io_args(parser)
    _add_eval_args(parser)
    _add_vlm_eval_args(parser)
    _add_pruning_args(parser)
    _add_quantization_args(parser)
    _add_workflow_args(parser)
    return parser


# ── config builder ──

def build_run_config(args) -> WorkflowConfig:
    has_pruning = args.pruning is not None
    has_quantization = args.quantization is not None

    # 校验
    if not has_pruning and not has_quantization and args.eval_ppl is False and args.eval_zero_shot is False and args.eval_vlm is False:
        raise ValueError("At least one of --pruning, --quantization, or an evaluation flag must be specified.")

    model_name = model_slug(args.model_path)
    base_common_args = vars(args).copy()

    stages: list[WorkflowStage] = []
    result_metadata: dict = {
        "model_path": args.model_path,
    }

    if has_pruning:
        pruning_method = get_pruning_method(args.pruning)
        prune_calib = args.pruning_calibration_dataset or args.calibration_dataset or pruning_method.default_calibration_dataset
        prune_samples = args.pruning_calibration_samples or args.calibration_samples
        prune_damp = args.pruning_damp_percent if args.pruning_damp_percent is not None else args.damp_percent
        pruning_args_dict = copy.deepcopy(base_common_args)
        pruning_args_dict.update({
            "model_path": args.model_path,
            "algorithm": args.pruning,
            "calibration_dataset": prune_calib,
            "calibration_samples": prune_samples,
            "damp_percent": prune_damp,
            "output_root": args.output_dir,
        })
        pruning_ns = argparse.Namespace(**pruning_args_dict)

        stages.append(WorkflowStage(
            stage_type="pruning",
            algorithm_name=args.pruning,
            parameters={
                "output_root": args.output_dir,
                "calibration_dataset": prune_calib,
                "calibration_samples": prune_samples,
                "damp_percent": prune_damp,
            },
        ))
        result_metadata["pruning_algorithm"] = args.pruning
        result_metadata["sparsity_ratio"] = args.sparsity_ratio

    if has_quantization:
        quant_method = get_quantization_method(args.quantization)
        quant_calib = args.quantization_calibration_dataset or args.calibration_dataset or quant_method.default_calibration_dataset
        quant_samples = args.quantization_calibration_samples or args.calibration_samples
        quant_damp = args.quantization_damp_percent if args.quantization_damp_percent is not None else args.damp_percent

        stages.append(WorkflowStage(
            stage_type="quantization",
            algorithm_name=args.quantization,
            parameters={
                "output_root": args.output_dir,
                "calibration_dataset": quant_calib,
                "calibration_samples": quant_samples,
                "damp_percent": quant_damp,
            },
        ))
        result_metadata["quantization_algorithm"] = args.quantization
        result_metadata["weight_bits"] = args.weight_bits
        result_metadata["activation_bits"] = args.activation_bits

    # 执行顺序
    if has_pruning and has_quantization:
        if args.execution_order == "quantization_then_pruning":
            stages.reverse()
        result_metadata["execution_order"] = args.execution_order

    # 确定 output_dir
    if len(stages) == 1:
        # 单阶段：使用算法自己的 resolve_output_dir
        first_method = (get_pruning_method if stages[0].stage_type == "pruning" else get_quantization_method)(stages[0].algorithm_name)
        first_stage_args = argparse.Namespace(**{**base_common_args, **stages[0].parameters, "model_path": args.model_path})
        output_dir = first_method.resolve_output_dir(first_stage_args)
    elif len(stages) > 1:
        # 多阶段：构建组合目录 <output_dir>/<model>/<execution_order>/<algo1>__<algo2>/<run_spec>
        algo_parts = "__".join(s.algorithm_name for s in stages)
        run_spec_parts = []
        for s in stages:
            if s.stage_type == "pruning":
                run_spec_parts.append(f"{s.algorithm_name}_s{args.sparsity_ratio}")
            else:
                run_spec_parts.append(f"{s.algorithm_name}_w{args.weight_bits}a{args.activation_bits}")
        run_spec = "_".join(run_spec_parts) + f"_seq{args.sequence_length}"
        output_dir = ensure_dir(Path(args.output_dir) / model_name / args.execution_order / algo_parts / run_spec)
    else:
        # 仅评测
        output_dir = ensure_dir(Path(args.output_dir) / model_name / "evaluate")

    flatten = len(stages) == 1

    return WorkflowConfig(
        model_path=args.model_path,
        common_args=base_common_args,
        stages=stages,
        output_dir=output_dir,
        result_metadata=result_metadata,
        flatten_single_stage=flatten,
        save_model=args.save_model,
    )


def validate_workflow_config(config: WorkflowConfig) -> None:
    if not config.stages and config.output_dir is None:
        raise ValueError("Evaluate-only mode requires --output_dir")
    if not config.flatten_single_stage and len(config.stages) > 1 and config.output_dir is None:
        raise ValueError("Multi-stage workflow requires an explicit output_dir")
    for stage in config.stages:
        if stage.stage_type not in VALID_STAGE_TYPES:
            raise ValueError(f"Unsupported stage_type: {stage.stage_type}")
        if stage.stage_type == "quantization" and stage.algorithm_name not in QUANTIZATION_METHOD_REGISTRY:
            raise ValueError(f"Unknown quantization algorithm: {stage.algorithm_name}")
        if stage.stage_type == "pruning" and stage.algorithm_name not in PRUNING_METHOD_REGISTRY:
            raise ValueError(f"Unknown pruning algorithm: {stage.algorithm_name}")
