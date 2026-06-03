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
    parser.add_argument("--device_map", default=None, help="device_map 传给 from_pretrained，如 'auto' 实现多卡分片")
    parser.add_argument(
        "--max_memory",
        default=None,
        help=(
            "Optional max_memory for device_map loading. Accepts JSON or "
            "comma-separated pairs such as '0:70GiB,1:70GiB,cpu:120GiB'."
        ),
    )
    parser.add_argument(
        "--offload_folder",
        default=None,
        help="Optional disk offload folder used by Hugging Face device_map loading.",
    )
    parser.add_argument(
        "--offload_state_dict",
        type=_bool_flag,
        default=None,
        help="Optional Hugging Face offload_state_dict flag for device_map loading.",
    )
    parser.add_argument(
        "--no_split_module_classes",
        nargs="+",
        default=None,
        help="Optional module class names that device_map should not split.",
    )
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
    parser.add_argument(
        "--vlm_resume",
        type=_bool_flag,
        default=False,
        help="Resume VLM evaluation from existing per-dataset artifacts in the same output_dir when possible.",
    )
    parser.add_argument("--vlm_work_dir", default=None)
    parser.add_argument("--vlm_eval_kit_root", default=DEFAULT_VLMEVALKIT_ROOT)
    parser.add_argument("--vlm_judge", default=None)
    parser.add_argument("--vlm_api_nproc", type=int, default=4)
    parser.add_argument("--vlm_verbose", type=_bool_flag, default=False)
    parser.add_argument("--vlm_ignore_failed", type=_bool_flag, default=False)
    parser.add_argument("--vlm_pred_format", default="xlsx", choices=["xlsx", "tsv", "json"])
    parser.add_argument(
        "--vlm_use_cache",
        type=_bool_flag,
        default=False,
        help="Enable generation KV cache for VLM evaluation. Can significantly speed up decoding.",
    )
    parser.add_argument(
        "--vlm_max_new_tokens",
        type=int,
        default=None,
        help="Optional override for VLM generation max_new_tokens on all datasets.",
    )
    parser.add_argument(
        "--vlm_sample_cleanup",
        type=_bool_flag,
        default=True,
        help="Run gc/empty_cache after each sample during VLM generation.",
    )
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
    parser.add_argument("--llmpruner_pruner_type", default="taylor", choices=["l2", "taylor"],
                        help="LLM-Pruner importance type: l2 (magnitude) or taylor.")
    parser.add_argument("--llmpruner_taylor", default="param_first",
                        choices=["param_first"],
                        help="LLM-Pruner Taylor expansion mode (only param_first supported currently).")
    parser.add_argument("--llmpruner_min_attention_heads", type=int, default=1,
                        help="LLM-Pruner: minimum attention groups to keep per layer.")
    parser.add_argument("--llmpruner_min_mlp_neurons", type=int, default=8,
                        help="LLM-Pruner: minimum MLP neurons to keep per layer.")


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
    parser.add_argument("--weight_clip", type=_bool_flag, default=False)
    parser.add_argument("--activation_clip_ratio", type=float, default=1.0)
    parser.add_argument("--key_clip_ratio", type=float, default=1.0)
    parser.add_argument("--value_clip_ratio", type=float, default=1.0)
    parser.add_argument("--weight_method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--use_activation_order", type=_bool_flag, default=False)
    # SmoothQuant
    parser.add_argument("--smoothquant_alpha", type=float, default=0.85)
    parser.add_argument("--smoothquant_act_scales_from", default=None)
    parser.add_argument("--smoothquant_save_act_scales", type=_bool_flag, default=True)
    # OmniQuant
    parser.add_argument("--omniquant_alpha", type=float, default=0.5)
    parser.add_argument("--omniquant_let_lr", type=float, default=5e-3)
    parser.add_argument("--omniquant_lwc_lr", type=float, default=1e-2)
    parser.add_argument("--omniquant_weight_decay", type=float, default=0.0)
    parser.add_argument("--omniquant_epochs", type=int, default=10)
    parser.add_argument("--omniquant_let", type=_bool_flag, default=False)
    parser.add_argument("--omniquant_lwc", type=_bool_flag, default=False)
    parser.add_argument("--omniquant_weight_symmetric", type=_bool_flag, default=None)
    parser.add_argument("--omniquant_aug_loss", type=_bool_flag, default=True)
    parser.add_argument("--omniquant_deactive_amp", type=_bool_flag, default=False)
    parser.add_argument("--omniquant_disable_zero_point", type=_bool_flag, default=False)
    parser.add_argument("--omniquant_save_diagnostics", type=_bool_flag, default=False)
    parser.add_argument("--omniquant_resume_from", default=None)
    parser.add_argument("--omniquant_act_scales_from", default=None)
    parser.add_argument("--omniquant_act_shifts_from", default=None)
    parser.add_argument("--omniquant_use_shift", type=_bool_flag, default=False)
    parser.add_argument("--omniquant_save_act_stats", type=_bool_flag, default=True)
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
    # SplitQuant
    parser.add_argument("--splitquant_epochs", type=int, default=15)
    parser.add_argument("--splitquant_calibration_batch_size", type=int, default=32)
    parser.add_argument("--splitquant_lr", type=float, default=5e-3)
    parser.add_argument("--splitquant_cali_trans", type=_bool_flag, default=True)
    parser.add_argument("--splitquant_add_diag", type=_bool_flag, default=True)
    parser.add_argument("--splitquant_lwc", type=_bool_flag, default=True)
    parser.add_argument("--splitquant_lac", type=_bool_flag, default=True)
    parser.add_argument("--splitquant_diag_init", default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--splitquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--splitquant_warmup", type=_bool_flag, default=False)
    parser.add_argument("--splitquant_deactive_amp", type=_bool_flag, default=True)
    parser.add_argument("--splitquant_separate_vtrans", type=_bool_flag, default=False)
    parser.add_argument("--splitquant_resume_from", default=None)
    parser.add_argument("--splitquant_reload_matrix_from", default=None)
    parser.add_argument("--splitquant_save_matrix", type=_bool_flag, default=False)
    parser.add_argument("--static_groups", type=_bool_flag, default=False)
    # QLoRA
    parser.add_argument("--qlora_train_file", default=None, help="Local supervised training file for QLoRA (.json/.jsonl/.csv/.parquet).")
    parser.add_argument("--qlora_eval_file", default=None, help="Optional local supervised evaluation file for QLoRA.")
    parser.add_argument("--qlora_eval_split_ratio", type=float, default=0.0, help="Optional holdout split ratio when only --qlora_train_file is provided.")
    parser.add_argument("--qlora_input_field", default="input", help="Input/prompt field name in the local QLoRA dataset.")
    parser.add_argument("--qlora_output_field", default="output", help="Target/response field name in the local QLoRA dataset.")
    parser.add_argument("--qlora_max_train_samples", type=int, default=None, help="Optional cap on QLoRA training samples.")
    parser.add_argument("--qlora_max_eval_samples", type=int, default=None, help="Optional cap on QLoRA eval samples.")
    parser.add_argument(
        "--qlora_plain_text_default_samples",
        type=int,
        default=1024,
        help=(
            "Default number of training samples for plain-text QLoRA when --qlora_train_file and "
            "--qlora_max_train_samples are both omitted."
        ),
    )
    parser.add_argument("--qlora_source_max_len", type=int, default=None, help="Optional source token cap for QLoRA. Defaults to sequence_length - qlora_target_max_len.")
    parser.add_argument("--qlora_target_max_len", type=int, default=256, help="Target token cap for QLoRA supervision.")
    parser.add_argument("--qlora_train_on_source", type=_bool_flag, default=False, help="Whether QLoRA should include prompt tokens in the loss.")
    parser.add_argument("--qlora_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--qlora_per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--qlora_gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--qlora_learning_rate", type=float, default=2e-4)
    parser.add_argument("--qlora_weight_decay", type=float, default=0.0)
    parser.add_argument("--qlora_num_train_epochs", type=float, default=1.0)
    parser.add_argument("--qlora_max_steps", type=int, default=-1)
    parser.add_argument("--qlora_logging_steps", type=int, default=10)
    parser.add_argument("--qlora_save_steps", type=int, default=250)
    parser.add_argument("--qlora_save_total_limit", type=int, default=2)
    parser.add_argument("--qlora_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--qlora_lr_scheduler_type", default="constant")
    parser.add_argument("--qlora_dataloader_num_workers", type=int, default=0)
    parser.add_argument("--qlora_gradient_checkpointing", type=_bool_flag, default=True)
    parser.add_argument("--qlora_lora_r", type=int, default=64)
    parser.add_argument("--qlora_lora_alpha", type=int, default=16)
    parser.add_argument("--qlora_lora_dropout", type=float, default=0.0)
    parser.add_argument("--qlora_double_quant", type=_bool_flag, default=True)
    parser.add_argument("--qlora_quant_type", default="nf4", choices=["fp4", "nf4"])
    parser.add_argument("--qlora_merge_adapter", type=_bool_flag, default=False)
    # QA-LoRA
    parser.add_argument(
        "--qalora_group_size",
        type=int,
        default=None,
        help="QA-LoRA adapter pooling group size. Defaults to --weight_group_size, then --group_size.",
    )
    # AWQ
    parser.add_argument("--awq_search", type=_bool_flag, default=True)
    parser.add_argument(
        "--awq_reuse_search_result",
        type=_bool_flag,
        default=False,
        help="Reuse an existing awq_search.pt under the target AWQ output_dir instead of rerunning AWQ search.",
    )
    parser.add_argument(
        "--awq_search_sequence_length",
        type=int,
        default=512,
        help="Calibration sequence length used by AWQ search. Kept separate from the global evaluation sequence length to match upstream AWQ defaults.",
    )
    parser.add_argument(
        "--awq_auto_scale",
        type=_bool_flag,
        default=True,
        help="Enable AWQ scale search during run_awq.",
    )
    parser.add_argument(
        "--awq_mse_range",
        type=_bool_flag,
        default=True,
        help="Enable AWQ clipping (MSE range search) during run_awq.",
    )
    parser.add_argument(
        "--awq_clip_targets",
        default="auto",
        help=(
            "AWQ clip target selection. "
            "Use 'auto' for model-specific defaults, 'none' to disable clip, 'all' for all supported linears, "
            "or a comma-separated list such as 'self_attn.v_proj,self_attn.o_proj,mlp.down_proj'."
        ),
    )
    parser.add_argument(
        "--awq_qwen3_5_quantize_linear_attn",
        type=_bool_flag,
        default=True,
        help=(
            "Enable Qwen3.5 linear_attn AWQ weight quantization. "
            "When enabled, AWQ quantizes linear_attn.in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj "
            "instead of keeping the entire linear-attention token mixer in higher precision. "
            "Set this flag to false to fall back to the old high-precision behavior."
        ),
    )
    parser.add_argument(
        "--awq_vlm_dataset_name",
        default=None,
        help=(
            "Optional VLM dataset for multimodal AWQ calibration on supported VLMs "
            "(currently Qwen2-VL / Qwen2.5-VL visual blocks and merger / connector). "
            "Falls back to --mquant_dataset_name when omitted."
        ),
    )
    parser.add_argument(
        "--awq_vlm_calib_num",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of multimodal samples used in AWQ visual / connector calibration. "
            "Falls back to --mquant_calib_num or --calibration_samples."
        ),
    )
    parser.add_argument(
        "--awq_vlm_quant_visual",
        type=_bool_flag,
        default=None,
        help="AWQ only: whether to quantize the visual encoder blocks during multimodal AWQ calibration.",
    )
    parser.add_argument(
        "--awq_vlm_quant_connector",
        type=_bool_flag,
        default=None,
        help="AWQ only: whether to quantize the visual merger / connector during multimodal AWQ calibration.",
    )
    parser.add_argument(
        "--awq_vlm_quant_llm",
        type=_bool_flag,
        default=None,
        help=(
            "AWQ only: whether to quantize the language decoder when multimodal AWQ calibration is enabled. "
            "This currently reuses the standard text AWQ path."
        ),
    )
    parser.add_argument(
        "--awq_visual_w_bits",
        type=int,
        default=None,
        help=(
            "Optional AWQ override for visual encoder weight bits during multimodal quantization. "
            "Defaults to --weight_bits when unset."
        ),
    )
    parser.add_argument(
        "--awq_connector_w_bits",
        type=int,
        default=None,
        help=(
            "Optional AWQ override for visual connector / merger weight bits during multimodal quantization. "
            "Defaults to --awq_visual_w_bits or --weight_bits when unset."
        ),
    )
    parser.add_argument(
        "--awq_llm_w_bits",
        type=int,
        default=None,
        help=(
            "Optional AWQ override for language decoder weight bits during multimodal quantization. "
            "Defaults to --weight_bits when unset."
        ),
    )
    # QuaRot / SpinQuant
    parser.add_argument("--rotation_mode", default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_checkpoint", default=None)
    parser.add_argument(
        "--quarot_k_tokenwise_per_head",
        type=_bool_flag,
        default=False,
        help="QuaRot only: when kv_group_size=-1, quantize each KV head independently per token instead of sharing one scale across all KV heads.",
    )
    parser.add_argument(
        "--quarot_k_hadamard",
        type=_bool_flag,
        default=True,
        help="QuaRot only: apply the exact Hadamard transform on the Q/K path before K-cache quantization.",
    )
    parser.add_argument(
        "--quarot_k_pre_rope",
        type=_bool_flag,
        default=False,
        help="QuaRot only: quantize the K path before RoPE instead of after RoPE.",
    )
    parser.add_argument(
        "--quarot_k_per_head_channel",
        type=_bool_flag,
        default=False,
        help="QuaRot only: quantize K with one scale per head/channel shared across the sequence chunk.",
    )
    parser.add_argument(
        "--quarot_k_equalize",
        type=_bool_flag,
        default=False,
        help="QuaRot only: collect offline K-channel stats and equalize Q/K per head/channel before K quantization.",
    )
    parser.add_argument(
        "--quarot_k_equalize_alpha",
        type=float,
        default=1.0,
        help="QuaRot only: strength of offline K-channel equalization. 0 disables the learned scaling, 1 fully equalizes K absmax per head.",
    )
    parser.add_argument(
        "--quarot_k_equalize_max_scale",
        type=float,
        default=8.0,
        help="QuaRot only: clamp equalization scale into [1/max_scale, max_scale]. Set <=0 to disable clamping.",
    )
    parser.add_argument(
        "--quarot_k_equalize_with_q",
        type=_bool_flag,
        default=False,
        help="QuaRot only: collect Q statistics and make K equalization GQA-aware instead of using K-only absmax stats.",
    )
    parser.add_argument(
        "--quarot_k_equalize_q_power",
        type=float,
        default=1.0,
        help="QuaRot only: exponent applied to Q statistics in Q-aware K equalization. Larger values preserve channels used by large-Q heads more aggressively.",
    )
    parser.add_argument(
        "--fp32_had",
        type=_bool_flag,
        default=False,
        help="QuaRot only: run online Hadamard transforms in FP32 to match upstream paper settings.",
    )
    parser.add_argument(
        "--quarot_disable_hidden_hadamard",
        type=_bool_flag,
        default=None,
        help="QuaRot only: disable language-side hidden Hadamard compensation and keep only the orthogonal basis rotation.",
    )
    parser.add_argument(
        "--quarot_qwen2_5_vl_rotate_visual_branch",
        type=_bool_flag,
        default=True,
        help="QuaRot only: rotate the Qwen2.5-VL visual encoder branch during basis change.",
    )
    parser.add_argument(
        "--quarot_qwen2_5_vl_rotate_merger_output",
        type=_bool_flag,
        default=True,
        help="QuaRot only: rotate Qwen2.5-VL visual merger output into the text hidden basis.",
    )
    parser.add_argument(
        "--quarot_qwen2_5_vl_disable_hidden_hadamard",
        type=_bool_flag,
        default=False,
        help="QuaRot only: disable Qwen2.5-VL language-side down_proj/o_proj exact-Hadamard and online Hadamard compensation for pure orthogonal rotation debugging.",
    )
    parser.add_argument(
        "--quarot_qwen2_5_vl_center_merger_output",
        type=_bool_flag,
        default=False,
        help="QuaRot only: center Qwen2.5-VL merger output in the text hidden space before rotation, mirroring token embedding mean-centering.",
    )
    parser.add_argument(
        "--quarot_qwen2_5_vl_first_layer_activation_bits",
        type=int,
        default=None,
        help="QuaRot only: override activation bits for Qwen2.5-VL language decoder layer 0 inputs. Useful for protecting the visual-to-text bridge while keeping deeper layers low-bit.",
    )
    parser.add_argument(
        "--quarot_qwen3_vl_down_proj_rtn_fallback",
        type=_bool_flag,
        default=False,
        help="QuaRot only: experiment flag for Qwen3-VL GPTQ runs. Quantize online-Hadamard down_proj with RTN instead of GPTQ.",
    )
    parser.add_argument(
        "--quarot_static_acts",
        type=_bool_flag,
        default=False,
        help=(
            "QuaRot only: collect layer-wise min/max on the text calibration dataset and "
            "use fixed input-activation scales during evaluation, similar to MQuant static activation quantization. "
            "This currently applies to decoder input activations only; KV-cache quantization remains dynamic."
        ),
    )
    # MQuant GPTQ-specific calibration (multimodal datasets)
    parser.add_argument(
        "--mquant_dataset_name",
        default=None,
        help="VLM dataset for MQuant GPTQ calibration (e.g. OCRBench/TextVQA_VAL/DocVQA_VAL/MME).",
    )
    parser.add_argument(
        "--mquant_calib_num",
        type=int,
        default=None,
        help="Optional cap on the number of VLM samples used in MQuant GPTQ calibration.",
    )
    parser.add_argument(
        "--mquant_max_new_tokens",
        type=int,
        default=20,
        help="Max new tokens per prompt when collecting MQuant GPTQ activations.",
    )
    parser.add_argument(
        "--gptq_vlm_dataset_name",
        default=None,
        help="Optional VLM dataset for multimodal GPTQ calibration (currently Qwen2-VL / Qwen2.5-VL). Falls back to --mquant_dataset_name when omitted.",
    )
    parser.add_argument(
        "--gptq_vlm_calib_num",
        type=int,
        default=None,
        help="Optional cap on the number of multimodal samples used in GPTQ calibration. Falls back to --mquant_calib_num or --calibration_samples.",
    )
    parser.add_argument(
        "--gptq_vlm_quant_visual",
        type=_bool_flag,
        default=None,
        help="GPTQ only: whether to quantize the visual encoder branch during multimodal calibration.",
    )
    parser.add_argument(
        "--gptq_vlm_quant_connector",
        type=_bool_flag,
        default=None,
        help="GPTQ only: whether to quantize the visual merger / connector branch during multimodal calibration.",
    )
    parser.add_argument(
        "--gptq_vlm_quant_llm",
        type=_bool_flag,
        default=None,
        help="GPTQ only: whether to quantize the language decoder branch during multimodal calibration.",
    )
    parser.add_argument(
        "--gptq_max_layers",
        type=int,
        default=None,
        help="Optional GPTQ text-backbone layer cap for smoke tests. Defaults to quantizing all layers.",
    )
    parser.add_argument(
        "--spinquant_vlm_dataset_name",
        default=None,
        help=(
            "Optional VLM dataset for multimodal SpinQuant calibration on supported VLMs "
            "(first pass: Qwen2-VL / Qwen2.5-VL / Qwen3-VL visual blocks, optional connector, and language decoder). "
            "Falls back to --gptq_vlm_dataset_name / --mquant_dataset_name when omitted."
        ),
    )
    parser.add_argument(
        "--spinquant_vlm_calib_num",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of multimodal samples used in SpinQuant visual / connector / llm calibration. "
            "Falls back to --gptq_vlm_calib_num / --mquant_calib_num / --calibration_samples."
        ),
    )
    parser.add_argument(
        "--spinquant_vlm_quant_visual",
        type=_bool_flag,
        default=None,
        help="SpinQuant only: whether to quantize the visual encoder blocks during multimodal calibration.",
    )
    parser.add_argument(
        "--spinquant_vlm_quant_connector",
        type=_bool_flag,
        default=None,
        help="SpinQuant only: whether to quantize the visual merger / connector during multimodal calibration.",
    )
    parser.add_argument(
        "--spinquant_vlm_quant_llm",
        type=_bool_flag,
        default=None,
        help=(
            "SpinQuant only: whether to quantize the language decoder branch during multimodal calibration. "
            "When disabled, SpinQuant can be used as a visual-only multimodal experiment."
        ),
    )
    parser.add_argument(
        "--mquant_visual_w_bits",
        type=int,
        default=None,
        help="Optional override for MQuant visual branch weight bits.",
    )
    parser.add_argument(
        "--mquant_visual_a_bits",
        type=int,
        default=None,
        help="Optional override for MQuant visual branch activation bits.",
    )
    parser.add_argument(
        "--mquant_llm_w_bits",
        type=int,
        default=None,
        help="Optional override for MQuant language branch weight bits.",
    )
    parser.add_argument(
        "--mquant_llm_a_bits",
        type=int,
        default=None,
        help="Optional override for MQuant language branch activation bits.",
    )
    parser.add_argument(
        "--mquant_visual_w_clip",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant visual branch weight clipping (MSE search).",
    )
    parser.add_argument(
        "--mquant_llm_w_clip",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant language branch weight clipping (MSE search).",
    )
    parser.add_argument(
        "--mquant_visual_static",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant visual activation static quantization.",
    )
    parser.add_argument(
        "--mquant_llm_static",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant language activation static quantization.",
    )
    parser.add_argument(
        "--mquant_weight_group_size",
        type=int,
        default=None,
        help="Optional override for MQuant weight groupsize.",
    )
    parser.add_argument(
        "--mquant_activation_group_size",
        type=int,
        default=None,
        help="Optional override for MQuant activation groupsize.",
    )
    parser.add_argument(
        "--mquant_quant_llm",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant language-branch quantization.",
    )
    parser.add_argument(
        "--mquant_quant_visual_clip",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant visual-branch quantization.",
    )
    parser.add_argument(
        "--mquant_quant_cross_attention",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant cross-attention quantization.",
    )
    parser.add_argument(
        "--mquant_not_fuse_layer_norms",
        type=_bool_flag,
        default=None,
        help="Optional override for skipping MQuant layer-norm fusion.",
    )
    parser.add_argument(
        "--mquant_no_fuse_visual_clip",
        type=_bool_flag,
        default=None,
        help="Optional override for skipping MQuant visual-branch fusion.",
    )
    parser.add_argument(
        "--mquant_no_fuse_visual_cross_attn",
        type=_bool_flag,
        default=None,
        help="Optional override for skipping MQuant visual cross-attention fusion.",
    )
    parser.add_argument(
        "--mquant_no_fuse_llm",
        type=_bool_flag,
        default=None,
        help="Optional override for skipping MQuant language-branch fusion.",
    )
    parser.add_argument(
        "--mquant_rotate",
        type=_bool_flag,
        default=None,
        help="Optional override for enabling or disabling MQuant rotation.",
    )
    parser.add_argument(
        "--mquant_rotate_visual_clip",
        type=_bool_flag,
        default=None,
        help="Optional override for visual-branch rotation.",
    )
    parser.add_argument(
        "--mquant_rotate_visual_cross_attn",
        type=_bool_flag,
        default=None,
        help="Optional override for visual cross-attention rotation.",
    )
    parser.add_argument(
        "--mquant_rotate_llm",
        type=_bool_flag,
        default=None,
        help="Optional override for language-branch rotation.",
    )
    parser.add_argument(
        "--mquant_act_per_tensor",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant per-tensor activation quantization.",
    )
    parser.add_argument(
        "--mquant_online_visual_hadamard",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant online visual Hadamard rotation.",
    )
    parser.add_argument(
        "--mquant_online_llm_hadamard",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant online language Hadamard rotation.",
    )
    parser.add_argument(
        "--mquant_fp32_had",
        type=_bool_flag,
        default=None,
        help="Optional override for running MQuant online Hadamard in FP32.",
    )
    parser.add_argument(
        "--mquant_visual_split",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant visual split mode.",
    )
    parser.add_argument(
        "--mquant_llm_split",
        type=_bool_flag,
        default=None,
        help="Optional override for MQuant language split mode.",
    )
    parser.add_argument(
        "--mquant_skip_names",
        nargs="+",
        default=None,
        help="Optional list of MQuant layer-name patterns to skip.",
    )
    parser.add_argument(
        "--mquant_act_skip_names",
        nargs="+",
        default=None,
        help="Optional list of MQuant activation-only layer-name patterns to skip.",
    )


def _add_workflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution_order", default="pruning_then_quantization", choices=EXECUTION_ORDER_CHOICES)
    # 单独指定时覆盖公共的 calibration 参数
    parser.add_argument("--pruning_calibration_dataset", default=None, choices=["wikitext2", "c4", "pileval", "pg19", "bookcorpus"])
    parser.add_argument("--quantization_calibration_dataset", default=None, choices=["wikitext2", "c4", "pileval", "pg19", "bookcorpus"])
    parser.add_argument("--pruning_calibration_samples", type=int, default=None)
    parser.add_argument("--quantization_calibration_samples", type=int, default=None)
    parser.add_argument("--pruning_damp_percent", type=float, default=None)
    parser.add_argument("--quantization_damp_percent", type=float, default=None)


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration_dataset", default=None, choices=["wikitext2", "c4", "pileval", "pg19", "bookcorpus"],
                        help="Calibration dataset. Each pruning/quantization method has its own default (e.g. shortgpt→pg19, flap→wikitext2).")
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--save_model", type=_bool_flag, default=False)
    parser.add_argument(
        "--export_real_quant",
        type=_bool_flag,
        default=False,
        help="Export a backend-loadable real quantized model after evaluation. MVP supports GPTQ W4A16 to vLLM.",
    )
    parser.add_argument(
        "--export_backend",
        default="vllm",
        choices=["vllm"],
        help="Backend format for --export_real_quant.",
    )
    parser.add_argument(
        "--export_quantized_model_dir",
        default=None,
        help="Output directory for the real quantized model. Defaults to <run_output>/real_quant_vllm_model.",
    )


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

    # n:m 半结构化剪枝仅对 wanda / sparsegpt / alps 生效，仅支持 2:4 和 4:8，且稀疏率必须为 0.5（与原 repo 一致）
    _nm_methods = {"wanda", "sparsegpt", "alps"}
    _valid_nm_patterns = {"2:4", "4:8"}
    if has_pruning and args.pruning in _nm_methods and args.structure_pattern != "unstructured":
        if args.structure_pattern not in _valid_nm_patterns:
            raise ValueError(
                f"不支持的 n:m 模式: {args.structure_pattern}，"
                f"仅支持 {', '.join(sorted(_valid_nm_patterns))}"
            )
        if args.sparsity_ratio != 0.5:
            raise ValueError(
                f"n:m 半结构化剪枝 ({args.structure_pattern}) 的稀疏率必须为 0.5，"
                f"当前为 {args.sparsity_ratio}"
            )

    if has_quantization and args.quantization in {"qlora", "qalora"}:
        if args.eval_vlm:
            raise ValueError(f"{args.quantization} v1 is text-only; set --eval_vlm false.")
        if args.weight_bits not in {2, 3, 4}:
            raise ValueError(f"{args.quantization} currently supports --weight_bits in {{2, 3, 4}}.")

    if args.export_real_quant:
        if not has_quantization or args.quantization != "gptq":
            raise ValueError("--export_real_quant MVP currently requires --quantization gptq.")
        if has_pruning:
            raise ValueError("--export_real_quant MVP currently supports single-stage GPTQ only.")
        if args.weight_bits != 4 or args.activation_bits != 16:
            raise ValueError("--export_real_quant MVP requires W4A16: --weight_bits 4 --activation_bits 16.")
        if not args.weight_symmetric:
            raise ValueError("--export_real_quant MVP requires --weight_symmetric true.")
        if args.use_activation_order:
            raise ValueError("--export_real_quant MVP does not support --use_activation_order true yet.")

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
