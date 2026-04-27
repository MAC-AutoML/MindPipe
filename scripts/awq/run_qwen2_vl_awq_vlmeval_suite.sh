#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_awq_vlm_suite_common.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/awq_qwen2_vl_vlmeval}"
REQUIRED_MODULES_STR="${REQUIRED_MODULES_STR:-torch transformers qwen_vl_utils}"

MODEL_PATH_CANDIDATES=(
  "/mnt/82_store/LLM-weights/Qwen2-VL-7B-Instruct"
  "/mnt/82_store/LLM-weights/Qwen2-VL-2B-Instruct"
  "/mnt/82_store/zy/model/Qwen2-VL-7B-Instruct"
  "/mnt/82_store/zy/model/Qwen2-VL-2B-Instruct"
  "/mnt/82_store/huggingface/datasets/Qwen/Qwen2-VL-7B-Instruct"
  "/mnt/82_store/huggingface/datasets/Qwen/Qwen2-VL-2B-Instruct"
)

mindpipe_awq_vlm_pick_model_path "${MODEL_PATH_CANDIDATES[@]}"
mindpipe_awq_vlm_main
