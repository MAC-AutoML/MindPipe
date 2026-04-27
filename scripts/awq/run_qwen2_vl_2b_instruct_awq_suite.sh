#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_awq_suite_common.sh"

MODEL_NAME="Qwen2-VL-2B-Instruct"
MODEL_TAG="qwen2_vl_2b_instruct"
MODEL_DISPLAY_NAME="$MODEL_NAME"
GPU_ID="${GPU_ID:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/awq_qwen2_vl_2b_instruct}"
REQUIRED_MODULES_STR="${REQUIRED_MODULES_STR:-torch transformers lm_eval qwen_vl_utils}"

MODEL_PATH_CANDIDATES=(
  "/mnt/82_store/LLM-weights/Qwen2-VL-2B-Instruct"
  "/mnt/82_store/LLM-weights/Qwen/Qwen2-VL-2B-Instruct"
  "/mnt/82_store/LLM-weights/qwen-2-vl-2b-instruct"
  "/mnt/82_store/zy/model/Qwen2-VL-2B-Instruct"
  "/mnt/82_store/huggingface/datasets/Qwen/Qwen2-VL-2B-Instruct"
)

mindpipe_awq_pick_model_path "${MODEL_PATH_CANDIDATES[@]}"
mindpipe_awq_main
