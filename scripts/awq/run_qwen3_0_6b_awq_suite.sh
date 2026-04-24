#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_awq_suite_common.sh"

MODEL_NAME="Qwen3-0.6B"
MODEL_TAG="qwen3_0_6b"
MODEL_DISPLAY_NAME="$MODEL_NAME"
GPU_ID="${GPU_ID:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/awq_qwen3_0_6b}"
REQUIRED_MODULES_STR="${REQUIRED_MODULES_STR:-torch transformers lm_eval}"

MODEL_PATH_CANDIDATES=(
  "/mnt/82_store/LLM-weights/Qwen3-0.6B"
  "/mnt/82_store/LLM-weights/Qwen/Qwen3-0.6B"
  "/mnt/82_store/zy/model/Qwen3-0.6B"
  "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-0.6B"
)

mindpipe_awq_pick_model_path "${MODEL_PATH_CANDIDATES[@]}"
mindpipe_awq_main
