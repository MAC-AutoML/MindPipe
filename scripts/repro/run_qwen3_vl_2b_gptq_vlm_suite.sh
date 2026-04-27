#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "/mnt/82_store/LLM-weights/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen3-VL-2B" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B" \
    "/mnt/82_store/zy/model/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/zy/model/Qwen3-VL-2B" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B"
  do
    if [[ -d "$candidate" ]]; then
      export MODEL_PATH="$candidate"
      break
    fi
  done
fi

export MPL_CONFIG_DIR="${MPL_CONFIG_DIR:-/tmp/mpl_qwen3_vl_2b_gptq_vlm_gpu${GPU_ID:-0}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../../new_results/quantization_suite/gptq_qwen3_vl_2b_chartqa100_vlm}"

echo "[INFO] Qwen3-VL pure GPTQ VLM suite wrapper"
echo "[INFO] current GPTQ visual coverage: visual blocks + merger/deepstack merger + llm"
echo "[INFO] model.visual.patch_embed.proj is Conv3d and remains FP in the current pure GPTQ path"

source "$SCRIPT_DIR/run_qwen2_vl_2b_gptq_vlm_suite.sh"
