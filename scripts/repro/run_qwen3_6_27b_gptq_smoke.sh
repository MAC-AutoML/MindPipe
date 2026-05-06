#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen3.6-27B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/gptq_qwen3_6_27b_smoke}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"

PRIMARY_GPUS="${PRIMARY_GPUS:-1}"
FALLBACK_GPUS="${FALLBACK_GPUS:-1,3}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-1}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
GPTQ_MAX_LAYERS="${GPTQ_MAX_LAYERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SEED="${SEED:-0}"

run_once() {
  local visible_gpus="$1"
  local run_name="$2"
  local run_output="$OUTPUT_ROOT/$run_name"
  local log_path="$run_output/run.log"
  mkdir -p "$run_output"

  local -a cmd=(
    env "CUDA_VISIBLE_DEVICES=$visible_gpus" "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --quantization gptq
    --model_path "$MODEL_PATH"
    --device "$DEVICE"
    --device_map "$DEVICE_MAP"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --calibration_dataset "$CALIBRATION_DATASET"
    --evaluation_dataset "$EVALUATION_DATASET"
    --calibration_samples "$CALIBRATION_SAMPLES"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$WEIGHT_BITS"
    --activation_bits 16
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --eval_ppl true
    --eval_zero_shot false
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )

  if [[ -n "$GPTQ_MAX_LAYERS" ]]; then
    cmd+=(--gptq_max_layers "$GPTQ_MAX_LAYERS")
  fi

  printf '[INFO] Running on CUDA_VISIBLE_DEVICES=%s\n' "$visible_gpus"
  printf '[INFO] Log: %s\n' "$log_path"
  "${cmd[@]}" >"$log_path" 2>&1
}

mkdir -p "$OUTPUT_ROOT"

if run_once "$PRIMARY_GPUS" "single_gpu"; then
  echo "[INFO] GPTQ smoke succeeded on $PRIMARY_GPUS"
  exit 0
fi

if grep -Eiq 'out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$OUTPUT_ROOT/single_gpu/run.log"; then
  echo "[WARN] Single-GPU run hit OOM; retrying on $FALLBACK_GPUS"
  run_once "$FALLBACK_GPUS" "dual_gpu"
  echo "[INFO] GPTQ smoke succeeded on $FALLBACK_GPUS"
  exit 0
fi

echo "[ERROR] Single-GPU run failed for a non-OOM reason. See $OUTPUT_ROOT/single_gpu/run.log"
exit 1
