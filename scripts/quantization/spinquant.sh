#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-4}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
QUERY_BITS="${QUERY_BITS:-16}"
KEY_BITS="${KEY_BITS:-16}"
VALUE_BITS="${VALUE_BITS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_METHOD="${WEIGHT_METHOD:-gptq}"
ROTATION_MODE="${ROTATION_MODE:-hadamard}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/quantization}"

CMD=(
  python "$REPO_ROOT/main.py"
  quantization
  --algorithm spinquant
  --model_path "$MODEL_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --calibration_dataset "$CALIBRATION_DATASET"
  --evaluation_dataset "$EVALUATION_DATASET"
  --calibration_samples "$CALIBRATION_SAMPLES"
  --sequence_length "$SEQUENCE_LENGTH"
  --batch_size "$BATCH_SIZE"
  --max_eval_chunks "$MAX_EVAL_CHUNKS"
  --weight_bits "$WEIGHT_BITS"
  --activation_bits "$ACTIVATION_BITS"
  --query_bits "$QUERY_BITS"
  --key_bits "$KEY_BITS"
  --value_bits "$VALUE_BITS"
  --group_size "$GROUP_SIZE"
  --weight_method "$WEIGHT_METHOD"
  --rotation_mode "$ROTATION_MODE"
  --output_root "$OUTPUT_ROOT"
)

if [[ -n "${ROTATION_CHECKPOINT:-}" ]]; then
  CMD+=(--rotation_checkpoint "$ROTATION_CHECKPOINT")
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
