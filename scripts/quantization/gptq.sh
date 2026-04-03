#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda:2}"
DTYPE="${DTYPE:-float16}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
WEIGHT_BITS="${WEIGHT_BITS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
DAMP_PERCENT="${DAMP_PERCENT:-0.05}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/my_results/quantization}"

CMD=(
  python "$REPO_ROOT/main.py"
  quantization
  --algorithm gptq
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
  --group_size "$GROUP_SIZE"
  --damp_percent "$DAMP_PERCENT"
  --output_root "$OUTPUT_ROOT"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
