#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-c4}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
STRUCTURE_PATTERN="${STRUCTURE_PATTERN:-unstructured}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
DAMP_PERCENT="${DAMP_PERCENT:-0.01}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/pruning}"

CMD=(
  python "$REPO_ROOT/main.py"
  pruning
  --algorithm sparsegpt
  --model_path "$MODEL_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --calibration_dataset "$CALIBRATION_DATASET"
  --evaluation_dataset "$EVALUATION_DATASET"
  --calibration_samples "$CALIBRATION_SAMPLES"
  --sequence_length "$SEQUENCE_LENGTH"
  --batch_size "$BATCH_SIZE"
  --max_eval_chunks "$MAX_EVAL_CHUNKS"
  --sparsity_ratio "$SPARSITY_RATIO"
  --structure_pattern "$STRUCTURE_PATTERN"
  --block_size "$BLOCK_SIZE"
  --damp_percent "$DAMP_PERCENT"
  --output_root "$OUTPUT_ROOT"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
