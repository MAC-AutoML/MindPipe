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
STRUCTURE_PATTERN="${STRUCTURE_PATTERN:-AL-AM}"
FLAP_METRICS="${FLAP_METRICS:-WIFV}"
FLAP_REMOVE_HEADS="${FLAP_REMOVE_HEADS:-8}"
PSEUDO_PRUNING="${PSEUDO_PRUNING:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/pruning}"

CMD=(
  python "$REPO_ROOT/main.py"
  --pruning flap
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
  --flap_metrics "$FLAP_METRICS"
  --flap_remove_heads "$FLAP_REMOVE_HEADS"
  --output_dir "$OUTPUT_DIR"
)

CMD+=(--pseudo_pruning "$PSEUDO_PRUNING")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
