#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/openbmb/MiniCPM-V}"
DEVICE="${DEVICE:-cuda:4}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
SEED="${SEED:-42}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
WEIGHT_BITS="${WEIGHT_BITS:-8}"
ACTIVATION_BITS="${ACTIVATION_BITS:-8}"
QUERY_BITS="${QUERY_BITS:-16}"
KEY_BITS="${KEY_BITS:-16}"
VALUE_BITS="${VALUE_BITS:-16}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.85}"
SMOOTHQUANT_SAVE_ACT_SCALES="${SMOOTHQUANT_SAVE_ACT_SCALES:-true}"
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-true}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
EVAL_VLM="${EVAL_VLM:-true}"
VLM_DATASETS="${VLM_DATASETS:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
VLM_MODE="${VLM_MODE:-all}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization}}"

if [[ "$WEIGHT_BITS" != "8" || "$ACTIVATION_BITS" != "8" ]]; then
  printf 'Warning: SmoothQuant currently supports only W8A8 in MindPipe; got W%sA%s.\n' \
    "$WEIGHT_BITS" "$ACTIVATION_BITS" >&2
  exit 2
fi

CMD=(
  python "$REPO_ROOT/main.py"
  --quantization smoothquant
  --model_path "$MODEL_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --data_path "$DATA_PATH"
  --seed "$SEED"
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
  --smoothquant_alpha "$SMOOTHQUANT_ALPHA"
  --smoothquant_save_act_scales "$SMOOTHQUANT_SAVE_ACT_SCALES"
  --output_dir "$OUTPUT_DIR"
  --eval_ppl "$EVAL_PPL"
  --eval_zero_shot "$EVAL_ZERO_SHOT"
  --eval_vlm "$EVAL_VLM"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
fi

if [[ -n "${SMOOTHQUANT_ACT_SCALES_FROM:-}" ]]; then
  CMD+=(--smoothquant_act_scales_from "$SMOOTHQUANT_ACT_SCALES_FROM")
fi

if [[ "$EVAL_ZERO_SHOT" == "true" ]]; then
  read -r -a ZERO_SHOT_TASK_ARRAY <<< "$ZERO_SHOT_TASKS"
  CMD+=(
    --zero_shot_tasks "${ZERO_SHOT_TASK_ARRAY[@]}"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
  )
fi

if [[ "$EVAL_VLM" == "true" ]]; then
  read -r -a VLM_DATASET_ARRAY <<< "$VLM_DATASETS"
  CMD+=(
    --vlm_datasets "${VLM_DATASET_ARRAY[@]}"
    --vlm_mode "$VLM_MODE"
    --vlm_api_nproc "$VLM_API_NPROC"
    --vlm_pred_format "$VLM_PRED_FORMAT"
    --vlm_verbose "${VLM_VERBOSE:-false}"
    --vlm_ignore_failed "${VLM_IGNORE_FAILED:-false}"
  )
  if [[ -n "${VLM_WORK_DIR:-}" ]]; then
    CMD+=(--vlm_work_dir "$VLM_WORK_DIR")
  fi
  if [[ -n "${VLM_EVAL_KIT_ROOT:-}" ]]; then
    CMD+=(--vlm_eval_kit_root "$VLM_EVAL_KIT_ROOT")
  fi
  if [[ -n "${VLM_JUDGE:-}" ]]; then
    CMD+=(--vlm_judge "$VLM_JUDGE")
  fi
fi

if [[ -n "${NUM_SAMPLES:-}" ]]; then
  CMD+=(--num_samples "$NUM_SAMPLES")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
