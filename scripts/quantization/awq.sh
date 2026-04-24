#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# openbmb/MiniCPM-V
# /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/openbmb/MiniCPM-V}"
DEVICE="${DEVICE:-cuda:6}"
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
WEIGHT_BITS="${WEIGHT_BITS:-2}"
GROUP_SIZE="${GROUP_SIZE:-128}"
AWQ_SEARCH="${AWQ_SEARCH:-true}"
AWQ_SEARCH_SEQUENCE_LENGTH="${AWQ_SEARCH_SEQUENCE_LENGTH:-512}"
AWQ_AUTO_SCALE="${AWQ_AUTO_SCALE:-true}"
AWQ_MSE_RANGE="${AWQ_MSE_RANGE:-true}"
AWQ_CLIP_TARGETS="${AWQ_CLIP_TARGETS:-auto}"
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-false}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
EVAL_VLM="${EVAL_VLM:-true}"
# SEEDBench_IMG
VLM_DATASETS="${VLM_DATASETS:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
VLM_MODE="${VLM_MODE:-all}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization}}"

CMD=(
  python "$REPO_ROOT/main.py"
  --quantization awq
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
  --group_size "$GROUP_SIZE"
  --output_dir "$OUTPUT_DIR"
  --weight_symmetric false
  --awq_search "$AWQ_SEARCH"
  --awq_search_sequence_length "$AWQ_SEARCH_SEQUENCE_LENGTH"
  --awq_auto_scale "$AWQ_AUTO_SCALE"
  --awq_mse_range "$AWQ_MSE_RANGE"
  --awq_clip_targets "$AWQ_CLIP_TARGETS"
  --eval_ppl "$EVAL_PPL"
  --eval_zero_shot "$EVAL_ZERO_SHOT"
  --eval_vlm "$EVAL_VLM"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
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
