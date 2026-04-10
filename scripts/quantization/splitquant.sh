#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALIBRATION_BATCH_SIZE="${FLATQUANT_CALIBRATION_BATCH_SIZE:-32}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
FLATQUANT_DIAG_INIT="${FLATQUANT_DIAG_INIT:-sq_style}"
FLATQUANT_DIAG_ALPHA="${FLATQUANT_DIAG_ALPHA:-0.3}"
FLATQUANT_WARMUP="${FLATQUANT_WARMUP:-false}"
FLATQUANT_DEACTIVE_AMP="${FLATQUANT_DEACTIVE_AMP:-true}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-4}"
QUERY_BITS="${QUERY_BITS:-16}"
KEY_BITS="${KEY_BITS:-4}"
VALUE_BITS="${VALUE_BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
ACTIVATION_GROUP_SIZE="${ACTIVATION_GROUP_SIZE:-$GROUP_SIZE}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-128}"
WEIGHT_METHOD="${WEIGHT_METHOD:-rtn}"
DAMP_PERCENT="${DAMP_PERCENT:-0.01}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-true}"
ACTIVATION_SYMMETRIC="${ACTIVATION_SYMMETRIC:-true}"
QUERY_SYMMETRIC="${QUERY_SYMMETRIC:-true}"
KEY_SYMMETRIC="${KEY_SYMMETRIC:-false}"
VALUE_SYMMETRIC="${VALUE_SYMMETRIC:-false}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-true}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-true}"
FLATQUANT_LWC="${FLATQUANT_LWC:-true}"
FLATQUANT_LAC="${FLATQUANT_LAC:-true}"
FLATQUANT_DIRECT_INV="${FLATQUANT_DIRECT_INV:-false}"
FLATQUANT_SEPARATE_VTRANS="${FLATQUANT_SEPARATE_VTRANS:-false}"
FLATQUANT_SAVE_MATRIX="${FLATQUANT_SAVE_MATRIX:-false}"
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-true}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
EVAL_VLM="${EVAL_VLM:-false}"
VLM_DATASETS="${VLM_DATASETS:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
VLM_MODE="${VLM_MODE:-all}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization}}"

if [[ "$WEIGHT_BITS" -lt 16 && "$WEIGHT_GROUP_SIZE" -le 0 ]]; then
  echo "splitquant requires WEIGHT_GROUP_SIZE>0 when WEIGHT_BITS<16" >&2
  exit 1
fi

if [[ "$ACTIVATION_BITS" -lt 16 && "$ACTIVATION_GROUP_SIZE" -le 0 ]]; then
  echo "splitquant requires ACTIVATION_GROUP_SIZE>0 when ACTIVATION_BITS<16" >&2
  exit 1
fi

if [[ "$WEIGHT_BITS" -lt 16 && "$ACTIVATION_BITS" -lt 16 && "$WEIGHT_GROUP_SIZE" != "$ACTIVATION_GROUP_SIZE" ]]; then
  echo "splitquant requires WEIGHT_GROUP_SIZE and ACTIVATION_GROUP_SIZE to match when both are quantized" >&2
  exit 1
fi

CMD=(
  python "$REPO_ROOT/main.py"
  --quantization splitquant
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
  --flatquant_epochs "$FLATQUANT_EPOCHS"
  --flatquant_calibration_batch_size "$FLATQUANT_CALIBRATION_BATCH_SIZE"
  --flatquant_lr "$FLATQUANT_LR"
  --flatquant_diag_init "$FLATQUANT_DIAG_INIT"
  --flatquant_diag_alpha "$FLATQUANT_DIAG_ALPHA"
  --flatquant_warmup "$FLATQUANT_WARMUP"
  --flatquant_deactive_amp "$FLATQUANT_DEACTIVE_AMP"
  --batch_size "$BATCH_SIZE"
  --max_eval_chunks "$MAX_EVAL_CHUNKS"
  --weight_bits "$WEIGHT_BITS"
  --activation_bits "$ACTIVATION_BITS"
  --query_bits "$QUERY_BITS"
  --key_bits "$KEY_BITS"
  --value_bits "$VALUE_BITS"
  --weight_group_size "$WEIGHT_GROUP_SIZE"
  --activation_group_size "$ACTIVATION_GROUP_SIZE"
  --kv_group_size "$KV_GROUP_SIZE"
  --weight_method "$WEIGHT_METHOD"
  --damp_percent "$DAMP_PERCENT"
  --weight_symmetric "$WEIGHT_SYMMETRIC"
  --activation_symmetric "$ACTIVATION_SYMMETRIC"
  --query_symmetric "$QUERY_SYMMETRIC"
  --key_symmetric "$KEY_SYMMETRIC"
  --value_symmetric "$VALUE_SYMMETRIC"
  --flatquant_cali_trans "$FLATQUANT_CALI_TRANS"
  --flatquant_add_diag "$FLATQUANT_ADD_DIAG"
  --flatquant_lwc "$FLATQUANT_LWC"
  --flatquant_lac "$FLATQUANT_LAC"
  --flatquant_direct_inv "$FLATQUANT_DIRECT_INV"
  --flatquant_separate_vtrans "$FLATQUANT_SEPARATE_VTRANS"
  --flatquant_save_matrix "$FLATQUANT_SAVE_MATRIX"
  --output_dir "$OUTPUT_DIR"
  --eval_ppl "$EVAL_PPL"
  --eval_zero_shot "$EVAL_ZERO_SHOT"
  --eval_vlm "$EVAL_VLM"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
fi

if [[ -n "${FLATQUANT_RESUME_FROM:-}" ]]; then
  CMD+=(--flatquant_resume_from "$FLATQUANT_RESUME_FROM")
fi

if [[ -n "${FLATQUANT_RELOAD_MATRIX_FROM:-}" ]]; then
  CMD+=(--flatquant_reload_matrix_from "$FLATQUANT_RELOAD_MATRIX_FROM")
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
