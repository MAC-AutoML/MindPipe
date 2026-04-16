#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Llama-2-7b-hf}"
DEVICE="${DEVICE:-cuda:7}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
SEED="${SEED:-42}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-wikitext2}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
WEIGHT_BITS="${WEIGHT_BITS:-16}"
ACTIVATION_BITS="${ACTIVATION_BITS:-4}"
QUERY_BITS="${QUERY_BITS:-16}"
KEY_BITS="${KEY_BITS:-16}"
VALUE_BITS="${VALUE_BITS:-16}"
GROUP_SIZE="${GROUP_SIZE:--1}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
ACTIVATION_GROUP_SIZE="${ACTIVATION_GROUP_SIZE:-$GROUP_SIZE}"
OMNIQUANT_WEIGHT_SYMMETRIC="${OMNIQUANT_WEIGHT_SYMMETRIC:-false}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-$OMNIQUANT_WEIGHT_SYMMETRIC}"
ACTIVATION_SYMMETRIC="${ACTIVATION_SYMMETRIC:-false}"
OMNIQUANT_EPOCHS="${OMNIQUANT_EPOCHS:-20}"
OMNIQUANT_ALPHA="${OMNIQUANT_ALPHA:-0.75}"
OMNIQUANT_LET="${OMNIQUANT_LET:-true}"
OMNIQUANT_LWC="${OMNIQUANT_LWC:-true}"
OMNIQUANT_LET_LR="${OMNIQUANT_LET_LR:-1e-3}"
OMNIQUANT_LWC_LR="${OMNIQUANT_LWC_LR:-1e-2}"
OMNIQUANT_WEIGHT_DECAY="${OMNIQUANT_WEIGHT_DECAY:-0.0}"
OMNIQUANT_AUG_LOSS="${OMNIQUANT_AUG_LOSS:-true}"
OMNIQUANT_SAVE_ACT_STATS="${OMNIQUANT_SAVE_ACT_STATS:-true}"
OMNIQUANT_SAVE_DIAGNOSTICS="${OMNIQUANT_SAVE_DIAGNOSTICS:-false}"
OMNIQUANT_DISABLE_ZERO_POINT="${OMNIQUANT_DISABLE_ZERO_POINT:-false}"
OMNIQUANT_DEACTIVE_AMP="${OMNIQUANT_DEACTIVE_AMP:-false}"
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-true}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-/mnt/82_store/wxx/HWQuant/Mindpipe/results}}"

if [[ "$QUERY_BITS" -lt 16 || "$KEY_BITS" -lt 16 || "$VALUE_BITS" -lt 16 ]]; then
  echo "omniquant follows upstream and does not expose independent Q/K/V cache quantization; keep QUERY_BITS/KEY_BITS/VALUE_BITS at 16" >&2
  exit 1
fi

CMD=(
  python "$REPO_ROOT/main.py"
  --quantization omniquant
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
  --group_size "$GROUP_SIZE"
  --weight_group_size "$WEIGHT_GROUP_SIZE"
  --activation_group_size "$ACTIVATION_GROUP_SIZE"
  --weight_symmetric "$WEIGHT_SYMMETRIC"
  --activation_symmetric "$ACTIVATION_SYMMETRIC"
  --omniquant_weight_symmetric "$OMNIQUANT_WEIGHT_SYMMETRIC"
  --omniquant_epochs "$OMNIQUANT_EPOCHS"
  --omniquant_alpha "$OMNIQUANT_ALPHA"
  --omniquant_let "$OMNIQUANT_LET"
  --omniquant_lwc "$OMNIQUANT_LWC"
  --omniquant_let_lr "$OMNIQUANT_LET_LR"
  --omniquant_lwc_lr "$OMNIQUANT_LWC_LR"
  --omniquant_weight_decay "$OMNIQUANT_WEIGHT_DECAY"
  --omniquant_aug_loss "$OMNIQUANT_AUG_LOSS"
  --omniquant_save_act_stats "$OMNIQUANT_SAVE_ACT_STATS"
  --omniquant_save_diagnostics "$OMNIQUANT_SAVE_DIAGNOSTICS"
  --omniquant_disable_zero_point "$OMNIQUANT_DISABLE_ZERO_POINT"
  --omniquant_deactive_amp "$OMNIQUANT_DEACTIVE_AMP"
  --output_dir "$OUTPUT_DIR"
  --eval_ppl "$EVAL_PPL"
  --eval_zero_shot "$EVAL_ZERO_SHOT"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
fi

if [[ -n "${OMNIQUANT_RESUME_FROM:-}" ]]; then
  CMD+=(--omniquant_resume_from "$OMNIQUANT_RESUME_FROM")
fi

if [[ -n "${OMNIQUANT_ACT_SCALES_FROM:-}" ]]; then
  CMD+=(--omniquant_act_scales_from "$OMNIQUANT_ACT_SCALES_FROM")
fi

if [[ -n "${OMNIQUANT_ACT_SHIFTS_FROM:-}" ]]; then
  CMD+=(--omniquant_act_shifts_from "$OMNIQUANT_ACT_SHIFTS_FROM")
fi

if [[ "$EVAL_ZERO_SHOT" == "true" ]]; then
  read -r -a ZERO_SHOT_TASK_ARRAY <<< "$ZERO_SHOT_TASKS"
  CMD+=(
    --zero_shot_tasks "${ZERO_SHOT_TASK_ARRAY[@]}"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
  )
fi

if [[ -n "${NUM_SAMPLES:-}" ]]; then
  CMD+=(--num_samples "$NUM_SAMPLES")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
