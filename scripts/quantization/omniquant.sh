#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/modelzoo/Qwen/Qwen2.5-VL-7B-Instruct}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/home/ma-user/work/data}"
SEED="${SEED:-42}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
QUERY_BITS="${QUERY_BITS:-16}"
KEY_BITS="${KEY_BITS:-16}"
VALUE_BITS="${VALUE_BITS:-16}"
GROUP_SIZE="${GROUP_SIZE:--1}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
ACTIVATION_GROUP_SIZE="${ACTIVATION_GROUP_SIZE:-$GROUP_SIZE}"
OMNIQUANT_WEIGHT_SYMMETRIC="${OMNIQUANT_WEIGHT_SYMMETRIC:-false}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-$OMNIQUANT_WEIGHT_SYMMETRIC}"
ACTIVATION_SYMMETRIC="${ACTIVATION_SYMMETRIC:-false}"
OMNIQUANT_EPOCHS_SET="${OMNIQUANT_EPOCHS+x}"
OMNIQUANT_ALPHA_SET="${OMNIQUANT_ALPHA+x}"
OMNIQUANT_LET_SET="${OMNIQUANT_LET+x}"
OMNIQUANT_LWC_SET="${OMNIQUANT_LWC+x}"
OMNIQUANT_LET_LR_SET="${OMNIQUANT_LET_LR+x}"
OMNIQUANT_LWC_LR_SET="${OMNIQUANT_LWC_LR+x}"
OMNIQUANT_AUG_LOSS_SET="${OMNIQUANT_AUG_LOSS+x}"
OMNIQUANT_USE_SHIFT_SET="${OMNIQUANT_USE_SHIFT+x}"
OMNIQUANT_DEACTIVE_AMP_SET="${OMNIQUANT_DEACTIVE_AMP+x}"
OMNIQUANT_EPOCHS="${OMNIQUANT_EPOCHS:-}"
OMNIQUANT_ALPHA="${OMNIQUANT_ALPHA:-}"
OMNIQUANT_LET="${OMNIQUANT_LET:-}"
OMNIQUANT_LWC="${OMNIQUANT_LWC:-}"
OMNIQUANT_LET_LR="${OMNIQUANT_LET_LR:-}"
OMNIQUANT_LWC_LR="${OMNIQUANT_LWC_LR:-}"
OMNIQUANT_WEIGHT_DECAY="${OMNIQUANT_WEIGHT_DECAY:-0.0}"
OMNIQUANT_AUG_LOSS="${OMNIQUANT_AUG_LOSS:-}"
OMNIQUANT_USE_SHIFT="${OMNIQUANT_USE_SHIFT:-}"
OMNIQUANT_SAVE_ACT_STATS="${OMNIQUANT_SAVE_ACT_STATS:-true}"
OMNIQUANT_SAVE_DIAGNOSTICS="${OMNIQUANT_SAVE_DIAGNOSTICS:-false}"
OMNIQUANT_DISABLE_ZERO_POINT="${OMNIQUANT_DISABLE_ZERO_POINT:-false}"
OMNIQUANT_DEACTIVE_AMP="${OMNIQUANT_DEACTIVE_AMP:-false}"
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-false}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
EVAL_VLM="${EVAL_VLM:-true}"
VLM_DATASETS="${VLM_DATASETS:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
VLM_MODE="${VLM_MODE:-all}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-$REPO_ROOT/my_results/quantization/npu}}"

set_if_unset() {
  local is_set="$1"
  local var_name="$2"
  local value="$3"
  if [[ -z "$is_set" ]]; then
    printf -v "$var_name" '%s' "$value"
  fi
}

apply_upstream_recipe_defaults() {
  local bit_config="w${WEIGHT_BITS}a${ACTIVATION_BITS}"
  case "$bit_config" in
    w2a16)
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "40"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.5"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "false"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "true"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "5e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "1e-2"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "true"
      ;;
    w3a16|w4a16)
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "20"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.5"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "false"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "true"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "5e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "1e-2"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "false"
      if [[ "$bit_config" == "w4a16" ]]; then
        # Llama-2 w4a16 can overflow on later layers under fp16 autocast.
        set_if_unset "$OMNIQUANT_DEACTIVE_AMP_SET" OMNIQUANT_DEACTIVE_AMP "true"
      fi
      ;;
    w4a4)
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "20"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.75"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "true"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "true"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "1e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "1e-2"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "false"
      set_if_unset "$OMNIQUANT_DEACTIVE_AMP_SET" OMNIQUANT_DEACTIVE_AMP "true"
      ;;
    w8a16)
      # Keep the high-bit recipe close to the upstream fallback defaults, but
      # enable learned weight clipping so it still follows the OmniQuant path.
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "20"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.5"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "false"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "true"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "5e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "1e-2"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "false"
      ;;
    w8a8)
      # High-bit activation quantization still benefits from LET, but this path
      # is much less sensitive than w6a6/w4a4, so keep the recipe conservative.
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "10"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.5"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "true"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "true"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "1e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "5e-3"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "false"
      set_if_unset "$OMNIQUANT_DEACTIVE_AMP_SET" OMNIQUANT_DEACTIVE_AMP "true"
      ;;
    w6a6)
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "20"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.5"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "true"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "true"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "1e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "5e-3"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "false"
      set_if_unset "$OMNIQUANT_DEACTIVE_AMP_SET" OMNIQUANT_DEACTIVE_AMP "true"
      ;;
    *)
      # Fall back to upstream main.py defaults when no dedicated LLaMA/Llama-2 recipe exists.
      set_if_unset "$OMNIQUANT_EPOCHS_SET" OMNIQUANT_EPOCHS "10"
      set_if_unset "$OMNIQUANT_ALPHA_SET" OMNIQUANT_ALPHA "0.5"
      set_if_unset "$OMNIQUANT_LET_SET" OMNIQUANT_LET "false"
      set_if_unset "$OMNIQUANT_LWC_SET" OMNIQUANT_LWC "false"
      set_if_unset "$OMNIQUANT_LET_LR_SET" OMNIQUANT_LET_LR "5e-3"
      set_if_unset "$OMNIQUANT_LWC_LR_SET" OMNIQUANT_LWC_LR "1e-2"
      set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "false"
      ;;
  esac
}

apply_upstream_recipe_defaults

set_if_unset "$OMNIQUANT_USE_SHIFT_SET" OMNIQUANT_USE_SHIFT "false"

apply_model_family_overrides() {
  local bit_config="w${WEIGHT_BITS}a${ACTIVATION_BITS}"
  local model_path_lower="${MODEL_PATH,,}"
  if [[ "$bit_config" != "w4a4" ]]; then
    return
  fi

  if [[ "$model_path_lower" == *"meta-llama-3.1"* ]]; then
    set_if_unset "$OMNIQUANT_AUG_LOSS_SET" OMNIQUANT_AUG_LOSS "true"
  fi
  if [[ "$model_path_lower" == *"qwen"* ]]; then
    set_if_unset "$OMNIQUANT_USE_SHIFT_SET" OMNIQUANT_USE_SHIFT "true"
  fi
  if [[ "$model_path_lower" == *"minicpm"* ]]; then
    set_if_unset "$OMNIQUANT_USE_SHIFT_SET" OMNIQUANT_USE_SHIFT "false"
  fi
}

apply_model_family_overrides

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
  --omniquant_use_shift "$OMNIQUANT_USE_SHIFT"
  --omniquant_save_act_stats "$OMNIQUANT_SAVE_ACT_STATS"
  --omniquant_save_diagnostics "$OMNIQUANT_SAVE_DIAGNOSTICS"
  --omniquant_disable_zero_point "$OMNIQUANT_DISABLE_ZERO_POINT"
  --omniquant_deactive_amp "$OMNIQUANT_DEACTIVE_AMP"
  --output_dir "$OUTPUT_DIR"
  --eval_ppl "$EVAL_PPL"
  --eval_zero_shot "$EVAL_ZERO_SHOT"
  --eval_vlm "$EVAL_VLM"
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
