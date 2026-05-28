#!/usr/bin/env bash
set -euo pipefail

# LLM-only compression LoRA entrypoint.
# Examples: Llama-2, Llama-3.1, Qwen2.5-Instruct, Qwen3 dense text models.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct}"
PRUNING="${PRUNING:-wanda}"
GPU_ID="${GPU_ID:-3,4,5}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/82_store/wxx/HWQuant/Mindpipe/results_lora}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"

WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
QUERY_BITS="${QUERY_BITS:-16}"
KEY_BITS="${KEY_BITS:-16}"
VALUE_BITS="${VALUE_BITS:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-42}"

FLATQUANT_CALIBRATION_DATASET="${FLATQUANT_CALIBRATION_DATASET:-pileval}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
FLATQUANT_CALIBRATION_SAMPLES="${FLATQUANT_CALIBRATION_SAMPLES:-128}"
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALIBRATION_BATCH_SIZE="${FLATQUANT_CALIBRATION_BATCH_SIZE:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-true}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-true}"
FLATQUANT_LWC="${FLATQUANT_LWC:-true}"
FLATQUANT_LAC="${FLATQUANT_LAC:-true}"
FLATQUANT_DIRECT_INV="${FLATQUANT_DIRECT_INV:-true}"
FLATQUANT_DEACTIVE_AMP="${FLATQUANT_DEACTIVE_AMP:-true}"
WEIGHT_METHOD="${WEIGHT_METHOD:-rtn}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-128}"

SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
PRUNING_CALIBRATION_DATASET="${PRUNING_CALIBRATION_DATASET:-c4}"
PRUNING_CALIBRATION_SAMPLES="${PRUNING_CALIBRATION_SAMPLES:-128}"
STRUCTURE_PATTERN="${STRUCTURE_PATTERN:-unstructured}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
PRUNING_DAMP_PERCENT="${PRUNING_DAMP_PERCENT:-0.01}"
FLAP_METRICS="${FLAP_METRICS:-WIFV}"
FLAP_REMOVE_HEADS="${FLAP_REMOVE_HEADS:-8}"
PSEUDO_PRUNING="${PSEUDO_PRUNING:-true}"

COMPRESSION_LORA_TRAIN_PLAN="${COMPRESSION_LORA_TRAIN_PLAN:-cpt,sft}"
COMPRESSION_LORA_CPT_TRAIN_FILE="${COMPRESSION_LORA_CPT_TRAIN_FILE:-/mnt/42_store/wxx/datasets/fineweb_edu/fineweb_edu_subset_10000_minint4_seed42.jsonl}"
COMPRESSION_LORA_CPT_SAMPLES="${COMPRESSION_LORA_CPT_SAMPLES:-64}"
COMPRESSION_LORA_CPT_LR="${COMPRESSION_LORA_CPT_LR:-3e-5}"
COMPRESSION_LORA_CPT_EPOCHS="${COMPRESSION_LORA_CPT_EPOCHS:-1}"
COMPRESSION_LORA_CPT_MAX_STEPS="${COMPRESSION_LORA_CPT_MAX_STEPS:--1}"
COMPRESSION_LORA_SFT_FORMAT="${COMPRESSION_LORA_SFT_FORMAT:-alpaca}"
COMPRESSION_LORA_SFT_TRAIN_FILE="${COMPRESSION_LORA_SFT_TRAIN_FILE:-/mnt/42_store/wxx/datasets/alpaca/alpaca.json}"
COMPRESSION_LORA_SFT_SAMPLES="${COMPRESSION_LORA_SFT_SAMPLES:-64}"
COMPRESSION_LORA_SFT_LR="${COMPRESSION_LORA_SFT_LR:-5e-5}"
COMPRESSION_LORA_SFT_EPOCHS="${COMPRESSION_LORA_SFT_EPOCHS:-1}"
COMPRESSION_LORA_SFT_MAX_STEPS="${COMPRESSION_LORA_SFT_MAX_STEPS:--1}"
COMPRESSION_LORA_SAVE_CPT_ADAPTER="${COMPRESSION_LORA_SAVE_CPT_ADAPTER:-true}"
COMPRESSION_LORA_RANK="${COMPRESSION_LORA_RANK:-64}"
COMPRESSION_LORA_ALPHA="${COMPRESSION_LORA_ALPHA:-64}"
COMPRESSION_LORA_DROPOUT="${COMPRESSION_LORA_DROPOUT:-0.05}"
COMPRESSION_LORA_INIT="${COMPRESSION_LORA_INIT:-lora}"
COMPRESSION_LORA_LR="${COMPRESSION_LORA_LR:-1e-4}"
COMPRESSION_LORA_EPOCHS="${COMPRESSION_LORA_EPOCHS:-3}"
COMPRESSION_LORA_BATCH_SIZE="${COMPRESSION_LORA_BATCH_SIZE:-1}"
COMPRESSION_LORA_GRAD_ACCUM="${COMPRESSION_LORA_GRAD_ACCUM:-4}"
COMPRESSION_LORA_LOGGING_STEPS="${COMPRESSION_LORA_LOGGING_STEPS:-1}"
COMPRESSION_LORA_GRADIENT_CHECKPOINTING="${COMPRESSION_LORA_GRADIENT_CHECKPOINTING:-false}"
COMPRESSION_LORA_SAVE_MERGED_MODEL="${COMPRESSION_LORA_SAVE_MERGED_MODEL:-false}"
COMPRESSION_LORA_ADAPTER_FROM="${COMPRESSION_LORA_ADAPTER_FROM:-}"
COMPRESSION_LORA_MASKS_FROM="${COMPRESSION_LORA_MASKS_FROM:-}"

EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-true}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-16}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
EVAL_VLM="${EVAL_VLM:-false}"

cmd=(
  python "$REPO_ROOT/main.py"
  --quantization flatquant
  --pruning "$PRUNING"
  --execution_order quantization_then_pruning
  --finetuning compression_lora
  --model_path "$MODEL_PATH"
  --device cuda:0
  --device_map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --data_path "$DATA_PATH"
  --calibration_dataset "$CALIBRATION_DATASET"
  --seed "$SEED"
  --output_dir "$OUTPUT_DIR"
  --sequence_length "$SEQUENCE_LENGTH"
  --batch_size "$BATCH_SIZE"
  --max_eval_chunks "$MAX_EVAL_CHUNKS"
  --weight_bits "$WEIGHT_BITS"
  --activation_bits "$ACTIVATION_BITS"
  --query_bits "$QUERY_BITS"
  --key_bits "$KEY_BITS"
  --value_bits "$VALUE_BITS"
  --kv_group_size "$KV_GROUP_SIZE"
  --weight_method "$WEIGHT_METHOD"
  --flatquant_epochs "$FLATQUANT_EPOCHS"
  --flatquant_calibration_batch_size "$FLATQUANT_CALIBRATION_BATCH_SIZE"
  --flatquant_lr "$FLATQUANT_LR"
  --flatquant_cali_trans "$FLATQUANT_CALI_TRANS"
  --flatquant_add_diag "$FLATQUANT_ADD_DIAG"
  --flatquant_lwc "$FLATQUANT_LWC"
  --flatquant_lac "$FLATQUANT_LAC"
  --flatquant_direct_inv "$FLATQUANT_DIRECT_INV"
  --flatquant_deactive_amp "$FLATQUANT_DEACTIVE_AMP"
  --quantization_calibration_dataset "$FLATQUANT_CALIBRATION_DATASET"
  --quantization_calibration_samples "$FLATQUANT_CALIBRATION_SAMPLES"
  --sparsity_ratio "$SPARSITY_RATIO"
  --pruning_calibration_dataset "$PRUNING_CALIBRATION_DATASET"
  --pruning_calibration_samples "$PRUNING_CALIBRATION_SAMPLES"
  --structure_pattern "$STRUCTURE_PATTERN"
  --block_size "$BLOCK_SIZE"
  --pruning_damp_percent "$PRUNING_DAMP_PERCENT"
  --flap_metrics "$FLAP_METRICS"
  --flap_remove_heads "$FLAP_REMOVE_HEADS"
  --pseudo_pruning "$PSEUDO_PRUNING"
  --compression_lora_train_plan "$COMPRESSION_LORA_TRAIN_PLAN"
  --compression_lora_cpt_train_file "$COMPRESSION_LORA_CPT_TRAIN_FILE"
  --compression_lora_cpt_samples "$COMPRESSION_LORA_CPT_SAMPLES"
  --compression_lora_cpt_learning_rate "$COMPRESSION_LORA_CPT_LR"
  --compression_lora_cpt_num_train_epochs "$COMPRESSION_LORA_CPT_EPOCHS"
  --compression_lora_cpt_max_steps "$COMPRESSION_LORA_CPT_MAX_STEPS"
  --compression_lora_sft_format "$COMPRESSION_LORA_SFT_FORMAT"
  --compression_lora_sft_train_file "$COMPRESSION_LORA_SFT_TRAIN_FILE"
  --compression_lora_sft_samples "$COMPRESSION_LORA_SFT_SAMPLES"
  --compression_lora_sft_learning_rate "$COMPRESSION_LORA_SFT_LR"
  --compression_lora_sft_num_train_epochs "$COMPRESSION_LORA_SFT_EPOCHS"
  --compression_lora_sft_max_steps "$COMPRESSION_LORA_SFT_MAX_STEPS"
  --compression_lora_save_cpt_adapter "$COMPRESSION_LORA_SAVE_CPT_ADAPTER"
  --compression_lora_rank "$COMPRESSION_LORA_RANK"
  --compression_lora_alpha "$COMPRESSION_LORA_ALPHA"
  --compression_lora_dropout "$COMPRESSION_LORA_DROPOUT"
  --compression_lora_init "$COMPRESSION_LORA_INIT"
  --compression_lora_learning_rate "$COMPRESSION_LORA_LR"
  --compression_lora_num_train_epochs "$COMPRESSION_LORA_EPOCHS"
  --compression_lora_per_device_train_batch_size "$COMPRESSION_LORA_BATCH_SIZE"
  --compression_lora_gradient_accumulation_steps "$COMPRESSION_LORA_GRAD_ACCUM"
  --compression_lora_logging_steps "$COMPRESSION_LORA_LOGGING_STEPS"
  --compression_lora_gradient_checkpointing "$COMPRESSION_LORA_GRADIENT_CHECKPOINTING"
  --compression_lora_save_merged_model "$COMPRESSION_LORA_SAVE_MERGED_MODEL"
  --eval_ppl "$EVAL_PPL"
  --eval_zero_shot "$EVAL_ZERO_SHOT"
  --eval_vlm "$EVAL_VLM"
)

if [[ -n "$COMPRESSION_LORA_ADAPTER_FROM" ]]; then
  cmd+=(--compression_lora_adapter_from "$COMPRESSION_LORA_ADAPTER_FROM")
fi
if [[ -n "$COMPRESSION_LORA_MASKS_FROM" ]]; then
  cmd+=(--compression_lora_masks_from "$COMPRESSION_LORA_MASKS_FROM")
fi
if [[ "$EVAL_ZERO_SHOT" == "true" ]]; then
  read -r -a zero_shot_task_array <<< "$ZERO_SHOT_TASKS"
  cmd+=(--zero_shot_tasks "${zero_shot_task_array[@]}")
  cmd+=(--zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE")
  cmd+=(--zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  cmd+=(--hf_token "$HF_TOKEN")
fi

printf 'cmd:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=$GPU_ID" "${cmd[@]}"
printf '\n'

env "CUDA_VISIBLE_DEVICES=$GPU_ID" "${cmd[@]}"
