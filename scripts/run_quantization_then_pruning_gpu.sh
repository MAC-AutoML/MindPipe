#!/usr/bin/env bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared experiment defaults con
OUTPUT_ROOT_DEFAULT="$REPO_ROOT/task_results/quantization"
DATA_PATH_DEFAULT="/mnt/42_store/lcw/data2/Huawei/datasets"
SEQUENCE_LENGTH_DEFAULT=512
DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-42}"

# Shared evaluation defaults
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

# FlatQuant defaults aligned with scripts/quantization/flatquant.sh
FLATQUANT_CALIBRATION_DATASET="${FLATQUANT_CALIBRATION_DATASET:-pileval}"
FLATQUANT_CALIBRATION_SAMPLES="${FLATQUANT_CALIBRATION_SAMPLES:-128}"
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALIBRATION_BATCH_SIZE="${FLATQUANT_CALIBRATION_BATCH_SIZE:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-128}"
WEIGHT_METHOD="${WEIGHT_METHOD:-rtn}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-true}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-true}"
FLATQUANT_LWC="${FLATQUANT_LWC:-true}"
FLATQUANT_LAC="${FLATQUANT_LAC:-true}"
FLATQUANT_DIRECT_INV="${FLATQUANT_DIRECT_INV:-true}"
FLATQUANT_DEACTIVE_AMP="${FLATQUANT_DEACTIVE_AMP:-true}"

# Pruning defaults
FLAP_CALIBRATION_DATASET="${FLAP_CALIBRATION_DATASET:-wikitext2}"
FLAP_CALIBRATION_SAMPLES="${FLAP_CALIBRATION_SAMPLES:-2048}"
FLAP_METRICS="${FLAP_METRICS:-WIFV}"
FLAP_REMOVE_HEADS="${FLAP_REMOVE_HEADS:-8}"
FLAP_PSEUDO_PRUNING="${FLAP_PSEUDO_PRUNING:-true}"

SPARSEGPT_CALIBRATION_DATASET="${SPARSEGPT_CALIBRATION_DATASET:-c4}"
SPARSEGPT_CALIBRATION_SAMPLES="${SPARSEGPT_CALIBRATION_SAMPLES:-128}"
SPARSEGPT_STRUCTURE_PATTERN="${SPARSEGPT_STRUCTURE_PATTERN:-unstructured}"
SPARSEGPT_BLOCK_SIZE="${SPARSEGPT_BLOCK_SIZE:-64}"
SPARSEGPT_DAMP_PERCENT="${SPARSEGPT_DAMP_PERCENT:-0.01}"

WANDA_CALIBRATION_DATASET="${WANDA_CALIBRATION_DATASET:-c4}"
WANDA_CALIBRATION_SAMPLES="${WANDA_CALIBRATION_SAMPLES:-128}"
WANDA_STRUCTURE_PATTERN="${WANDA_STRUCTURE_PATTERN:-unstructured}"


pick_first_existing_path() {
  local fallback_path="$1"
  shift
  local candidate
  for candidate in "$fallback_path" "$@"; do
    if [[ -d "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  printf '%s' "$fallback_path"
}


# Experiment matrix
MODELS=(
  #"/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct"
  #"/mnt/82_store/LLM-weights/Llama-2-7b-hf"
  #"/mnt/82_store/LLM-weights/Meta-Llama-3.1-8B-Instruct"
  #"/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct"
  #"/mnt/82_store/LLM-weights/openbmb/MiniCPM-V"
  "$(pick_first_existing_path \
    "/mnt/82_store/LLM-weights/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen3-VL-2B" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B" \
    "/mnt/82_store/zy/model/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/zy/model/Qwen3-VL-2B" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B")"
  "$(pick_first_existing_path \
    "/mnt/42_store/wxx/modelzoo/Qwen/Qwen3-0.6B" \
    "/mnt/82_store/LLM-weights/Qwen3-0.6B" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3-0.6B" \
    "/mnt/82_store/zy/model/Qwen3-0.6B" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-0.6B")"
  "$(pick_first_existing_path \
    "/mnt/82_store/LLM-weights/Qwen3.5-4B" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3.5-4B" \
    "/mnt/82_store/LLM-weights/Qwen3_5-4B" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3_5-4B" \
    "/mnt/82_store/zy/model/Qwen3.5-4B" \
    "/mnt/82_store/zy/model/Qwen3_5-4B" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3.5-4B" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen3_5-4B")"
)
FLATQUANT_CONFIGS=(
  "4 16 16 16 16 w4a16"
  "4 4 16 16 16 w4a4"
  "8 8 16 16 16 w8a8"
)

FLAP_SPARSITIES=(0.2)
SPARSEGPT_SPARSITIES=(0.2 0.4 0.5)
WANDA_SPARSITIES=(0.2 0.4 0.5)

# Worker scheduling
ENABLE_FLAP="${ENABLE_FLAP:-true}"
ENABLE_SPARSEGPT="${ENABLE_SPARSEGPT:-true}"
ENABLE_WANDA="${ENABLE_WANDA:-true}"

FLAP_GPUS="${FLAP_GPUS:-2}"
SPARSEGPT_GPUS="${SPARSEGPT_GPUS:-3,5}"
WANDA_GPUS="${WANDA_GPUS:-3,7}"

# Execution control
FORCE_RERUN="${FORCE_RERUN:-false}"
DRY_RUN="${DRY_RUN:-false}"

LAST_RUN_STATUS=""
WORKER_PIDS=()
WORKER_LABELS=()
WORKER_GPUS=()


output_root() {
  printf '%s' "${OUTPUT_DIR:-${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}}"
}


format_ratio() {
  local ratio="$1"
  if [[ "$ratio" == *.* ]]; then
    while [[ "$ratio" == *0 ]]; do
      ratio="${ratio%0}"
    done
    ratio="${ratio%.}"
  fi
  printf '%s' "${ratio//./p}"
}


# Match workflow/builder.py multi-stage output layout.
metrics_path_for() {
  local model_path="$1"
  local pruning_algorithm="$2"
  local weight_bits="$3"
  local activation_bits="$4"
  local sparsity_ratio="$5"
  local out_root
  out_root="$(output_root)"
  local model_name
  model_name="$(basename "$model_path")"

  printf '%s/%s/quantization_then_pruning/flatquant__%s/flatquant_w%sa%s_%s_s%s_seq%s/metrics.json' \
    "$out_root" \
    "$model_name" \
    "$pruning_algorithm" \
    "$weight_bits" \
    "$activation_bits" \
    "$pruning_algorithm" \
    "$sparsity_ratio" \
    "${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
}


is_complete() {
  local metrics_path="$1"
  local require_vlm="${2:-false}"
  [[ -f "$metrics_path" ]] || return 1
  if [[ "$EVAL_PPL" == "true" ]] && ! grep -q '"perplexity"' "$metrics_path"; then
    return 1
  fi
  if [[ "$EVAL_ZERO_SHOT" == "true" ]] && ! grep -q '"zero_shot"' "$metrics_path"; then
    return 1
  fi
  if [[ "$require_vlm" == "true" ]] && ! grep -q '"vlm_eval"' "$metrics_path"; then
    return 1
  fi
  return 0
}


is_vlm_model() {
  local model_name
  model_name="$(basename "$1")"
  model_name="${model_name,,}"

  if [[ "$model_name" == *"qwen"* && "$model_name" == *"vl"* ]]; then
    return 0
  fi
  if [[ "$model_name" == *"minicpm-v"* || "$model_name" == *"minicpm_v"* ]]; then
    return 0
  fi
  return 1
}


should_eval_vlm() {
  local model_path="$1"
  if [[ "$EVAL_VLM" != "true" ]]; then
    return 1
  fi
  is_vlm_model "$model_path"
}


resolve_visible_devices() {
  local gpu_spec="$1"
  if [[ "$gpu_spec" == "cpu" ]]; then
    return 1
  fi
  if [[ "$gpu_spec" == cuda:* ]]; then
    printf '%s' "${gpu_spec#cuda:}"
    return 0
  fi
  printf '%s' "$gpu_spec"
}


resolve_runtime_device() {
  local gpu_spec="$1"
  if [[ "$gpu_spec" == "cpu" ]]; then
    printf 'cpu'
  else
    printf 'cuda:0'
  fi
}


append_device_env() {
  local -n env_ref="$1"
  local gpu_spec="$2"
  local visible_devices
  if visible_devices="$(resolve_visible_devices "$gpu_spec")"; then
    env_ref+=("CUDA_VISIBLE_DEVICES=$visible_devices")
  fi
}


parse_gpu_specs() {
  local gpu_csv="$1"
  local -n gpu_specs_ref="$2"
  gpu_specs_ref=()
  if [[ -z "${gpu_csv//,/}" ]]; then
    return 0
  fi
  IFS=',' read -r -a gpu_specs_ref <<< "$gpu_csv"
  local index
  for index in "${!gpu_specs_ref[@]}"; do
    gpu_specs_ref[$index]="${gpu_specs_ref[$index]//[[:space:]]/}"
  done
}


append_zero_shot_args() {
  local -n cmd_ref="$1"
  if [[ "$EVAL_ZERO_SHOT" != "true" ]]; then
    return 0
  fi
  local -a zero_shot_task_array=()
  read -r -a zero_shot_task_array <<< "$ZERO_SHOT_TASKS"
  cmd_ref+=(
    --zero_shot_tasks "${zero_shot_task_array[@]}"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
  )
}


append_vlm_args() {
  local -n cmd_ref="$1"
  local effective_eval_vlm="$2"
  if [[ "$effective_eval_vlm" != "true" ]]; then
    return 0
  fi
  local -a vlm_dataset_array=()
  read -r -a vlm_dataset_array <<< "$VLM_DATASETS"
  cmd_ref+=(
    --vlm_datasets "${vlm_dataset_array[@]}"
    --vlm_mode "$VLM_MODE"
    --vlm_api_nproc "$VLM_API_NPROC"
    --vlm_pred_format "$VLM_PRED_FORMAT"
    --vlm_verbose "${VLM_VERBOSE:-false}"
    --vlm_ignore_failed "${VLM_IGNORE_FAILED:-false}"
  )
  if [[ -n "${VLM_WORK_DIR:-}" ]]; then
    cmd_ref+=(--vlm_work_dir "$VLM_WORK_DIR")
  fi
  if [[ -n "${VLM_EVAL_KIT_ROOT:-}" ]]; then
    cmd_ref+=(--vlm_eval_kit_root "$VLM_EVAL_KIT_ROOT")
  fi
  if [[ -n "${VLM_JUDGE:-}" ]]; then
    cmd_ref+=(--vlm_judge "$VLM_JUDGE")
  fi
}


run_experiment() {
  local pruning_algorithm="$1"
  local model_path="$2"
  local weight_bits="$3"
  local activation_bits="$4"
  local query_bits="$5"
  local key_bits="$6"
  local value_bits="$7"
  local quant_label="$8"
  local sparsity_ratio="$9"
  local gpu_spec="${10}"
  local model_name
  model_name="$(basename "$model_path")"
  local ratio_tag
  ratio_tag="$(format_ratio "$sparsity_ratio")"
  local run_id="${model_name}__flatquant__${quant_label}__${pruning_algorithm}__s${ratio_tag}"
  local metrics_path
  metrics_path="$(metrics_path_for "$model_path" "$pruning_algorithm" "$weight_bits" "$activation_bits" "$sparsity_ratio")"
  local effective_eval_vlm="false"
  if should_eval_vlm "$model_path"; then
    effective_eval_vlm="true"
  fi

  if [[ "$FORCE_RERUN" != "true" ]] && is_complete "$metrics_path" "$effective_eval_vlm"; then
    printf '[skip][%s][gpu=%s] %s\n' "$pruning_algorithm" "$gpu_spec" "$run_id"
    LAST_RUN_STATUS="skip"
    return 0
  fi

  local pruning_calibration_dataset
  local pruning_calibration_samples
  local -a pruning_args=()
  case "$pruning_algorithm" in
    flap)
      pruning_calibration_dataset="$FLAP_CALIBRATION_DATASET"
      pruning_calibration_samples="$FLAP_CALIBRATION_SAMPLES"
      pruning_args=(
        --flap_metrics "$FLAP_METRICS"
        --flap_remove_heads "$FLAP_REMOVE_HEADS"
        --pseudo_pruning "$FLAP_PSEUDO_PRUNING"
      )
      ;;
    sparsegpt)
      pruning_calibration_dataset="$SPARSEGPT_CALIBRATION_DATASET"
      pruning_calibration_samples="$SPARSEGPT_CALIBRATION_SAMPLES"
      pruning_args=(
        --structure_pattern "$SPARSEGPT_STRUCTURE_PATTERN"
        --block_size "$SPARSEGPT_BLOCK_SIZE"
        --pruning_damp_percent "$SPARSEGPT_DAMP_PERCENT"
      )
      ;;
    wanda)
      pruning_calibration_dataset="$WANDA_CALIBRATION_DATASET"
      pruning_calibration_samples="$WANDA_CALIBRATION_SAMPLES"
      pruning_args=(
        --structure_pattern "$WANDA_STRUCTURE_PATTERN"
      )
      ;;
    *)
      printf 'Unknown pruning algorithm: %s\n' "$pruning_algorithm" >&2
      LAST_RUN_STATUS="fail"
      return 1
      ;;
  esac

  local out_root
  out_root="$(output_root)"
  local runtime_device
  runtime_device="$(resolve_runtime_device "$gpu_spec")"
  local -a env_vars=()
  append_device_env env_vars "$gpu_spec"

  local -a cmd=(
    python "$REPO_ROOT/main.py"
    --quantization flatquant
    --pruning "$pruning_algorithm"
    --execution_order quantization_then_pruning
    --model_path "$model_path"
    --device "$runtime_device"
    --device_map "$DEVICE_MAP"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --data_path "$DATA_PATH"
    --seed "$SEED"
    --output_dir "$out_root"
    --evaluation_dataset "$EVALUATION_DATASET"
    --sequence_length "${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$weight_bits"
    --activation_bits "$activation_bits"
    --query_bits "$query_bits"
    --key_bits "$key_bits"
    --value_bits "$value_bits"
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
    --sparsity_ratio "$sparsity_ratio"
    --pruning_calibration_dataset "$pruning_calibration_dataset"
    --pruning_calibration_samples "$pruning_calibration_samples"
    --eval_ppl "$EVAL_PPL"
    --eval_zero_shot "$EVAL_ZERO_SHOT"
    --eval_vlm "$effective_eval_vlm"
  )
  cmd+=("${pruning_args[@]}")

  if [[ -n "${HF_TOKEN:-}" ]]; then
    cmd+=(--hf_token "$HF_TOKEN")
  fi

  append_zero_shot_args cmd
  append_vlm_args cmd "$effective_eval_vlm"

  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    cmd+=(--num_samples "$NUM_SAMPLES")
  fi

  printf '[run][%s][gpu=%s] %s\n' "$pruning_algorithm" "$gpu_spec" "$run_id"
  printf '  out: %s\n' "$metrics_path"
  printf '  eval_ppl: %s\n' "$EVAL_PPL"
  printf '  eval_vlm: %s\n' "$effective_eval_vlm"
  printf '  cmd:'
  printf ' %q' env "${env_vars[@]}" "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    LAST_RUN_STATUS="success"
    return 0
  fi

  set +e
  env "${env_vars[@]}" "${cmd[@]}" 2>&1 | sed -u "s/^/[${run_id}][gpu=${gpu_spec}] /"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ $exit_code -ne 0 ]]; then
    printf '[fail][%s][gpu=%s] %s\n' "$pruning_algorithm" "$gpu_spec" "$run_id"
    LAST_RUN_STATUS="fail"
    return 1
  fi

  printf '[ok][%s][gpu=%s] %s\n' "$pruning_algorithm" "$gpu_spec" "$run_id"
  LAST_RUN_STATUS="success"
  return 0
}


run_algorithm_queue() {
  local pruning_algorithm="$1"
  local gpu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local failure_count=0
  local success_count=0
  local skip_count=0
  local model_path
  local config
  local sparsity_ratio
  local job_index=0

  case "$pruning_algorithm" in
    flap)
      for model_path in "${MODELS[@]}"; do
        for config in "${FLATQUANT_CONFIGS[@]}"; do
          local weight_bits
          local activation_bits
          local query_bits
          local key_bits
          local value_bits
          local quant_label
          read -r weight_bits activation_bits query_bits key_bits value_bits quant_label <<< "$config"
          for sparsity_ratio in "${FLAP_SPARSITIES[@]}"; do
            if (( job_index % worker_count == worker_index )); then
              if ! run_experiment flap "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$quant_label" "$sparsity_ratio" "$gpu_spec"; then
                ((failure_count += 1))
              fi
              case "$LAST_RUN_STATUS" in
                success) ((success_count += 1)) ;;
                skip) ((skip_count += 1)) ;;
              esac
            fi
            ((job_index += 1))
          done
        done
      done
      ;;
    sparsegpt)
      for model_path in "${MODELS[@]}"; do
        for config in "${FLATQUANT_CONFIGS[@]}"; do
          local weight_bits
          local activation_bits
          local query_bits
          local key_bits
          local value_bits
          local quant_label
          read -r weight_bits activation_bits query_bits key_bits value_bits quant_label <<< "$config"
          for sparsity_ratio in "${SPARSEGPT_SPARSITIES[@]}"; do
            if (( job_index % worker_count == worker_index )); then
              if ! run_experiment sparsegpt "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$quant_label" "$sparsity_ratio" "$gpu_spec"; then
                ((failure_count += 1))
              fi
              case "$LAST_RUN_STATUS" in
                success) ((success_count += 1)) ;;
                skip) ((skip_count += 1)) ;;
              esac
            fi
            ((job_index += 1))
          done
        done
      done
      ;;
    wanda)
      for model_path in "${MODELS[@]}"; do
        for config in "${FLATQUANT_CONFIGS[@]}"; do
          local weight_bits
          local activation_bits
          local query_bits
          local key_bits
          local value_bits
          local quant_label
          read -r weight_bits activation_bits query_bits key_bits value_bits quant_label <<< "$config"
          for sparsity_ratio in "${WANDA_SPARSITIES[@]}"; do
            if (( job_index % worker_count == worker_index )); then
              if ! run_experiment wanda "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$quant_label" "$sparsity_ratio" "$gpu_spec"; then
                ((failure_count += 1))
              fi
              case "$LAST_RUN_STATUS" in
                success) ((success_count += 1)) ;;
                skip) ((skip_count += 1)) ;;
              esac
            fi
            ((job_index += 1))
          done
        done
      done
      ;;
    *)
      printf 'Unknown algorithm queue: %s\n' "$pruning_algorithm" >&2
      return 1
      ;;
  esac

  printf '[worker-summary] %s worker=%s/%s gpu=%s success=%s skip=%s fail=%s\n' \
    "$pruning_algorithm" \
    "$((worker_index + 1))" \
    "$worker_count" \
    "$gpu_spec" \
    "$success_count" \
    "$skip_count" \
    "$failure_count"
  return "$failure_count"
}


launch_worker() {
  local pruning_algorithm="$1"
  local gpu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local worker_label="${pruning_algorithm}_worker$((worker_index + 1))_of_${worker_count}"
  printf '[worker-start] %s on gpu=%s\n' "$worker_label" "$gpu_spec"
  run_algorithm_queue "$pruning_algorithm" "$gpu_spec" "$worker_index" "$worker_count" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("$worker_label")
  WORKER_GPUS+=("$gpu_spec")
}


launch_algorithm_workers() {
  local pruning_algorithm="$1"
  local gpu_csv="$2"
  local -a gpu_specs=()
  parse_gpu_specs "$gpu_csv" gpu_specs
  if ((${#gpu_specs[@]} == 0)); then
    printf '[worker-skip] %s has no gpu configured\n' "$pruning_algorithm"
    return 0
  fi

  local worker_count="${#gpu_specs[@]}"
  local index
  for index in "${!gpu_specs[@]}"; do
    launch_worker "$pruning_algorithm" "${gpu_specs[$index]}" "$index" "$worker_count"
  done
}


main() {
  local exit_code
  local had_failure=false

  if [[ "$ENABLE_FLAP" == "true" ]]; then
    launch_algorithm_workers flap "$FLAP_GPUS"
  fi

  if [[ "$ENABLE_SPARSEGPT" == "true" ]]; then
    launch_algorithm_workers sparsegpt "$SPARSEGPT_GPUS"
  fi

  if [[ "$ENABLE_WANDA" == "true" ]]; then
    launch_algorithm_workers wanda "$WANDA_GPUS"
  fi

  if ((${#WORKER_PIDS[@]} == 0)); then
    printf 'No pruning workers enabled.\n'
    return 0
  fi

  local index
  for index in "${!WORKER_PIDS[@]}"; do
    set +e
    wait "${WORKER_PIDS[$index]}"
    exit_code=$?
    set -e
    if [[ $exit_code -ne 0 ]]; then
      printf '[worker-fail] %s on gpu=%s exited with code %s\n' \
        "${WORKER_LABELS[$index]}" \
        "${WORKER_GPUS[$index]}" \
        "$exit_code"
      had_failure=true
    else
      printf '[worker-done] %s on gpu=%s\n' \
        "${WORKER_LABELS[$index]}" \
        "${WORKER_GPUS[$index]}"
    fi
  done

  if [[ "$had_failure" == "true" ]]; then
    return 1
  fi
  return 0
}


main "$@"
