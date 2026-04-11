#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Shared experiment defaults
OUTPUT_ROOT_DEFAULT="$REPO_ROOT/my_results/quantization/npu"
DATA_PATH_DEFAULT="/home/ma-user/work/data"
SEQUENCE_LENGTH_DEFAULT=2048
DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

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

# Algorithm-specific defaults
AWQ_SEARCH="${AWQ_SEARCH:-true}"
FLATQUANT_QUERY_BITS_DEFAULT=16
FLATQUANT_KEY_BITS_DEFAULT=4
FLATQUANT_VALUE_BITS_DEFAULT=4
SMOOTHQUANT_QUERY_BITS_DEFAULT=16
SMOOTHQUANT_KEY_BITS_DEFAULT=16
SMOOTHQUANT_VALUE_BITS_DEFAULT=16
SMOOTHQUANT_ALPHA_DEFAULT="${SMOOTHQUANT_ALPHA_DEFAULT:-0.85}"
SPLITQUANT_QUERY_BITS_DEFAULT=16
SPLITQUANT_KEY_BITS_DEFAULT=4
SPLITQUANT_VALUE_BITS_DEFAULT=4
SPLITQUANT_GROUP_SIZE="${SPLITQUANT_GROUP_SIZE:-64}"
SPLITQUANT_WEIGHT_GROUP_SIZE="${SPLITQUANT_WEIGHT_GROUP_SIZE:-$SPLITQUANT_GROUP_SIZE}"
SPLITQUANT_ACTIVATION_GROUP_SIZE="${SPLITQUANT_ACTIVATION_GROUP_SIZE:-$SPLITQUANT_GROUP_SIZE}"
SPLITQUANT_KV_GROUP_SIZE="${SPLITQUANT_KV_GROUP_SIZE:-128}"

# Experiment matrix
MODELS=(
  "/home/ma-user/work/modelzoo/Qwen/Qwen2.5-VL-7B-Instruct"
  # "/home/ma-user/work/modelzoo/Qwen/Qwen2.5-7B-Instruct"
  # "/home/ma-user/work/modelzoo/Meta/Llama-2-7b-hf"
  # "/home/ma-user/work/modelzoo/Meta/Meta-Llama-3.1-8B-Instruct"
  # "/home/ma-user/work/modelzoo/openbmb/MiniCPM-V"
)
# AWQ_BITS=(2 3 4)
# GPTQ_BITS=(2 3 4)
AWQ_BITS=(4)
GPTQ_BITS=(4)
SMOOTHQUANT_CONFIGS=(
  "8 8 16 16 16 ${SMOOTHQUANT_ALPHA_DEFAULT} w8a8"
)
FLATQUANT_CONFIGS=(
  "2 16 16 4 4 w2a16"
  "3 16 16 4 4 w3a16"
  "4 16 16 4 4 w4a16"
  "4 4 16 4 4 w4a4"
  "8 8 16 4 4 w8a8"
  "16 16 16 16 16 w16a16"
)
SPLITQUANT_CONFIGS=(
  "2 16 16 4 4 w2a16"
  "3 16 16 4 4 w3a16"
  "4 16 16 4 4 w4a16"
  "4 4 16 4 4 w4a4"
  "8 8 16 4 4 w8a8"
  "16 16 16 16 16 w16a16"
)

# Worker scheduling
ENABLE_AWQ="${ENABLE_AWQ:-true}"
ENABLE_GPTQ="${ENABLE_GPTQ:-false}"
ENABLE_SMOOTHQUANT="${ENABLE_SMOOTHQUANT:-false}"
ENABLE_FLATQUANT="${ENABLE_FLATQUANT:-false}"
ENABLE_SPLITQUANT="${ENABLE_SPLITQUANT:-false}"

AWQ_NPUS="${AWQ_NPUS:-0}"
GPTQ_NPUS="${GPTQ_NPUS:-1}"
SMOOTHQUANT_NPUS="${SMOOTHQUANT_NPUS:-1}"
FLATQUANT_NPUS="${FLATQUANT_NPUS:-1}"
SPLITQUANT_NPUS="${SPLITQUANT_NPUS:-1}"

# Execution control
FORCE_RERUN="${FORCE_RERUN:-false}"
DRY_RUN="${DRY_RUN:-false}"

LAST_RUN_STATUS=""
WORKER_PIDS=()
WORKER_LABELS=()
WORKER_NPUS=()


output_root() {
  printf '%s' "${OUTPUT_DIR:-${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}}"
}


format_alpha() {
  local alpha="$1"
  if [[ "$alpha" == *.* ]]; then
    while [[ "$alpha" == *0 ]]; do
      alpha="${alpha%0}"
    done
    alpha="${alpha%.}"
  fi
  printf '%s' "${alpha//./p}"
}


metrics_path_for() {
  local algorithm="$1"
  local model_path="$2"
  local weight_bits="$3"
  local activation_bits="$4"
  local default_query_bits="$FLATQUANT_QUERY_BITS_DEFAULT"
  local default_key_bits="$FLATQUANT_KEY_BITS_DEFAULT"
  local default_value_bits="$FLATQUANT_VALUE_BITS_DEFAULT"
  if [[ "$algorithm" == "smoothquant" ]]; then
    default_query_bits="$SMOOTHQUANT_QUERY_BITS_DEFAULT"
    default_key_bits="$SMOOTHQUANT_KEY_BITS_DEFAULT"
    default_value_bits="$SMOOTHQUANT_VALUE_BITS_DEFAULT"
  elif [[ "$algorithm" == "splitquant" ]]; then
    default_query_bits="$SPLITQUANT_QUERY_BITS_DEFAULT"
    default_key_bits="$SPLITQUANT_KEY_BITS_DEFAULT"
    default_value_bits="$SPLITQUANT_VALUE_BITS_DEFAULT"
  fi
  local query_bits="${5:-$default_query_bits}"
  local key_bits="${6:-$default_key_bits}"
  local value_bits="${7:-$default_value_bits}"
  local smoothquant_alpha="${8:-$SMOOTHQUANT_ALPHA_DEFAULT}"
  local out_root
  out_root="$(output_root)"
  local model_name
  model_name="$(basename "$model_path")"

  if [[ "$algorithm" == "smoothquant" ]]; then
    local alpha_tag
    alpha_tag="$(format_alpha "$smoothquant_alpha")"
    printf '%s/%s/%s/%s_w%sa%s_seq%s_alpha%s/metrics.json' \
      "$out_root" \
      "$model_name" \
      "$algorithm" \
      "$algorithm" \
      "$weight_bits" \
      "$activation_bits" \
      "${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}" \
      "$alpha_tag"
    return 0
  fi

  if [[ "$algorithm" == "flatquant" || "$algorithm" == "splitquant" ]]; then
    printf '%s/%s/%s/%s_w%sa%s_q%sk%sv%s_seq%s/metrics.json' \
      "$out_root" \
      "$model_name" \
      "$algorithm" \
      "$algorithm" \
      "$weight_bits" \
      "$activation_bits" \
      "$query_bits" \
      "$key_bits" \
      "$value_bits" \
      "${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
  else
    printf '%s/%s/%s/%s_w%sa%s_seq%s/metrics.json' \
      "$out_root" \
      "$model_name" \
      "$algorithm" \
      "$algorithm" \
      "$weight_bits" \
      "$activation_bits" \
      "${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
  fi
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


append_passthrough_env() {
  local -n env_ref="$1"
  local -a common_keys=(
    HF_TOKEN
    OUTPUT_ROOT
    OUTPUT_DIR
    DATA_PATH
    CALIBRATION_DATASET
    EVALUATION_DATASET
    CALIBRATION_SAMPLES
    BATCH_SIZE
    MAX_EVAL_CHUNKS
    ATTN_IMPLEMENTATION
  )
  local -a zero_shot_keys=(
    ZERO_SHOT_TASKS
    ZERO_SHOT_BATCH_SIZE
    ZERO_SHOT_NUM_FEWSHOT
    ZERO_SHOT_LIMIT
  )
  local -a vlm_keys=(
    VLM_DATASETS
    VLM_MODE
    VLM_WORK_DIR
    VLM_EVAL_KIT_ROOT
    VLM_JUDGE
    VLM_API_NPROC
    VLM_VERBOSE
    VLM_IGNORE_FAILED
    VLM_PRED_FORMAT
  )
  local -a algorithm_keys=(
    AWQ_SEARCH
    SMOOTHQUANT_SAVE_ACT_SCALES
    SMOOTHQUANT_ACT_SCALES_FROM
  )
  local key
  for key in "${common_keys[@]}" "${zero_shot_keys[@]}" "${vlm_keys[@]}" "${algorithm_keys[@]}"; do
    if [[ -n "${!key:-}" ]]; then
      env_ref+=("$key=${!key}")
    fi
  done
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
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    return 1
  fi
  if [[ "$npu_spec" == npu:* ]]; then
    printf '%s' "${npu_spec#npu:}"
    return 0
  fi
  printf '%s' "$npu_spec"
}


resolve_runtime_device() {
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    printf 'cpu'
  else
    printf 'npu:0'
  fi
}


append_device_env() {
  local -n env_ref="$1"
  local npu_spec="$2"
  local runtime_device
  runtime_device="$(resolve_runtime_device "$npu_spec")"
  env_ref+=("DEVICE=$runtime_device")

  local visible_devices
  if visible_devices="$(resolve_visible_devices "$npu_spec")"; then
    env_ref+=("ASCEND_RT_VISIBLE_DEVICES=$visible_devices")
  fi
}


parse_npu_specs() {
  local npu_csv="$1"
  local -n npu_specs_ref="$2"
  npu_specs_ref=()
  if [[ -z "${npu_csv//,/}" ]]; then
    return 0
  fi
  IFS=',' read -r -a npu_specs_ref <<< "$npu_csv"
  local index
  for index in "${!npu_specs_ref[@]}"; do
    npu_specs_ref[$index]="${npu_specs_ref[$index]//[[:space:]]/}"
  done
}


run_experiment() {
  local algorithm="$1"
  local model_path="$2"
  local weight_bits="$3"
  local activation_bits="$4"
  local query_bits="$5"
  local key_bits="$6"
  local value_bits="$7"
  local label="$8"
  local npu_spec="$9"
  local smoothquant_alpha="${10:-$SMOOTHQUANT_ALPHA_DEFAULT}"
  local script_path="$REPO_ROOT/scripts/quantization/${algorithm}.sh"
  local model_name
  model_name="$(basename "$model_path")"
  local run_id="${model_name}__${algorithm}__${label}"
  local metrics_path
  local effective_eval_vlm="false"
  local out_root
  out_root="$(output_root)"
  metrics_path="$(metrics_path_for "$algorithm" "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$smoothquant_alpha")"
  if should_eval_vlm "$model_path"; then
    effective_eval_vlm="true"
  fi

  if [[ "$FORCE_RERUN" != "true" ]] && is_complete "$metrics_path" "$effective_eval_vlm"; then
    printf '[skip][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
    LAST_RUN_STATUS="skip"
    return 0
  fi

  local -a env_vars=(
    "MODEL_PATH=$model_path"
    "WEIGHT_BITS=$weight_bits"
    "DATA_PATH=$DATA_PATH"
    "SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
    "EVAL_PPL=$EVAL_PPL"
    "EVAL_ZERO_SHOT=$EVAL_ZERO_SHOT"
    "EVAL_VLM=$effective_eval_vlm"
    "OUTPUT_ROOT=$out_root"
    "OUTPUT_DIR=$out_root"
  )
  append_device_env env_vars "$npu_spec"
  if [[ "$algorithm" == "flatquant" || "$algorithm" == "splitquant" || "$algorithm" == "smoothquant" ]]; then
    env_vars+=(
      "ACTIVATION_BITS=$activation_bits"
      "QUERY_BITS=$query_bits"
      "KEY_BITS=$key_bits"
      "VALUE_BITS=$value_bits"
    )
  fi
  if [[ "$algorithm" == "smoothquant" ]]; then
    env_vars+=("SMOOTHQUANT_ALPHA=$smoothquant_alpha")
  fi
  if [[ "$algorithm" == "splitquant" ]]; then
    env_vars+=(
      "GROUP_SIZE=$SPLITQUANT_GROUP_SIZE"
      "WEIGHT_GROUP_SIZE=$SPLITQUANT_WEIGHT_GROUP_SIZE"
      "ACTIVATION_GROUP_SIZE=$SPLITQUANT_ACTIVATION_GROUP_SIZE"
      "KV_GROUP_SIZE=$SPLITQUANT_KV_GROUP_SIZE"
    )
  fi
  append_passthrough_env env_vars

  printf '[run][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
  printf '  out: %s\n' "$metrics_path"
  printf '  eval_ppl: %s\n' "$EVAL_PPL"
  printf '  eval_vlm: %s\n' "$effective_eval_vlm"
  printf '  cmd:'
  printf ' %q' env "${env_vars[@]}" bash "$script_path"
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    LAST_RUN_STATUS="success"
    return 0
  fi

  set +e
  env "${env_vars[@]}" bash "$script_path" 2>&1 | sed -u "s/^/[${run_id}][npu=${npu_spec}] /"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ $exit_code -ne 0 ]]; then
    printf '[fail][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
    LAST_RUN_STATUS="fail"
    return 1
  fi

  printf '[ok][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
  LAST_RUN_STATUS="success"
  return 0
}


run_algorithm_queue() {
  local algorithm="$1"
  local npu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local failure_count=0
  local success_count=0
  local skip_count=0
  local model_path
  local bit
  local weight_bits
  local activation_bits
  local query_bits
  local key_bits
  local value_bits
  local config
  local job_index=0

  case "$algorithm" in
    awq)
      for model_path in "${MODELS[@]}"; do
        for bit in "${AWQ_BITS[@]}"; do
          if (( job_index % worker_count == worker_index )); then
            if ! run_experiment awq "$model_path" "$bit" 16 16 16 16 "w${bit}a16" "$npu_spec"; then
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
      ;;
    gptq)
      for model_path in "${MODELS[@]}"; do
        for bit in "${GPTQ_BITS[@]}"; do
          if (( job_index % worker_count == worker_index )); then
            if ! run_experiment gptq "$model_path" "$bit" 16 16 16 16 "w${bit}a16" "$npu_spec"; then
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
      ;;
    smoothquant)
      for model_path in "${MODELS[@]}"; do
        for config in "${SMOOTHQUANT_CONFIGS[@]}"; do
          local smoothquant_alpha
          read -r weight_bits activation_bits query_bits key_bits value_bits smoothquant_alpha label <<< "$config"
          if (( job_index % worker_count == worker_index )); then
            if ! run_experiment smoothquant "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$label" "$npu_spec" "$smoothquant_alpha"; then
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
      ;;
    flatquant)
      for model_path in "${MODELS[@]}"; do
        for config in "${FLATQUANT_CONFIGS[@]}"; do
          read -r weight_bits activation_bits query_bits key_bits value_bits label <<< "$config"
          if (( job_index % worker_count == worker_index )); then
            if ! run_experiment flatquant "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$label" "$npu_spec"; then
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
      ;;
    splitquant)
      for model_path in "${MODELS[@]}"; do
        for config in "${SPLITQUANT_CONFIGS[@]}"; do
          read -r weight_bits activation_bits query_bits key_bits value_bits label <<< "$config"
          if (( job_index % worker_count == worker_index )); then
            if ! run_experiment splitquant "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$label" "$npu_spec"; then
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
      ;;
    *)
      printf 'Unknown algorithm queue: %s\n' "$algorithm" >&2
      return 1
      ;;
  esac

  printf '[worker-summary] %s worker=%s/%s npu=%s success=%s skip=%s fail=%s\n' \
    "$algorithm" \
    "$((worker_index + 1))" \
    "$worker_count" \
    "$npu_spec" \
    "$success_count" \
    "$skip_count" \
    "$failure_count"
  return "$failure_count"
}


launch_worker() {
  local algorithm="$1"
  local npu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local worker_label="${algorithm}_worker$((worker_index + 1))_of_${worker_count}"
  printf '[worker-start] %s on npu=%s\n' "$worker_label" "$npu_spec"
  run_algorithm_queue "$algorithm" "$npu_spec" "$worker_index" "$worker_count" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("$worker_label")
  WORKER_NPUS+=("$npu_spec")
}


launch_algorithm_workers() {
  local algorithm="$1"
  local npu_csv="$2"
  local -a npu_specs=()
  parse_npu_specs "$npu_csv" npu_specs
  if ((${#npu_specs[@]} == 0)); then
    printf '[worker-skip] %s has no npu configured\n' "$algorithm"
    return 0
  fi

  local worker_count="${#npu_specs[@]}"
  local index
  for index in "${!npu_specs[@]}"; do
    launch_worker "$algorithm" "${npu_specs[$index]}" "$index" "$worker_count"
  done
}


main() {
  local exit_code
  local had_failure=false

  if [[ "$ENABLE_AWQ" == "true" ]]; then
    launch_algorithm_workers awq "$AWQ_NPUS"
  fi

  if [[ "$ENABLE_GPTQ" == "true" ]]; then
    launch_algorithm_workers gptq "$GPTQ_NPUS"
  fi

  if [[ "$ENABLE_SMOOTHQUANT" == "true" ]]; then
    launch_algorithm_workers smoothquant "$SMOOTHQUANT_NPUS"
  fi

  if [[ "$ENABLE_FLATQUANT" == "true" ]]; then
    launch_algorithm_workers flatquant "$FLATQUANT_NPUS"
  fi

  if [[ "$ENABLE_SPLITQUANT" == "true" ]]; then
    launch_algorithm_workers splitquant "$SPLITQUANT_NPUS"
  fi

  if ((${#WORKER_PIDS[@]} == 0)); then
    printf 'No algorithm workers enabled.\n'
    return 0
  fi

  local index
  for index in "${!WORKER_PIDS[@]}"; do
    set +e
    wait "${WORKER_PIDS[$index]}"
    exit_code=$?
    set -e
    if [[ $exit_code -ne 0 ]]; then
      printf '[worker-fail] %s on npu=%s exited with code %s\n' \
        "${WORKER_LABELS[$index]}" \
        "${WORKER_NPUS[$index]}" \
        "$exit_code"
      had_failure=true
    else
      printf '[worker-done] %s on npu=%s\n' \
        "${WORKER_LABELS[$index]}" \
        "${WORKER_NPUS[$index]}"
    fi
  done

  if [[ "$had_failure" == "true" ]]; then
    return 1
  fi
  return 0
}


main "$@"
