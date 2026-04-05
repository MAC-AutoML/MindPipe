#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT_DEFAULT="$REPO_ROOT/my_results/quantization"
SEQUENCE_LENGTH_DEFAULT=2048
FLATQUANT_QUERY_BITS_DEFAULT=16
FLATQUANT_KEY_BITS_DEFAULT=4
FLATQUANT_VALUE_BITS_DEFAULT=4

MODELS=(
  "/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct"
  "/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct"
  "/mnt/82_store/LLM-weights/Llama-2-7b-hf"
  "/mnt/82_store/LLM-weights/Meta-Llama-3.1-8B-Instruct"
  "/mnt/82_store/LLM-weights/openbmb/MiniCPM-V"
)
AWQ_BITS=(2 3 4)
GPTQ_BITS=(2 3 4)
FLATQUANT_CONFIGS=(
  "2 16 16 4 4 w2a16"
  "3 16 16 4 4 w3a16"
  "4 16 16 4 4 w4a16"
  "4 4 16 4 4 w4a4"
  "8 8 16 4 4 w8a8"
  "16 16 16 16 16 w16a16"
)

ENABLE_AWQ="${ENABLE_AWQ:-true}"
ENABLE_GPTQ="${ENABLE_GPTQ:-true}"
ENABLE_FLATQUANT="${ENABLE_FLATQUANT:-true}"

AWQ_GPUS="${AWQ_GPUS:-0}"
GPTQ_GPUS="${GPTQ_GPUS:-7}"
FLATQUANT_GPUS="${FLATQUANT_GPUS:-1,2}"

EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-true}"
EVAL_VLM="${EVAL_VLM:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"
DRY_RUN="${DRY_RUN:-false}"

LAST_RUN_STATUS=""
WORKER_PIDS=()
WORKER_LABELS=()
WORKER_GPUS=()


output_root() {
  printf '%s' "${OUTPUT_DIR:-${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}}"
}


metrics_path_for() {
  local algorithm="$1"
  local model_path="$2"
  local weight_bits="$3"
  local activation_bits="$4"
  local query_bits="${5:-$FLATQUANT_QUERY_BITS_DEFAULT}"
  local key_bits="${6:-$FLATQUANT_KEY_BITS_DEFAULT}"
  local value_bits="${7:-$FLATQUANT_VALUE_BITS_DEFAULT}"
  local out_root
  out_root="$(output_root)"
  local model_name
  model_name="$(basename "$model_path")"

  if [[ "$algorithm" == "flatquant" ]]; then
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
  local key
  for key in HF_TOKEN OUTPUT_ROOT OUTPUT_DIR ZERO_SHOT_TASKS ZERO_SHOT_BATCH_SIZE ZERO_SHOT_NUM_FEWSHOT ZERO_SHOT_LIMIT VLM_DATASETS VLM_MODE VLM_WORK_DIR VLM_EVAL_KIT_ROOT VLM_JUDGE VLM_API_NPROC VLM_VERBOSE VLM_IGNORE_FAILED VLM_PRED_FORMAT; do
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
  local runtime_device
  runtime_device="$(resolve_runtime_device "$gpu_spec")"
  env_ref+=("DEVICE=$runtime_device")

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


run_experiment() {
  local algorithm="$1"
  local model_path="$2"
  local weight_bits="$3"
  local activation_bits="$4"
  local query_bits="$5"
  local key_bits="$6"
  local value_bits="$7"
  local label="$8"
  local gpu_spec="$9"
  local script_path="$REPO_ROOT/scripts/quantization/${algorithm}.sh"
  local model_name
  model_name="$(basename "$model_path")"
  local run_id="${model_name}__${algorithm}__${label}"
  local metrics_path
  local effective_eval_vlm="false"
  local out_root
  out_root="$(output_root)"
  metrics_path="$(metrics_path_for "$algorithm" "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits")"
  if should_eval_vlm "$model_path"; then
    effective_eval_vlm="true"
  fi

  if [[ "$FORCE_RERUN" != "true" ]] && is_complete "$metrics_path" "$effective_eval_vlm"; then
    printf '[skip][%s][gpu=%s] %s\n' "$algorithm" "$gpu_spec" "$run_id"
    LAST_RUN_STATUS="skip"
    return 0
  fi

  local -a env_vars=(
    "MODEL_PATH=$model_path"
    "WEIGHT_BITS=$weight_bits"
    "SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
    "EVAL_ZERO_SHOT=$EVAL_ZERO_SHOT"
    "EVAL_VLM=$effective_eval_vlm"
    "OUTPUT_ROOT=$out_root"
    "OUTPUT_DIR=$out_root"
  )
  append_device_env env_vars "$gpu_spec"
  if [[ "$algorithm" == "flatquant" ]]; then
    env_vars+=(
      "ACTIVATION_BITS=$activation_bits"
      "QUERY_BITS=$query_bits"
      "KEY_BITS=$key_bits"
      "VALUE_BITS=$value_bits"
    )
  fi
  append_passthrough_env env_vars

  printf '[run][%s][gpu=%s] %s\n' "$algorithm" "$gpu_spec" "$run_id"
  printf '  out: %s\n' "$metrics_path"
  printf '  eval_vlm: %s\n' "$effective_eval_vlm"
  printf '  cmd:'
  printf ' %q' env "${env_vars[@]}" bash "$script_path"
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    LAST_RUN_STATUS="success"
    return 0
  fi

  set +e
  env "${env_vars[@]}" bash "$script_path" 2>&1 | sed -u "s/^/[${run_id}][gpu=${gpu_spec}] /"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ $exit_code -ne 0 ]]; then
    printf '[fail][%s][gpu=%s] %s\n' "$algorithm" "$gpu_spec" "$run_id"
    LAST_RUN_STATUS="fail"
    return 1
  fi

  printf '[ok][%s][gpu=%s] %s\n' "$algorithm" "$gpu_spec" "$run_id"
  LAST_RUN_STATUS="success"
  return 0
}


run_algorithm_queue() {
  local algorithm="$1"
  local gpu_spec="$2"
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
            if ! run_experiment awq "$model_path" "$bit" 16 16 16 16 "w${bit}a16" "$gpu_spec"; then
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
            if ! run_experiment gptq "$model_path" "$bit" 16 16 16 16 "w${bit}a16" "$gpu_spec"; then
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
            if ! run_experiment flatquant "$model_path" "$weight_bits" "$activation_bits" "$query_bits" "$key_bits" "$value_bits" "$label" "$gpu_spec"; then
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

  printf '[worker-summary] %s worker=%s/%s gpu=%s success=%s skip=%s fail=%s\n' \
    "$algorithm" \
    "$((worker_index + 1))" \
    "$worker_count" \
    "$gpu_spec" \
    "$success_count" \
    "$skip_count" \
    "$failure_count"
  return "$failure_count"
}


launch_worker() {
  local algorithm="$1"
  local gpu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local worker_label="${algorithm}_worker$((worker_index + 1))_of_${worker_count}"
  printf '[worker-start] %s on gpu=%s\n' "$worker_label" "$gpu_spec"
  run_algorithm_queue "$algorithm" "$gpu_spec" "$worker_index" "$worker_count" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("$worker_label")
  WORKER_GPUS+=("$gpu_spec")
}


launch_algorithm_workers() {
  local algorithm="$1"
  local gpu_csv="$2"
  local -a gpu_specs=()
  parse_gpu_specs "$gpu_csv" gpu_specs
  if ((${#gpu_specs[@]} == 0)); then
    printf '[worker-skip] %s has no gpu configured\n' "$algorithm"
    return 0
  fi

  local worker_count="${#gpu_specs[@]}"
  local index
  for index in "${!gpu_specs[@]}"; do
    launch_worker "$algorithm" "${gpu_specs[$index]}" "$index" "$worker_count"
  done
}


main() {
  local exit_code
  local had_failure=false

  if [[ "$ENABLE_AWQ" == "true" ]]; then
    launch_algorithm_workers awq "$AWQ_GPUS"
  fi

  if [[ "$ENABLE_GPTQ" == "true" ]]; then
    launch_algorithm_workers gptq "$GPTQ_GPUS"
  fi

  if [[ "$ENABLE_FLATQUANT" == "true" ]]; then
    launch_algorithm_workers flatquant "$FLATQUANT_GPUS"
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
