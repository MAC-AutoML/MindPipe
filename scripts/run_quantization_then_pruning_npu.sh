#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_DEFAULT="$(command -v python || true)"
if [[ -z "$PYTHON_BIN_DEFAULT" && -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN_DEFAULT="$CONDA_PREFIX/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT:-python}}"
# shellcheck source=./_npu_device_utils.sh
source "$REPO_ROOT/scripts/_npu_device_utils.sh"

# Shared experiment defaults
OUTPUT_ROOT_DEFAULT="$REPO_ROOT/my_results/workflow/npu"
DATA_PATH_DEFAULT="$REPO_ROOT/data"
SEQUENCE_LENGTH_DEFAULT=512
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SEED="${SEED:-42}"
DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
QUANTIZATION_CALIBRATION_SAMPLES="${QUANTIZATION_CALIBRATION_SAMPLES:-128}"
PRUNING_CALIBRATION_SAMPLES="${PRUNING_CALIBRATION_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EXECUTION_ORDER="quantization_then_pruning"

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
HF_ENDPOINT_DEFAULT="${HF_ENDPOINT_DEFAULT:-https://hf-mirror.com}"

# FlatQuant defaults aligned with scripts/quantization/flatquant.sh
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALIBRATION_BATCH_SIZE="${FLATQUANT_CALIBRATION_BATCH_SIZE:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
WEIGHT_METHOD="${WEIGHT_METHOD:-rtn}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-true}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-true}"
FLATQUANT_LWC="${FLATQUANT_LWC:-true}"
FLATQUANT_LAC="${FLATQUANT_LAC:-true}"
FLATQUANT_DIRECT_INV="${FLATQUANT_DIRECT_INV:-true}"
FLATQUANT_DEACTIVE_AMP="${FLATQUANT_DEACTIVE_AMP:-true}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-128}"

# Pruning defaults aligned with standalone pruning scripts
SPARSEGPT_BLOCK_SIZE="${SPARSEGPT_BLOCK_SIZE:-64}"
PRUNING_DAMP_PERCENT="${PRUNING_DAMP_PERCENT:-0.01}"
SPARSEGPT_STRUCTURE_PATTERN="${SPARSEGPT_STRUCTURE_PATTERN:-unstructured}"
WANDA_STRUCTURE_PATTERN="${WANDA_STRUCTURE_PATTERN:-unstructured}"
FLAP_STRUCTURE_PATTERN="${FLAP_STRUCTURE_PATTERN:-AL-AM}"
FLAP_METRICS="${FLAP_METRICS:-WIFV}"
FLAP_REMOVE_HEADS="${FLAP_REMOVE_HEADS:-8}"
PSEUDO_PRUNING="${PSEUDO_PRUNING:-true}"


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
  "/home/ma-user/work/modelzoo/Qwen/Qwen2.5-7B-Instruct"
  "/home/ma-user/work/modelzoo/Meta/Llama-2-7b-hf"
  "/home/ma-user/work/modelzoo/Meta/Meta-Llama-3.1-8B-Instruct"
  "/home/ma-user/work/modelzoo/Qwen/Qwen2.5-VL-7B-Instruct"
  "/home/ma-user/work/modelzoo/openbmb/MiniCPM-V"
  "$(pick_first_existing_path \
    "/home/ma-user/work/modelzoo/Qwen/Qwen3-VL-2B-Instruct" \
    "/home/ma-user/work/modelzoo/Qwen/Qwen3-VL-2B")"
  "$(pick_first_existing_path \
    "/home/ma-user/work/modelzoo/Qwen/Qwen3-0.6B")"
  "$(pick_first_existing_path \
    "/home/ma-user/work/modelzoo/Qwen/Qwen3.5-4B" \
    "/home/ma-user/work/modelzoo/Qwen/Qwen3_5-4B")"
)
FLATQUANT_CONFIGS=(
  "4 16 16 4 4 w4a16"
  "4 4 16 4 4 w4a4"
  "8 8 16 4 4 w8a8"
)

PRUNING_CONFIGS=(
  "flap 0.2"
  "sparsegpt 0.2"
  "sparsegpt 0.4"
  "sparsegpt 0.5"
  "wanda 0.2"
  "wanda 0.4"
  "wanda 0.5"
)

# Worker scheduling
WORKFLOW_NPUS="${WORKFLOW_NPUS:-4,5,6,7}"

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


has_local_c4() {
  [[ -f "$DATA_PATH/c4/c4-train.00000-of-01024.json.gz" || -f "$DATA_PATH/c4/en/c4-train.00000-of-01024.json.gz" ]]
}


has_local_wikitext2() {
  [[ -f "$DATA_PATH/wikitext2/wiki.train.raw" && -f "$DATA_PATH/wikitext2/wiki.valid.raw" && -f "$DATA_PATH/wikitext2/wiki.test.raw" ]]
}


has_local_pileval() {
  [[ -f "$DATA_PATH/pileval/val.jsonl" ]]
}


resolve_quantization_calibration_dataset() {
  if [[ -n "${QUANTIZATION_CALIBRATION_DATASET:-}" ]]; then
    printf '%s' "$QUANTIZATION_CALIBRATION_DATASET"
    return 0
  fi
  if has_local_pileval; then
    printf 'pileval'
  else
    printf 'wikitext2'
  fi
}


resolve_pruning_calibration_dataset() {
  local pruning_algorithm="$1"
  local explicit_value=""
  case "$pruning_algorithm" in
    flap)
      explicit_value="${FLAP_CALIBRATION_DATASET:-${PRUNING_CALIBRATION_DATASET:-}}"
      if [[ -n "$explicit_value" ]]; then
        printf '%s' "$explicit_value"
      else
        printf 'wikitext2'
      fi
      ;;
    sparsegpt)
      explicit_value="${SPARSEGPT_CALIBRATION_DATASET:-${PRUNING_CALIBRATION_DATASET:-}}"
      if [[ -n "$explicit_value" ]]; then
        printf '%s' "$explicit_value"
      elif has_local_c4; then
        printf 'c4'
      else
        printf 'wikitext2'
      fi
      ;;
    wanda)
      explicit_value="${WANDA_CALIBRATION_DATASET:-${PRUNING_CALIBRATION_DATASET:-}}"
      if [[ -n "$explicit_value" ]]; then
        printf '%s' "$explicit_value"
      elif has_local_c4; then
        printf 'c4'
      else
        printf 'wikitext2'
      fi
      ;;
    *)
      return 1
      ;;
  esac
}


format_decimal() {
  local value="$1"
  if [[ "$value" == *.* ]]; then
    while [[ "$value" == *0 ]]; do
      value="${value%0}"
    done
    value="${value%.}"
  fi
  printf '%s' "$value"
}


require_python_import() {
  local module_name="$1"
  local install_hint="$2"
  if "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    printf '[preflight-ok] python module `%s` is available\n' "$module_name"
    return 0
  fi
  printf '[preflight-fail] missing python module `%s`. %s\n' "$module_name" "$install_hint" >&2
  return 1
}


check_vlm_eval_runtime() {
  local vlm_eval_kit_root="${VLM_EVAL_KIT_ROOT:-$REPO_ROOT/third_party/VLMEvalKit}"
  if [[ ! -d "$vlm_eval_kit_root" ]]; then
    printf '[preflight-fail] VLMEvalKit root does not exist: %s\n' "$vlm_eval_kit_root" >&2
    return 1
  fi

  if PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    VLM_EVAL_KIT_ROOT_TO_CHECK="$vlm_eval_kit_root" \
    "$PYTHON_BIN" -c 'import os; from evaluation.vlm_eval import _load_vlmeval_modules; _load_vlmeval_modules(os.environ["VLM_EVAL_KIT_ROOT_TO_CHECK"])' >/dev/null 2>&1; then
    printf '[preflight-ok] VLM evaluation runtime is available: %s\n' "$vlm_eval_kit_root"
    return 0
  fi

  printf '[preflight-fail] VLM evaluation runtime check failed for %s\n' "$vlm_eval_kit_root" >&2
  printf '  hint: install missing VLMEvalKit dependencies or set VLM_EVAL_KIT_ROOT correctly.\n' >&2
  return 1
}


require_dataset_availability() {
  local dataset_name="$1"
  local usage_label="$2"
  local normalized_name="${dataset_name,,}"

  case "$normalized_name" in
    wikitext2)
      if has_local_wikitext2; then
        printf '[preflight-ok] %s dataset `%s` found under %s/wikitext2\n' "$usage_label" "$dataset_name" "$DATA_PATH"
      else
        printf '[preflight-warn] %s dataset `%s` is not present under %s/wikitext2; the workflow will try Hugging Face at runtime.\n' "$usage_label" "$dataset_name" "$DATA_PATH" >&2
      fi
      ;;
    c4)
      if has_local_c4; then
        printf '[preflight-ok] %s dataset `%s` found under %s/c4\n' "$usage_label" "$dataset_name" "$DATA_PATH"
      else
        printf '[preflight-warn] %s dataset `%s` is not present under %s/c4; the workflow will try Hugging Face at runtime.\n' "$usage_label" "$dataset_name" "$DATA_PATH" >&2
      fi
      ;;
    pileval)
      if has_local_pileval; then
        printf '[preflight-ok] %s dataset `%s` found under %s/pileval\n' "$usage_label" "$dataset_name" "$DATA_PATH"
      else
        printf '[preflight-fail] %s dataset `%s` not found under %s/pileval\n' "$usage_label" "$dataset_name" "$DATA_PATH" >&2
        return 1
      fi
      ;;
    *)
      printf '[preflight-warn] skipping local availability check for %s dataset `%s`\n' "$usage_label" "$dataset_name" >&2
      ;;
  esac
}


preflight_checks() {
  local failed=0
  local model_path
  local need_vlm=false
  local quantization_dataset
  local pruning_config
  local pruning_algorithm
  local sparsity_ratio
  local pruning_dataset
  local -a workflow_npu_specs=()

  printf '[preflight] repo=%s\n' "$REPO_ROOT"
  printf '[preflight] data_path=%s\n' "$DATA_PATH"
  printf '[preflight] workflow_npus=%s\n' "$WORKFLOW_NPUS"
  printf '[preflight] python=%s\n' "$PYTHON_BIN"

  parse_npu_specs "$WORKFLOW_NPUS" workflow_npu_specs
  validate_npu_specs "workflow" workflow_npu_specs || failed=1

  for model_path in "${MODELS[@]}"; do
    if [[ -d "$model_path" ]]; then
      printf '[preflight-ok] model path exists: %s\n' "$model_path"
    else
      printf '[preflight-fail] model path does not exist: %s\n' "$model_path" >&2
      failed=1
    fi
    if should_eval_vlm "$model_path"; then
      need_vlm=true
    fi
  done

  if [[ "$EVAL_ZERO_SHOT" == "true" ]]; then
    require_python_import "lm_eval" "Install lm-evaluation-harness into the same Python environment running this script." || failed=1
  fi

  if [[ "$need_vlm" == "true" ]]; then
    check_vlm_eval_runtime || failed=1
  fi

  require_dataset_availability "$EVALUATION_DATASET" "evaluation" || failed=1

  quantization_dataset="$(resolve_quantization_calibration_dataset)"
  require_dataset_availability "$quantization_dataset" "quantization calibration" || failed=1

  for pruning_config in "${PRUNING_CONFIGS[@]}"; do
    read -r pruning_algorithm sparsity_ratio <<< "$pruning_config"
    pruning_dataset="$(resolve_pruning_calibration_dataset "$pruning_algorithm")"
    require_dataset_availability "$pruning_dataset" "pruning calibration (${pruning_algorithm})" || failed=1
  done

  if [[ $failed -ne 0 ]]; then
    printf '[preflight-fail] aborting before launching workflow workers\n' >&2
    return 1
  fi

  printf '[preflight-ok] workflow launch checks passed\n'
  return 0
}


metrics_path_for() {
  local model_path="$1"
  local weight_bits="$2"
  local activation_bits="$3"
  local pruning_algorithm="$4"
  local sparsity_ratio="$5"
  local out_root
  out_root="$(output_root)"
  local model_name
  model_name="$(basename "$model_path")"
  local sparsity_tag
  sparsity_tag="$(format_decimal "$sparsity_ratio")"

  # Mirrors workflow/builder.py multi-stage output layout.
  printf '%s/%s/%s/flatquant__%s/flatquant_w%sa%s_%s_s%s_seq%s/metrics.json' \
    "$out_root" \
    "$model_name" \
    "$EXECUTION_ORDER" \
    "$pruning_algorithm" \
    "$weight_bits" \
    "$activation_bits" \
    "$pruning_algorithm" \
    "$sparsity_tag" \
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
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    return 1
  fi
  local requested_id="$npu_spec"
  if [[ "$requested_id" == npu:* ]]; then
    requested_id="${requested_id#npu:}"
  fi
  if mindpipe_resolve_visible_device_id "$requested_id"; then
    return 0
  fi
  printf '[npu-map-fail] unable to map `%s` to ASCEND_RT_VISIBLE_DEVICES. %s\n' "$npu_spec" "$MINDPIPE_NPU_ID_MAP_ERROR" >&2
  return 1
}


resolve_runtime_device() {
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    printf 'cpu'
  else
    printf 'npu:0'
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


validate_npu_specs() {
  local label="$1"
  local -n npu_specs_ref="$2"
  local -A seen_visible_devices=()
  local -a mapping_entries=()
  local npu_spec
  local visible_device

  if (( ${#npu_specs_ref[@]} == 0 )); then
    printf '[preflight-warn] no %s NPU specs were configured\n' "$label" >&2
    return 0
  fi

  for npu_spec in "${npu_specs_ref[@]}"; do
    if [[ "$npu_spec" == "cpu" ]]; then
      mapping_entries+=("cpu")
      continue
    fi

    if ! visible_device="$(resolve_visible_devices "$npu_spec")"; then
      printf '[preflight-fail] %s NPU spec `%s` is not usable on this host\n' "$label" "$npu_spec" >&2
      return 1
    fi

    if [[ -n "${seen_visible_devices[$visible_device]+x}" ]]; then
      printf '[preflight-fail] %s NPU specs `%s` and `%s` both resolve to logical device %s\n' "$label" "${seen_visible_devices[$visible_device]}" "$npu_spec" "$visible_device" >&2
      return 1
    fi

    seen_visible_devices["$visible_device"]="$npu_spec"
    mapping_entries+=("${npu_spec}->${visible_device}")
  done

  local mapping_summary
  printf -v mapping_summary '%s, ' "${mapping_entries[@]}"
  mapping_summary="${mapping_summary%, }"
  printf '[preflight-ok] %s NPU mapping: %s\n' "$label" "$mapping_summary"
  return 0
}


append_optional_arg() {
  local -n target_ref="$1"
  local flag="$2"
  local value="${3:-}"
  if [[ -n "$value" ]]; then
    target_ref+=("$flag" "$value")
  fi
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
  append_optional_arg cmd_ref --vlm_work_dir "${VLM_WORK_DIR:-}"
  append_optional_arg cmd_ref --vlm_eval_kit_root "${VLM_EVAL_KIT_ROOT:-}"
  append_optional_arg cmd_ref --vlm_judge "${VLM_JUDGE:-}"
}


append_pruning_args() {
  local -n cmd_ref="$1"
  local pruning_algorithm="$2"
  local pruning_calibration_dataset
  pruning_calibration_dataset="$(resolve_pruning_calibration_dataset "$pruning_algorithm")"

  case "$pruning_algorithm" in
    flap)
      cmd_ref+=(
        --structure_pattern "$FLAP_STRUCTURE_PATTERN"
        --flap_metrics "$FLAP_METRICS"
        --flap_remove_heads "$FLAP_REMOVE_HEADS"
        --pseudo_pruning "$PSEUDO_PRUNING"
      )
      cmd_ref+=(--pruning_calibration_dataset "$pruning_calibration_dataset")
      ;;
    sparsegpt)
      cmd_ref+=(
        --structure_pattern "$SPARSEGPT_STRUCTURE_PATTERN"
        --block_size "$SPARSEGPT_BLOCK_SIZE"
        --pruning_damp_percent "$PRUNING_DAMP_PERCENT"
      )
      cmd_ref+=(--pruning_calibration_dataset "$pruning_calibration_dataset")
      ;;
    wanda)
      cmd_ref+=(
        --structure_pattern "$WANDA_STRUCTURE_PATTERN"
      )
      cmd_ref+=(--pruning_calibration_dataset "$pruning_calibration_dataset")
      ;;
    *)
      printf 'Unknown pruning algorithm: %s\n' "$pruning_algorithm" >&2
      return 1
      ;;
  esac
}


run_experiment() {
  local model_path="$1"
  local weight_bits="$2"
  local activation_bits="$3"
  local query_bits="$4"
  local key_bits="$5"
  local value_bits="$6"
  local quant_label="$7"
  local pruning_algorithm="$8"
  local sparsity_ratio="$9"
  local npu_spec="${10}"
  local out_root
  out_root="$(output_root)"
  local model_name
  model_name="$(basename "$model_path")"
  local runtime_device
  runtime_device="$(resolve_runtime_device "$npu_spec")"
  local effective_eval_vlm="false"
  local metrics_path
  local quantization_calibration_dataset
  metrics_path="$(metrics_path_for "$model_path" "$weight_bits" "$activation_bits" "$pruning_algorithm" "$sparsity_ratio")"
  quantization_calibration_dataset="$(resolve_quantization_calibration_dataset)"
  if should_eval_vlm "$model_path"; then
    effective_eval_vlm="true"
  fi

  local sparsity_tag
  sparsity_tag="$(format_decimal "$sparsity_ratio")"
  local run_id="${model_name}__flatquant__${quant_label}__${pruning_algorithm}_s${sparsity_tag}"

  if [[ "$FORCE_RERUN" != "true" ]] && is_complete "$metrics_path" "$effective_eval_vlm"; then
    printf '[skip][workflow][npu=%s] %s\n' "$npu_spec" "$run_id"
    LAST_RUN_STATUS="skip"
    return 0
  fi

  local -a env_vars=()
  if [[ "$npu_spec" != "cpu" ]]; then
    local visible_devices
    visible_devices="$(resolve_visible_devices "$npu_spec")" || {
      LAST_RUN_STATUS="fail"
      return 1
    }
    env_vars+=("ASCEND_RT_VISIBLE_DEVICES=$visible_devices")
  fi
  env_vars+=("HF_ENDPOINT=${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}")

  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --model_path "$model_path"
    --device "$runtime_device"
    --device_map "$DEVICE_MAP"
    --dtype "$DTYPE"
    --log_level "$LOG_LEVEL"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --data_path "$DATA_PATH"
    --seed "$SEED"
    --output_dir "$out_root"
    --quantization flatquant
    --pruning "$pruning_algorithm"
    --execution_order "$EXECUTION_ORDER"
    --evaluation_dataset "$EVALUATION_DATASET"
    --sequence_length "${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --eval_ppl "$EVAL_PPL"
    --eval_zero_shot "$EVAL_ZERO_SHOT"
    --eval_vlm "$effective_eval_vlm"
    --quantization_calibration_dataset "$quantization_calibration_dataset"
    --quantization_calibration_samples "$QUANTIZATION_CALIBRATION_SAMPLES"
    --pruning_calibration_samples "$PRUNING_CALIBRATION_SAMPLES"
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
    --sparsity_ratio "$sparsity_ratio"
  )
  append_optional_arg cmd --hf_token "${HF_TOKEN:-}"
  append_zero_shot_args cmd
  append_vlm_args cmd "$effective_eval_vlm"
  append_pruning_args cmd "$pruning_algorithm"
  append_optional_arg cmd --num_samples "${NUM_SAMPLES:-}"

  printf '[run][workflow][npu=%s] %s\n' "$npu_spec" "$run_id"
  printf '  out: %s\n' "$metrics_path"
  printf '  eval_ppl: %s\n' "$EVAL_PPL"
  printf '  eval_zero_shot: %s\n' "$EVAL_ZERO_SHOT"
  printf '  eval_vlm: %s\n' "$effective_eval_vlm"
  printf '  cmd:'
  printf ' %q' env "${env_vars[@]}" "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    LAST_RUN_STATUS="success"
    return 0
  fi

  set +e
  env "${env_vars[@]}" "${cmd[@]}" 2>&1 | sed -u "s/^/[${run_id}][npu=${npu_spec}] /"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ $exit_code -ne 0 ]]; then
    printf '[fail][workflow][npu=%s] %s\n' "$npu_spec" "$run_id"
    LAST_RUN_STATUS="fail"
    return 1
  fi

  printf '[ok][workflow][npu=%s] %s\n' "$npu_spec" "$run_id"
  LAST_RUN_STATUS="success"
  return 0
}


run_workflow_queue() {
  local npu_spec="$1"
  local worker_index="$2"
  local worker_count="$3"
  local failure_count=0
  local success_count=0
  local skip_count=0
  local model_path
  local quant_config
  local pruning_config
  local weight_bits
  local activation_bits
  local query_bits
  local key_bits
  local value_bits
  local quant_label
  local pruning_algorithm
  local sparsity_ratio
  local job_index=0

  for model_path in "${MODELS[@]}"; do
    for quant_config in "${FLATQUANT_CONFIGS[@]}"; do
      read -r weight_bits activation_bits query_bits key_bits value_bits quant_label <<< "$quant_config"
      for pruning_config in "${PRUNING_CONFIGS[@]}"; do
        read -r pruning_algorithm sparsity_ratio <<< "$pruning_config"
        if (( job_index % worker_count == worker_index )); then
          if ! run_experiment \
            "$model_path" \
            "$weight_bits" \
            "$activation_bits" \
            "$query_bits" \
            "$key_bits" \
            "$value_bits" \
            "$quant_label" \
            "$pruning_algorithm" \
            "$sparsity_ratio" \
            "$npu_spec"; then
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

  printf '[worker-summary] workflow worker=%s/%s npu=%s success=%s skip=%s fail=%s\n' \
    "$((worker_index + 1))" \
    "$worker_count" \
    "$npu_spec" \
    "$success_count" \
    "$skip_count" \
    "$failure_count"
  return "$failure_count"
}


launch_worker() {
  local npu_spec="$1"
  local worker_index="$2"
  local worker_count="$3"
  local worker_label="workflow_worker$((worker_index + 1))_of_${worker_count}"
  printf '[worker-start] %s on npu=%s\n' "$worker_label" "$npu_spec"
  run_workflow_queue "$npu_spec" "$worker_index" "$worker_count" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("$worker_label")
  WORKER_NPUS+=("$npu_spec")
}


launch_workers() {
  local npu_csv="$1"
  local -a npu_specs=()
  parse_npu_specs "$npu_csv" npu_specs
  if ((${#npu_specs[@]} == 0)); then
    printf '[worker-skip] no npu configured\n'
    return 0
  fi

  local worker_count="${#npu_specs[@]}"
  local index
  for index in "${!npu_specs[@]}"; do
    launch_worker "${npu_specs[$index]}" "$index" "$worker_count"
  done
}


main() {
  local exit_code
  local had_failure=false

  preflight_checks
  launch_workers "$WORKFLOW_NPUS"

  if ((${#WORKER_PIDS[@]} == 0)); then
    printf 'No workflow workers enabled.\n'
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
