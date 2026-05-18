#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_NAME="$(basename "$0")"
ALGORITHM="sliderquant"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-999999}"
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-128}"
ACTIVATION_GROUP_SIZE="${ACTIVATION_GROUP_SIZE:-128}"
KV_GROUP_SIZE="${KV_GROUP_SIZE:-128}"
ROTATION_MODE="${ROTATION_MODE:-hadamard}"
SLIDERQUANT_EPOCHS="${SLIDERQUANT_EPOCHS:-10}"
SLIDERQUANT_QUANT_STEP="${SLIDERQUANT_QUANT_STEP:-2}"
SLIDERQUANT_NUM_LAYER="${SLIDERQUANT_NUM_LAYER:-1}"
SLIDERQUANT_USE_LORA="${SLIDERQUANT_USE_LORA:-false}"
SLIDERQUANT_LOW_MEMORY="${SLIDERQUANT_LOW_MEMORY:-true}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
RUN_FP16_BASELINE="${RUN_FP16_BASELINE:-true}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/sliderquant_text_${SUITE_VARIANT}_${RUN_ID}}"

# Matches current text suites while excluding hellaswag.
ZERO_SHOT_TASKS_STR="${ZERO_SHOT_TASKS_STR:-boolq piqa rte winogrande arc_easy arc_challenge openbookqa}"
read -r -a ZERO_SHOT_TASKS <<< "$ZERO_SHOT_TASKS_STR"

MODEL_SPECS=(
  "qwen2_5_0_5b|/mnt/82_store/LLM-weights/Qwen/Qwen2.5-0.5B"
  "llama3_2_1b|/mnt/82_store/LLM-weights/Llama-3.2-1B"
  "llama2_7b|/mnt/82_store/LLM-weights/Llama-2-7b-hf"
)

QUANT_CONFIGS=(
  "w3a16_g128_seq2048|3|16|16|16|16"
  "w4a16_g128_seq2048|4|16|16|16|16"
  "w4a4_g128_seq2048|4|4|4|4|4"
  "w8a8_g128_seq2048|8|8|8|8|8"
)

find_metrics_file() {
  local run_output="$1"
  find "$run_output" -type f -name metrics.json 2>/dev/null | head -n 1 || true
}

is_metrics_complete() {
  local metrics_path="$1"
  [[ -f "$metrics_path" ]] || return 1
  grep -q '"perplexity"' "$metrics_path" || return 1
  grep -q '"zero_shot"' "$metrics_path" || return 1
}

should_run_model() {
  local model_key="$1"
  if [[ -z "${MODEL_KEYS_STR:-}" ]]; then
    return 0
  fi
  local key
  for key in $MODEL_KEYS_STR; do
    [[ "$key" == "$model_key" ]] && return 0
  done
  return 1
}

append_optional_args() {
  local -n cmd_ref="$1"
  if [[ -n "$MAX_EVAL_CHUNKS" && "$MAX_EVAL_CHUNKS" != "none" ]]; then
    cmd_ref+=(--max_eval_chunks "$MAX_EVAL_CHUNKS")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    cmd_ref+=(--hf_token "$HF_TOKEN")
  fi
  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    cmd_ref+=(--num_samples "$NUM_SAMPLES")
  fi
}

run_command() {
  local log_path="$1"
  shift
  local -a cmd=("$@")

  printf '[INFO] Running command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] DRY_RUN=true, skipped execution. log_path=$log_path"
    return 0
  fi

  mkdir -p "$(dirname "$log_path")"
  "${cmd[@]}" 2>&1 | tee "$log_path"
  return "${PIPESTATUS[0]}"
}

record_status() {
  local model_key="$1"
  local run_tag="$2"
  local status="$3"
  local metrics_path="$4"
  printf '%s\t%s\t%s\t%s\n' "$model_key" "$run_tag" "$status" "$metrics_path" >> "$OUTPUT_ROOT/status.tsv"
}

run_fp_baseline() {
  local model_key="$1"
  local model_path="$2"
  local run_tag="fp16_seq2048"
  local run_output="$OUTPUT_ROOT/$model_key/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $model_key/$run_tag (found complete metrics): $metrics_path"
    record_status "$model_key" "$run_tag" "SKIP" "$metrics_path"
    return 0
  fi

  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --model_path "$model_path"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --evaluation_dataset "$EVALUATION_DATASET"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --eval_ppl true
    --eval_zero_shot true
    --zero_shot_tasks "${ZERO_SHOT_TASKS[@]}"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )
  append_optional_args cmd

  local status=0
  run_command "$run_output/run.log" "${cmd[@]}" || status=$?
  metrics_path="$(find_metrics_file "$run_output")"
  record_status "$model_key" "$run_tag" "$status" "${metrics_path:-NA}"
  return "$status"
}

run_quant_config() {
  local model_key="$1"
  local model_path="$2"
  local run_tag="$3"
  local weight_bits="$4"
  local activation_bits="$5"
  local query_bits="$6"
  local key_bits="$7"
  local value_bits="$8"
  local suffix="$run_tag"
  [[ "$SLIDERQUANT_ROTATE" == "true" ]] && suffix="${suffix}_rot"
  local run_output="$OUTPUT_ROOT/$model_key/$suffix"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $model_key/$suffix (found complete metrics): $metrics_path"
    record_status "$model_key" "$suffix" "SKIP" "$metrics_path"
    return 0
  fi

  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --quantization "$ALGORITHM"
    --model_path "$model_path"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --calibration_dataset "$CALIBRATION_DATASET"
    --evaluation_dataset "$EVALUATION_DATASET"
    --calibration_samples "$CALIBRATION_SAMPLES"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --weight_bits "$weight_bits"
    --activation_bits "$activation_bits"
    --query_bits "$query_bits"
    --key_bits "$key_bits"
    --value_bits "$value_bits"
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --activation_group_size "$ACTIVATION_GROUP_SIZE"
    --kv_group_size "$KV_GROUP_SIZE"
    --sliderquant_epochs "$SLIDERQUANT_EPOCHS"
    --sliderquant_quant_step "$SLIDERQUANT_QUANT_STEP"
    --sliderquant_num_layer "$SLIDERQUANT_NUM_LAYER"
    --sliderquant_use_lora "$SLIDERQUANT_USE_LORA"
    --sliderquant_low_memory "$SLIDERQUANT_LOW_MEMORY"
    --sliderquant_rotate "$SLIDERQUANT_ROTATE"
    --rotation_mode "$ROTATION_MODE"
    --eval_ppl true
    --eval_zero_shot true
    --zero_shot_tasks "${ZERO_SHOT_TASKS[@]}"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )
  append_optional_args cmd

  local status=0
  run_command "$run_output/run.log" "${cmd[@]}" || status=$?
  metrics_path="$(find_metrics_file "$run_output")"
  record_status "$model_key" "$suffix" "$status" "${metrics_path:-NA}"
  return "$status"
}

summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("[SUMMARY] model_key\trun_tag\tperplexity\tzero_shot_acc_avg\tmetrics_path")
for metrics_path in sorted(root.rglob("metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    parts = metrics_path.relative_to(root).parts
    model_key = parts[0] if parts else "unknown"
    run_tag = parts[1] if len(parts) > 1 else "unknown"
    ppl = payload.get("perplexity")
    zero_shot = payload.get("zero_shot") or {}
    acc_avg = zero_shot.get("acc_avg") if isinstance(zero_shot, dict) else None
    ppl_str = f"{ppl:.6f}" if isinstance(ppl, (int, float)) else "NA"
    acc_str = f"{acc_avg:.4f}" if isinstance(acc_avg, (int, float)) else "NA"
    print(f"{model_key}\t{run_tag}\t{ppl_str}\t{acc_str}\t{metrics_path}")
PY
}

check_setup() {
  if [[ -z "${SUITE_VARIANT:-}" ]]; then
    echo "[ERROR] SUITE_VARIANT must be set by the entry script."
    exit 1
  fi
  if [[ -z "${SLIDERQUANT_ROTATE:-}" ]]; then
    echo "[ERROR] SLIDERQUANT_ROTATE must be set by the entry script."
    exit 1
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi
  if [[ "$SEQUENCE_LENGTH" != "2048" ]]; then
    echo "[ERROR] SEQUENCE_LENGTH must be 2048. current=$SEQUENCE_LENGTH"
    exit 1
  fi
  if [[ "$CALIBRATION_SAMPLES" != "128" ]]; then
    echo "[ERROR] CALIBRATION_SAMPLES must be 128. current=$CALIBRATION_SAMPLES"
    exit 1
  fi
  if [[ "$CALIBRATION_DATASET" != "pileval" ]]; then
    echo "[ERROR] CALIBRATION_DATASET must be pileval. current=$CALIBRATION_DATASET"
    exit 1
  fi
  for spec in "${MODEL_SPECS[@]}"; do
    local model_key="${spec%%|*}"
    local model_path="${spec#*|}"
    should_run_model "$model_key" || continue
    if [[ ! -d "$model_path" ]]; then
      echo "[ERROR] Model path not found for $model_key: $model_path"
      exit 1
    fi
  done
}

dependency_check() {
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

for name in ("torch", "transformers", "lm_eval"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)
print("[INFO] dependency check passed: torch/transformers/lm_eval")
PY
}

export_runtime_env() {
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  export PYTHONNOUSERSITE=1
  export PYTHONHASHSEED="$SEED"
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_DISABLE_TELEMETRY=1
}

main() {
  check_setup
  export_runtime_env
  dependency_check
  mkdir -p "$OUTPUT_ROOT"
  : > "$OUTPUT_ROOT/status.tsv"

  echo "[INFO] script=$SCRIPT_NAME"
  echo "[INFO] suite_variant=$SUITE_VARIANT rotate=$SLIDERQUANT_ROTATE rotation_mode=$ROTATION_MODE"
  echo "[INFO] gpu_id=$GPU_ID device=$DEVICE"
  echo "[INFO] output_root=$OUTPUT_ROOT"
  echo "[INFO] zero_shot_tasks=${ZERO_SHOT_TASKS[*]}"
  echo "[INFO] configs: fp16 + ${QUANT_CONFIGS[*]}"

  local failed=0
  local spec
  for spec in "${MODEL_SPECS[@]}"; do
    local model_key="${spec%%|*}"
    local model_path="${spec#*|}"
    should_run_model "$model_key" || continue

    echo "[INFO] ===== model=$model_key path=$model_path ====="
    if [[ "$RUN_FP16_BASELINE" == "true" ]]; then
      run_fp_baseline "$model_key" "$model_path" || failed=$((failed + 1))
      if [[ "$CONTINUE_ON_ERROR" != "true" && "$failed" -gt 0 ]]; then
        exit 1
      fi
    fi

    local config
    for config in "${QUANT_CONFIGS[@]}"; do
      IFS='|' read -r run_tag w_bits a_bits q_bits k_bits v_bits <<< "$config"
      run_quant_config "$model_key" "$model_path" "$run_tag" "$w_bits" "$a_bits" "$q_bits" "$k_bits" "$v_bits" || failed=$((failed + 1))
      if [[ "$CONTINUE_ON_ERROR" != "true" && "$failed" -gt 0 ]]; then
        exit 1
      fi
    done
  done

  summarize_results
  if [[ "$failed" -gt 0 ]]; then
    echo "[WARN] suite finished with failed_jobs=$failed"
    return 1
  fi
  echo "[INFO] suite finished successfully"
}

main "$@"
