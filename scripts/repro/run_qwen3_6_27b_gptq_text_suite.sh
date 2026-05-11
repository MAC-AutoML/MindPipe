#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_NAME="$(basename "$0")"
ALGORITHM="gptq"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen3.6-27B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/gptq_qwen3_6_27b_text_suite}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"

GPU_IDS="${GPU_IDS:-1,3}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQ_LENGTHS_STR="${SEQ_LENGTHS_STR:-512 2048}"
WEIGHT_BITS_STR="${WEIGHT_BITS_STR:-2 3 4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-true}"
WEIGHT_METHOD="${WEIGHT_METHOD:-gptq}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
ZERO_SHOT_TASKS_STR="${ZERO_SHOT_TASKS_STR:-boolq piqa rte winogrande arc_easy arc_challenge openbookqa}"
RUN_FP_BASELINE="${RUN_FP_BASELINE:-true}"
RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-true}"
SEED="${SEED:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
DRY_RUN="${DRY_RUN:-false}"

declare -a SEQ_LENGTHS=()
declare -a WEIGHT_BITS_LIST=()
declare -a ZERO_SHOT_TASKS=()
declare -a RUN_TAGS=()
declare -a FAILED_RUNS=()

parse_arrays() {
  read -r -a SEQ_LENGTHS <<< "$SEQ_LENGTHS_STR"
  read -r -a WEIGHT_BITS_LIST <<< "$WEIGHT_BITS_STR"
  read -r -a ZERO_SHOT_TASKS <<< "$ZERO_SHOT_TASKS_STR"
}

validate_config() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] MODEL_PATH not found: $MODEL_PATH"
    exit 1
  fi
  if [[ "$CALIBRATION_DATASET" != "pileval" ]]; then
    echo "[ERROR] CALIBRATION_DATASET must be pileval. current=$CALIBRATION_DATASET"
    exit 1
  fi
  if [[ "$CALIBRATION_SAMPLES" != "128" ]]; then
    echo "[ERROR] CALIBRATION_SAMPLES must be 128. current=$CALIBRATION_SAMPLES"
    exit 1
  fi
  if [[ "$ACTIVATION_BITS" != "16" ]]; then
    echo "[ERROR] ACTIVATION_BITS must be 16. current=$ACTIVATION_BITS"
    exit 1
  fi
  if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
    echo "[ERROR] GROUP_SIZE/WEIGHT_GROUP_SIZE must both be 128. current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
    exit 1
  fi
}

prepare_runtime_env() {
  export PYTHONNOUSERSITE=1
  export PYTHONHASHSEED="$SEED"
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_DISABLE_TELEMETRY=1
  if [[ "$DEVICE" == cuda:* ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
  fi
}

check_dependencies() {
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

find_metrics_file() {
  local run_output="$1"
  find "$run_output" -type f -name metrics.json 2>/dev/null | head -n 1 || true
}

is_metrics_complete() {
  local metrics_path="$1"
  [[ -f "$metrics_path" ]] || return 1
  grep -q '"perplexity"' "$metrics_path" || return 1
  if [[ "$RUN_ZERO_SHOT" == "true" ]]; then
    grep -q '"zero_shot"' "$metrics_path" || return 1
  fi
}

run_command() {
  local run_tag="$1"
  local log_path="$2"
  shift 2
  local -a cmd=("$@")
  local status=0

  printf '[INFO] Running %s:' "$run_tag"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] DRY_RUN=true, skipped execution. log_path=$log_path"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[ERROR] $run_tag failed with status=$status. log_path=$log_path"
    FAILED_RUNS+=("$run_tag")
    if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
      return 0
    fi
    exit "$status"
  fi
}

append_optional_args() {
  local -n cmd_ref="$1"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    cmd_ref+=(--hf_token "$HF_TOKEN")
  fi
  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    cmd_ref+=(--num_samples "$NUM_SAMPLES")
  fi
}

append_zero_shot_args() {
  local -n cmd_ref="$1"
  cmd_ref+=(--eval_zero_shot "$RUN_ZERO_SHOT")
  if [[ "$RUN_ZERO_SHOT" == "true" ]]; then
    cmd_ref+=(
      --zero_shot_tasks "${ZERO_SHOT_TASKS[@]}"
      --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
      --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    )
  fi
}

run_fp_baseline() {
  local sequence_length="$1"
  local run_tag="full_precision_seq${sequence_length}"
  local run_output="$OUTPUT_ROOT/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $run_tag (found complete metrics): $metrics_path"
    RUN_TAGS+=("$run_tag")
    return 0
  fi

  mkdir -p "$run_output"
  local log_path="$run_output/run.log"
  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --model_path "$MODEL_PATH"
    --device "$DEVICE"
    --device_map "$DEVICE_MAP"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --evaluation_dataset "$EVALUATION_DATASET"
    --sequence_length "$sequence_length"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --eval_ppl true
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )
  append_zero_shot_args cmd
  append_optional_args cmd
  run_command "$run_tag" "$log_path" "${cmd[@]}"
  RUN_TAGS+=("$run_tag")
}

run_quant_config() {
  local weight_bits="$1"
  local sequence_length="$2"
  local run_tag="w${weight_bits}a${ACTIVATION_BITS}_g${WEIGHT_GROUP_SIZE}_seq${sequence_length}"
  local run_output="$OUTPUT_ROOT/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $run_tag (found complete metrics): $metrics_path"
    RUN_TAGS+=("$run_tag")
    return 0
  fi

  mkdir -p "$run_output"
  local log_path="$run_output/run.log"
  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --quantization "$ALGORITHM"
    --model_path "$MODEL_PATH"
    --device "$DEVICE"
    --device_map "$DEVICE_MAP"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --calibration_dataset "$CALIBRATION_DATASET"
    --evaluation_dataset "$EVALUATION_DATASET"
    --calibration_samples "$CALIBRATION_SAMPLES"
    --sequence_length "$sequence_length"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$weight_bits"
    --activation_bits "$ACTIVATION_BITS"
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --weight_symmetric "$WEIGHT_SYMMETRIC"
    --weight_method "$WEIGHT_METHOD"
    --eval_ppl true
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )
  append_zero_shot_args cmd
  append_optional_args cmd
  run_command "$run_tag" "$log_path" "${cmd[@]}"
  RUN_TAGS+=("$run_tag")
}

summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "${RUN_TAGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
run_tags = sys.argv[2:]

print("[SUMMARY] run_tag\tperplexity\tzero_shot_acc_avg\tmetrics_path")
for tag in run_tags:
    metrics_files = sorted((output_root / tag).rglob("metrics.json"))
    if not metrics_files:
        print(f"{tag}\tMISSING\tMISSING\t-")
        continue
    metrics_path = metrics_files[0]
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    ppl = payload.get("perplexity")
    zero_shot = payload.get("zero_shot") or {}
    acc_avg = zero_shot.get("acc_avg") if isinstance(zero_shot, dict) else None
    ppl_str = f"{ppl:.6f}" if isinstance(ppl, (int, float)) else "NA"
    acc_str = f"{acc_avg:.4f}" if isinstance(acc_avg, (int, float)) else "NA"
    print(f"{tag}\t{ppl_str}\t{acc_str}\t{metrics_path}")
PY
  if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
    echo "[WARN] Failed runs: ${FAILED_RUNS[*]}"
  fi
}

print_banner() {
  echo "[INFO] $SCRIPT_NAME"
  echo "[INFO] model=$MODEL_PATH"
  echo "[INFO] algorithm=$ALGORITHM weight_method=$WEIGHT_METHOD"
  echo "[INFO] device=$DEVICE device_map=$DEVICE_MAP cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[INFO] dtype=$DTYPE attn_implementation=$ATTN_IMPLEMENTATION"
  echo "[INFO] output_root=$OUTPUT_ROOT"
  echo "[INFO] calibration_dataset=$CALIBRATION_DATASET calibration_samples=$CALIBRATION_SAMPLES"
  echo "[INFO] evaluation_dataset=$EVALUATION_DATASET max_eval_chunks=$MAX_EVAL_CHUNKS"
  echo "[INFO] zero_shot_tasks=${ZERO_SHOT_TASKS[*]}"
  echo "[INFO] group_size=$GROUP_SIZE weight_group_size=$WEIGHT_GROUP_SIZE activation_bits=$ACTIVATION_BITS"
  echo "[INFO] plan: full_precision(seq=${SEQ_LENGTHS[*]}) + GPTQ w{${WEIGHT_BITS_LIST[*]}}a16 g${WEIGHT_GROUP_SIZE} @ seq=${SEQ_LENGTHS[*]}"
}

main() {
  parse_arrays
  validate_config
  prepare_runtime_env
  check_dependencies
  mkdir -p "$OUTPUT_ROOT"
  print_banner

  if [[ "$RUN_FP_BASELINE" == "true" ]]; then
    for sequence_length in "${SEQ_LENGTHS[@]}"; do
      run_fp_baseline "$sequence_length"
    done
  fi

  for sequence_length in "${SEQ_LENGTHS[@]}"; do
    for weight_bits in "${WEIGHT_BITS_LIST[@]}"; do
      run_quant_config "$weight_bits" "$sequence_length"
    done
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] DRY_RUN=true, summary skipped."
    return 0
  fi
  summarize_results
}

main "$@"
