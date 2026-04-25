#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPT_NAME="$(basename "$0")"
ALGORITHM="gptq"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
GPU_ID="${GPU_ID:-0}"
NPU_ID="${NPU_ID:-0}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
WEIGHT_METHOD="${WEIGHT_METHOD:-gptq}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
# Keep hellaswag out by default to reduce runtime.
ZERO_SHOT_TASKS_STR="${ZERO_SHOT_TASKS_STR:-boolq piqa rte winogrande arc_easy arc_challenge openbookqa}"
FP_SEQUENCE_LENGTH="${FP_SEQUENCE_LENGTH:-2048}"
SEQ_LENGTHS_STR="${SEQ_LENGTHS_STR:-512 2048}"
WEIGHT_BITS_STR="${WEIGHT_BITS_STR:-2 3 4}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/gptq_qwen2_vl_2b_instruct}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "/mnt/82_store/LLM-weights/Qwen2-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen2-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/qwen-2-vl-2b-instruct" \
    "/mnt/82_store/zy/model/Qwen2-VL-2B-Instruct" \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen2-VL-2B-Instruct"
  do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [[ "$CALIBRATION_SAMPLES" != "128" ]]; then
  echo "[ERROR] CALIBRATION_SAMPLES must be 128 for this suite. current=$CALIBRATION_SAMPLES"
  exit 1
fi
if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
  echo "[ERROR] GROUP_SIZE/WEIGHT_GROUP_SIZE must both be 128. current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
  exit 1
fi

read -r -a ZERO_SHOT_TASKS <<< "$ZERO_SHOT_TASKS_STR"
read -r -a SEQ_LENGTHS <<< "$SEQ_LENGTHS_STR"
read -r -a WEIGHT_BITS_LIST <<< "$WEIGHT_BITS_STR"
RUN_TAGS=()

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

  "${cmd[@]}" | tee "$log_path"
}

run_fp_baseline() {
  local run_tag="fp_baseline_seq${FP_SEQUENCE_LENGTH}"
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
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --evaluation_dataset "$EVALUATION_DATASET"
    --sequence_length "$FP_SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --eval_ppl true
    --eval_zero_shot true
    --zero_shot_tasks "${ZERO_SHOT_TASKS[@]}"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )

  if [[ -n "${HF_TOKEN:-}" ]]; then
    cmd+=(--hf_token "$HF_TOKEN")
  fi
  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    cmd+=(--num_samples "$NUM_SAMPLES")
  fi

  run_command "$log_path" "${cmd[@]}"
  RUN_TAGS+=("$run_tag")
}

run_quant_config() {
  local weight_bits="$1"
  local sequence_length="$2"
  local run_tag="w${weight_bits}g${WEIGHT_GROUP_SIZE}_seq${sequence_length}"
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
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --weight_method "$WEIGHT_METHOD"
    --eval_ppl true
    --eval_zero_shot true
    --zero_shot_tasks "${ZERO_SHOT_TASKS[@]}"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --eval_vlm false
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )

  if [[ -n "${HF_TOKEN:-}" ]]; then
    cmd+=(--hf_token "$HF_TOKEN")
  fi
  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    cmd+=(--num_samples "$NUM_SAMPLES")
  fi

  run_command "$log_path" "${cmd[@]}"
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
    run_dir = output_root / tag
    metrics_files = sorted(run_dir.rglob("metrics.json"))
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
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi
if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found for Qwen2-VL-2B-Instruct: ${MODEL_PATH:-<empty>}"
  echo "Set MODEL_PATH manually, e.g.:"
  echo "  MODEL_PATH=/path/to/Qwen2-VL-2B-Instruct"
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "$DEVICE" == cuda:* ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
if [[ "$DEVICE" == npu:* ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
fi

if [[ "$DRY_RUN" != "true" ]]; then
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = ("torch", "transformers", "lm_eval", "qwen_vl_utils")
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)
print("[INFO] dependency check passed: torch/transformers/lm_eval/qwen_vl_utils")
PY
fi

mkdir -p "$OUTPUT_ROOT"

echo "[INFO] $SCRIPT_NAME"
echo "[INFO] model=$MODEL_PATH"
echo "[INFO] algorithm=$ALGORITHM"
echo "[INFO] weight_method=$WEIGHT_METHOD"
echo "[INFO] device=$DEVICE gpu_id=$GPU_ID"
echo "[INFO] output_root=$OUTPUT_ROOT"
echo "[INFO] calibration_dataset=$CALIBRATION_DATASET calibration_samples=$CALIBRATION_SAMPLES"
echo "[INFO] zero_shot_tasks=${ZERO_SHOT_TASKS[*]}"
echo "[INFO] group_size=$GROUP_SIZE (weight_group_size=$WEIGHT_GROUP_SIZE)"
echo "[INFO] plan: fp_baseline(seq=$FP_SEQUENCE_LENGTH) + quant(w{2,3,4}g${WEIGHT_GROUP_SIZE} @ seq=512,2048) => 7 runs"

run_fp_baseline
for sequence_length in "${SEQ_LENGTHS[@]}"; do
  for weight_bits in "${WEIGHT_BITS_LIST[@]}"; do
    run_quant_config "$weight_bits" "$sequence_length"
  done
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, summary skipped."
  exit 0
fi

summarize_results
