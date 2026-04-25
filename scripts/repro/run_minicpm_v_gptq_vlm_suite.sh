#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPT_NAME="$(basename "$0")"
ALGORITHM="gptq"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
GPU_ID="${GPU_ID:-0}"
NPU_ID="${NPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
SEED="${SEED:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
GPTQ_VLM_DATASET_NAME="${GPTQ_VLM_DATASET_NAME:-ChartQA_TEST}"
GPTQ_VLM_CALIB_NUM="${GPTQ_VLM_CALIB_NUM:-100}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"

WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
DAMP_PERCENT="${DAMP_PERCENT:-0.05}"

RUN_FP16_BASELINE="${RUN_FP16_BASELINE:-true}"
# Format: run_tag:quant_visual:quant_connector:quant_llm
RUN_CONFIGS_STR="${RUN_CONFIGS_STR:-full:true:true:true}"

VLM_MODE="${VLM_MODE:-all}"
VLM_DATASETS_STR="${VLM_DATASETS_STR:-ChartQA}"
VLM_API_NPROC="${VLM_API_NPROC:-1}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
VLM_EVAL_KIT_ROOT="${VLM_EVAL_KIT_ROOT:-/mnt/42_store/zy/HUAWEI/work1/MQuant/third/VLMEvalKit}"
VLM_USE_CACHE="${VLM_USE_CACHE:-}"
VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-}"
VLM_SAMPLE_CLEANUP="${VLM_SAMPLE_CLEANUP:-}"
VLM_VERBOSE="${VLM_VERBOSE:-}"
VLM_IGNORE_FAILED="${VLM_IGNORE_FAILED:-}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

LMU_DATA_DIR="${LMU_DATA_DIR:-$REPO_ROOT/.lmu_data}"
MPL_CONFIG_DIR="${MPL_CONFIG_DIR:-/tmp/mpl_minicpm_v_gptq_vlm_gpu${GPU_ID}}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
SMOKE="${SMOKE:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/gptq_minicpm_v_chartqa100_vlm}"

if [[ "$SMOKE" == "true" ]]; then
  GPTQ_VLM_CALIB_NUM="${SMOKE_GPTQ_VLM_CALIB_NUM:-8}"
  NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-20}"
  VLM_API_NPROC="${SMOKE_VLM_API_NPROC:-1}"
  VLM_USE_CACHE="${VLM_USE_CACHE:-true}"
  VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-16}"
  VLM_SAMPLE_CLEANUP="${VLM_SAMPLE_CLEANUP:-false}"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "/mnt/82_store/LLM-weights/openbmb/MiniCPM-V" \
    "/mnt/82_store/LLM-weights/openbmb/MiniCPM-V-2" \
    "/mnt/82_store/zy/model/openbmb/MiniCPM-V" \
    "/mnt/82_store/huggingface/datasets/openbmb/MiniCPM-V"
  do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi

normalize_dataset_name() {
  local raw="$1"
  local lowered="${raw,,}"
  case "$lowered" in
    chartqa|chartqa_test)
      echo "ChartQA_TEST"
      ;;
    infovqa|infovqa_val|infovqa_test)
      echo "InfoVQA_VAL"
      ;;
    ocrbench)
      echo "OCRBench"
      ;;
    textvqa|textvqa_val)
      echo "TextVQA_VAL"
      ;;
    *)
      echo "$raw"
      ;;
  esac
}

read -r -a VLM_DATASETS_RAW <<< "$VLM_DATASETS_STR"
declare -A _seen_ds=()
VLM_DATASETS=()
for ds in "${VLM_DATASETS_RAW[@]}"; do
  canonical="$(normalize_dataset_name "$ds")"
  if [[ -n "${_seen_ds[$canonical]:-}" ]]; then
    continue
  fi
  _seen_ds["$canonical"]=1
  VLM_DATASETS+=("$canonical")
done
GPTQ_VLM_DATASET_NAME="$(normalize_dataset_name "$GPTQ_VLM_DATASET_NAME")"

read -r -a RUN_CONFIGS <<< "$RUN_CONFIGS_STR"
RUN_TAGS=()

if [[ "$SEQUENCE_LENGTH" != "128" ]]; then
  echo "[ERROR] SEQUENCE_LENGTH must be 128 for this suite. current=$SEQUENCE_LENGTH"
  exit 1
fi
if [[ "$WEIGHT_BITS" != "4" || "$ACTIVATION_BITS" != "16" ]]; then
  echo "[ERROR] WEIGHT_BITS/ACTIVATION_BITS must be 4/16 for this suite. current=$WEIGHT_BITS/$ACTIVATION_BITS"
  exit 1
fi
if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
  echo "[ERROR] GROUP_SIZE/WEIGHT_GROUP_SIZE must both be 128. current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
  exit 1
fi
if [[ "$GPTQ_VLM_CALIB_NUM" -le 0 ]]; then
  echo "[ERROR] GPTQ_VLM_CALIB_NUM must be positive. current=$GPTQ_VLM_CALIB_NUM"
  exit 1
fi

find_metrics_file() {
  local run_output="$1"
  find "$run_output" -type f -name metrics.json 2>/dev/null | head -n 1 || true
}

is_metrics_complete() {
  local metrics_path="$1"
  [[ -f "$metrics_path" ]] || return 1
  grep -q '"vlm_eval"' "$metrics_path" || return 1
  grep -q '"inference_completed": true' "$metrics_path" || return 1
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

  "${cmd[@]}" 2>&1 | tee "$log_path"
}

append_optional_args() {
  local -n _cmd_ref="$1"

  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    _cmd_ref+=(--num_samples "$NUM_SAMPLES")
  fi
  if [[ -n "${VLM_WORK_DIR:-}" ]]; then
    _cmd_ref+=(--vlm_work_dir "$VLM_WORK_DIR")
  fi
  if [[ -n "${VLM_JUDGE:-}" ]]; then
    _cmd_ref+=(--vlm_judge "$VLM_JUDGE")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    _cmd_ref+=(--hf_token "$HF_TOKEN")
  fi
  if [[ -n "${VLM_USE_CACHE:-}" ]]; then
    _cmd_ref+=(--vlm_use_cache "$VLM_USE_CACHE")
  fi
  if [[ -n "${VLM_MAX_NEW_TOKENS:-}" ]]; then
    _cmd_ref+=(--vlm_max_new_tokens "$VLM_MAX_NEW_TOKENS")
  fi
  if [[ -n "${VLM_SAMPLE_CLEANUP:-}" ]]; then
    _cmd_ref+=(--vlm_sample_cleanup "$VLM_SAMPLE_CLEANUP")
  fi
  if [[ -n "${VLM_VERBOSE:-}" ]]; then
    _cmd_ref+=(--vlm_verbose "$VLM_VERBOSE")
  fi
  if [[ -n "${VLM_IGNORE_FAILED:-}" ]]; then
    _cmd_ref+=(--vlm_ignore_failed "$VLM_IGNORE_FAILED")
  fi
}

run_fp16_baseline() {
  local run_tag="fp16_seq${SEQUENCE_LENGTH}"
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
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --eval_ppl false
    --eval_zero_shot false
    --eval_vlm true
    --vlm_datasets "${VLM_DATASETS[@]}"
    --vlm_mode "$VLM_MODE"
    --vlm_api_nproc "$VLM_API_NPROC"
    --vlm_pred_format "$VLM_PRED_FORMAT"
    --vlm_eval_kit_root "$VLM_EVAL_KIT_ROOT"
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )
  append_optional_args cmd
  run_command "$log_path" "${cmd[@]}"
  RUN_TAGS+=("$run_tag")
}

run_gptq_config() {
  local run_tag="$1"
  local quant_visual="$2"
  local quant_connector="$3"
  local quant_llm="$4"

  local run_output="$OUTPUT_ROOT/${run_tag}_w${WEIGHT_BITS}a${ACTIVATION_BITS}_seq${SEQUENCE_LENGTH}"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"
  local resolved_tag
  resolved_tag="$(basename "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $resolved_tag (found complete metrics): $metrics_path"
    RUN_TAGS+=("$resolved_tag")
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
    --calibration_samples "$CALIBRATION_SAMPLES"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$WEIGHT_BITS"
    --activation_bits "$ACTIVATION_BITS"
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --damp_percent "$DAMP_PERCENT"
    --gptq_vlm_dataset_name "$GPTQ_VLM_DATASET_NAME"
    --gptq_vlm_calib_num "$GPTQ_VLM_CALIB_NUM"
    --gptq_vlm_quant_visual "$quant_visual"
    --gptq_vlm_quant_connector "$quant_connector"
    --gptq_vlm_quant_llm "$quant_llm"
    --eval_ppl false
    --eval_zero_shot false
    --eval_vlm true
    --vlm_datasets "${VLM_DATASETS[@]}"
    --vlm_mode "$VLM_MODE"
    --vlm_api_nproc "$VLM_API_NPROC"
    --vlm_pred_format "$VLM_PRED_FORMAT"
    --vlm_eval_kit_root "$VLM_EVAL_KIT_ROOT"
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )
  append_optional_args cmd
  run_command "$log_path" "${cmd[@]}"
  RUN_TAGS+=("$resolved_tag")
}

summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "${RUN_TAGS[@]}" -- "${VLM_DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

args = list(sys.argv[1:])
if "--" in args:
    sep = args.index("--")
    output_root = Path(args[0])
    run_tags = args[1:sep]
    datasets = args[sep + 1 :]
else:
    output_root = Path(args[0])
    run_tags = args[1:]
    datasets = []

def compact(obj):
    if obj is None:
        return "NA"
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(obj)

print("[SUMMARY] run_tag\tdataset\tevaluation\tmultimodal_calibration\tmetrics_path")
for tag in run_tags:
    run_dir = output_root / tag
    metrics_files = sorted(run_dir.rglob("metrics.json"))
    if not metrics_files:
        for ds in datasets:
            print(f"{tag}\t{ds}\tMISSING\tMISSING\t-")
        continue

    metrics_path = metrics_files[0]
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    vlm_eval = payload.get("vlm_eval") or {}
    records = vlm_eval.get("datasets") if isinstance(vlm_eval, dict) else {}
    records = records if isinstance(records, dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    multimodal = artifacts.get("multimodal_calibration")

    for ds in datasets:
        ds_record = records.get(ds) if isinstance(records.get(ds), dict) else {}
        evaluation = ds_record.get("evaluation")
        print(
            f"{tag}\t{ds}\t{compact(evaluation)}\t{compact(multimodal)}\t{metrics_path}"
        )
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi
if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MiniCPM-V model path not found."
  echo "Set MODEL_PATH manually, e.g.:"
  echo "  MODEL_PATH=/mnt/82_store/LLM-weights/openbmb/MiniCPM-V"
  exit 1
fi
if [[ ! -d "$VLM_EVAL_KIT_ROOT" ]]; then
  echo "[ERROR] VLM_EVAL_KIT_ROOT not found: $VLM_EVAL_KIT_ROOT"
  echo "Set VLM_EVAL_KIT_ROOT manually."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LMU_DATA_DIR" "$MPL_CONFIG_DIR"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export LMUData="$LMU_DATA_DIR"
export MPLCONFIGDIR="$MPL_CONFIG_DIR"
if [[ "$DEVICE" == cuda:* ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
if [[ "$DEVICE" == npu:* ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
fi

if [[ "$DRY_RUN" != "true" ]]; then
  "$PYTHON_BIN" - "$DEVICE" <<'PY'
import importlib
import sys

device = sys.argv[1] if len(sys.argv) > 1 else "auto"
required = ("torch", "transformers", "timm")
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)

import torch
import transformers
print(f"[INFO] transformers={transformers.__version__}")
print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()} cuda.device_count={torch.cuda.device_count()}")
if device.startswith("cuda") and not torch.cuda.is_available():
    print("[ERROR] DEVICE requests CUDA but torch.cuda.is_available() is False.")
    sys.exit(2)
print("[INFO] dependency check passed: torch/transformers/timm")
PY
fi

echo "[INFO] $SCRIPT_NAME"
echo "[INFO] model=$MODEL_PATH"
echo "[INFO] algorithm=$ALGORITHM"
echo "[INFO] output_root=$OUTPUT_ROOT"
echo "[INFO] LMUData=$LMUData"
echo "[INFO] MPLCONFIGDIR=$MPLCONFIGDIR"
echo "[INFO] sequence_length=$SEQUENCE_LENGTH"
echo "[INFO] weight_bits=$WEIGHT_BITS activation_bits=$ACTIVATION_BITS"
echo "[INFO] group_size=$GROUP_SIZE weight_group_size=$WEIGHT_GROUP_SIZE"
echo "[INFO] gptq_vlm_dataset_name=$GPTQ_VLM_DATASET_NAME"
echo "[INFO] gptq_vlm_calib_num=$GPTQ_VLM_CALIB_NUM calibration_samples=$CALIBRATION_SAMPLES"
echo "[INFO] num_samples=${NUM_SAMPLES:-all}"
echo "[INFO] datasets=${VLM_DATASETS[*]}"
echo "[INFO] run_fp16_baseline=$RUN_FP16_BASELINE"
echo "[INFO] run_configs=${RUN_CONFIGS[*]}"
echo "[INFO] MiniCPM-V pure GPTQ current target: visual + connector + llm full branch"

if [[ ! -f "$VLM_EVAL_KIT_ROOT/.env" ]]; then
  echo "[WARN] Missing $VLM_EVAL_KIT_ROOT/.env (usually non-fatal for local model eval)."
fi

if [[ "$RUN_FP16_BASELINE" == "true" ]]; then
  run_fp16_baseline
fi

for entry in "${RUN_CONFIGS[@]}"; do
  IFS=':' read -r run_tag quant_visual quant_connector quant_llm <<< "$entry"
  if [[ -z "${run_tag:-}" || -z "${quant_visual:-}" || -z "${quant_connector:-}" || -z "${quant_llm:-}" ]]; then
    echo "[ERROR] Invalid RUN_CONFIGS entry: $entry"
    echo "Expected format: run_tag:quant_visual:quant_connector:quant_llm"
    exit 3
  fi
  run_gptq_config "$run_tag" "$quant_visual" "$quant_connector" "$quant_llm"
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, summary skipped."
  exit 0
fi

summarize_results
