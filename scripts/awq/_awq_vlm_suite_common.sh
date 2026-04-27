#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MINDPIPE_AWQ_VLM_SUITE_COMMON_SH_LOADED:-}" ]]; then
  return 0
fi
MINDPIPE_AWQ_VLM_SUITE_COMMON_SH_LOADED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALGORITHM="awq"

mindpipe_awq_vlm_init_defaults() {
  SCRIPT_NAME="${SCRIPT_NAME:-$(basename "$0")}"
  PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
  GPU_ID="${GPU_ID:-6}"
  NPU_ID="${NPU_ID:-0}"
  DEVICE="${DEVICE:-cuda:0}"
  DTYPE="${DTYPE:-float16}"
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
  DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
  SEED="${SEED:-0}"
  LOG_LEVEL="${LOG_LEVEL:-INFO}"

  CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
  EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
  CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
  SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"

  GROUP_SIZE="${GROUP_SIZE:-128}"
  WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
  WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-false}"
  WEIGHT_BITS_STR="${WEIGHT_BITS_STR:-4 3 2}"

  AWQ_SEARCH="${AWQ_SEARCH:-true}"
  AWQ_REUSE_SEARCH_RESULT="${AWQ_REUSE_SEARCH_RESULT:-true}"
  AWQ_SEARCH_SEQUENCE_LENGTH="${AWQ_SEARCH_SEQUENCE_LENGTH:-$SEQUENCE_LENGTH}"
  AWQ_AUTO_SCALE="${AWQ_AUTO_SCALE:-true}"
  AWQ_MSE_RANGE="${AWQ_MSE_RANGE:-true}"
  AWQ_CLIP_TARGETS="${AWQ_CLIP_TARGETS:-auto}"
  AWQ_QWEN3_5_QUANTIZE_LINEAR_ATTN="${AWQ_QWEN3_5_QUANTIZE_LINEAR_ATTN:-true}"

  VLM_MODE="${VLM_MODE:-all}"
  VLM_RESUME="${VLM_RESUME:-true}"
  VLM_DATASETS_STR="${VLM_DATASETS_STR:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
  VLM_API_NPROC="${VLM_API_NPROC:-4}"
  VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
  VLM_EVAL_KIT_ROOT="${VLM_EVAL_KIT_ROOT:-/mnt/42_store/zy/HUAWEI/work1/MQuant/third/VLMEvalKit}"
  VLM_USE_CACHE="${VLM_USE_CACHE:-}"
  VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-}"
  VLM_SAMPLE_CLEANUP="${VLM_SAMPLE_CLEANUP:-}"

  LMU_DATA_DIR="${LMU_DATA_DIR:-$REPO_ROOT/.lmu_data}"
  SKIP_EXISTING="${SKIP_EXISTING:-true}"
  DRY_RUN="${DRY_RUN:-false}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/awq_vlm_suite}"
  REQUIRED_MODULES_STR="${REQUIRED_MODULES_STR:-torch transformers}"
}

mindpipe_awq_vlm_pick_model_path() {
  local -a candidates=("$@")
  if [[ -n "${MODEL_PATH:-}" ]]; then
    return 0
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      return 0
    fi
  done
}

mindpipe_awq_vlm_normalize_dataset_name() {
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

mindpipe_awq_vlm_parse_arrays() {
  read -r -a WEIGHT_BITS_LIST <<< "$WEIGHT_BITS_STR"
  read -r -a REQUIRED_MODULES <<< "$REQUIRED_MODULES_STR"
  read -r -a VLM_DATASETS_RAW <<< "$VLM_DATASETS_STR"

  declare -gA _MINDPIPE_AWQ_VLM_SEEN_DATASETS=()
  VLM_DATASETS=()
  for ds in "${VLM_DATASETS_RAW[@]}"; do
    local canonical
    canonical="$(mindpipe_awq_vlm_normalize_dataset_name "$ds")"
    if [[ -n "${_MINDPIPE_AWQ_VLM_SEEN_DATASETS[$canonical]:-}" ]]; then
      continue
    fi
    _MINDPIPE_AWQ_VLM_SEEN_DATASETS["$canonical"]=1
    VLM_DATASETS+=("$canonical")
  done

  RUN_TAGS=("fp16_seq${SEQUENCE_LENGTH}")
  for weight_bits in "${WEIGHT_BITS_LIST[@]}"; do
    RUN_TAGS+=("w${weight_bits}a16_seq${SEQUENCE_LENGTH}")
  done
}

mindpipe_awq_vlm_assert_config() {
  if [[ "$SEQUENCE_LENGTH" != "2048" ]]; then
    echo "[ERROR] SEQUENCE_LENGTH must be 2048 for this suite. current=$SEQUENCE_LENGTH"
    exit 1
  fi
  if [[ "$CALIBRATION_DATASET" != "pileval" ]]; then
    echo "[ERROR] CALIBRATION_DATASET must be pileval for this suite. current=$CALIBRATION_DATASET"
    exit 1
  fi
  if [[ "$CALIBRATION_SAMPLES" != "128" ]]; then
    echo "[ERROR] CALIBRATION_SAMPLES must be 128 for this suite. current=$CALIBRATION_SAMPLES"
    exit 1
  fi
  if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
    echo "[ERROR] GROUP_SIZE/WEIGHT_GROUP_SIZE must both be 128. current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
    exit 1
  fi
  if [[ "${#VLM_DATASETS[@]}" -eq 0 ]]; then
    echo "[ERROR] VLM_DATASETS_STR resolved to an empty dataset list."
    exit 1
  fi
}

mindpipe_awq_vlm_find_metrics_file() {
  local run_output="$1"
  find "$run_output" -type f -name metrics.json 2>/dev/null | head -n 1 || true
}

mindpipe_awq_vlm_is_metrics_complete() {
  local metrics_path="$1"
  shift
  "$PYTHON_BIN" - "$metrics_path" "$VLM_MODE" "$@" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
mode = sys.argv[2]
datasets = sys.argv[3:]
if not metrics_path.is_file():
    sys.exit(1)

try:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

vlm_eval = payload.get("vlm_eval") or {}
records = vlm_eval.get("datasets") or {}
if not isinstance(records, dict):
    sys.exit(1)

def complete(record):
    if not isinstance(record, dict):
        return False
    if mode == "infer":
        return bool(record.get("inference_completed"))
    if mode == "eval":
        return "evaluation" in record or "evaluation_skipped" in record
    return bool(record.get("inference_completed")) and (
        "evaluation" in record or "evaluation_skipped" in record
    )

sys.exit(0 if all(complete(records.get(name)) for name in datasets) else 1)
PY
}

mindpipe_awq_vlm_run_command() {
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

mindpipe_awq_vlm_append_optional_args() {
  local -n cmd_ref=$1
  if [[ -n "${NUM_SAMPLES:-}" ]]; then
    cmd_ref+=(--num_samples "$NUM_SAMPLES")
  fi
  if [[ -n "${VLM_WORK_DIR:-}" ]]; then
    cmd_ref+=(--vlm_work_dir "$VLM_WORK_DIR")
  fi
  if [[ -n "${VLM_JUDGE:-}" ]]; then
    cmd_ref+=(--vlm_judge "$VLM_JUDGE")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    cmd_ref+=(--hf_token "$HF_TOKEN")
  fi
  if [[ -n "${VLM_USE_CACHE:-}" ]]; then
    cmd_ref+=(--vlm_use_cache "$VLM_USE_CACHE")
  fi
  if [[ -n "${VLM_RESUME:-}" ]]; then
    cmd_ref+=(--vlm_resume "$VLM_RESUME")
  fi
  if [[ -n "${VLM_MAX_NEW_TOKENS:-}" ]]; then
    cmd_ref+=(--vlm_max_new_tokens "$VLM_MAX_NEW_TOKENS")
  fi
  if [[ -n "${VLM_SAMPLE_CLEANUP:-}" ]]; then
    cmd_ref+=(--vlm_sample_cleanup "$VLM_SAMPLE_CLEANUP")
  fi
}

mindpipe_awq_vlm_run_fp16_baseline() {
  local run_tag="fp16_seq${SEQUENCE_LENGTH}"
  local run_output="$OUTPUT_ROOT/$run_tag"
  local metrics_path
  metrics_path="$(mindpipe_awq_vlm_find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && mindpipe_awq_vlm_is_metrics_complete "$metrics_path" "${VLM_DATASETS[@]}"; then
    echo "[INFO] Skip $run_tag (found complete metrics): $metrics_path"
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

  mindpipe_awq_vlm_append_optional_args cmd
  mindpipe_awq_vlm_run_command "$log_path" "${cmd[@]}"
}

mindpipe_awq_vlm_run_quantized() {
  local weight_bits="$1"
  local run_tag="w${weight_bits}a16_seq${SEQUENCE_LENGTH}"
  local run_output="$OUTPUT_ROOT/$run_tag"
  local metrics_path
  metrics_path="$(mindpipe_awq_vlm_find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && mindpipe_awq_vlm_is_metrics_complete "$metrics_path" "${VLM_DATASETS[@]}"; then
    echo "[INFO] Skip $run_tag (found complete metrics): $metrics_path"
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
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$weight_bits"
    --activation_bits 16
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --weight_symmetric "$WEIGHT_SYMMETRIC"
    --awq_search "$AWQ_SEARCH"
    --awq_reuse_search_result "$AWQ_REUSE_SEARCH_RESULT"
    --awq_search_sequence_length "$AWQ_SEARCH_SEQUENCE_LENGTH"
    --awq_auto_scale "$AWQ_AUTO_SCALE"
    --awq_mse_range "$AWQ_MSE_RANGE"
    --awq_clip_targets "$AWQ_CLIP_TARGETS"
    --awq_qwen3_5_quantize_linear_attn "$AWQ_QWEN3_5_QUANTIZE_LINEAR_ATTN"
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

  mindpipe_awq_vlm_append_optional_args cmd
  mindpipe_awq_vlm_run_command "$log_path" "${cmd[@]}"
}

mindpipe_awq_vlm_summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "${RUN_TAGS[@]}" -- "${VLM_DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

args = list(sys.argv[1:])
sep = args.index("--") if "--" in args else len(args)
output_root = Path(args[0])
run_tags = args[1:sep]
datasets = args[sep + 1 :] if sep < len(args) else []
summary_path = output_root / "summary.md"


def compact(obj):
    if obj is None:
        return "NA"
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(obj)


def find_payload(tag):
    run_dir = output_root / tag
    metrics_files = sorted(run_dir.rglob("metrics.json"))
    if not metrics_files:
        return None, None
    metrics_path = metrics_files[0]
    return json.loads(metrics_path.read_text(encoding="utf-8")), metrics_path


def extract_eval(payload, dataset_name):
    if not isinstance(payload, dict):
        return None
    vlm_eval = payload.get("vlm_eval") or {}
    if not isinstance(vlm_eval, dict):
        return None
    records = vlm_eval.get("datasets") or {}
    if not isinstance(records, dict):
        return None
    record = records.get(dataset_name) or {}
    if not isinstance(record, dict):
        return None
    return record.get("evaluation")


lines = [
    "# AWQ VLM evaluation summary",
    "",
    "| run_tag | dataset | evaluation | metrics_path |",
    "| --- | --- | --- | --- |",
]

print("[SUMMARY] run_tag\tdataset\tevaluation\tmetrics_path")
for tag in run_tags:
    payload, metrics_path = find_payload(tag)
    metrics_text = str(metrics_path) if metrics_path else "MISSING"
    for dataset_name in datasets:
        evaluation = extract_eval(payload, dataset_name)
        evaluation_text = compact(evaluation)
        print(f"{tag}\t{dataset_name}\t{evaluation_text}\t{metrics_text}")
        lines.append(f"| {tag} | {dataset_name} | `{evaluation_text}` | `{metrics_text}` |")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[SUMMARY] markdown={summary_path}")
PY
}

mindpipe_awq_vlm_prepare_runtime_env() {
  export PYTHONNOUSERSITE=1
  export PYTHONHASHSEED="$SEED"
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_DISABLE_TELEMETRY=1
  export LMUData="$LMU_DATA_DIR"

  if [[ "$DEVICE" == cuda:* ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
  fi
  if [[ "$DEVICE" == npu:* ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
  fi
}

mindpipe_awq_vlm_check_dependencies() {
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "${REQUIRED_MODULES[@]}" <<'PY'
import importlib
import sys

modules = sys.argv[1:]
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)
print("[INFO] dependency check passed:", "/".join(modules))
PY
}

mindpipe_awq_vlm_validate_entry() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi
  if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH:-<empty>}"
    echo "Set MODEL_PATH manually."
    exit 1
  fi
  if [[ ! -d "$VLM_EVAL_KIT_ROOT" ]]; then
    echo "[ERROR] VLM_EVAL_KIT_ROOT not found: $VLM_EVAL_KIT_ROOT"
    echo "Set VLM_EVAL_KIT_ROOT manually."
    exit 1
  fi
}

mindpipe_awq_vlm_print_banner() {
  echo "[INFO] $SCRIPT_NAME"
  echo "[INFO] MODEL_PATH=$MODEL_PATH"
  echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "[INFO] DATASETS=${VLM_DATASETS[*]}"
  echo "[INFO] sequence_length=$SEQUENCE_LENGTH calibration_dataset=$CALIBRATION_DATASET calibration_samples=$CALIBRATION_SAMPLES"
  echo "[INFO] group_size=$GROUP_SIZE weight_group_size=$WEIGHT_GROUP_SIZE"
  echo "[INFO] awq_search=$AWQ_SEARCH awq_reuse_search_result=$AWQ_REUSE_SEARCH_RESULT awq_auto_scale=$AWQ_AUTO_SCALE awq_mse_range=$AWQ_MSE_RANGE awq_clip_targets=$AWQ_CLIP_TARGETS awq_qwen3_5_quantize_linear_attn=$AWQ_QWEN3_5_QUANTIZE_LINEAR_ATTN"
  echo "[INFO] vlm_mode=$VLM_MODE vlm_resume=$VLM_RESUME"
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[INFO] LMUData=$LMUData"
  echo "[INFO] run_order=${RUN_TAGS[*]}"
}

mindpipe_awq_vlm_main() {
  mindpipe_awq_vlm_init_defaults
  mindpipe_awq_vlm_parse_arrays
  mindpipe_awq_vlm_assert_config
  mindpipe_awq_vlm_validate_entry
  mindpipe_awq_vlm_prepare_runtime_env
  mindpipe_awq_vlm_check_dependencies

  mkdir -p "$OUTPUT_ROOT"
  mindpipe_awq_vlm_print_banner

  mindpipe_awq_vlm_run_fp16_baseline
  for weight_bits in "${WEIGHT_BITS_LIST[@]}"; do
    mindpipe_awq_vlm_run_quantized "$weight_bits"
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] DRY_RUN=true, summary skipped."
    return 0
  fi

  mindpipe_awq_vlm_summarize_results
}
