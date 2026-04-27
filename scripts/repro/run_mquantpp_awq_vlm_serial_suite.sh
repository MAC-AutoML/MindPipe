#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPT_NAME="$(basename "$0")"
ALGORITHM="awq"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
GPU_ID="${GPU_ID:-1}"
NPU_ID="${NPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
SEED="${SEED:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Text AWQ search still uses the standard text calibration path.
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"

# Visual / connector AWQ uses multimodal calibration samples.
AWQ_VLM_DATASET_NAME="${AWQ_VLM_DATASET_NAME:-OCRBench}"
AWQ_VLM_CALIB_NUM="${AWQ_VLM_CALIB_NUM:-128}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"

GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-false}"

AWQ_SEARCH="${AWQ_SEARCH:-true}"
AWQ_REUSE_SEARCH_RESULT="${AWQ_REUSE_SEARCH_RESULT:-true}"
AWQ_SEARCH_SEQUENCE_LENGTH="${AWQ_SEARCH_SEQUENCE_LENGTH:-512}"
AWQ_AUTO_SCALE="${AWQ_AUTO_SCALE:-true}"
AWQ_MSE_RANGE="${AWQ_MSE_RANGE:-true}"
AWQ_CLIP_TARGETS="${AWQ_CLIP_TARGETS:-auto}"
AWQ_QWEN3_5_QUANTIZE_LINEAR_ATTN="${AWQ_QWEN3_5_QUANTIZE_LINEAR_ATTN:-true}"

VLM_MODE="${VLM_MODE:-all}"
VLM_DATASETS_STR="${VLM_DATASETS_STR:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
VLM_EVAL_KIT_ROOT="${VLM_EVAL_KIT_ROOT:-/mnt/42_store/zy/HUAWEI/work1/MQuant/third/VLMEvalKit}"
VLM_USE_CACHE="${VLM_USE_CACHE:-}"
VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-}"
VLM_SAMPLE_CLEANUP="${VLM_SAMPLE_CLEANUP:-}"
VLM_VERBOSE="${VLM_VERBOSE:-}"
VLM_IGNORE_FAILED="${VLM_IGNORE_FAILED:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

# Explicit model list for the current MQuant++ pure-AWQ validation round.
MODEL_KEYS_STR="${MODEL_KEYS_STR:-qwen3_vl_2b qwen2_5_vl_7b qwen2_vl_2b minicpm_v}"

# Format: run_tag:visual_w_bits:visual_a_bits:llm_w_bits:llm_a_bits
QUANT_CONFIGS_STR="${QUANT_CONFIGS_STR:-vis_w8a8_lang_w4a8:8:8:4:8 vis_w4a8_lang_w4a8:4:8:4:8}"

STRICT_ACTIVATION_COMPAT="${STRICT_ACTIVATION_COMPAT:-false}"
RUN_FP16_BASELINE="${RUN_FP16_BASELINE:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/new_results/quantization_suite/mquantpp_awq_vlm_serial_suite}"
LMU_DATA_DIR="${LMU_DATA_DIR:-$REPO_ROOT/.lmu_data}"
MPL_CONFIG_DIR="${MPL_CONFIG_DIR:-/tmp/mpl_mquantpp_awq_vlm_gpu${GPU_ID}}"
REQUIRED_MODULES_STR="${REQUIRED_MODULES_STR:-torch transformers}"

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

resolve_model_name() {
  case "$1" in
    qwen3_vl_2b)
      echo "Qwen3-VL-2B-Instruct"
      ;;
    qwen2_5_vl_7b)
      echo "Qwen2.5-VL-7B-Instruct"
      ;;
    qwen2_vl_2b)
      echo "Qwen2-VL-2B-Instruct"
      ;;
    minicpm_v)
      echo "MiniCPM-V"
      ;;
    *)
      echo "[ERROR] Unsupported model key: $1" >&2
      return 1
      ;;
  esac
}

resolve_model_candidates() {
  case "$1" in
    qwen3_vl_2b)
      cat <<'EOF'
/mnt/82_store/LLM-weights/Qwen3-VL-2B-Instruct
/mnt/82_store/LLM-weights/Qwen3-VL-2B
/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B-Instruct
/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B
/mnt/82_store/zy/model/Qwen3-VL-2B-Instruct
/mnt/82_store/zy/model/Qwen3-VL-2B
/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B-Instruct
/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B
EOF
      ;;
    qwen2_5_vl_7b)
      cat <<'EOF'
/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct
/mnt/82_store/zy/model/Qwen2.5-VL-7B-Instruct
/mnt/82_store/huggingface/datasets/Qwen/Qwen2.5-VL-7B-Instruct
EOF
      ;;
    qwen2_vl_2b)
      cat <<'EOF'
/mnt/82_store/LLM-weights/Qwen2-VL-2B-Instruct
/mnt/82_store/LLM-weights/Qwen/Qwen2-VL-2B-Instruct
/mnt/82_store/zy/model/Qwen2-VL-2B-Instruct
/mnt/82_store/huggingface/datasets/Qwen/Qwen2-VL-2B-Instruct
EOF
      ;;
    minicpm_v)
      cat <<'EOF'
/mnt/82_store/LLM-weights/openbmb/MiniCPM-V
/mnt/82_store/LLM-weights/openbmb/MiniCPM-V-2
/mnt/82_store/zy/model/openbmb/MiniCPM-V
/mnt/82_store/huggingface/datasets/openbmb/MiniCPM-V
EOF
      ;;
    *)
      echo "[ERROR] Unsupported model key: $1" >&2
      return 1
      ;;
  esac
}

pick_model_path() {
  local model_key="$1"
  local model_path=""
  local candidate=""
  while IFS= read -r candidate; do
    [[ -z "$candidate" ]] && continue
    if [[ -d "$candidate" ]]; then
      model_path="$candidate"
      break
    fi
  done < <(resolve_model_candidates "$model_key")

  if [[ -z "$model_path" ]]; then
    echo "[ERROR] MODEL_PATH not found for model_key=$model_key" >&2
    return 1
  fi
  printf '%s\n' "$model_path"
}

find_metrics_file() {
  local run_output="$1"
  find "$run_output" -type f -name metrics.json 2>/dev/null | head -n 1 || true
}

is_metrics_complete() {
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
  local -n cmd_ref="$1"
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
  if [[ -n "${VLM_MAX_NEW_TOKENS:-}" ]]; then
    cmd_ref+=(--vlm_max_new_tokens "$VLM_MAX_NEW_TOKENS")
  fi
  if [[ -n "${VLM_SAMPLE_CLEANUP:-}" ]]; then
    cmd_ref+=(--vlm_sample_cleanup "$VLM_SAMPLE_CLEANUP")
  fi
  if [[ -n "${VLM_VERBOSE:-}" ]]; then
    cmd_ref+=(--vlm_verbose "$VLM_VERBOSE")
  fi
  if [[ -n "${VLM_IGNORE_FAILED:-}" ]]; then
    cmd_ref+=(--vlm_ignore_failed "$VLM_IGNORE_FAILED")
  fi
}

run_fp16_baseline() {
  local model_key="$1"
  local model_name="$2"
  local model_path="$3"
  local model_output_root="$4"
  local run_tag="fp16_seq${SEQUENCE_LENGTH}"
  local run_output="$model_output_root/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path" "${VLM_DATASETS[@]}"; then
    echo "[INFO] Skip ${model_key}/${run_tag} (found complete metrics): $metrics_path"
    return 0
  fi

  mkdir -p "$run_output"
  local log_path="$run_output/run.log"
  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --model_path "$model_path"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --evaluation_dataset wikitext2
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
}

run_quant_config() {
  local model_key="$1"
  local model_name="$2"
  local model_path="$3"
  local model_output_root="$4"
  local run_tag="$5"
  local visual_w_bits="$6"
  local visual_a_bits="$7"
  local llm_w_bits="$8"
  local llm_a_bits="$9"

  local run_output="$model_output_root/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path" "${VLM_DATASETS[@]}"; then
    echo "[INFO] Skip ${model_key}/${run_tag} (found complete metrics): $metrics_path"
    return 0
  fi

  mkdir -p "$run_output"

  if [[ "$visual_a_bits" != "16" || "$llm_a_bits" != "16" ]]; then
    local act_warn
    act_warn="[WARN] ${model_key}/${run_tag}: current MindPipe pure-AWQ path only fake-quantizes weights. Requested visual A${visual_a_bits} / llm A${llm_a_bits} will execute as A16 at runtime; effective run is visual+connector W${visual_w_bits}A16 and llm W${llm_w_bits}A16."
    echo "$act_warn"
    printf '%s\n' "$act_warn" > "$run_output/activation_limit.txt"
    if [[ "$STRICT_ACTIVATION_COMPAT" == "true" ]]; then
      echo "[ERROR] STRICT_ACTIVATION_COMPAT=true but pure AWQ activation quantization is not implemented."
      exit 1
    fi
  fi

  printf '%s\n' "$model_name" > "$model_output_root/model_name.txt"

  local log_path="$run_output/run.log"
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
    --evaluation_dataset wikitext2
    --calibration_samples "$CALIBRATION_SAMPLES"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$llm_w_bits"
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
    --awq_vlm_dataset_name "$AWQ_VLM_DATASET_NAME"
    --awq_vlm_calib_num "$AWQ_VLM_CALIB_NUM"
    --awq_vlm_quant_visual true
    --awq_vlm_quant_connector true
    --awq_vlm_quant_llm true
    --awq_visual_w_bits "$visual_w_bits"
    --awq_connector_w_bits "$visual_w_bits"
    --awq_llm_w_bits "$llm_w_bits"
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
}

summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" -- "${VLM_DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

args = list(sys.argv[1:])
sep = args.index("--")
output_root = Path(args[0])
datasets = args[sep + 1 :]
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
    "# MQuant++ pure-AWQ VLM serial suite summary",
    "",
    "| model_dir | run_tag | dataset | evaluation | metrics_path |",
    "| --- | --- | --- | --- | --- |",
]

print("[SUMMARY] model_dir\trun_tag\tdataset\tevaluation\tmetrics_path")
for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
    for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
        metrics_files = sorted(run_dir.rglob("metrics.json"))
        payload = None
        metrics_path = "MISSING"
        if metrics_files:
            metrics_path = str(metrics_files[0])
            payload = json.loads(metrics_files[0].read_text(encoding="utf-8"))
        for dataset_name in datasets:
            evaluation = extract_eval(payload, dataset_name)
            evaluation_text = compact(evaluation)
            print(f"{model_dir.name}\t{run_dir.name}\t{dataset_name}\t{evaluation_text}\t{metrics_path}")
            lines.append(
                f"| {model_dir.name} | {run_dir.name} | {dataset_name} | `{evaluation_text}` | `{metrics_path}` |"
            )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[SUMMARY] markdown={summary_path}")
PY
}

prepare_runtime_env() {
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
}

check_dependencies() {
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "${REQUIRED_MODULES[@]}" <<'PY'
import importlib
import sys

for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)
print("[INFO] dependency check passed:", "/".join(sys.argv[1:]))
PY
}

validate_entry() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi
  if [[ ! -d "$VLM_EVAL_KIT_ROOT" ]]; then
    echo "[ERROR] VLM_EVAL_KIT_ROOT not found: $VLM_EVAL_KIT_ROOT"
    exit 1
  fi
  if [[ "$SEQUENCE_LENGTH" != "2048" ]]; then
    echo "[ERROR] SEQUENCE_LENGTH must be 2048 for this suite. current=$SEQUENCE_LENGTH"
    exit 1
  fi
  if [[ "$CALIBRATION_SAMPLES" != "128" || "$AWQ_VLM_CALIB_NUM" != "128" ]]; then
    echo "[ERROR] CALIBRATION_SAMPLES and AWQ_VLM_CALIB_NUM must both be 128. current=$CALIBRATION_SAMPLES/$AWQ_VLM_CALIB_NUM"
    exit 1
  fi
  if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
    echo "[ERROR] GROUP_SIZE/WEIGHT_GROUP_SIZE must both be 128. current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
    exit 1
  fi
  if [[ "${#MODEL_KEYS[@]}" -eq 0 ]]; then
    echo "[ERROR] MODEL_KEYS_STR resolved to an empty model list."
    exit 1
  fi
  if [[ "${#VLM_DATASETS[@]}" -eq 0 ]]; then
    echo "[ERROR] VLM_DATASETS_STR resolved to an empty dataset list."
    exit 1
  fi
}

print_banner() {
  echo "[INFO] $SCRIPT_NAME"
  echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "[INFO] MODEL_KEYS=${MODEL_KEYS[*]}"
  echo "[INFO] VLM_DATASETS=${VLM_DATASETS[*]}"
  echo "[INFO] multimodal_calibration=${AWQ_VLM_DATASET_NAME} x ${AWQ_VLM_CALIB_NUM}"
  echo "[INFO] text_awq_search_calibration=${CALIBRATION_DATASET} x ${CALIBRATION_SAMPLES}"
  echo "[INFO] group_size=$GROUP_SIZE weight_group_size=$WEIGHT_GROUP_SIZE"
  echo "[INFO] awq_search=$AWQ_SEARCH awq_reuse_search_result=$AWQ_REUSE_SEARCH_RESULT awq_auto_scale=$AWQ_AUTO_SCALE awq_mse_range=$AWQ_MSE_RANGE awq_clip_targets=$AWQ_CLIP_TARGETS"
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[INFO] LMUData=$LMUData"
  echo "[INFO] MPLCONFIGDIR=$MPLCONFIGDIR"
  echo "[INFO] quant_configs=${QUANT_CONFIGS[*]}"
  echo "[INFO] NOTE: current MindPipe pure-AWQ path only fake-quantizes weights. Requested A8 in run tags is recorded as target config, but runtime still executes A16."
}

main() {
  AWQ_VLM_DATASET_NAME="$(normalize_dataset_name "$AWQ_VLM_DATASET_NAME")"

  read -r -a MODEL_KEYS <<< "$MODEL_KEYS_STR"
  read -r -a QUANT_CONFIGS <<< "$QUANT_CONFIGS_STR"
  read -r -a REQUIRED_MODULES <<< "$REQUIRED_MODULES_STR"
  read -r -a VLM_DATASETS_RAW <<< "$VLM_DATASETS_STR"

  declare -gA _SEEN_DATASETS=()
  VLM_DATASETS=()
  for ds in "${VLM_DATASETS_RAW[@]}"; do
    local canonical
    canonical="$(normalize_dataset_name "$ds")"
    if [[ -n "${_SEEN_DATASETS[$canonical]:-}" ]]; then
      continue
    fi
    _SEEN_DATASETS["$canonical"]=1
    VLM_DATASETS+=("$canonical")
  done

  validate_entry
  prepare_runtime_env
  check_dependencies

  mkdir -p "$OUTPUT_ROOT"
  print_banner

  local model_key=""
  for model_key in "${MODEL_KEYS[@]}"; do
    local model_name
    local model_path
    local model_output_root

    model_name="$(resolve_model_name "$model_key")"
    model_path="$(pick_model_path "$model_key")"
    model_output_root="$OUTPUT_ROOT/$model_key"
    mkdir -p "$model_output_root"
    printf '%s\n' "$model_name" > "$model_output_root/model_name.txt"

    echo "[INFO] ===== model_key=$model_key model_name=$model_name model_path=$model_path ====="

    if [[ "$RUN_FP16_BASELINE" == "true" ]]; then
      run_fp16_baseline "$model_key" "$model_name" "$model_path" "$model_output_root"
    fi

    local entry=""
    for entry in "${QUANT_CONFIGS[@]}"; do
      IFS=":" read -r run_tag visual_w_bits visual_a_bits llm_w_bits llm_a_bits <<< "$entry"
      if [[ -z "${run_tag:-}" || -z "${visual_w_bits:-}" || -z "${visual_a_bits:-}" || -z "${llm_w_bits:-}" || -z "${llm_a_bits:-}" ]]; then
        echo "[ERROR] Invalid QUANT_CONFIGS_STR entry: $entry"
        exit 1
      fi
      run_quant_config \
        "$model_key" \
        "$model_name" \
        "$model_path" \
        "$model_output_root" \
        "$run_tag" \
        "$visual_w_bits" \
        "$visual_a_bits" \
        "$llm_w_bits" \
        "$llm_a_bits"
    done
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] DRY_RUN=true, summary skipped."
    return 0
  fi

  summarize_results
}

main "$@"
