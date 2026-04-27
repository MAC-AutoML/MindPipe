#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPT_NAME="$(basename "$0")"
ALGORITHM="awq"

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
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-32}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

WEIGHT_BITS="${WEIGHT_BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-}"
AWQ_SEARCH="${AWQ_SEARCH:-true}"
AWQ_SEARCH_SEQUENCE_LENGTH="${AWQ_SEARCH_SEQUENCE_LENGTH:-512}"
AWQ_AUTO_SCALE="${AWQ_AUTO_SCALE:-true}"
AWQ_MSE_RANGE="${AWQ_MSE_RANGE:-true}"
AWQ_CLIP_TARGETS="${AWQ_CLIP_TARGETS:-auto}"

VLM_MODE="${VLM_MODE:-all}"
VLM_DATASETS_STR="${VLM_DATASETS_STR:-ChartQA_TEST}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
VLM_EVAL_KIT_ROOT="${VLM_EVAL_KIT_ROOT:-/mnt/42_store/zy/HUAWEI/work1/MQuant/third/VLMEvalKit}"
VLM_USE_CACHE="${VLM_USE_CACHE:-}"
VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-}"
VLM_SAMPLE_CLEANUP="${VLM_SAMPLE_CLEANUP:-}"

LMU_DATA_DIR="${LMU_DATA_DIR:-$REPO_ROOT/.lmu_data}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Example:
#   GPU_ID=6 MODEL_PATH=/mnt/82_store/LLM-weights/Qwen2-VL-7B-Instruct \
#   VLM_DATASETS_STR="ChartQA_TEST" NUM_SAMPLES=100 \
#   bash scripts/repro/run_vlm_fp16_vs_awq_subset.sh

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "/mnt/82_store/LLM-weights/Qwen2-VL-7B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/LLM-weights/openbmb/MiniCPM-V" \
    "/mnt/82_store/zy/model/Qwen2-VL-7B-Instruct" \
    "/mnt/82_store/zy/model/Qwen2.5-VL-7B-Instruct" \
    "/mnt/82_store/zy/model/Qwen3-VL-2B-Instruct" \
    "/mnt/82_store/zy/model/openbmb/MiniCPM-V"
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

sanitize_tag() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//:/_}"
  raw="${raw//,/__}"
  raw="${raw//[!A-Za-z0-9._-]/_}"
  echo "$raw"
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

if [[ "${#VLM_DATASETS[@]}" -eq 0 ]]; then
  echo "[ERROR] VLM_DATASETS_STR resolved to an empty dataset list."
  exit 1
fi

MODEL_BASENAME="$(basename "${MODEL_PATH:-unknown_model}")"
MODEL_TAG="$(sanitize_tag "$MODEL_BASENAME")"
DATASET_TAG="$(sanitize_tag "${VLM_DATASETS[*]}")"
DEFAULT_OUTPUT_ROOT="$REPO_ROOT/new_results/vlm_fp16_vs_awq_subset/${MODEL_TAG}__${DATASET_TAG}__n${NUM_SAMPLES}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"

FP_RUN_TAG="fp16_seq${SEQUENCE_LENGTH}_n${NUM_SAMPLES}"
AWQ_RUN_TAG="awq_w4a16_g128_seq${SEQUENCE_LENGTH}_n${NUM_SAMPLES}"
RUN_TAGS=("$FP_RUN_TAG" "$AWQ_RUN_TAG")

# Keep the semantics of this helper explicit and fixed.
if [[ "$WEIGHT_BITS" != "4" ]]; then
  echo "[ERROR] WEIGHT_BITS must be 4 for this helper. current=$WEIGHT_BITS"
  exit 1
fi
if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
  echo "[ERROR] GROUP_SIZE and WEIGHT_GROUP_SIZE must both be 128 for this helper."
  echo "current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
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

  _cmd_ref+=(--num_samples "$NUM_SAMPLES")

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
}

run_fp16_baseline() {
  local run_tag="$FP_RUN_TAG"
  local run_output="$OUTPUT_ROOT/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
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

  append_optional_args cmd
  run_command "$log_path" "${cmd[@]}"
}

run_awq_quantized() {
  local run_tag="$AWQ_RUN_TAG"
  local run_output="$OUTPUT_ROOT/$run_tag"
  local metrics_path
  metrics_path="$(find_metrics_file "$run_output")"

  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
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
    --weight_bits "$WEIGHT_BITS"
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --awq_search "$AWQ_SEARCH"
    --awq_search_sequence_length "$AWQ_SEARCH_SEQUENCE_LENGTH"
    --awq_auto_scale "$AWQ_AUTO_SCALE"
    --awq_mse_range "$AWQ_MSE_RANGE"
    --awq_clip_targets "$AWQ_CLIP_TARGETS"
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

  if [[ -n "${WEIGHT_SYMMETRIC:-}" ]]; then
    cmd+=(--weight_symmetric "$WEIGHT_SYMMETRIC")
  fi
  append_optional_args cmd
  run_command "$log_path" "${cmd[@]}"
}

summarize_results() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$FP_RUN_TAG" "$AWQ_RUN_TAG" "$NUM_SAMPLES" -- "${VLM_DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

args = list(sys.argv[1:])
sep = args.index("--") if "--" in args else len(args)
output_root = Path(args[0])
fp_tag = args[1]
awq_tag = args[2]
num_samples = args[3]
datasets = args[sep + 1 :] if sep < len(args) else []
summary_path = output_root / "comparison_summary.md"


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


def extract_dataset_record(payload, dataset_name):
    if not isinstance(payload, dict):
        return {}
    vlm_eval = payload.get("vlm_eval") or {}
    if not isinstance(vlm_eval, dict):
        return {}
    records = vlm_eval.get("datasets") or {}
    if not isinstance(records, dict):
        return {}
    record = records.get(dataset_name) or {}
    return record if isinstance(record, dict) else {}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pick_primary_metric(evaluation):
    preferred_keys = (
        "Overall",
        "overall",
        "Final Score Norm",
        "Final Score",
        "score",
        "accuracy",
        "acc",
        "val",
        "test",
    )
    if isinstance(evaluation, dict):
        records = evaluation.get("records")
        if isinstance(records, list) and records:
            first = records[0]
            if isinstance(first, dict):
                for key in preferred_keys:
                    value = first.get(key)
                    if is_number(value):
                        return key, float(value)
                for key, value in first.items():
                    if is_number(value):
                        return key, float(value)
        for key in preferred_keys:
            value = evaluation.get(key)
            if is_number(value):
                return key, float(value)
        for key, value in evaluation.items():
            if is_number(value):
                return key, float(value)
    return None, None


def fmt_metric(metric_name, score):
    if metric_name is None or score is None:
        return "NA"
    return f"{metric_name}={score:.4f}"


fp_payload, fp_metrics_path = find_payload(fp_tag)
awq_payload, awq_metrics_path = find_payload(awq_tag)

lines = [
    "# FP16 vs AWQ subset comparison",
    "",
    f"- num_samples per dataset: `{num_samples}`",
    f"- fp16 metrics: `{fp_metrics_path}`" if fp_metrics_path else "- fp16 metrics: `MISSING`",
    f"- awq metrics: `{awq_metrics_path}`" if awq_metrics_path else "- awq metrics: `MISSING`",
    "",
    "| dataset | fp16 | awq_w4a16_g128 | delta (awq-fp16) |",
    "| --- | ---: | ---: | ---: |",
]

print("[SUMMARY] dataset\tfp16\tawq_w4a16_g128\tdelta")
for dataset_name in datasets:
    fp_record = extract_dataset_record(fp_payload, dataset_name)
    awq_record = extract_dataset_record(awq_payload, dataset_name)
    fp_eval = fp_record.get("evaluation")
    awq_eval = awq_record.get("evaluation")
    fp_metric, fp_score = pick_primary_metric(fp_eval)
    awq_metric, awq_score = pick_primary_metric(awq_eval)
    delta = awq_score - fp_score if fp_score is not None and awq_score is not None else None

    fp_text = fmt_metric(fp_metric, fp_score)
    awq_text = fmt_metric(awq_metric, awq_score)
    delta_text = f"{delta:.4f}" if delta is not None else "NA"
    print(f"{dataset_name}\t{fp_text}\t{awq_text}\t{delta_text}")
    lines.append(f"| {dataset_name} | {fp_text} | {awq_text} | {delta_text} |")

    if fp_eval is not None or awq_eval is not None:
        lines.append("")
        lines.append(f"## {dataset_name} raw evaluation")
        lines.append("")
        lines.append(f"- fp16: `{compact(fp_eval)}`")
        lines.append(f"- awq: `{compact(awq_eval)}`")

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[SUMMARY] markdown={summary_path}")
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi
if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH:-<empty>}"
  echo "Set MODEL_PATH manually, e.g.:"
  echo "  MODEL_PATH=/mnt/82_store/LLM-weights/Qwen2-VL-7B-Instruct"
  exit 1
fi
if [[ ! -d "$VLM_EVAL_KIT_ROOT" ]]; then
  echo "[ERROR] VLM_EVAL_KIT_ROOT not found: $VLM_EVAL_KIT_ROOT"
  echo "Set VLM_EVAL_KIT_ROOT manually."
  exit 1
fi

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

if [[ "$DRY_RUN" != "true" ]]; then
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

for name in ("torch", "transformers"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)

print("[INFO] dependency check passed: torch, transformers")
PY
fi

echo "[INFO] $SCRIPT_NAME"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[INFO] DATASETS=${VLM_DATASETS[*]}"
echo "[INFO] NUM_SAMPLES=$NUM_SAMPLES"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] LMUData=$LMUData"

run_fp16_baseline
run_awq_quantized
summarize_results
