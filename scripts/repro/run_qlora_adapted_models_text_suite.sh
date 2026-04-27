#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPT_NAME="$(basename "$0")"
ALGORITHM="qlora"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
GPU_ID="${GPU_ID:-1}"
NPU_ID="${NPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-pileval}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-1024}"
QLORA_MAX_TRAIN_SAMPLES="${QLORA_MAX_TRAIN_SAMPLES:-1024}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-1}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
ZERO_SHOT_TASKS_STR="${ZERO_SHOT_TASKS_STR:-boolq piqa rte winogrande arc_easy arc_challenge openbookqa}"
SEED="${SEED:-0}"
GROUP_SIZE="${GROUP_SIZE:-128}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-$GROUP_SIZE}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
WEIGHT_SYMMETRIC="${WEIGHT_SYMMETRIC:-true}"
WEIGHT_CLIP="${WEIGHT_CLIP:-false}"

QLORA_PER_DEVICE_TRAIN_BATCH_SIZE="${QLORA_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
QLORA_PER_DEVICE_EVAL_BATCH_SIZE="${QLORA_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
QLORA_GRADIENT_ACCUMULATION_STEPS="${QLORA_GRADIENT_ACCUMULATION_STEPS:-16}"
QLORA_NUM_TRAIN_EPOCHS="${QLORA_NUM_TRAIN_EPOCHS:-1}"
QLORA_MAX_STEPS="${QLORA_MAX_STEPS:--1}"
QLORA_LOGGING_STEPS="${QLORA_LOGGING_STEPS:-10}"
QLORA_SAVE_STEPS="${QLORA_SAVE_STEPS:-1000}"
QLORA_SAVE_TOTAL_LIMIT="${QLORA_SAVE_TOTAL_LIMIT:-1}"
QLORA_LEARNING_RATE="${QLORA_LEARNING_RATE:-2e-4}"
QLORA_WEIGHT_DECAY="${QLORA_WEIGHT_DECAY:-0.0}"
QLORA_WARMUP_RATIO="${QLORA_WARMUP_RATIO:-0.03}"
QLORA_LR_SCHEDULER_TYPE="${QLORA_LR_SCHEDULER_TYPE:-constant}"
QLORA_GRADIENT_CHECKPOINTING="${QLORA_GRADIENT_CHECKPOINTING:-true}"
QLORA_LORA_R="${QLORA_LORA_R:-64}"
QLORA_LORA_ALPHA="${QLORA_LORA_ALPHA:-16}"
QLORA_LORA_DROPOUT="${QLORA_LORA_DROPOUT:-0.0}"
QLORA_DOUBLE_QUANT="${QLORA_DOUBLE_QUANT:-true}"
QLORA_QUANT_TYPE="${QLORA_QUANT_TYPE:-nf4}"
QLORA_MERGE_ADAPTER="${QLORA_MERGE_ADAPTER:-false}"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_FP_BASELINE="${RUN_FP_BASELINE:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
DRY_RUN="${DRY_RUN:-false}"
MODEL_FILTER="${MODEL_FILTER:-}"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/new_results/quantization_suite/qlora_text_suite}"

MODEL_KEYS=(
  llama2
  llama_family
  qwen2_5
  qwen3
  qwen2_5_vl
  qwen3_vl
  qwen3_5
)
RUN_TAGS=(
  full_precision_seq2048
  w2a16_seq2048
  w3a16_seq2048
  w4a16_seq2048
)

declare -a ZERO_SHOT_TASKS=()
declare -a FILTERED_MODEL_KEYS=()
declare -a FAILED_MODELS=()
declare -a CANDIDATE_PATHS=()
declare -a QUANT_CONFIGS=()

MODEL_KEY=""
MODEL_NAME=""
MODEL_PATH=""
MODEL_OUTPUT_ROOT=""

parse_arrays() {
  read -r -a ZERO_SHOT_TASKS <<< "$ZERO_SHOT_TASKS_STR"
  QUANT_CONFIGS=(
    "w2a16_seq2048 2 fake_quant"
    "w3a16_seq2048 3 fake_quant"
    "w4a16_seq2048 4 bnb_4bit"
  )
}

check_fixed_setup() {
  if [[ "$CALIBRATION_DATASET" != "pileval" ]]; then
    echo "[ERROR] CALIBRATION_DATASET must be pileval for this suite. current=$CALIBRATION_DATASET"
    exit 1
  fi
  if [[ "$CALIBRATION_SAMPLES" != "1024" ]]; then
    echo "[ERROR] CALIBRATION_SAMPLES must be 1024 for this suite. current=$CALIBRATION_SAMPLES"
    exit 1
  fi
  if [[ "$QLORA_MAX_TRAIN_SAMPLES" != "1024" ]]; then
    echo "[ERROR] QLORA_MAX_TRAIN_SAMPLES must be 1024 for this suite. current=$QLORA_MAX_TRAIN_SAMPLES"
    exit 1
  fi
  if [[ "$EVALUATION_DATASET" != "wikitext2" ]]; then
    echo "[ERROR] EVALUATION_DATASET must be wikitext2 for this suite. current=$EVALUATION_DATASET"
    exit 1
  fi
  if [[ "$SEQUENCE_LENGTH" != "2048" ]]; then
    echo "[ERROR] SEQUENCE_LENGTH must be 2048 for this suite. current=$SEQUENCE_LENGTH"
    exit 1
  fi
  if [[ "$ACTIVATION_BITS" != "16" ]]; then
    echo "[ERROR] ACTIVATION_BITS must be 16 for this suite. current=$ACTIVATION_BITS"
    exit 1
  fi
  if [[ "$GROUP_SIZE" != "128" || "$WEIGHT_GROUP_SIZE" != "128" ]]; then
    echo "[ERROR] GROUP_SIZE/WEIGHT_GROUP_SIZE must both be 128. current=$GROUP_SIZE/$WEIGHT_GROUP_SIZE"
    exit 1
  fi
}

dependency_check() {
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = ("torch", "transformers", "lm_eval", "peft", "bitsandbytes")
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"[ERROR] dependency check failed for {name}: {exc}")
        sys.exit(2)
print("[INFO] dependency check passed: torch/transformers/lm_eval/peft/bitsandbytes")
PY
}

export_runtime_env() {
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
}

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

  "${cmd[@]}" 2>&1 | tee "$log_path"
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

model_override_for_key() {
  case "$1" in
    llama2) echo "${LLAMA2_MODEL_PATH:-}" ;;
    llama_family) echo "${LLAMA_FAMILY_MODEL_PATH:-}" ;;
    qwen2_5) echo "${QWEN2_5_MODEL_PATH:-}" ;;
    qwen3) echo "${QWEN3_MODEL_PATH:-}" ;;
    qwen2_5_vl) echo "${QWEN2_5_VL_MODEL_PATH:-}" ;;
    qwen3_vl) echo "${QWEN3_VL_MODEL_PATH:-}" ;;
    qwen3_5) echo "${QWEN3_5_MODEL_PATH:-}" ;;
    *)
      echo ""
      ;;
  esac
}

resolve_model_config() {
  local model_key="$1"
  local model_override=""

  MODEL_KEY="$model_key"
  MODEL_NAME=""
  MODEL_PATH=""
  MODEL_OUTPUT_ROOT=""
  CANDIDATE_PATHS=()

  case "$model_key" in
    llama2)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Llama-2-7b-hf"
        "/mnt/82_store/LLM-weights/meta-llama/Llama-2-7b-hf"
        "/mnt/82_store/zy/model/Llama-2-7b-hf"
        "/mnt/82_store/huggingface/datasets/meta-llama/Llama-2-7b-hf"
      )
      ;;
    llama_family)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Meta-Llama-3.1-8B"
        "/mnt/82_store/LLM-weights/meta-llama/Meta-Llama-3.1-8B"
        "/mnt/82_store/LLM-weights/Meta-Llama-3.1-8B-Instruct"
        "/mnt/82_store/zy/model/Meta-Llama-3.1-8B"
        "/mnt/82_store/huggingface/datasets/meta-llama/Meta-Llama-3.1-8B"
      )
      ;;
    qwen2_5)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct"
        "/mnt/82_store/LLM-weights/Qwen/Qwen2.5-7B-Instruct"
        "/mnt/82_store/zy/model/Qwen2.5-7B-Instruct"
        "/mnt/82_store/huggingface/datasets/Qwen/Qwen2.5-7B-Instruct"
      )
      ;;
    qwen3)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Qwen3-8B"
        "/mnt/82_store/LLM-weights/Qwen/Qwen3-8B"
        "/mnt/82_store/zy/model/Qwen3-8B"
        "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-8B"
      )
      ;;
    qwen2_5_vl)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct"
        "/mnt/82_store/zy/model/Qwen2.5-VL-7B-Instruct"
        "/mnt/82_store/huggingface/datasets/Qwen/Qwen2.5-VL-7B-Instruct"
      )
      ;;
    qwen3_vl)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Qwen3-VL-2B-Instruct"
        "/mnt/82_store/LLM-weights/Qwen/Qwen3-VL-2B-Instruct"
        "/mnt/82_store/zy/model/Qwen3-VL-2B-Instruct"
        "/mnt/82_store/huggingface/datasets/Qwen/Qwen3-VL-2B-Instruct"
      )
      ;;
    qwen3_5)
      CANDIDATE_PATHS=(
        "/mnt/82_store/LLM-weights/Qwen3.5-4B"
        "/mnt/82_store/LLM-weights/Qwen/Qwen3.5-4B"
        "/mnt/82_store/LLM-weights/Qwen3_5-4B"
        "/mnt/82_store/LLM-weights/Qwen/Qwen3_5-4B"
        "/mnt/82_store/zy/model/Qwen3.5-4B"
        "/mnt/82_store/zy/model/Qwen3_5-4B"
        "/mnt/82_store/huggingface/datasets/Qwen/Qwen3.5-4B"
        "/mnt/82_store/huggingface/datasets/Qwen/Qwen3_5-4B"
      )
      ;;
    *)
      echo "[ERROR] Unknown model key: $model_key"
      return 1
      ;;
  esac

  model_override="$(model_override_for_key "$model_key")"
  if [[ -n "$model_override" ]]; then
    MODEL_PATH="$model_override"
  else
    local candidate=""
    for candidate in "${CANDIDATE_PATHS[@]}"; do
      if [[ -d "$candidate" ]]; then
        MODEL_PATH="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] MODEL_PATH not found for model_key=$model_key"
    echo "[ERROR] Tried candidates:"
    local candidate=""
    for candidate in "${CANDIDATE_PATHS[@]}"; do
      echo "  - $candidate"
    done
    return 1
  fi

  MODEL_NAME="$(basename "$MODEL_PATH")"
  MODEL_OUTPUT_ROOT="$OUTPUT_BASE/$MODEL_KEY"
}

apply_model_filter() {
  local normalized_filter=""
  local -a requested=()
  local key=""
  local item=""
  local matched=""

  if [[ -z "$MODEL_FILTER" ]]; then
    FILTERED_MODEL_KEYS=("${MODEL_KEYS[@]}")
    return 0
  fi

  normalized_filter="${MODEL_FILTER//,/ }"
  read -r -a requested <<< "$normalized_filter"
  FILTERED_MODEL_KEYS=()

  for key in "${MODEL_KEYS[@]}"; do
    matched="false"
    for item in "${requested[@]}"; do
      if [[ "$item" == "$key" ]]; then
        matched="true"
        break
      fi
    done
    if [[ "$matched" == "true" ]]; then
      FILTERED_MODEL_KEYS+=("$key")
    fi
  done

  if [[ "${#FILTERED_MODEL_KEYS[@]}" -eq 0 ]]; then
    echo "[ERROR] MODEL_FILTER did not match any supported model key: $MODEL_FILTER"
    echo "[ERROR] Supported model keys: ${MODEL_KEYS[*]}"
    exit 1
  fi
}

run_fp_baseline() {
  local run_tag="full_precision_seq2048"
  local run_output="$MODEL_OUTPUT_ROOT/$run_tag"
  local metrics_path=""
  local log_path=""
  local -a cmd=()

  metrics_path="$(find_metrics_file "$run_output")"
  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $MODEL_KEY/$run_tag (found complete metrics): $metrics_path"
    return 0
  fi

  if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$run_output"
  fi
  log_path="$run_output/run.log"
  cmd=(
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
  run_command "$log_path" "${cmd[@]}"
}

run_quant_config() {
  local run_tag="$1"
  local weight_bits="$2"
  local expected_backend="$3"
  local run_output="$MODEL_OUTPUT_ROOT/$run_tag"
  local metrics_path=""
  local log_path=""
  local -a cmd=()

  metrics_path="$(find_metrics_file "$run_output")"
  if [[ "$SKIP_EXISTING" == "true" && -n "$metrics_path" ]] && is_metrics_complete "$metrics_path"; then
    echo "[INFO] Skip $MODEL_KEY/$run_tag (found complete metrics): $metrics_path"
    return 0
  fi

  if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$run_output"
  fi
  log_path="$run_output/run.log"
  cmd=(
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
    --evaluation_dataset "$EVALUATION_DATASET"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --max_eval_chunks "$MAX_EVAL_CHUNKS"
    --weight_bits "$weight_bits"
    --activation_bits "$ACTIVATION_BITS"
    --group_size "$GROUP_SIZE"
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --weight_symmetric "$WEIGHT_SYMMETRIC"
    --weight_clip "$WEIGHT_CLIP"
    --eval_ppl true
    --eval_zero_shot true
    --zero_shot_tasks "${ZERO_SHOT_TASKS[@]}"
    --zero_shot_num_fewshot "$ZERO_SHOT_NUM_FEWSHOT"
    --zero_shot_batch_size "$ZERO_SHOT_BATCH_SIZE"
    --eval_vlm false
    --qlora_max_train_samples "$QLORA_MAX_TRAIN_SAMPLES"
    --qlora_per_device_train_batch_size "$QLORA_PER_DEVICE_TRAIN_BATCH_SIZE"
    --qlora_per_device_eval_batch_size "$QLORA_PER_DEVICE_EVAL_BATCH_SIZE"
    --qlora_gradient_accumulation_steps "$QLORA_GRADIENT_ACCUMULATION_STEPS"
    --qlora_num_train_epochs "$QLORA_NUM_TRAIN_EPOCHS"
    --qlora_max_steps "$QLORA_MAX_STEPS"
    --qlora_logging_steps "$QLORA_LOGGING_STEPS"
    --qlora_save_steps "$QLORA_SAVE_STEPS"
    --qlora_save_total_limit "$QLORA_SAVE_TOTAL_LIMIT"
    --qlora_learning_rate "$QLORA_LEARNING_RATE"
    --qlora_weight_decay "$QLORA_WEIGHT_DECAY"
    --qlora_warmup_ratio "$QLORA_WARMUP_RATIO"
    --qlora_lr_scheduler_type "$QLORA_LR_SCHEDULER_TYPE"
    --qlora_gradient_checkpointing "$QLORA_GRADIENT_CHECKPOINTING"
    --qlora_lora_r "$QLORA_LORA_R"
    --qlora_lora_alpha "$QLORA_LORA_ALPHA"
    --qlora_lora_dropout "$QLORA_LORA_DROPOUT"
    --qlora_double_quant "$QLORA_DOUBLE_QUANT"
    --qlora_quant_type "$QLORA_QUANT_TYPE"
    --qlora_merge_adapter "$QLORA_MERGE_ADAPTER"
    --output_dir "$run_output"
    --log_level "$LOG_LEVEL"
  )

  append_optional_args cmd
  echo "[INFO] Expected QLoRA backend for $MODEL_KEY/$run_tag: $expected_backend"
  run_command "$log_path" "${cmd[@]}"
}

summarize_model_results() {
  "$PYTHON_BIN" - "$MODEL_OUTPUT_ROOT" "$MODEL_NAME" "${RUN_TAGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
model_name = sys.argv[2]
run_tags = sys.argv[3:]

print(f"[SUMMARY] {model_name}")
print("[SUMMARY] run_tag\tbackend\tperplexity\tzero_shot_acc_avg\tmetrics_path")

def load_artifacts(metrics_path, payload):
    artifacts = payload.get("artifacts") or {}
    if artifacts:
        return artifacts
    artifacts_name = payload.get("artifacts_path")
    if not artifacts_name:
        return {}
    artifacts_path = metrics_path.parent / artifacts_name
    if not artifacts_path.is_file():
        return {}
    return json.loads(artifacts_path.read_text(encoding="utf-8"))

for tag in run_tags:
    run_dir = output_root / tag
    metrics_files = sorted(run_dir.rglob("metrics.json"))
    if not metrics_files:
        print(f"{tag}\tMISSING\tMISSING\tMISSING\t-")
        continue

    metrics_path = metrics_files[0]
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    ppl = payload.get("perplexity")
    zero_shot = payload.get("zero_shot") or {}
    acc_avg = zero_shot.get("acc_avg") if isinstance(zero_shot, dict) else None
    artifacts = load_artifacts(metrics_path, payload)
    qlora_cfg = artifacts.get("qlora_config") or {}
    backend = qlora_cfg.get("backend", "full_precision")

    ppl_str = f"{ppl:.6f}" if isinstance(ppl, (int, float)) else "NA"
    acc_str = f"{acc_avg:.4f}" if isinstance(acc_avg, (int, float)) else "NA"
    print(f"{tag}\t{backend}\t{ppl_str}\t{acc_str}\t{metrics_path}")
PY
}

write_suite_summary() {
  local summary_path="$OUTPUT_BASE/suite_summary.tsv"
  "$PYTHON_BIN" - "$OUTPUT_BASE" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []

def load_artifacts(metrics_path, payload):
    artifacts = payload.get("artifacts") or {}
    if artifacts:
        return artifacts
    artifacts_name = payload.get("artifacts_path")
    if not artifacts_name:
        return {}
    artifacts_path = metrics_path.parent / artifacts_name
    if not artifacts_path.is_file():
        return {}
    return json.loads(artifacts_path.read_text(encoding="utf-8"))

for metrics_path in sorted(output_base.rglob("metrics.json")):
    rel = metrics_path.relative_to(output_base)
    if len(rel.parts) < 2:
        continue
    model_key = rel.parts[0]
    run_tag = rel.parts[1]
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    artifacts = load_artifacts(metrics_path, payload)
    qlora_cfg = artifacts.get("qlora_config") or {}
    zero_shot = payload.get("zero_shot") or {}
    rows.append(
        {
            "model_key": model_key,
            "run_tag": run_tag,
            "backend": qlora_cfg.get("backend", "full_precision"),
            "weight_bits": payload.get("weight_bits", qlora_cfg.get("weight_bits")),
            "perplexity": payload.get("perplexity"),
            "zero_shot_acc_avg": zero_shot.get("acc_avg") if isinstance(zero_shot, dict) else None,
            "model_path": payload.get("model_path"),
            "metrics_path": str(metrics_path),
        }
    )

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as handle:
    handle.write("model_key\trun_tag\tbackend\tweight_bits\tperplexity\tzero_shot_acc_avg\tmodel_path\tmetrics_path\n")
    for row in rows:
        handle.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                row["model_key"],
                row["run_tag"],
                row["backend"],
                "" if row["weight_bits"] is None else row["weight_bits"],
                "" if row["perplexity"] is None else row["perplexity"],
                "" if row["zero_shot_acc_avg"] is None else row["zero_shot_acc_avg"],
                "" if row["model_path"] is None else row["model_path"],
                row["metrics_path"],
            )
        )
print(f"[INFO] Wrote suite summary: {summary_path}")
PY
}

run_model_suite() {
  local model_key="$1"
  local config=""
  local run_tag=""
  local weight_bits=""
  local backend=""

  resolve_model_config "$model_key"

  echo
  echo "[INFO] ===== model_key=$MODEL_KEY model_name=$MODEL_NAME ====="
  echo "[INFO] model_path=$MODEL_PATH"
  echo "[INFO] output_root=$MODEL_OUTPUT_ROOT"

  if [[ "$RUN_FP_BASELINE" == "true" ]]; then
    run_fp_baseline
  fi
  for config in "${QUANT_CONFIGS[@]}"; do
    read -r run_tag weight_bits backend <<< "$config"
    run_quant_config "$run_tag" "$weight_bits" "$backend"
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] DRY_RUN=true, summary skipped for $MODEL_KEY."
    return 0
  fi

  summarize_model_results
}

main() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi

  parse_arrays
  apply_model_filter
  check_fixed_setup
  export_runtime_env
  dependency_check

  mkdir -p "$OUTPUT_BASE"

  echo "[INFO] $SCRIPT_NAME"
  echo "[INFO] algorithm=$ALGORITHM"
  echo "[INFO] output_base=$OUTPUT_BASE"
  echo "[INFO] model_keys=${FILTERED_MODEL_KEYS[*]}"
  echo "[INFO] zero_shot_tasks=${ZERO_SHOT_TASKS[*]}"
  echo "[INFO] calibration_dataset=$CALIBRATION_DATASET calibration_samples=$CALIBRATION_SAMPLES qlora_max_train_samples=$QLORA_MAX_TRAIN_SAMPLES"
  echo "[INFO] evaluation_dataset=$EVALUATION_DATASET sequence_length=$SEQUENCE_LENGTH max_eval_chunks=$MAX_EVAL_CHUNKS"
  echo "[INFO] weight policy: group_size=$GROUP_SIZE weight_group_size=$WEIGHT_GROUP_SIZE activation_bits=$ACTIVATION_BITS"
  echo "[INFO] qlora policy: w2/w3=fake_quant, w4=bitsandbytes_4bit"
  echo "[INFO] run_tags=${RUN_TAGS[*]}"

  for MODEL_KEY in "${FILTERED_MODEL_KEYS[@]}"; do
    if ! run_model_suite "$MODEL_KEY"; then
      echo "[ERROR] Model suite failed: $MODEL_KEY"
      FAILED_MODELS+=("$MODEL_KEY")
      if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
        exit 1
      fi
    fi
  done

  if [[ "$DRY_RUN" != "true" ]]; then
    write_suite_summary
  fi

  if [[ "${#FAILED_MODELS[@]}" -gt 0 ]]; then
    echo "[ERROR] Failed model suites: ${FAILED_MODELS[*]}"
    exit 1
  fi

  echo
  echo "[INFO] All requested model suites completed."
}

main "$@"
