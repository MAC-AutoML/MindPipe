#!/usr/bin/env bash
# Usage:
#   1. Edit the experiment matrix below as needed:
#        - MODELS
#        - FLATQUANT_CONFIGS
#        - PRUNING_CONFIGS
#   2. Choose NPUs and run:
#        # Per-algorithm scheduling (like scripts/run_quantization_then_pruning_gpu.sh):
#        #   FLAP_NPUS=0 SPARSEGPT_NPUS=1 WANDA_NPUS=2 ALPS_NPUS=3 \
#        #   bash scripts/run_quantization_then_pruning_npu.sh
#        # Expose multiple NPUs to a single worker for device_map sharding:
#        #   WANDA_NPUS=0+1 SPARSEGPT_NPUS=2+3 ALPS_NPUS=4+5 bash scripts/run_quantization_then_pruning_npu.sh
#        # Legacy scheduling (single shared NPU pool):
#        #   WORKFLOW_NPUS=0,1,2,3 bash scripts/run_quantization_then_pruning_npu.sh
#   3. Common overrides:
#        DRY_RUN=true bash scripts/run_quantization_then_pruning_npu.sh
#        MODE=save_model bash scripts/run_quantization_then_pruning_npu.sh
#        SAVE_MODEL_OUTPUT_ROOT=/path/to/save_model_root \
#        MODE=save_model bash scripts/run_quantization_then_pruning_npu.sh
#        FLATQUANT_REUSE_CHECKPOINTS=false bash scripts/run_quantization_then_pruning_npu.sh
#        FLATQUANT_CHECKPOINT_ROOT=/path/to/flatquant_root \
#        FLATQUANT_REQUIRE_CHECKPOINTS=true \
#        WORKFLOW_NPUS=0,1 \
#        bash scripts/run_quantization_then_pruning_npu.sh
# Notes:
#   - MODE=full (default): run the normal evaluation pipeline and do not save model.
#   - MODE=save_model: skip all evaluations and only save the compressed model.
#     Outputs are written under <base_output_root>/save_model_only by default.
#   - When FLATQUANT_REUSE_CHECKPOINTS=true, the script prefers
#     flat_matrices.pth, then falls back to flat_parameters.pth.
#   - In save_model mode, a run is considered complete only if both
#     metrics.json and saved_model/ weights exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python selection: prefer the `mindpipe` conda env when present on this host,
# but allow overriding via PYTHON_BIN.
PYTHON_BIN_DEFAULT=""
if [[ -x "/home/ma-user/anaconda3/envs/mindpipe/bin/python" ]]; then
  PYTHON_BIN_DEFAULT="/home/ma-user/anaconda3/envs/mindpipe/bin/python"
else
  PYTHON_BIN_DEFAULT="$(command -v python || true)"
  if [[ -z "$PYTHON_BIN_DEFAULT" && -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN_DEFAULT="$CONDA_PREFIX/bin/python"
  fi
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
MAX_MEMORY="${MAX_MEMORY:-}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-}"
OFFLOAD_STATE_DICT="${OFFLOAD_STATE_DICT:-}"
NO_SPLIT_MODULE_CLASSES="${NO_SPLIT_MODULE_CLASSES:-}"
DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
QUANTIZATION_CALIBRATION_SAMPLES="${QUANTIZATION_CALIBRATION_SAMPLES:-128}"
PRUNING_CALIBRATION_SAMPLES="${PRUNING_CALIBRATION_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EXECUTION_ORDER="quantization_then_pruning"
PYTHON_UNBUFFERED="${PYTHON_UNBUFFERED:-true}"

# Shared evaluation defaults
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-true}"
ZERO_SHOT_TASKS="${ZERO_SHOT_TASKS:-boolq rte winogrande arc_easy arc_challenge openbookqa}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-4}"
ZERO_SHOT_NUM_FEWSHOT="${ZERO_SHOT_NUM_FEWSHOT:-0}"
EVAL_VLM="${EVAL_VLM:-true}"
VLM_DATASETS="${VLM_DATASETS:-OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL}"
VLM_MODE="${VLM_MODE:-all}"
VLM_API_NPROC="${VLM_API_NPROC:-4}"
VLM_PRED_FORMAT="${VLM_PRED_FORMAT:-xlsx}"
HF_ENDPOINT_DEFAULT="${HF_ENDPOINT_DEFAULT:-https://hf-mirror.com}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
AUTO_NPU_ALLOC_CONF="${AUTO_NPU_ALLOC_CONF:-true}"
PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-}"
PYTORCH_NPU_ALLOC_CONF_DEFAULT="${PYTORCH_NPU_ALLOC_CONF_DEFAULT:-max_split_size_mb:32,garbage_collection_threshold:0.6}"
AUTO_MAX_MEMORY="${AUTO_MAX_MEMORY:-true}"
AUTO_MAX_MEMORY_HEADROOM_GIB="${AUTO_MAX_MEMORY_HEADROOM_GIB:-8}"
AUTO_MULTIDEVICE_DEVICE_MAP="${AUTO_MULTIDEVICE_DEVICE_MAP:-true}"
NPU_MULTIDEVICE_DEVICE_MAP="${NPU_MULTIDEVICE_DEVICE_MAP:-balanced}"

# HuggingFace networking defaults for ModelArts-like environments:
# - A forced corporate proxy (e.g. proxy.modelarts.com) can return 503 for HF Hub requests.
# - Use hf-mirror by default and bypass proxies for HF domains via NO_PROXY/no_proxy.
# Prefer domain suffix entries (".example.com") for broad client compatibility.
HF_NO_PROXY_DEFAULT="${HF_NO_PROXY_DEFAULT:-hf-mirror.com,.hf-mirror.com,huggingface.co,.huggingface.co}"
HF_TIMEOUT_DEFAULT="${HF_TIMEOUT_DEFAULT:-120}"

# Local modelzoo defaults (override via MODELZOO_ROOT=/path/to/modelzoo)
MODELZOO_ROOT_DEFAULT="/home/ma-user/work/modelzoo"
MODELZOO_ROOT="${MODELZOO_ROOT:-$MODELZOO_ROOT_DEFAULT}"
MODEL_SYNC="${MODEL_SYNC:-false}"
MODEL_SYNC_PY="${MODEL_SYNC_PY:-$REPO_ROOT/scripts/sync_models.py}"

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
FLATQUANT_REUSE_CHECKPOINTS="${FLATQUANT_REUSE_CHECKPOINTS:-true}"
FLATQUANT_CHECKPOINT_ROOT="${FLATQUANT_CHECKPOINT_ROOT:-}"
FLATQUANT_REQUIRE_CHECKPOINTS="${FLATQUANT_REQUIRE_CHECKPOINTS:-false}"

# Pruning defaults aligned with standalone pruning scripts
SPARSEGPT_BLOCK_SIZE="${SPARSEGPT_BLOCK_SIZE:-64}"
PRUNING_DAMP_PERCENT="${PRUNING_DAMP_PERCENT:-0.01}"
SPARSEGPT_STRUCTURE_PATTERN="${SPARSEGPT_STRUCTURE_PATTERN:-unstructured}"
WANDA_STRUCTURE_PATTERN="${WANDA_STRUCTURE_PATTERN:-unstructured}"
# Do not default ALPS to C4 unless local C4 files are present.
# The current MindPipe C4 online loader uses a legacy config name ("allenai--c4")
# which is not available in recent `datasets` versions; if you want C4, place the
# shard under `$DATA_PATH/c4/en/c4-train.00000-of-01024.json.gz`.
ALPS_CALIBRATION_DATASET="${ALPS_CALIBRATION_DATASET:-}"
ALPS_STRUCTURE_PATTERN="${ALPS_STRUCTURE_PATTERN:-unstructured}"
FLAP_STRUCTURE_PATTERN="${FLAP_STRUCTURE_PATTERN:-AL-AM}"
FLAP_METRICS="${FLAP_METRICS:-WIFV}"
FLAP_REMOVE_HEADS="${FLAP_REMOVE_HEADS:-8}"
PSEUDO_PRUNING="${PSEUDO_PRUNING:-true}"
ALPS_RHO="${ALPS_RHO:-0.1}"


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
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen2.5-7B-Instruct" \
  #   "/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Meta/Llama-2-7b-hf" \
  #   "/mnt/82_store/LLM-weights/Llama-2-7b-hf")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Meta/Meta-Llama-3.1-8B-Instruct" \
  #   "/mnt/82_store/LLM-weights/Meta-Llama-3.1-8B-Instruct")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen2.5-VL-7B-Instruct" \
  #   "/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/openbmb/MiniCPM-V" \
  #   "/mnt/82_store/LLM-weights/openbmb/MiniCPM-V")"
  # "$(pick_first_existing_path \
  #   "${MODELZOO_ROOT}/Qwen/Qwen3.6-27B" \
  #   "/mnt/82_store/LLM-weights/Qwen3.6-27B" \
  #   "/mnt/82_store/LLM-weights/Qwen/Qwen3.6-27B")"
  "$(pick_first_existing_path \
    "${MODELZOO_ROOT}/Qwen/Qwen3-30B-A3B" \
    "/mnt/82_store/LLM-weights/Qwen3-30B-A3B" \
    "/mnt/82_store/LLM-weights/Qwen/Qwen3-30B-A3B")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen3-VL-2B-Instruct" \
  #   "/mnt/82_store/LLM-weights/Qwen3-VL-2B-Instruct" \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen3-VL-2B")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen3-0.6B" \
  #   "/mnt/82_store/LLM-weights/Qwen3-0.6B" \
  #   "/mnt/82_store/LLM-weights/Qwen/Qwen3-0.6B")"
  # "$(pick_first_existing_path \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen3.5-4B" \
  #   "/home/ma-user/work/modelzoo/Qwen/Qwen3_5-4B" \
  #   "/mnt/82_store/LLM-weights/Qwen3.5-4B" \
  #   "/mnt/82_store/LLM-weights/Qwen/Qwen3.5-4B" \
  #   "/mnt/82_store/LLM-weights/Qwen3_5-4B" \
  #   "/mnt/82_store/LLM-weights/Qwen/Qwen3_5-4B")"
)
FLATQUANT_CONFIGS=(
  "4 16 16 4 4 w4a16"
  "8 8 16 4 4 w8a8"
)

PRUNING_CONFIGS=(
  "sparsegpt 0.5"
  "alps 0.5"
)

# Worker scheduling
ENABLE_FLAP="${ENABLE_FLAP:-false}"
ENABLE_SPARSEGPT="${ENABLE_SPARSEGPT:-true}"
ENABLE_WANDA="${ENABLE_WANDA:-false}"
ENABLE_ALPS="${ENABLE_ALPS:-false}"

FLAP_NPUS="${FLAP_NPUS:-}"
SPARSEGPT_NPUS="${SPARSEGPT_NPUS:-}"
WANDA_NPUS="${WANDA_NPUS:-}"
ALPS_NPUS="${ALPS_NPUS:-}"

default_workflow_npus() {
  # Prefer exposing all logical NPUs to a single worker so large models can shard via device_map.
  local logical_count
  logical_count="$(PYTHON_BIN="$PYTHON_BIN" mindpipe_query_npu_logical_count 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$logical_count" =~ ^[0-9]+$ ]] && (( logical_count > 0 )); then
    local spec=""
    local i
    for ((i=0; i<logical_count; i++)); do
      if (( i > 0 )); then
        spec+="+"
      fi
      spec+="$i"
    done
    printf '%s' "$spec"
    return 0
  fi
  printf '0'
  return 0
}

WORKFLOW_NPUS="${WORKFLOW_NPUS:-}"
if [[ -z "${WORKFLOW_NPUS//[[:space:]]/}" ]]; then
  WORKFLOW_NPUS="$(default_workflow_npus)"
fi

# Execution control
FORCE_RERUN="${FORCE_RERUN:-false}"
DRY_RUN="${DRY_RUN:-false}"
MODE="${MODE:-full}"
SAVE_MODEL_OUTPUT_ROOT="${SAVE_MODEL_OUTPUT_ROOT:-}"

case "$MODE" in
  full)
    MODE_EVAL_PPL="$EVAL_PPL"
    MODE_EVAL_ZERO_SHOT="$EVAL_ZERO_SHOT"
    MODE_EVAL_VLM="$EVAL_VLM"
    MODE_SAVE_MODEL="false"
    ;;
  save_model)
    MODE_EVAL_PPL="false"
    MODE_EVAL_ZERO_SHOT="false"
    MODE_EVAL_VLM="false"
    MODE_SAVE_MODEL="true"
    ;;
  *)
    printf 'Unsupported MODE: %s (expected: full or save_model)\n' "$MODE" >&2
    exit 2
    ;;
esac

LAST_RUN_STATUS=""
WORKER_PIDS=()
WORKER_LABELS=()
WORKER_NPUS=()


output_root() {
  local base_root="${OUTPUT_DIR:-${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}}"
  if [[ "$MODE" == "save_model" ]]; then
    printf '%s' "${SAVE_MODEL_OUTPUT_ROOT:-$base_root/save_model_only}"
  else
    printf '%s' "$base_root"
  fi
}


flatquant_checkpoint_root() {
  local base_root="${OUTPUT_DIR:-${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}}"
  if [[ -n "$FLATQUANT_CHECKPOINT_ROOT" ]]; then
    printf '%s' "$FLATQUANT_CHECKPOINT_ROOT"
  elif [[ "$MODE" == "save_model" ]]; then
    printf '%s' "$base_root"
  else
    printf '%s' "$(output_root)"
  fi
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
    alps)
      explicit_value="${ALPS_CALIBRATION_DATASET:-${PRUNING_CALIBRATION_DATASET:-}}"
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

default_npu_max_memory_map_for_visible_devices() {
  local visible_devices_csv="$1"
  local headroom_gib="$2"

  if [[ -z "$visible_devices_csv" ]]; then
    return 1
  fi

  local python_bin="$PYTHON_BIN"
  if [[ -z "$python_bin" ]]; then
    python_bin="$(command -v python || true)"
  fi
  if [[ -z "$python_bin" ]]; then
    return 1
  fi

  env \
    ASCEND_RT_VISIBLE_DEVICES="$visible_devices_csv" \
    MINDPIPE_HEADROOM_GIB="$headroom_gib" \
    PYTHONWARNINGS=ignore \
    "$python_bin" - <<'PY' 2>/dev/null
import os
import torch
import torch_npu  # noqa: F401

headroom = int(os.environ.get("MINDPIPE_HEADROOM_GIB", "8"))
device_count = int(torch.npu.device_count())

parts = []
for index in range(device_count):
    total = int(torch.npu.get_device_properties(index).total_memory)
    gib_total = total // (1024**3)
    gib_limit = max(int(gib_total) - headroom, 1)
    parts.append(f"{index}:{gib_limit}GiB")
print(",".join(parts))
PY
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

maybe_sync_models() {
  if [[ "$MODEL_SYNC" != "true" ]]; then
    return 0
  fi

  if [[ ! -f "$MODEL_SYNC_PY" ]]; then
    printf '[preflight-fail] MODEL_SYNC requested but helper script missing: %s\n' "$MODEL_SYNC_PY" >&2
    return 1
  fi

  printf '[preflight] MODEL_SYNC=true; attempting to sync missing models into %s\n' "$MODELZOO_ROOT"
  "$PYTHON_BIN" "$MODEL_SYNC_PY" --modelzoo-root "$MODELZOO_ROOT" --endpoint "${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}" || return 1
  return 0
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
  printf '[preflight] mode=%s\n' "$MODE"
  printf '[preflight] python=%s\n' "$PYTHON_BIN"

  maybe_sync_models || failed=1

  if has_any_algorithm_npus_configured; then
    printf '[preflight] flap_npus=%s\n' "$FLAP_NPUS"
    printf '[preflight] sparsegpt_npus=%s\n' "$SPARSEGPT_NPUS"
    printf '[preflight] wanda_npus=%s\n' "$WANDA_NPUS"
    printf '[preflight] alps_npus=%s\n' "$ALPS_NPUS"

    if [[ "$ENABLE_FLAP" == "true" ]]; then
      local -a flap_npu_specs=()
      parse_npu_specs "$FLAP_NPUS" flap_npu_specs
      validate_npu_specs "flap" flap_npu_specs || failed=1
    fi

    if [[ "$ENABLE_SPARSEGPT" == "true" ]]; then
      local -a sparsegpt_npu_specs=()
      parse_npu_specs "$SPARSEGPT_NPUS" sparsegpt_npu_specs
      validate_npu_specs "sparsegpt" sparsegpt_npu_specs || failed=1
    fi

    if [[ "$ENABLE_WANDA" == "true" ]]; then
      local -a wanda_npu_specs=()
      parse_npu_specs "$WANDA_NPUS" wanda_npu_specs
      validate_npu_specs "wanda" wanda_npu_specs || failed=1
    fi

    if [[ "$ENABLE_ALPS" == "true" ]]; then
      local -a alps_npu_specs=()
      parse_npu_specs "$ALPS_NPUS" alps_npu_specs
      validate_npu_specs "alps" alps_npu_specs || failed=1
    fi
  else
    printf '[preflight] workflow_npus=%s\n' "$WORKFLOW_NPUS"
    parse_npu_specs "$WORKFLOW_NPUS" workflow_npu_specs
    validate_npu_specs "workflow" workflow_npu_specs || failed=1
  fi

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

  if [[ "$MODE_EVAL_ZERO_SHOT" == "true" ]]; then
    require_python_import "lm_eval" "Install lm-evaluation-harness into the same Python environment running this script." || failed=1
  fi

  if [[ "$need_vlm" == "true" ]]; then
    check_vlm_eval_runtime || failed=1
  fi

  if [[ "$MODE_EVAL_PPL" == "true" ]]; then
    require_dataset_availability "$EVALUATION_DATASET" "evaluation" || failed=1
  fi

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


saved_model_dir_for() {
  local metrics_path="$1"
  printf '%s/saved_model' "$(dirname "$metrics_path")"
}


is_complete() {
  local metrics_path="$1"
  local require_vlm="${2:-false}"
  local require_saved_model="${3:-false}"
  [[ -f "$metrics_path" ]] || return 1
  if [[ "$MODE_EVAL_PPL" == "true" ]] && ! grep -q '"perplexity"' "$metrics_path"; then
    return 1
  fi
  if [[ "$MODE_EVAL_ZERO_SHOT" == "true" ]] && ! grep -q '"zero_shot"' "$metrics_path"; then
    return 1
  fi
  if [[ "$require_vlm" == "true" ]] && ! grep -q '"vlm_eval"' "$metrics_path"; then
    return 1
  fi
  if [[ "$require_saved_model" == "true" ]]; then
    local saved_model_dir
    saved_model_dir="$(saved_model_dir_for "$metrics_path")"
    [[ -f "$saved_model_dir/config.json" ]] || return 1
    find "$saved_model_dir" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q . || return 1
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
  if [[ "$MODE_EVAL_VLM" != "true" ]]; then
    return 1
  fi
  is_vlm_model "$model_path"
}


has_any_algorithm_npus_configured() {
  [[ -n "${FLAP_NPUS//[[:space:],]/}" ]] || [[ -n "${SPARSEGPT_NPUS//[[:space:],]/}" ]] || [[ -n "${WANDA_NPUS//[[:space:],]/}" ]] || [[ -n "${ALPS_NPUS//[[:space:],]/}" ]]
}


resolve_visible_devices() {
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    return 1
  fi

  # Allow grouping multiple visible NPUs for a single worker using '+'.
  # Example: "4+5" -> "4,5", "npu:4+npu:5" -> "4,5".
  local normalized="${npu_spec//npu:/}"
  normalized="${normalized//[[:space:]]/}"
  if [[ -z "$normalized" ]]; then
    printf '[npu-map-fail] empty NPU spec after normalization: `%s`\n' "$npu_spec" >&2
    return 1
  fi

  local -a requested_ids=()
  IFS='+' read -r -a requested_ids <<< "$normalized"

  local -a visible_ids=()
  local -A seen_visible_ids=()
  local requested_id
  local visible_id
  for requested_id in "${requested_ids[@]}"; do
    [[ -n "$requested_id" ]] || continue
    if ! visible_id="$(mindpipe_resolve_visible_device_id "$requested_id")"; then
      printf '[npu-map-fail] unable to map `%s` to ASCEND_RT_VISIBLE_DEVICES. %s\n' "$requested_id" "$MINDPIPE_NPU_ID_MAP_ERROR" >&2
      return 1
    fi
    if [[ -n "${seen_visible_ids[$visible_id]+x}" ]]; then
      printf '[npu-map-fail] NPU spec `%s` resolves to duplicate logical device id %s\n' "$npu_spec" "$visible_id" >&2
      return 1
    fi
    seen_visible_ids["$visible_id"]=1
    visible_ids+=("$visible_id")
  done

  if ((${#visible_ids[@]} == 0)); then
    printf '[npu-map-fail] NPU spec `%s` did not contain any valid device ids\n' "$npu_spec" >&2
    return 1
  fi

  local joined=""
  local index
  for index in "${!visible_ids[@]}"; do
    if (( index > 0 )); then
      joined+=","
    fi
    joined+="${visible_ids[$index]}"
  done
  printf '%s' "$joined"
  return 0
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
  local visible_device_csv
  local -a visible_device_list=()
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

    if ! visible_device_csv="$(resolve_visible_devices "$npu_spec")"; then
      printf '[preflight-fail] %s NPU spec `%s` is not usable on this host\n' "$label" "$npu_spec" >&2
      return 1
    fi

    visible_device_list=()
    IFS=',' read -r -a visible_device_list <<< "$visible_device_csv"
    for visible_device in "${visible_device_list[@]}"; do
      [[ -n "$visible_device" ]] || continue
      if [[ -n "${seen_visible_devices[$visible_device]+x}" ]]; then
        printf '[preflight-fail] %s NPU specs `%s` and `%s` both resolve to logical device %s\n' \
          "$label" \
          "${seen_visible_devices[$visible_device]}" \
          "$npu_spec" \
          "$visible_device" >&2
        return 1
      fi
      seen_visible_devices["$visible_device"]="$npu_spec"
    done
    mapping_entries+=("${npu_spec}->${visible_device_csv}")
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
    alps)
      cmd_ref+=(
        --structure_pattern "$ALPS_STRUCTURE_PATTERN"
        --rho "$ALPS_RHO"
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
  local effective_device_map="$DEVICE_MAP"
  local effective_max_memory="$MAX_MEMORY"
  local effective_eval_vlm="false"
  local metrics_path
  local saved_model_dir
  local quantization_calibration_dataset
  metrics_path="$(metrics_path_for "$model_path" "$weight_bits" "$activation_bits" "$pruning_algorithm" "$sparsity_ratio")"
  saved_model_dir="$(saved_model_dir_for "$metrics_path")"
  quantization_calibration_dataset="$(resolve_quantization_calibration_dataset)"
  if should_eval_vlm "$model_path"; then
    effective_eval_vlm="true"
  fi

  local sparsity_tag
  sparsity_tag="$(format_decimal "$sparsity_ratio")"
  local run_id="${model_name}__flatquant__${quant_label}__${pruning_algorithm}_s${sparsity_tag}"

  if [[ "$FORCE_RERUN" != "true" ]] && is_complete "$metrics_path" "$effective_eval_vlm" "$MODE_SAVE_MODEL"; then
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

    if [[ "$AUTO_MULTIDEVICE_DEVICE_MAP" == "true" && "$effective_device_map" == "auto" && "$visible_devices" == *","* ]]; then
      effective_device_map="$NPU_MULTIDEVICE_DEVICE_MAP"
    fi

    if [[ -z "$effective_max_memory" && "$AUTO_MAX_MEMORY" == "true" && "$visible_devices" == *","* ]]; then
      effective_max_memory="$(default_npu_max_memory_map_for_visible_devices "$visible_devices" "$AUTO_MAX_MEMORY_HEADROOM_GIB" || true)"
      if [[ -z "$effective_max_memory" ]]; then
        printf '[warn][max_memory] failed to infer max_memory for visible_devices=%s; fall back to transformers defaults.\n' "$visible_devices" >&2
      fi
    fi

    local effective_npu_alloc_conf=""
    if [[ -n "$PYTORCH_NPU_ALLOC_CONF" ]]; then
      effective_npu_alloc_conf="$PYTORCH_NPU_ALLOC_CONF"
    elif [[ -n "$PYTORCH_ALLOC_CONF" ]]; then
      # Backward-compat: map the generic knob to the actual NPU allocator env var.
      effective_npu_alloc_conf="$PYTORCH_ALLOC_CONF"
    elif [[ "$AUTO_NPU_ALLOC_CONF" == "true" ]]; then
      effective_npu_alloc_conf="$PYTORCH_NPU_ALLOC_CONF_DEFAULT"
    fi
    if [[ -n "$effective_npu_alloc_conf" ]]; then
      env_vars+=("PYTORCH_NPU_ALLOC_CONF=$effective_npu_alloc_conf")
    fi
  fi
  if [[ "$PYTHON_UNBUFFERED" == "true" ]]; then
    env_vars+=("PYTHONUNBUFFERED=1")
  fi
  # Apply HuggingFace mirror + proxy bypass settings.
  local hf_endpoint
  hf_endpoint="${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}"
  local hf_timeout
  hf_timeout="${HF_TIMEOUT:-$HF_TIMEOUT_DEFAULT}"
  local hf_no_proxy
  hf_no_proxy="${HF_NO_PROXY:-$HF_NO_PROXY_DEFAULT}"

  env_vars+=("HF_ENDPOINT=$hf_endpoint")
  if [[ -n "${NO_PROXY:-}" ]]; then
    env_vars+=("NO_PROXY=${NO_PROXY},${hf_no_proxy}")
  else
    env_vars+=("NO_PROXY=${hf_no_proxy}")
  fi
  if [[ -n "${no_proxy:-}" ]]; then
    env_vars+=("no_proxy=${no_proxy},${hf_no_proxy}")
  else
    env_vars+=("no_proxy=${hf_no_proxy}")
  fi
  if [[ -n "$hf_timeout" ]]; then
    env_vars+=("HF_HUB_DOWNLOAD_TIMEOUT=$hf_timeout")
    env_vars+=("HF_HUB_REQUEST_TIMEOUT=$hf_timeout")
    env_vars+=("HF_HUB_ETAG_TIMEOUT=$hf_timeout")
  fi
  if [[ -n "$PYTORCH_ALLOC_CONF" ]]; then
    env_vars+=("PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF")
  fi

  local -a cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    --model_path "$model_path"
    --device "$runtime_device"
    --device_map "$effective_device_map"
  )
  if [[ -n "$effective_max_memory" ]]; then
    cmd+=(--max_memory "$effective_max_memory")
  fi
  if [[ -n "$OFFLOAD_FOLDER" ]]; then
    cmd+=(--offload_folder "$OFFLOAD_FOLDER")
  fi
  if [[ -n "$OFFLOAD_STATE_DICT" ]]; then
    cmd+=(--offload_state_dict "$OFFLOAD_STATE_DICT")
  fi
  if [[ -n "$NO_SPLIT_MODULE_CLASSES" ]]; then
    local normalized_no_split
    normalized_no_split="${NO_SPLIT_MODULE_CLASSES//,/ }"
    local -a no_split_module_classes=()
    read -r -a no_split_module_classes <<< "$normalized_no_split"
    if ((${#no_split_module_classes[@]} > 0)); then
      cmd+=(--no_split_module_classes "${no_split_module_classes[@]}")
    fi
  fi
  cmd+=(
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
    --eval_ppl "$MODE_EVAL_PPL"
    --eval_zero_shot "$MODE_EVAL_ZERO_SHOT"
    --eval_vlm "$effective_eval_vlm"
    --save_model "$MODE_SAVE_MODEL"
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
  if [[ "$FLATQUANT_REUSE_CHECKPOINTS" == "true" ]]; then
    local rotation_root
    rotation_root="$(flatquant_checkpoint_root)"
    local flatquant_dir="${rotation_root}/${model_name}/flatquant/flatquant_w${weight_bits}a${activation_bits}_q${query_bits}k${key_bits}v${value_bits}_seq${SEQUENCE_LENGTH:-$SEQUENCE_LENGTH_DEFAULT}"
    if [[ -f "$flatquant_dir/flat_matrices.pth" ]]; then
      cmd+=(--flatquant_reload_matrix_from "$flatquant_dir")
    elif [[ -f "$flatquant_dir/flat_parameters.pth" ]]; then
      cmd+=(--flatquant_resume_from "$flatquant_dir")
    elif [[ "$FLATQUANT_REQUIRE_CHECKPOINTS" == "true" ]]; then
      printf '[fail][flatquant-reuse] missing flatquant checkpoint in %s\n' "$flatquant_dir" >&2
      LAST_RUN_STATUS="fail"
      return 1
    else
      printf '[warn][flatquant-reuse] missing flatquant checkpoint in %s; fall back to calibration\n' "$flatquant_dir" >&2
    fi
  fi
  append_optional_arg cmd --hf_token "${HF_TOKEN:-}"
  append_zero_shot_args cmd
  append_vlm_args cmd "$effective_eval_vlm"
  append_pruning_args cmd "$pruning_algorithm"
  append_optional_arg cmd --num_samples "${NUM_SAMPLES:-}"

  printf '[run][workflow][npu=%s] %s\n' "$npu_spec" "$run_id"
  printf '  mode: %s\n' "$MODE"
  printf '  out: %s\n' "$metrics_path"
  printf '  eval_ppl: %s\n' "$MODE_EVAL_PPL"
  printf '  eval_zero_shot: %s\n' "$MODE_EVAL_ZERO_SHOT"
  printf '  eval_vlm: %s\n' "$effective_eval_vlm"
  printf '  save_model: %s\n' "$MODE_SAVE_MODEL"
  if [[ "$MODE_SAVE_MODEL" == "true" ]]; then
    printf '  saved_model_dir: %s\n' "$saved_model_dir"
  fi
  printf '  cmd:'
  printf ' %q' env "${env_vars[@]}" "${cmd[@]}"
  printf '\n'
  if [[ "$pruning_algorithm" == "alps" && "$runtime_device" == npu:* ]]; then
    printf '  note: alps on NPU runs a CPU eigendecomposition (very slow); expect long silent periods before `pruning layer ...` logs.\n'
  fi

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
        case "$pruning_algorithm" in
          flap)
            if [[ "$ENABLE_FLAP" != "true" ]]; then
              continue
            fi
            ;;
          sparsegpt)
            if [[ "$ENABLE_SPARSEGPT" != "true" ]]; then
              continue
            fi
            ;;
          wanda)
            if [[ "$ENABLE_WANDA" != "true" ]]; then
              continue
            fi
            ;;
          alps)
            if [[ "$ENABLE_ALPS" != "true" ]]; then
              continue
            fi
            ;;
        esac
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


run_algorithm_queue() {
  local pruning_algorithm="$1"
  local npu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
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
  local pruning_algorithm_from_config
  local sparsity_ratio
  local job_index=0

  for model_path in "${MODELS[@]}"; do
    for quant_config in "${FLATQUANT_CONFIGS[@]}"; do
      read -r weight_bits activation_bits query_bits key_bits value_bits quant_label <<< "$quant_config"
      for pruning_config in "${PRUNING_CONFIGS[@]}"; do
        read -r pruning_algorithm_from_config sparsity_ratio <<< "$pruning_config"
        if [[ "$pruning_algorithm_from_config" != "$pruning_algorithm" ]]; then
          continue
        fi
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

  printf '[worker-summary] %s worker=%s/%s npu=%s success=%s skip=%s fail=%s\n' \
    "$pruning_algorithm" \
    "$((worker_index + 1))" \
    "$worker_count" \
    "$npu_spec" \
    "$success_count" \
    "$skip_count" \
    "$failure_count"
  return "$failure_count"
}


launch_worker() {
  local pruning_algorithm="$1"
  local npu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local worker_label="${pruning_algorithm}_worker$((worker_index + 1))_of_${worker_count}"
  printf '[worker-start] %s on npu=%s\n' "$worker_label" "$npu_spec"
  run_algorithm_queue "$pruning_algorithm" "$npu_spec" "$worker_index" "$worker_count" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("$worker_label")
  WORKER_NPUS+=("$npu_spec")
}


launch_algorithm_workers() {
  local pruning_algorithm="$1"
  local npu_csv="$2"
  local -a npu_specs=()
  parse_npu_specs "$npu_csv" npu_specs
  if ((${#npu_specs[@]} == 0)); then
    printf '[worker-skip] %s has no npu configured\n' "$pruning_algorithm"
    return 0
  fi

  validate_npu_specs "$pruning_algorithm" npu_specs || return 1

  local worker_count="${#npu_specs[@]}"
  local index
  for index in "${!npu_specs[@]}"; do
    launch_worker "$pruning_algorithm" "${npu_specs[$index]}" "$index" "$worker_count"
  done
}


launch_workers() {
  local npu_csv="$1"
  local -a npu_specs=()
  parse_npu_specs "$npu_csv" npu_specs
  if ((${#npu_specs[@]} == 0)); then
    printf '[worker-skip] no npu configured\n'
    return 0
  fi

  validate_npu_specs "workflow" npu_specs || return 1

  local worker_count="${#npu_specs[@]}"
  local index
  for index in "${!npu_specs[@]}"; do
    local npu_spec="${npu_specs[$index]}"
    local worker_label="workflow_worker$((index + 1))_of_${worker_count}"
    printf '[worker-start] %s on npu=%s\n' "$worker_label" "$npu_spec"
    run_workflow_queue "$npu_spec" "$index" "$worker_count" &
    WORKER_PIDS+=("$!")
    WORKER_LABELS+=("$worker_label")
    WORKER_NPUS+=("$npu_spec")
  done
}


main() {
  local exit_code
  local had_failure=false

  preflight_checks
  if has_any_algorithm_npus_configured; then
    if [[ "$ENABLE_FLAP" == "true" ]]; then
      launch_algorithm_workers flap "$FLAP_NPUS"
    fi
    if [[ "$ENABLE_SPARSEGPT" == "true" ]]; then
      launch_algorithm_workers sparsegpt "$SPARSEGPT_NPUS"
    fi
    if [[ "$ENABLE_WANDA" == "true" ]]; then
      launch_algorithm_workers wanda "$WANDA_NPUS"
    fi
    if [[ "$ENABLE_ALPS" == "true" ]]; then
      launch_algorithm_workers alps "$ALPS_NPUS"
    fi
  else
    launch_workers "$WORKFLOW_NPUS"
  fi

  if ((${#WORKER_PIDS[@]} == 0)); then
    printf 'No workers enabled.\n'
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
