#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_ROOT="${CONFIG_ROOT:-$REPO_ROOT/configs}"
DEFAULT_LOCAL_CONFIG="$CONFIG_ROOT/common/local.npu.yaml"
if [[ ! -f "$DEFAULT_LOCAL_CONFIG" ]]; then
  DEFAULT_LOCAL_CONFIG="$CONFIG_ROOT/common/local.yaml"
fi
LOCAL_CONFIG="${LOCAL_CONFIG:-$DEFAULT_LOCAL_CONFIG}"

if [[ ! -f "$LOCAL_CONFIG" ]]; then
  echo "Missing local config: $LOCAL_CONFIG" >&2
  echo "Create it from $CONFIG_ROOT/common/local.example.yaml before running this script." >&2
  exit 1
fi

# Shared model list. These are config model names under configs/algorithms/*/models/.
MODELS=(
  "Llama-2-7b-hf"
  "Meta-Llama-3.1-8B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-VL-7B-Instruct"
  "MiniCPM-V"
  "Qwen3-VL-2B-Instruct"
  "Qwen3-0.6B"
  "Qwen3.5-4B"
)

# Per-algorithm model overrides. Leave empty to reuse MODELS.
AWQ_MODELS=()
GPTQ_MODELS=()
SMOOTHQUANT_MODELS=()
OMNIQUANT_MODELS=()
FLATQUANT_MODELS=()
SPLITQUANT_MODELS=()

# Per-algorithm recipe lists. Recipe names must exist in the corresponding model yaml.
AWQ_RECIPES=(
  "w2a16"
  "w3a16"
  "w4a16"
)
GPTQ_RECIPES=(
  "w2a16"
  "w3a16"
  "w4a16"
)
SMOOTHQUANT_RECIPES=(
  "w8a8"
)
OMNIQUANT_RECIPES=(
  "w3a16"
  "w4a16"
  "w6a6"
  "w8a8"
)
FLATQUANT_RECIPES=(
  "w3a16"
  "w8a8"
)
SPLITQUANT_RECIPES=(
  "w8a8"
)

# Optional global overrides applied to every run via repeated --set key=value.
EXTRA_SETS=(
  # "calibration_samples=64"
  # "eval_zero_shot=false"
  # "output_dir=/path/to/results_npu"
  # "data_path=/path/to/data"
)

# Optional algorithm-specific overrides.
AWQ_EXTRA_SETS=()
GPTQ_EXTRA_SETS=()
SMOOTHQUANT_EXTRA_SETS=()
OMNIQUANT_EXTRA_SETS=()
FLATQUANT_EXTRA_SETS=()
SPLITQUANT_EXTRA_SETS=()

# Worker scheduling.
ENABLE_AWQ="${ENABLE_AWQ:-false}"
ENABLE_GPTQ="${ENABLE_GPTQ:-false}"
ENABLE_SMOOTHQUANT="${ENABLE_SMOOTHQUANT:-false}"
ENABLE_OMNIQUANT="${ENABLE_OMNIQUANT:-false}"
ENABLE_FLATQUANT="${ENABLE_FLATQUANT:-false}"
ENABLE_SPLITQUANT="${ENABLE_SPLITQUANT:-false}"

AWQ_NPUS="${AWQ_NPUS:-0}"
GPTQ_NPUS="${GPTQ_NPUS:-1}"
SMOOTHQUANT_NPUS="${SMOOTHQUANT_NPUS:-2}"
OMNIQUANT_NPUS="${OMNIQUANT_NPUS:-3}"
FLATQUANT_NPUS="${FLATQUANT_NPUS:-4}"
SPLITQUANT_NPUS="${SPLITQUANT_NPUS:-5}"

FORCE_RERUN="${FORCE_RERUN:-false}"
DRY_RUN="${DRY_RUN:-false}"

LAST_RUN_STATUS=""
WORKER_PIDS=()
WORKER_LABELS=()
WORKER_NPUS=()


algorithm_var_name() {
  local algorithm="$1"
  local suffix="$2"
  printf '%s%s' "${algorithm^^}" "$suffix"
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


resolve_visible_devices() {
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    return 1
  fi
  if [[ "$npu_spec" == npu:* ]]; then
    printf '%s' "${npu_spec#npu:}"
    return 0
  fi
  printf '%s' "$npu_spec"
}


resolve_runtime_device() {
  local npu_spec="$1"
  if [[ "$npu_spec" == "cpu" ]]; then
    printf 'cpu'
  else
    printf 'npu:0'
  fi
}


resolve_algorithm_models() {
  local algorithm="$1"
  local -n out_ref="$2"
  local models_var
  models_var="$(algorithm_var_name "$algorithm" "_MODELS")"
  local -n models_ref="$models_var"
  out_ref=()
  if ((${#models_ref[@]} > 0)); then
    out_ref=("${models_ref[@]}")
    return 0
  fi
  out_ref=("${MODELS[@]}")
}


resolve_algorithm_recipes() {
  local algorithm="$1"
  local -n out_ref="$2"
  local recipes_var
  recipes_var="$(algorithm_var_name "$algorithm" "_RECIPES")"
  local -n recipes_ref="$recipes_var"
  out_ref=("${recipes_ref[@]}")
}


resolve_algorithm_override_sets() {
  local algorithm="$1"
  local -n out_ref="$2"
  local sets_var
  sets_var="$(algorithm_var_name "$algorithm" "_EXTRA_SETS")"
  local -n sets_ref="$sets_var"
  out_ref=("${EXTRA_SETS[@]}" "${sets_ref[@]}")
}


resolve_job_metadata() {
  local algorithm="$1"
  local model="$2"
  local recipe="$3"
  local runtime_device="$4"
  shift 4

  "$PYTHON_BIN" - "$REPO_ROOT" "$CONFIG_ROOT" "$LOCAL_CONFIG" "$algorithm" "$model" "$recipe" "$runtime_device" "$@" <<'PY'
import os
import sys
from pathlib import Path
from string import Template

import yaml

_repo_root, config_root, local_config, algorithm, model, recipe, _runtime_device, *extra_sets = sys.argv[1:]


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping yaml: {path}")
    return payload


def parse_override(entry):
    if "=" not in entry:
        raise ValueError(f"Invalid override: {entry!r}")
    key, raw_value = entry.split("=", 1)
    value = yaml.safe_load(raw_value)
    if key.strip() in {"data_path", "output_dir", "vlm_eval_kit_root", "model_path"} and isinstance(value, str):
        value = expand_path(value)
    return key.strip(), value


def model_slug(model_path):
    return Path(str(model_path).rstrip("/")).name


def format_alpha(alpha):
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def expand_path(value):
    env = dict(os.environ)
    env.setdefault("REPO_ROOT", _repo_root)
    return os.path.expanduser(Template(value).safe_substitute(env))


config_root_path = Path(config_root)
common_base_path = config_root_path / "common" / "base.yaml"
local_config_path = Path(local_config)
if not local_config_path.is_absolute():
    local_config_path = config_root_path / local_config_path
algorithm_base_path = config_root_path / "algorithms" / algorithm / "base.yaml"
model_config_path = config_root_path / "algorithms" / algorithm / "models" / f"{model}.yaml"

merged = {}
all_model_path_overrides = {}
for path in (common_base_path, local_config_path, algorithm_base_path):
    payload = load_yaml(path)
    defaults = payload.get("defaults") or {}
    if defaults:
        defaults = {
            key: (expand_path(value) if key in {"data_path", "output_dir", "vlm_eval_kit_root", "model_path"} and isinstance(value, str) else value)
            for key, value in defaults.items()
        }
        merged = deep_merge(merged, defaults)
    model_paths = payload.get("model_paths") or {}
    if not isinstance(model_paths, dict):
        raise ValueError(f"'model_paths' must be a mapping in {path}")
    if model_paths:
        all_model_path_overrides.update(
            {name: expand_path(value) if isinstance(value, str) else value for name, value in model_paths.items()}
        )

model_payload = load_yaml(model_config_path)
model_paths = model_payload.get("model_paths") or {}
if not isinstance(model_paths, dict):
    raise ValueError(f"'model_paths' must be a mapping in {model_config_path}")
if model_paths:
    all_model_path_overrides.update(
        {name: expand_path(value) if isinstance(value, str) else value for name, value in model_paths.items()}
    )
model_defaults = model_payload.get("defaults") or {}
if model_defaults:
    model_defaults = {
        key: (expand_path(value) if key in {"data_path", "output_dir", "vlm_eval_kit_root", "model_path"} and isinstance(value, str) else value)
        for key, value in model_defaults.items()
    }
    merged = deep_merge(merged, model_defaults)
recipes = model_payload.get("recipes") or {}
recipe_payload = recipes.get(recipe) or {}
merged = deep_merge(merged, recipe_payload)
model_meta = model_payload.get("model") or {}
model_path = all_model_path_overrides.get(model, model_meta["model_path"])
merged["model_path"] = expand_path(model_path) if isinstance(model_path, str) else model_path

for entry in extra_sets:
    key, value = parse_override(entry)
    merged[key] = value

group_size = int(merged.get("group_size", 128))
if merged.get("weight_group_size") is None:
    merged["weight_group_size"] = group_size
if merged.get("activation_group_size") is None:
    merged["activation_group_size"] = group_size
if merged.get("kv_group_size") is None:
    merged["kv_group_size"] = group_size

output_root = Path(merged["output_dir"])
weight_bits = int(merged["weight_bits"])
activation_bits = int(merged.get("activation_bits", 16))
query_bits = int(merged.get("query_bits", 16))
key_bits = int(merged.get("key_bits", 16))
value_bits = int(merged.get("value_bits", 16))
sequence_length = int(merged["sequence_length"])
model_name = model_slug(merged["model_path"])

if algorithm == "smoothquant":
    alpha = float(merged["smoothquant_alpha"])
    run_spec = f"{algorithm}_w{weight_bits}a{activation_bits}_seq{sequence_length}_alpha{format_alpha(alpha)}"
elif algorithm in {"flatquant", "splitquant"}:
    run_spec = (
        f"{algorithm}_w{weight_bits}a{activation_bits}"
        f"_q{query_bits}k{key_bits}v{value_bits}_seq{sequence_length}"
    )
elif algorithm == "omniquant":
    weight_group_size = int(merged["weight_group_size"])
    group_suffix = ""
    if weight_bits < 16 and weight_group_size > 0:
        group_suffix = f"g{weight_group_size}"
    run_spec = f"{algorithm}_w{weight_bits}a{activation_bits}{group_suffix}_seq{sequence_length}"
else:
    run_spec = f"{algorithm}_w{weight_bits}a{activation_bits}_seq{sequence_length}"

output_dir = output_root / model_name / algorithm / run_spec

print(str(output_dir))
print(str(output_dir / "metrics.json"))
print("true" if bool(merged.get("eval_ppl", False)) else "false")
print("true" if bool(merged.get("eval_zero_shot", False)) else "false")
print("true" if bool(merged.get("eval_vlm", False)) else "false")
PY
}


is_complete() {
  local metrics_path="$1"
  local require_ppl="$2"
  local require_zero_shot="$3"
  local require_vlm="$4"
  [[ -f "$metrics_path" ]] || return 1
  if [[ "$require_ppl" == "true" ]] && ! grep -q '"perplexity"' "$metrics_path"; then
    return 1
  fi
  if [[ "$require_zero_shot" == "true" ]] && ! grep -q '"zero_shot"' "$metrics_path"; then
    return 1
  fi
  if [[ "$require_vlm" == "true" ]] && ! grep -q '"vlm_eval"' "$metrics_path"; then
    return 1
  fi
  return 0
}


run_experiment() {
  local algorithm="$1"
  local model="$2"
  local recipe="$3"
  local npu_spec="$4"
  local runtime_device
  runtime_device="$(resolve_runtime_device "$npu_spec")"

  local -a override_sets=()
  resolve_algorithm_override_sets "$algorithm" override_sets

  local -a metadata=()
  mapfile -t metadata < <(resolve_job_metadata "$algorithm" "$model" "$recipe" "$runtime_device" "${override_sets[@]}")
  if ((${#metadata[@]} < 5)); then
    echo "Failed to resolve metadata for $algorithm/$model/$recipe" >&2
    LAST_RUN_STATUS="fail"
    return 1
  fi

  local output_dir="${metadata[0]}"
  local metrics_path="${metadata[1]}"
  local eval_ppl="${metadata[2]}"
  local eval_zero_shot="${metadata[3]}"
  local eval_vlm="${metadata[4]}"
  local run_id="${model}__${algorithm}__${recipe}"

  if [[ "$FORCE_RERUN" != "true" ]] && is_complete "$metrics_path" "$eval_ppl" "$eval_zero_shot" "$eval_vlm"; then
    printf '[skip][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
    LAST_RUN_STATUS="skip"
    return 0
  fi

  local -a cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/scripts/run_from_config.py"
    --config-root "$CONFIG_ROOT"
    --local-config "$LOCAL_CONFIG"
    --algorithm "$algorithm"
    --model "$model"
    --recipe "$recipe"
    --set "device=$runtime_device"
  )

  local override
  for override in "${override_sets[@]}"; do
    cmd+=(--set "$override")
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    cmd+=(--dry-run)
  fi

  local visible_devices=""
  if visible_devices="$(resolve_visible_devices "$npu_spec")"; then
    :
  else
    visible_devices=""
  fi

  printf '[run][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
  printf '  out: %s\n' "$metrics_path"
  printf '  output_dir: %s\n' "$output_dir"
  printf '  eval_ppl: %s\n' "$eval_ppl"
  printf '  eval_zero_shot: %s\n' "$eval_zero_shot"
  printf '  eval_vlm: %s\n' "$eval_vlm"
  printf '  cmd:'
  if [[ -n "$visible_devices" ]]; then
    printf ' %q' env ASCEND_RT_VISIBLE_DEVICES="$visible_devices" "${cmd[@]}"
  else
    printf ' %q' "${cmd[@]}"
  fi
  printf '\n'

  if [[ "$DRY_RUN" == "true" ]]; then
    LAST_RUN_STATUS="success"
    return 0
  fi

  local exit_code
  set +e
  if [[ -n "$visible_devices" ]]; then
    env ASCEND_RT_VISIBLE_DEVICES="$visible_devices" "${cmd[@]}" 2>&1 | sed -u "s/^/[${run_id}][npu=${npu_spec}] /"
  else
    "${cmd[@]}" 2>&1 | sed -u "s/^/[${run_id}][npu=${npu_spec}] /"
  fi
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ $exit_code -ne 0 ]]; then
    printf '[fail][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
    LAST_RUN_STATUS="fail"
    return 1
  fi

  printf '[ok][%s][npu=%s] %s\n' "$algorithm" "$npu_spec" "$run_id"
  LAST_RUN_STATUS="success"
  return 0
}


run_algorithm_queue() {
  local algorithm="$1"
  local npu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local failure_count=0
  local success_count=0
  local skip_count=0
  local job_index=0

  local -a models=()
  local -a recipes=()
  resolve_algorithm_models "$algorithm" models
  resolve_algorithm_recipes "$algorithm" recipes

  if ((${#models[@]} == 0)); then
    printf '[worker-skip] %s has no models configured\n' "$algorithm"
    return 0
  fi
  if ((${#recipes[@]} == 0)); then
    printf '[worker-skip] %s has no recipes configured\n' "$algorithm"
    return 0
  fi

  local model
  local recipe
  for model in "${models[@]}"; do
    for recipe in "${recipes[@]}"; do
      if (( job_index % worker_count == worker_index )); then
        if ! run_experiment "$algorithm" "$model" "$recipe" "$npu_spec"; then
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

  printf '[worker-summary] %s worker=%s/%s npu=%s success=%s skip=%s fail=%s\n' \
    "$algorithm" \
    "$((worker_index + 1))" \
    "$worker_count" \
    "$npu_spec" \
    "$success_count" \
    "$skip_count" \
    "$failure_count"
  return "$failure_count"
}


launch_worker() {
  local algorithm="$1"
  local npu_spec="$2"
  local worker_index="$3"
  local worker_count="$4"
  local worker_label="${algorithm}_worker$((worker_index + 1))_of_${worker_count}"
  printf '[worker-start] %s on npu=%s\n' "$worker_label" "$npu_spec"
  run_algorithm_queue "$algorithm" "$npu_spec" "$worker_index" "$worker_count" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("$worker_label")
  WORKER_NPUS+=("$npu_spec")
}


launch_algorithm_workers() {
  local algorithm="$1"
  local npu_csv="$2"
  local -a npu_specs=()
  parse_npu_specs "$npu_csv" npu_specs
  if ((${#npu_specs[@]} == 0)); then
    printf '[worker-skip] %s has no npu configured\n' "$algorithm"
    return 0
  fi

  local worker_count="${#npu_specs[@]}"
  local index
  for index in "${!npu_specs[@]}"; do
    launch_worker "$algorithm" "${npu_specs[$index]}" "$index" "$worker_count"
  done
}


main() {
  local exit_code
  local had_failure=false

  if [[ "$ENABLE_AWQ" == "true" ]]; then
    launch_algorithm_workers awq "$AWQ_NPUS"
  fi
  if [[ "$ENABLE_GPTQ" == "true" ]]; then
    launch_algorithm_workers gptq "$GPTQ_NPUS"
  fi
  if [[ "$ENABLE_SMOOTHQUANT" == "true" ]]; then
    launch_algorithm_workers smoothquant "$SMOOTHQUANT_NPUS"
  fi
  if [[ "$ENABLE_OMNIQUANT" == "true" ]]; then
    launch_algorithm_workers omniquant "$OMNIQUANT_NPUS"
  fi
  if [[ "$ENABLE_FLATQUANT" == "true" ]]; then
    launch_algorithm_workers flatquant "$FLATQUANT_NPUS"
  fi
  if [[ "$ENABLE_SPLITQUANT" == "true" ]]; then
    launch_algorithm_workers splitquant "$SPLITQUANT_NPUS"
  fi

  if ((${#WORKER_PIDS[@]} == 0)); then
    printf 'No algorithm workers enabled.\n'
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
# Maintenance touch for repository metadata refresh.
