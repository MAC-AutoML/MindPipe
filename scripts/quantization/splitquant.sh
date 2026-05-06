#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Edit this block for the experiment you want to run.
PYTHON_BIN="python"
GPU_ID="0"
MODEL="Qwen3.5-4B"
RECIPE="w8a8"
DRY_RUN="false"

CONFIG_ROOT="$REPO_ROOT/configs"
DEFAULT_LOCAL_CONFIG="$CONFIG_ROOT/common/local.gpu.yaml"
if [[ ! -f "$DEFAULT_LOCAL_CONFIG" ]]; then
  DEFAULT_LOCAL_CONFIG="$CONFIG_ROOT/common/local.yaml"
fi
LOCAL_CONFIG="${LOCAL_CONFIG:-$DEFAULT_LOCAL_CONFIG}"

# Keep machine-specific paths in configs/common/local.gpu.yaml.
# You can also override LOCAL_CONFIG explicitly if needed.
if [[ ! -f "$LOCAL_CONFIG" ]]; then
  echo "Missing local config: $LOCAL_CONFIG" >&2
  echo "Create it from $CONFIG_ROOT/common/local.example.yaml before running this script." >&2
  exit 1
fi

# Optional per-run overrides passed as repeated `--set key=value`.
EXTRA_SETS=(
  # "calibration_samples=64"
  # "eval_zero_shot=false"
  # "eval_vlm=true"
)

# Optional extra raw CLI args for `scripts/run_from_config.py`.
EXTRA_ARGS=(
  # "--dry-run"
)

CMD=(
  "$PYTHON_BIN"
  "$REPO_ROOT/scripts/run_from_config.py"
  --config-root "$CONFIG_ROOT"
  --local-config "$LOCAL_CONFIG"
  --algorithm splitquant
  --model "$MODEL"
  --recipe "$RECIPE"
  --set "device=cuda:0"
)

if [[ "$DRY_RUN" == "true" ]]; then
  CMD+=(--dry-run)
fi

for override in "${EXTRA_SETS[@]}"; do
  CMD+=(--set "$override")
done

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running:'
printf ' %q' env CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}"
printf '\n'

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}"
# Replace legacy quantization shell scripts with a config-based GPU/NPU runner covering AWQ, GPTQ, FlatQuant, SplitQuant, SmoothQuant, and OmniQuant.
