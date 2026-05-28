#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH must be set." >&2
  echo "Example: MODEL_PATH=/path/to/model GPU_ID=0 bash $0" >&2
  exit 2
fi

read_model_field() {
  local field="$1"
  python - "$MODEL_PATH" "$field" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
field = sys.argv[2]
config_path = model_path / "config.json"
if not config_path.exists():
    print("")
    raise SystemExit(0)
with config_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)
value = config.get(field)
if isinstance(value, list):
    print(" ".join(str(item) for item in value))
elif value is None:
    print("")
else:
    print(str(value))
PY
}

MODEL_TYPE="$(read_model_field model_type)"
ARCHITECTURES="$(read_model_field architectures)"
MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_NAME_LOWER="$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"

if [[ "$MODEL_TYPE" == "qwen3_5_moe" || "$MODEL_TYPE" == "qwen3_5_moe_text" ]]; then
  echo "ERROR: qwen3_5_moe models are not supported by compression_lora scripts yet: $MODEL_PATH" >&2
  exit 3
fi

if [[ "$MODEL_TYPE" == "qwen3_5" ]]; then
  TARGET_SCRIPT="$SCRIPT_DIR/qwen3_5/flatquant_lora_qwen3_5_gpu.sh"
elif [[ "$MODEL_TYPE" == "qwen2_vl" || "$MODEL_TYPE" == "qwen2_5_vl" || "$MODEL_TYPE" == "qwen3_vl" || "$MODEL_TYPE" == "minicpmv" || "$MODEL_TYPE" == "minicpm" ]]; then
  TARGET_SCRIPT="$SCRIPT_DIR/vlm/flatquant_lora_vlm_gpu.sh"
elif [[ "$ARCHITECTURES" == *"MiniCPMV"* || "$MODEL_NAME_LOWER" == *"minicpm"* ]]; then
  TARGET_SCRIPT="$SCRIPT_DIR/vlm/flatquant_lora_vlm_gpu.sh"
elif [[ "$MODEL_NAME_LOWER" == *"vl"* ]]; then
  TARGET_SCRIPT="$SCRIPT_DIR/vlm/flatquant_lora_vlm_gpu.sh"
else
  TARGET_SCRIPT="$SCRIPT_DIR/llm/flatquant_lora_llm_gpu.sh"
fi

echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] model_type=${MODEL_TYPE:-unknown} architectures=${ARCHITECTURES:-unknown}"
echo "[INFO] dispatching to $TARGET_SCRIPT"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

exec "$TARGET_SCRIPT" "$@"
