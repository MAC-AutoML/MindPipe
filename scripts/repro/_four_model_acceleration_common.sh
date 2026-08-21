#!/usr/bin/env bash

set -euo pipefail

MINDPIPE_ROOT=${MINDPIPE_ROOT:-$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")}
PYTHON=${PYTHON:-$(command -v python3)}
VLLM_ROOT=${VLLM_ROOT:?Set VLLM_ROOT to a compatible vLLM source tree}
VLLM_ASCEND_ROOT=${VLLM_ASCEND_ROOT:?Set VLLM_ASCEND_ROOT to a compatible vLLM-Ascend source tree}
PYTHON_HOME=${PYTHON_HOME:-$(dirname "$(dirname "$PYTHON")")}
ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}

test -x "$PYTHON"
test -d "$VLLM_ROOT/vllm"
test -d "$VLLM_ASCEND_ROOT/vllm_ascend"
test -f "$ASCEND_ENV"

# shellcheck disable=SC1090
source "$ASCEND_ENV"

RUNTIME_PYTHONPATH="$VLLM_ROOT:$VLLM_ASCEND_ROOT:$MINDPIPE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUNTIME_LD_LIBRARY_PATH="$VLLM_ASCEND_ROOT/vllm_ascend:$VLLM_ASCEND_ROOT/vllm_ascend/lib64:$PYTHON_HOME/lib/python3.11/site-packages/torch/lib:$PYTHON_HOME/lib/python3.11/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}"

"$PYTHON" "$MINDPIPE_ROOT/acceleration/verify_runtime.py" \
  --vllm-root "$VLLM_ROOT" \
  --vllm-ascend-root "$VLLM_ASCEND_ROOT" >/dev/null

wait_for_idle_npus() {
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if (( $(npu-smi info | rg -c 'No running processes found in NPU' || true) >= 4 )); then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for all four NPUs to become idle" >&2
  return 1
}

prepare_output() {
  local output=$1
  if [[ -d "$output" ]] && find "$output" -mindepth 1 -print -quit | rg -q .; then
    echo "Refusing to reuse non-empty output directory: $output" >&2
    return 2
  fi
  mkdir -p "$output"
}

reject_replicated_qwen3_environment() {
  local name
  for name in \
    MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS \
    MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES \
    MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS \
    MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE; do
    if [[ -n "${!name:-}" && "${!name}" != 0 ]]; then
      echo "Refusing duplicated-parameter Qwen3 setting: $name=${!name}" >&2
      return 2
    fi
  done
}
