#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: configure the environment below, then run this script with no arguments.' \
    'Required: MODEL_FP16, MODEL_W8A8, SEMANTIC_SUMMARY, VLLM_PYTHONPATH' \
    'Optional: PYTHON, OUT_ROOT, ASCEND_ENV, ROUND_RETRIES, RETRY_DELAY_SECONDS'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# != 0 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON="${PYTHON:-python3}"
MODEL_FP16="${MODEL_FP16:?Set MODEL_FP16 to the Qwen3 MoE FP16 checkpoint}"
MODEL_W8A8="${MODEL_W8A8:?Set MODEL_W8A8 to the exported Qwen3 MoE W8A8 checkpoint}"
OUT_ROOT="${OUT_ROOT:-$ROOT/my_results/qwen3_strict_alternating_c32_20260723}"
SEMANTIC_SUMMARY="${SEMANTIC_SUMMARY:?Set SEMANTIC_SUMMARY to a passing request-parallel gate summary}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
VLLM_PYTHONPATH="${VLLM_PYTHONPATH:-}"
ROUND_RETRIES="${ROUND_RETRIES:-1}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"

[[ -d "$MODEL_FP16" ]] || { echo "MODEL_FP16 is not a directory: $MODEL_FP16" >&2; exit 2; }
[[ -d "$MODEL_W8A8" ]] || { echo "MODEL_W8A8 is not a directory: $MODEL_W8A8" >&2; exit 2; }
[[ -f "$SEMANTIC_SUMMARY" ]] || { echo "SEMANTIC_SUMMARY is not a file: $SEMANTIC_SUMMARY" >&2; exit 2; }
[[ -f "$ASCEND_ENV" ]] || { echo "ASCEND_ENV is not a file: $ASCEND_ENV" >&2; exit 2; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
[[ "$ROUND_RETRIES" =~ ^[0-9]+$ ]] || { echo "ROUND_RETRIES must be a non-negative integer" >&2; exit 2; }
[[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]] || { echo "RETRY_DELAY_SECONDS must be a non-negative integer" >&2; exit 2; }

[[ -n "$VLLM_PYTHONPATH" ]] || {
  echo "Set VLLM_PYTHONPATH to the vLLM and vLLM-Ascend source directories" >&2
  exit 2
}
IFS=':' read -r -a runtime_paths <<< "$VLLM_PYTHONPATH"
(( ${#runtime_paths[@]} == 2 )) || {
  echo "VLLM_PYTHONPATH must contain exactly two source directories" >&2
  exit 2
}
for runtime_path in "${runtime_paths[@]}"; do
  [[ -n "$runtime_path" && -d "$runtime_path" ]] || {
    echo "VLLM_PYTHONPATH entry is not a directory: $runtime_path" >&2
    exit 2
  }
done
export PYTHONPATH="$VLLM_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

# Source the selected toolkit once and fail before launching a multi-device run.
source "$ASCEND_ENV"
"$PYTHON" -c 'import vllm; import vllm_ascend'
COMMON=(
  --python "$PYTHON"
  --served_model_name qwen3-30b-a3b
  --device 0,1
  --host 127.0.0.1
  --input_len 2048
  --output_len 16
  --num_prompts 64
  --warmup_num_prompts 64
  --warmup_max_concurrency 32
  --request_rate inf
  --skip_initial_test
  --seed 20260712
  --max_concurrency 32
  --max_model_len 2304
  --max_num_batched_tokens 65536
  --max_num_seqs 32
  --gpu_memory_utilization 0.8
  --tensor_parallel_size 2
  --enable_expert_parallel
  --additional_config '{"torchair_graph_config":{"enabled":false},"ascend_scheduler_config":{"enabled":true},"refresh":true}'
  --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}'
  --disable_prefix_caching
  --disable_chunked_prefill
  --aiv
  --startup_timeout 900
  --rerun
)
for runtime_path in "${runtime_paths[@]}"; do
  COMMON+=(--runtime_pythonpath "$runtime_path")
done

run_one() {
  local mode="$1"
  local pair="$2"
  local port="$3"
  local model output_dir tag
  local -a env_args
  if [[ "$mode" == fp16 ]]; then
    model="$MODEL_FP16"
    env_args=(
      --env HCCL_OP_EXPANSION_MODE=AIV
      --env MINDPIPE_ENGINE_IDLE_COALESCE_US=30000
      --env MINDPIPE_ENGINE_IDLE_COALESCE_TARGET_ADDS=31
      --env MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL=0
      --env MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH=0
      --env VLLM_ASCEND_ENABLE_FLASHCOMM=0
      --env VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP=0
      --env VLLM_ASCEND_ENABLE_PREFETCH_MLP=0
      --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0
      --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE_W8A8=0
      --env MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS=0
    )
  else
    model="$MODEL_W8A8"
    env_args=(
      --env HCCL_OP_EXPANSION_MODE=AIV
      --env MINDPIPE_ENGINE_IDLE_COALESCE_US=30000
      --env MINDPIPE_ENGINE_IDLE_COALESCE_TARGET_ADDS=31
      --env MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL=1
      --env MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH=1
      --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT=1
      --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_TARGETS=qkv
      --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_MAX_TOKENS=0
      --env MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS=1
      --env MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES=0
      --env MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS=1
      --env MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE=0
      --env MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING=0
      --env MINDPIPE_QWEN3_MOE_GLOBAL_ROUTING_QUANT=0
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_PEER_REDUCE_SCATTER=0
      --env MINDPIPE_QWEN3_MOE_GMM2_TUNING=0
      --env VLLM_ASCEND_ENABLE_FLASHCOMM=0
      --env VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP=0
      --env VLLM_ASCEND_ENABLE_PREFETCH_MLP=0
      --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0
      --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE_W8A8=0
    )
  fi
  output_dir="$OUT_ROOT/pair${pair}_${mode}"
  tag="qwen3_30b_a3b_strict_alternating_c32_pair${pair}_${mode}_20260723"
  local attempt=0 status
  while (( attempt <= ROUND_RETRIES )); do
    if "$PYTHON" "$ROOT/scripts/repro/benchmark_vllm_online_serving.py" \
      "${COMMON[@]}" --mode "$mode" --model "$model" \
      --output_dir "$output_dir" --tag "$tag" --port "$port" \
      "${env_args[@]}"; then
      return 0
    else
      status=$?
    fi
    if (( attempt == ROUND_RETRIES )); then
      echo "Round pair${pair}_${mode} failed after $((attempt + 1)) attempt(s)." >&2
      return "$status"
    fi
    attempt=$((attempt + 1))
    echo "Round pair${pair}_${mode} failed with status $status; retrying in ${RETRY_DELAY_SECONDS}s (attempt $((attempt + 1))/$((ROUND_RETRIES + 1)))." >&2
    sleep "$RETRY_DELAY_SECONDS"
  done
}

mkdir -p "$OUT_ROOT"
run_one fp16 1 19071
run_one w8a8 1 19072
run_one w8a8 2 19073
run_one fp16 2 19074
run_one fp16 3 19075
run_one w8a8 3 19076
"$PYTHON" "$ROOT/scripts/repro/summarize_qwen3_strict_alternating_c32_20260723.py" \
  --root "$OUT_ROOT" --output "$OUT_ROOT/RESULT.json" \
  --semantic-summary "$SEMANTIC_SUMMARY"
