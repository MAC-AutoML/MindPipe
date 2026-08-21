#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=_four_model_acceleration_common.sh
source "$SCRIPT_DIR/_four_model_acceleration_common.sh"

BF16_MODEL=${BF16_MODEL:?Set BF16_MODEL to the Mixtral BF16 model directory}
W8A8_MODEL=${W8A8_MODEL:?Set W8A8_MODEL to the Mixtral W8A8 model directory}
REQUESTS=${REQUESTS:-$MINDPIPE_ROOT/acceleration/configs/mixtral_64x2048x16_c64.jsonl}
OUT=${OUT:-$MINDPIPE_ROOT/results/repro/mixtral_1p5x}
PORT_BASE=${PORT_BASE:-19210}
BENCH="$SCRIPT_DIR/benchmark_vllm_acceleration_serving.py"

for path in "$BF16_MODEL/config.json" "$W8A8_MODEL/config.json" "$REQUESTS" "$BENCH"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done
prepare_output "$OUT"

run_one() {
  local role=$1 index=$2 port=$3 mode model tag enabled
  if [[ "$role" == candidate ]]; then
    mode=w8a8; model=$W8A8_MODEL; enabled=1
  else
    mode=fp16; model=$BF16_MODEL; enabled=0
  fi
  tag="mixtral_${role}${index}"
  mkdir -p "$OUT/$tag"
  wait_for_idle_npus
  "$PYTHON" "$BENCH" \
    --python "$PYTHON" --mode "$mode" --dtype bfloat16 --model "$model" \
    --served_model_name mixtral --device 0,1,2,3 --port "$port" \
    --output_dir "$OUT/$tag" --tag "$tag" --input_len 2048 \
    --output_len 16 --num_prompts 64 --warmup_num_prompts 64 \
    --warmup_max_concurrency 64 --request-file "$REQUESTS" \
    --request-timeout 1800 --request_rate inf --seed 20260804 \
    --max_concurrency 64 --max_model_len 2304 \
    --max_num_batched_tokens 65536 --max_num_seqs 64 \
    --gpu_memory_utilization 0.8 --tensor_parallel_size 4 \
    --disable_prefix_caching --disable_chunked_prefill --enforce_eager \
    --startup_timeout 1800 \
    --env "PYTHONPATH=$RUNTIME_PYTHONPATH" \
    --env "LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 --env PYTHONNOUSERSITE=1 \
    --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 \
    --env VLLM_ASCEND_ENABLE_W8A8_MATMUL_ALLREDUCE=1 \
    --env VLLM_ASCEND_ENABLE_FLASHCOMM=0 \
    --env VLLM_ASCEND_MOE_PREFILL_COMM_METHOD= \
    --env MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE_PENDING_DEPTH=2 \
    --env MINDPIPE_MIXTRAL_TP4_ROUTING_METADATA_MODE=2 \
    --env MINDPIPE_ENGINE_IDLE_COALESCE_MS=20 \
    --env MINDPIPE_STAGE_TIMING_PATH= \
    --env MINDPIPE_PROFILE_TRACE=0 \
    --env "MINDPIPE_MIXTRAL_MOE_PREQUANT_ROUTING=$enabled" \
    --env "MINDPIPE_W8A8_FIA_FASTPATH=$enabled" \
    --env "MINDPIPE_MIXTRAL_ATTN_COMM_QUANT=$enabled" \
    --env "MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT=$enabled" \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_MAX_TOKENS=0 \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_MIN_TOKENS=0 \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_TARGETS=qkv \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_TRACE_PATH= \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_TRACE_LIMIT=0 \
    --env "MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE=$enabled" \
    --env MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE_MIN_TOKENS=16384 \
    --env MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE_TILE_COUNT_OVERRIDE= \
    --env MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE_TRACE_PATH= \
    --env MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE_TRACE_LIMIT=0
  wait_for_idle_npus
}

# Five pairs in the same alternating order as the accepted campaign.
run_one control 1 "$PORT_BASE"
run_one candidate 1 "$((PORT_BASE + 1))"
run_one candidate 2 "$((PORT_BASE + 2))"
run_one control 2 "$((PORT_BASE + 3))"
run_one control 3 "$((PORT_BASE + 4))"
run_one candidate 3 "$((PORT_BASE + 5))"
run_one candidate 4 "$((PORT_BASE + 6))"
run_one control 4 "$((PORT_BASE + 7))"
run_one control 5 "$((PORT_BASE + 8))"
run_one candidate 5 "$((PORT_BASE + 9))"

"$PYTHON" "$SCRIPT_DIR/summarize_paired_speedup.py" \
  --model Mixtral-8x7B \
  --control "$OUT/mixtral_control1/mixtral_control1_fp16_summary.json" \
  --control "$OUT/mixtral_control2/mixtral_control2_fp16_summary.json" \
  --control "$OUT/mixtral_control3/mixtral_control3_fp16_summary.json" \
  --control "$OUT/mixtral_control4/mixtral_control4_fp16_summary.json" \
  --control "$OUT/mixtral_control5/mixtral_control5_fp16_summary.json" \
  --candidate "$OUT/mixtral_candidate1/mixtral_candidate1_w8a8_summary.json" \
  --candidate "$OUT/mixtral_candidate2/mixtral_candidate2_w8a8_summary.json" \
  --candidate "$OUT/mixtral_candidate3/mixtral_candidate3_w8a8_summary.json" \
  --candidate "$OUT/mixtral_candidate4/mixtral_candidate4_w8a8_summary.json" \
  --candidate "$OUT/mixtral_candidate5/mixtral_candidate5_w8a8_summary.json" \
  --minimum-speedup 1.5 --output "$OUT/RESULT.json"
