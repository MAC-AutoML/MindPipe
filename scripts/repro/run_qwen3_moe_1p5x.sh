#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=_four_model_acceleration_common.sh
source "$SCRIPT_DIR/_four_model_acceleration_common.sh"

reject_replicated_qwen3_environment
FP16_MODEL=${FP16_MODEL:?Set FP16_MODEL to the Qwen3-MoE FP16 model directory}
W8A8_MODEL=${W8A8_MODEL:?Set W8A8_MODEL to the Qwen3-MoE W8A8 model directory}
REQUESTS=${REQUESTS:-$MINDPIPE_ROOT/acceleration/configs/qwen3_64x2048x16_c32.jsonl}
OUT=${OUT:-$MINDPIPE_ROOT/results/repro/qwen3_moe_1p5x}
PORT_BASE=${PORT_BASE:-19701}
BENCH="$SCRIPT_DIR/benchmark_vllm_acceleration_serving.py"

for path in "$FP16_MODEL/config.json" "$W8A8_MODEL/config.json" "$REQUESTS" "$BENCH"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done
prepare_output "$OUT"

run_one() {
  local role=$1 pair=$2 port=$3 mode model tag enabled
  if [[ "$role" == candidate ]]; then
    mode=w8a8; model=$W8A8_MODEL; enabled=1
  else
    mode=fp16; model=$FP16_MODEL; enabled=0
  fi
  tag="qwen3_moe_pair${pair}_${role}"
  mkdir -p "$OUT/$tag"
  wait_for_idle_npus
  "$PYTHON" "$BENCH" \
    --python "$PYTHON" --mode "$mode" --model "$model" \
    --served_model_name qwen3-30b-a3b --device 0,1 --port "$port" \
    --output_dir "$OUT/$tag" --tag "$tag" --dtype float16 \
    --input_len 2048 --output_len 16 --num_prompts 64 \
    --warmup_num_prompts 64 --warmup_max_concurrency 32 \
    --request_rate inf --request-file "$REQUESTS" --request-timeout 1800 \
    --seed 20260712 --max_concurrency 32 --max_model_len 2304 \
    --max_num_batched_tokens 65536 --max_num_seqs 32 \
    --gpu_memory_utilization 0.8 --tensor_parallel_size 2 \
    --enable_expert_parallel --disable_prefix_caching \
    --disable_chunked_prefill --aiv --startup_timeout 1800 \
    --additional_config '{"torchair_graph_config":{"enabled":false},"ascend_scheduler_config":{"enabled":true},"refresh":true}' \
    --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}' \
    --quality_prompt "The capital of France is" \
    --quality_prompt "Complete exactly: 2 + 2 =" --quality_max_tokens 8 \
    --env "PYTHONPATH=$RUNTIME_PYTHONPATH" \
    --env "LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 --env PYTHONNOUSERSITE=1 \
    --env HCCL_OP_EXPANSION_MODE=AIV \
    --env VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP=0 \
    --env VLLM_ASCEND_ENABLE_PREFETCH_MLP=0 \
    --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 \
    --env VLLM_ASCEND_ENABLE_W8A8_MATMUL_ALLREDUCE=0 \
    --env MINDPIPE_QWEN3_ATTN_QUANTIZED_TP2_ALLREDUCE=0 \
    --env MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING=0 \
    --env MINDPIPE_QWEN3_MOE_PREQUANT_MULTICAST=0 \
    --env MINDPIPE_STAGE_TIMING_PATH= \
    --env "MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE=$enabled" \
    --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE_MIN_TOKENS=8192 \
    --env "VLLM_ASCEND_ENABLE_FLASHCOMM=$enabled" \
    --env "VLLM_DISABLE_COMPILE_CACHE=$enabled" \
    --env "MINDPIPE_QWEN3_SP_FAST_ROPE=$enabled" \
    --env MINDPIPE_QWEN3_SP_SPARSE_LOGITS=0 \
    --env "MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER=$enabled"
  wait_for_idle_npus
}

run_one control 1 "$PORT_BASE"
run_one candidate 1 "$((PORT_BASE + 1))"
run_one candidate 2 "$((PORT_BASE + 2))"
run_one control 2 "$((PORT_BASE + 3))"
run_one control 3 "$((PORT_BASE + 4))"
run_one candidate 3 "$((PORT_BASE + 5))"

"$PYTHON" "$SCRIPT_DIR/summarize_paired_speedup.py" \
  --model Qwen3-30B-A3B-MoE \
  --control "$OUT/qwen3_moe_pair1_control/qwen3_moe_pair1_control_fp16_summary.json" \
  --control "$OUT/qwen3_moe_pair2_control/qwen3_moe_pair2_control_fp16_summary.json" \
  --control "$OUT/qwen3_moe_pair3_control/qwen3_moe_pair3_control_fp16_summary.json" \
  --candidate "$OUT/qwen3_moe_pair1_candidate/qwen3_moe_pair1_candidate_w8a8_summary.json" \
  --candidate "$OUT/qwen3_moe_pair2_candidate/qwen3_moe_pair2_candidate_w8a8_summary.json" \
  --candidate "$OUT/qwen3_moe_pair3_candidate/qwen3_moe_pair3_candidate_w8a8_summary.json" \
  --minimum-speedup 1.5 --output "$OUT/RESULT.json"
