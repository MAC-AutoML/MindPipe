#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=_four_model_acceleration_common.sh
source "$SCRIPT_DIR/_four_model_acceleration_common.sh"

FP16_MODEL=${FP16_MODEL:?Set FP16_MODEL to the Qwen2.5-VL-7B FP16 model directory}
W8A8_MODEL=${W8A8_MODEL:?Set W8A8_MODEL to the Qwen2.5-VL-7B W8A8 model directory}
IMAGE_DIR=${IMAGE_DIR:?Set IMAGE_DIR to the fixed benchmark image directory}
BRIDGE=${BRIDGE:-$MINDPIPE_ROOT/acceleration/custom_ops/qwen2_5_vl/aclnn_grouped_swiglu_out_bridge.cpp}
OUT=${OUT:-$MINDPIPE_ROOT/results/repro/qwen25_vl_7b_1p5x}
DEVICE=${DEVICE:-0}
PORT_BASE=${PORT_BASE:-18950}
TOOLCHAIN=${TOOLCHAIN:?Set TOOLCHAIN to the GCC 11 cross-toolchain bin directory}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-32}
BENCH="$SCRIPT_DIR/benchmark_vllm_vl_online_serving.py"

for path in "$FP16_MODEL/config.json" "$W8A8_MODEL/config.json" "$BRIDGE" "$BENCH"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done
test -d "$IMAGE_DIR"
prepare_output "$OUT"
wait_for_idle_npus

ASCEND_RT_VISIBLE_DEVICES="$DEVICE" \
PYTHONPATH="$RUNTIME_PYTHONPATH" LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" \
MINDPIPE_ACLNN_GCC_TOOLCHAIN="$TOOLCHAIN" \
"$PYTHON" "$SCRIPT_DIR/verify_qwen25_vl_w8a8_loop_out_hybrid.py" \
  --device 0 --small-tokens 68 --large-tokens 161415 --min-tokens 32768 \
  --source "$BRIDGE" --vllm-root "$VLLM_ROOT" \
  --vllm-ascend-root "$VLLM_ASCEND_ROOT" \
  --output "$OUT/operator_correctness.json"
wait_for_idle_npus

run_one() {
  local role=$1 pair=$2 port=$3 mode model tag
  if [[ "$role" == candidate ]]; then
    mode=w8a8
    model=$W8A8_MODEL
  else
    mode=fp16
    model=$FP16_MODEL
  fi
  tag="qwen25_vl_7b_pair${pair}_${role}"
  "$PYTHON" "$BENCH" \
    --python "$PYTHON" --mode "$mode" --model "$model" \
    --served_model_name qwen25-vl --output_dir "$OUT" --tag "$tag" \
    --device "$DEVICE" --port "$port" --api_server_count 4 \
    --image_dir "$IMAGE_DIR" --images_per_prompt 1 \
    --question "请根据图片内容完成结构化理解任务，并用简短中文回答。" \
    --num_prompts 68 --warmup_num_prompts 68 --max_concurrency 68 \
    --dispatch_wave_size 0 --max_tokens "$OUTPUT_TOKENS" \
    --min_tokens "$OUTPUT_TOKENS" \
    --text_repetitions 88 --max_model_len 3104 \
    --max_num_batched_tokens 211072 --max_num_seqs 68 \
    --gpu_memory_utilization 0.796 --tensor_parallel_size 1 \
    --limit_mm_per_prompt '{"image":1}' --disable_chunked_prefill \
    --disable_prefix_caching --generation_config vllm \
    --repetition_penalty 1.0 --seed 0 \
    --compilation_config '{"level":3,"cudagraph_capture_sizes":[136,128,120,112,104,96,88,80,72,68,64,56,48,40,32,24,16,8,4,2,1]}' \
    --env "PYTHONPATH=$RUNTIME_PYTHONPATH" \
    --env "LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 --env PYTHONNOUSERSITE=1 \
    --env USE_OPTIMIZED_MODEL=1 --env VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=0 \
    --env VLLM_ASCEND_ENABLE_PREFETCH_MLP=0 \
    --env VLLM_ASCEND_ENABLE_FLASHCOMM=0 \
    --env VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=0 \
    --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 \
    --env MINDPIPE_QWEN2_MLP_FUSED_W8A8=0 \
    --env MINDPIPE_QWEN2_MLP_FUSED_ALLREDUCE_W8A8=0 \
    --env MINDPIPE_QWEN2_ATTN_FUSED_ALLREDUCE_W8A8=0 \
    --env MINDPIPE_QWEN2_ATTN_COMM_QUANT=0 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_W8A8=$([[ "$role" == candidate ]] && echo 1 || echo 0) \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_CHUNKS=4 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_BATCHED=1 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_UNROLL4=0 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_MODE=prefill \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_PREFILL_MIN_TOKENS=256 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_REPLICATION=repeat \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT=$([[ "$role" == candidate ]] && echo 1 || echo 0) \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_MIN_TOKENS=32768 \
    --env "MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_SOURCE=$BRIDGE" \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_TRACE_PATH= \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_TRACE_LIMIT=0 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_DISABLE_ACLGRAPH=0 \
    --env "MINDPIPE_ACLNN_GCC_TOOLCHAIN=$TOOLCHAIN" \
    --env TASK_QUEUE_ENABLE=1 \
    --env MINDPIPE_QWEN2_LM_HEAD_CHUNKED_W8A8=0 \
    --env MINDPIPE_QWEN2_VISION_MLP_BIAS_FUSED_W8A8=0 \
    --env MINDPIPE_QWEN2_VISION_PROJ_FUSED_ALLREDUCE_W8A8=0 \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT=0 \
    --env MINDPIPE_QWEN2_VISION_RUNTIME_HEAD_PADDING=0 \
    --env MINDPIPE_ENGINE_IDLE_COALESCE_MS=0 \
    --startup_timeout 1800 --request_timeout 900
  wait_for_idle_npus
}

run_one candidate 1 "$PORT_BASE"
run_one control 1 "$((PORT_BASE + 1))"
run_one control 2 "$((PORT_BASE + 2))"
run_one candidate 2 "$((PORT_BASE + 3))"
run_one candidate 3 "$((PORT_BASE + 4))"
run_one control 3 "$((PORT_BASE + 5))"

"$PYTHON" "$SCRIPT_DIR/summarize_paired_speedup.py" \
  --model Qwen2.5-VL-7B \
  --control "$OUT/qwen25_vl_7b_pair1_control_fp16_summary.json" \
  --control "$OUT/qwen25_vl_7b_pair2_control_fp16_summary.json" \
  --control "$OUT/qwen25_vl_7b_pair3_control_fp16_summary.json" \
  --candidate "$OUT/qwen25_vl_7b_pair1_candidate_w8a8_summary.json" \
  --candidate "$OUT/qwen25_vl_7b_pair2_candidate_w8a8_summary.json" \
  --candidate "$OUT/qwen25_vl_7b_pair3_candidate_w8a8_summary.json" \
  --minimum-speedup 1.5 --output "$OUT/RESULT.json"
