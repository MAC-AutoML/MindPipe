#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=_four_model_acceleration_common.sh
source "$SCRIPT_DIR/_four_model_acceleration_common.sh"

FP16_MODEL=${FP16_MODEL:?Set FP16_MODEL to the Qwen2.5-VL-72B FP16 model directory}
W8A8_MODEL=${W8A8_MODEL:?Set W8A8_MODEL to the Qwen2.5-VL-72B W8A8 model directory}
IMAGE_DIR=${IMAGE_DIR:?Set IMAGE_DIR to the fixed benchmark image directory}
OUT=${OUT:-$MINDPIPE_ROOT/results/repro/qwen25_vl_72b_1p5x}
PORT_BASE=${PORT_BASE:-19050}
BENCH="$SCRIPT_DIR/benchmark_vllm_vl_online_serving.py"

for path in "$FP16_MODEL/config.json" "$W8A8_MODEL/config.json" "$BENCH"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done
test -d "$IMAGE_DIR"
prepare_output "$OUT"
wait_for_idle_npus

PYTHONPATH="$RUNTIME_PYTHONPATH" LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
"$PYTHON" "$SCRIPT_DIR/verify_qwen25_vl_72b_tp4_fused_allreduce.py" \
  --tokens 64 1687 --output "$OUT/operator_correctness.json"
wait_for_idle_npus

PYTHONPATH="$RUNTIME_PYTHONPATH" LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" \
ASCEND_RT_VISIBLE_DEVICES=0 \
"$PYTHON" "$SCRIPT_DIR/verify_qwen25_vl_fia_fastpath.py" \
  --device 0 --output "$OUT/fia_fastpath_correctness.json"
wait_for_idle_npus

run_one() {
  local role=$1 pair=$2 port=$3 mode model tag fused fia
  if [[ "$role" == candidate ]]; then
    mode=w8a8; model=$W8A8_MODEL; fused=1; fia=1
  else
    mode=fp16; model=$FP16_MODEL; fused=0; fia=0
  fi
  tag="qwen25_vl_72b_pair${pair}_${role}"
  "$PYTHON" "$BENCH" \
    --python "$PYTHON" --mode "$mode" --model "$model" \
    --served_model_name qwen25-vl --output_dir "$OUT" --tag "$tag" \
    --device 0,1,2,3 --port "$port" --api_server_count 1 \
    --image_dir "$IMAGE_DIR" --images_per_prompt 1 \
    --question "请根据图片内容完成结构化理解任务，并用简短中文回答。" \
    --num_prompts 37 --warmup_num_prompts 37 --max_concurrency 37 \
    --dispatch_wave_size 0 --max_tokens 16 --min_tokens 16 \
    --text_repetitions 48 --max_model_len 3104 \
    --max_num_batched_tokens 16384 --max_num_seqs 37 \
    --gpu_memory_utilization 0.72 --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 --limit_mm_per_prompt '{"image":1}' \
    --disable_chunked_prefill --disable_prefix_caching --enforce_eager \
    --generation_config vllm --repetition_penalty 1.0 --seed 0 \
    --env "PYTHONPATH=$RUNTIME_PYTHONPATH" \
    --env "LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 --env PYTHONNOUSERSITE=1 \
    --env VLLM_ASCEND_ENABLE_FLASHCOMM=0 \
    --env VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=0 \
    --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 \
    --env MINDPIPE_QWEN2_MLP_CHUNKED_W8A8=0 \
    --env MINDPIPE_QWEN2_MLP_FUSED_W8A8=0 \
    --env "MINDPIPE_QWEN2_MLP_FUSED_ALLREDUCE_W8A8=$fused" \
    --env "MINDPIPE_QWEN2_ATTN_FUSED_ALLREDUCE_W8A8=$fused" \
    --env MINDPIPE_QWEN2_ATTN_COMM_QUANT=0 \
    --env MINDPIPE_QWEN2_FUSED_ALLREDUCE_ACLGRAPH=0 \
    --env MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT=0 \
    --env "MINDPIPE_W8A8_FIA_FASTPATH=$fia" \
    --env MINDPIPE_STAGE_TIMING_PATH= \
    --startup_timeout 2400 --request_timeout 1200
  wait_for_idle_npus
}

run_one candidate 1 "$PORT_BASE"
run_one control 1 "$((PORT_BASE + 1))"
run_one control 2 "$((PORT_BASE + 2))"
run_one candidate 2 "$((PORT_BASE + 3))"
run_one candidate 3 "$((PORT_BASE + 4))"
run_one control 3 "$((PORT_BASE + 5))"

"$PYTHON" "$SCRIPT_DIR/summarize_paired_speedup.py" \
  --model Qwen2.5-VL-72B \
  --control "$OUT/qwen25_vl_72b_pair1_control_fp16_summary.json" \
  --control "$OUT/qwen25_vl_72b_pair2_control_fp16_summary.json" \
  --control "$OUT/qwen25_vl_72b_pair3_control_fp16_summary.json" \
  --candidate "$OUT/qwen25_vl_72b_pair1_candidate_w8a8_summary.json" \
  --candidate "$OUT/qwen25_vl_72b_pair2_candidate_w8a8_summary.json" \
  --candidate "$OUT/qwen25_vl_72b_pair3_candidate_w8a8_summary.json" \
  --minimum-speedup 1.5 --output "$OUT/RESULT.json"
