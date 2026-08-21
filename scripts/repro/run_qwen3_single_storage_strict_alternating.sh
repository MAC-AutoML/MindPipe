#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=_four_model_acceleration_common.sh
source "$SCRIPT_DIR/_four_model_acceleration_common.sh"

reject_replicated_qwen3_environment
FP16_MODEL=${FP16_MODEL:?Set FP16_MODEL to the Qwen3-MoE FP16 model directory}
W8A8_MODEL=${W8A8_MODEL:?Set W8A8_MODEL to the Qwen3-MoE W8A8 model directory}
REQUESTS=${REQUESTS:-$MINDPIPE_ROOT/acceleration/configs/qwen3_64x2048x16_c32.jsonl}
OUT=${OUT:-$MINDPIPE_ROOT/results/repro/qwen3_single_storage_strict}
PORT_BASE=${PORT_BASE:-19701}
BENCH="$SCRIPT_DIR/benchmark_vllm_acceleration_serving.py"
INSPECTOR="$SCRIPT_DIR/inspect_qwen3_moe_w8a8_runtime.py"
SUMMARIZER="$SCRIPT_DIR/summarize_qwen3_single_storage_strict_alternating.py"
QKV_AUDITOR="$SCRIPT_DIR/audit_qwen3_quantized_qkv_profile.py"
MECHANISM_AUDITOR="$SCRIPT_DIR/audit_qwen3_single_storage_mechanisms.py"
STORAGE_AUDIT="$OUT/single_storage_audit.json"
PROFILE_CONTROL="$OUT/mechanism_profile_control"
PROFILE_CANDIDATE="$OUT/mechanism_profile_candidate"
QKV_AUDIT="$OUT/quantized_qkv_profile_audit.json"
MECHANISM_AUDIT="$OUT/natural_mechanism_audit.json"
REUSE_VALIDATED_AUDITS=${REUSE_VALIDATED_AUDITS:-0}

for path in \
  "$FP16_MODEL/config.json" "$W8A8_MODEL/config.json" "$REQUESTS" \
  "$BENCH" "$INSPECTOR" "$SUMMARIZER" "$QKV_AUDITOR" \
  "$MECHANISM_AUDITOR"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done
if [[ "$REUSE_VALIDATED_AUDITS" == 1 ]]; then
  mkdir -p "$OUT"
  for path in "$OUT"/pair{1,2,3}_{fp16,w8a8}; do
    if [[ -e "$path" ]]; then
      echo "Refusing audit reuse after formal runs have started: $path" >&2
      exit 2
    fi
  done
  "$PYTHON" - "$STORAGE_AUDIT" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("passed") is not True:
    raise SystemExit(f"missing or failed reusable storage audit: {path}")
PY
else
  prepare_output "$OUT"
fi

COMMON_ENV=(
  --env "PYTHONPATH=$RUNTIME_PYTHONPATH"
  --env "LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH"
  --env TORCH_DEVICE_BACKEND_AUTOLOAD=0
  --env PYTHONNOUSERSITE=1
  --env HCCL_OP_EXPANSION_MODE=AIV
  --env MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS=0
  --env MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES=0
  --env MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS=0
  --env MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE=0
  --env MINDPIPE_QWEN3_MOE_SINGLE_STORAGE_AUDIT=0
  --env VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP=0
  --env VLLM_ASCEND_ENABLE_PREFETCH_MLP=0
  --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0
  --env VLLM_ASCEND_ENABLE_W8A8_MATMUL_ALLREDUCE=0
  --env VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE_W8A8=0
  --env MINDPIPE_QWEN3_ATTN_QUANTIZED_TP2_ALLREDUCE=0
  --env MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING=0
  --env MINDPIPE_QWEN3_MOE_PREQUANT_MULTICAST=0
  --env MINDPIPE_QWEN3_MOE_MULTICAST_REDUCE_SCATTER=0
  --env MINDPIPE_QWEN3_MOE_MULTICAST_BATCHED_P2P=0
  --env MINDPIPE_STAGE_TIMING_PATH=
  --env MINDPIPE_STAGE_TIMING_SYNC=0
)

COMMON_ARGS=(
  --python "$PYTHON"
  --served_model_name qwen3-30b-a3b
  --device 0,1
  --dtype float16
  --input_len 2048
  --output_len 16
  --num_prompts 64
  --warmup_num_prompts 64
  --warmup_max_concurrency 32
  --request_rate inf
  --request-file "$REQUESTS"
  --request-timeout 1800
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
  --startup_timeout 1800
  --quality_prompt "The capital of France is"
  --quality_prompt "Complete exactly: 2 + 2 ="
  --quality_max_tokens 8
)

if [[ "$REUSE_VALIDATED_AUDITS" != 1 ]]; then
  wait_for_idle_npus
  PYTHONPATH="$RUNTIME_PYTHONPATH" \
  LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" \
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  PYTHONNOUSERSITE=1 \
  HCCL_OP_EXPANSION_MODE=AIV \
  MINDPIPE_QWEN3_MOE_SINGLE_STORAGE_AUDIT=1 \
  MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS=0 \
  MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES=0 \
  MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS=0 \
  MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE=0 \
  "$PYTHON" "$INSPECTOR" \
    --model "$W8A8_MODEL" --dtype float16 \
    --tensor-parallel-size 2 --max-model-len 512 \
    --gpu-memory-utilization 0.8 --output-json "$STORAGE_AUDIT"
  wait_for_idle_npus
fi

run_mechanism_profile() {
  local variant=$1 port=$2 output raw tag
  local -a mechanism_env
  output="$OUT/mechanism_profile_${variant}"
  raw="$output/raw"
  tag="qwen3_single_storage_mechanism_${variant}"
  if [[ "$variant" == control ]]; then
    mechanism_env=(
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE=0
      --env MINDPIPE_QWEN3_SP_FAST_ROPE=0
      --env MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER=0
    )
  else
    mechanism_env=(
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE=1
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE_MIN_TOKENS=8192
      --env MINDPIPE_QWEN3_SP_FAST_ROPE=1
      --env MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER=1
    )
  fi
  mkdir -p "$output"
  wait_for_idle_npus
  "$PYTHON" "$BENCH" \
    "${COMMON_ARGS[@]}" --mode w8a8 --model "$W8A8_MODEL" --port "$port" \
    --output_dir "$output" --tag "$tag" --profile \
    "${COMMON_ENV[@]}" \
    --env VLLM_ASCEND_ENABLE_FLASHCOMM=1 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env MINDPIPE_QWEN3_SP_SPARSE_LOGITS=0 \
    --env "VLLM_TORCH_PROFILER_DIR=$raw" \
    --env VLLM_TORCH_PROFILER_WITH_STACK=0 \
    --env VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0 \
    --env MINDPIPE_TORCH_PROFILER_LEVEL=1 \
    --env MINDPIPE_TORCH_PROFILER_MSTX=1 \
    --env MINDPIPE_PROFILE_TRACE=1 \
    "${mechanism_env[@]}"
  wait_for_idle_npus
}

analyse_profile() {
  local raw=$1
  "$PYTHON" - "$raw" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
databases = sorted(root.rglob("ascend_pytorch_profiler_*.db"))
if len(databases) != 2:
    from torch_npu.profiler.profiler import analyse
    profiles = sorted(path for path in root.rglob("*_ascend_pt") if path.is_dir())
    if len(profiles) != 2:
        raise SystemExit(
            f"expected two raw rank profiles under {root}, found {len(profiles)}"
        )
    for profile in profiles:
        analyse(str(profile))
    databases = sorted(root.rglob("ascend_pytorch_profiler_*.db"))
if len(databases) != 2:
    raise SystemExit(
        f"expected two profiler databases under {root}, found {len(databases)}"
    )
for database in databases:
    if database.stat().st_size == 0:
        raise SystemExit(f"empty profiler database: {database}")
PY
}

if [[ "$REUSE_VALIDATED_AUDITS" != 1 ]]; then
  run_mechanism_profile control "$((PORT_BASE + 10))"
  run_mechanism_profile candidate "$((PORT_BASE + 11))"
fi
analyse_profile "$PROFILE_CONTROL/raw"
analyse_profile "$PROFILE_CANDIDATE/raw"
"$PYTHON" "$QKV_AUDITOR" \
  --control-root "$PROFILE_CONTROL/raw" \
  --candidate-root "$PROFILE_CANDIDATE/raw" \
  --output-json "$QKV_AUDIT"
"$PYTHON" "$MECHANISM_AUDITOR" \
  --control-root "$PROFILE_CONTROL/raw" \
  --candidate-root "$PROFILE_CANDIDATE/raw" \
  --output-json "$MECHANISM_AUDIT"
wait_for_idle_npus

run_one() {
  local mode=$1 pair=$2 port=$3 sequence_index=$4 model tag output
  local -a mode_env
  if [[ "$mode" == fp16 ]]; then
    model=$FP16_MODEL
    mode_env=(
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE=0
      --env VLLM_ASCEND_ENABLE_FLASHCOMM=0
      --env VLLM_DISABLE_COMPILE_CACHE=0
      --env MINDPIPE_QWEN3_SP_FAST_ROPE=0
      --env MINDPIPE_QWEN3_SP_SPARSE_LOGITS=0
      --env MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER=0
    )
  else
    model=$W8A8_MODEL
    mode_env=(
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE=1
      --env MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE_MIN_TOKENS=8192
      --env VLLM_ASCEND_ENABLE_FLASHCOMM=1
      --env VLLM_DISABLE_COMPILE_CACHE=1
      --env MINDPIPE_QWEN3_SP_FAST_ROPE=1
      --env MINDPIPE_QWEN3_SP_SPARSE_LOGITS=0
      --env MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER=1
    )
  fi
  tag="qwen3_single_storage_pair${pair}_${mode}"
  output="$OUT/pair${pair}_${mode}"
  mkdir -p "$output"
  wait_for_idle_npus
  "$PYTHON" "$BENCH" \
    "${COMMON_ARGS[@]}" --mode "$mode" --model "$model" --port "$port" \
    --output_dir "$output" --tag "$tag" \
    "${COMMON_ENV[@]}" "${mode_env[@]}"
  "$PYTHON" - "$output/${tag}_${mode}_summary.json" "$sequence_index" <<'PY'
import json
import os
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
data["acceptance_sequence_index"] = int(sys.argv[2])
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
os.replace(temporary, path)
PY
  wait_for_idle_npus
}

run_one fp16 1 "$PORT_BASE" 0
run_one w8a8 1 "$((PORT_BASE + 1))" 1
run_one w8a8 2 "$((PORT_BASE + 2))" 2
run_one fp16 2 "$((PORT_BASE + 3))" 3
run_one fp16 3 "$((PORT_BASE + 4))" 4
run_one w8a8 3 "$((PORT_BASE + 5))" 5

"$PYTHON" "$SUMMARIZER" \
  --root "$OUT" --request-file "$REQUESTS" \
  --fp16-model "$FP16_MODEL" --w8a8-model "$W8A8_MODEL" \
  --storage-audit "$STORAGE_AUDIT" \
  --qkv-profile-audit "$QKV_AUDIT" \
  --mechanism-audit "$MECHANISM_AUDIT" \
  --output "$OUT/RESULT.json"
