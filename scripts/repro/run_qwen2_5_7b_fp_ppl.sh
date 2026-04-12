#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/42_store/lcw/miniconda3/envs/mindpipe/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bfloat16}"
EVALUATION_DATASET="${EVALUATION_DATASET:-wikitext2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
MAX_EVAL_CHUNKS="${MAX_EVAL_CHUNKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DATA_PATH="${DATA_PATH:-/mnt/42_store/lcw/data2/Huawei/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/dev/shm/zy_tmp/mindpipe_fp_qwen2_5_7b_eval4_seq2048_repro}"
EXPECTED_TRANSFORMERS="${EXPECTED_TRANSFORMERS:-5.5.2}"
DRY_RUN="${DRY_RUN:-false}"

export PYTHONNOUSERSITE=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "/mnt/82_store/huggingface/datasets/Qwen/Qwen2.5-7B-Instruct" \
    "/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct"
  do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] Qwen2.5-7B model path not found."
  echo "Set MODEL_PATH manually, e.g.:"
  echo "  MODEL_PATH=/path/to/Qwen2.5-7B-Instruct"
  exit 1
fi

TF_VERSION="$("$PYTHON_BIN" - <<'PY'
import transformers
print(transformers.__version__)
PY
)"

if [[ "$TF_VERSION" != "$EXPECTED_TRANSFORMERS" ]]; then
  echo "[ERROR] transformers version mismatch: current=$TF_VERSION expected=$EXPECTED_TRANSFORMERS"
  echo "Please run:"
  echo "  PYTHONNOUSERSITE=1 $PYTHON_BIN -m pip install -U transformers==$EXPECTED_TRANSFORMERS"
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/main.py"
  --model_path "$MODEL_PATH"
  --device cuda:0
  --dtype "$DTYPE"
  --seed "$SEED"
  --data_path "$DATA_PATH"
  --evaluation_dataset "$EVALUATION_DATASET"
  --sequence_length "$SEQUENCE_LENGTH"
  --batch_size "$BATCH_SIZE"
  --max_eval_chunks "$MAX_EVAL_CHUNKS"
  --eval_ppl true
  --eval_zero_shot false
  --eval_vlm false
  --output_dir "$OUTPUT_DIR"
  --log_level "$LOG_LEVEL"
)

printf '[INFO] Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, command not executed."
  exit 0
fi

"${CMD[@]}" | tee "$OUTPUT_DIR/run.log"

METRICS_PATH="$(find "$OUTPUT_DIR" -type f -name metrics.json | head -n 1 || true)"
if [[ -z "$METRICS_PATH" || ! -f "$METRICS_PATH" ]]; then
  echo "[ERROR] metrics.json not found under $OUTPUT_DIR"
  exit 3
fi

echo "[INFO] Metrics file: $METRICS_PATH"
"$PYTHON_BIN" - "$METRICS_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

ppl = metrics.get("perplexity")
dataset = metrics.get("evaluation_dataset")
seq_len = metrics.get("sequence_length")
chunks = metrics.get("evaluated_chunks")
dtype = metrics.get("dtype")
device = metrics.get("device")

print(
    f"[RESULT] perplexity={ppl:.12f} dataset={dataset} "
    f"sequence_length={seq_len} evaluated_chunks={chunks} device={device} dtype={dtype}"
)
PY
