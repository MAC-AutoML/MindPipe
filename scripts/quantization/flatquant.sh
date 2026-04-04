#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda:2}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-42}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALIBRATION_BATCH_SIZE="${FLATQUANT_CALIBRATION_BATCH_SIZE:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-5e-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/my_results/quantization}"

CMD=(
  python "$REPO_ROOT/main.py"
  --quantization flatquant
  --model_path "$MODEL_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --calibration_dataset wikitext2
  --evaluation_dataset wikitext2
  --calibration_samples "$CALIBRATION_SAMPLES"
  --sequence_length "$SEQUENCE_LENGTH"
  --flatquant_epochs "$FLATQUANT_EPOCHS"
  --flatquant_calibration_batch_size "$FLATQUANT_CALIBRATION_BATCH_SIZE"
  --flatquant_lr "$FLATQUANT_LR"
  --batch_size 1
  --max_eval_chunks 64
  --weight_bits 3
  --activation_bits 16
  --query_bits 16
  --key_bits 4
  --value_bits 4
  --kv_group_size 128
  --weight_method rtn
  --flatquant_cali_trans true
  --flatquant_add_diag true
  --flatquant_lwc true
  --flatquant_lac true
  --flatquant_direct_inv true
  --flatquant_deactive_amp true
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf_token "$HF_TOKEN")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
