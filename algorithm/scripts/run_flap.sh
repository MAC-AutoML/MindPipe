#!/bin/bash

# 获取脚本所在目录的上级目录（algorithm根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_VISIBLE_DEVICES=2 python "$ROOT_DIR/main.py" \
    --task pruning \
    --algorithm flap \
    --prune_method flap \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --pruning_ratio 0.2 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 128 \
    --save_model "$ROOT_DIR/output/llama2-7b-flap-0.2" \
    --eval \
    --generate \
    --prompts "What is artificial intelligence?" \
    --max_new_tokens 128 \
    --save_generation "$ROOT_DIR/output/llama2-7b-flap-0.2/generation_results.txt"
