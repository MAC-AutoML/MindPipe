#!/bin/bash

# 获取脚本所在目录的上级目录（algorithm根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_VISIBLE_DEVICES=1 python "$ROOT_DIR/main.py" \
    --task pruning \
    --algorithm wanda \
    --prune_method wanda \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --nsamples 128 \
    --save_model "$ROOT_DIR/output/llama2-7b-wanda-0.5" \
    --save "$ROOT_DIR/output/llama2-7b-wanda-0.5" \
    --generate \
    --prompts "What is artificial intelligence?" \
    --max_new_tokens 128 \
    --save_generation "$ROOT_DIR/output/llama2-7b-wanda-0.5/generation_results.txt"
