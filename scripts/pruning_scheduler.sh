#!/bin/bash
# 剪枝实验调度器 - 硬编码任务列表，自动分配空闲GPU对(01/23/45/67)
# 运行: nohup bash pruning_scheduler.sh > /tmp/scheduler.log 2>&1 &

M=/mnt/42_store/lcw/data2/Huawei/MindPipe/main.py
D=/mnt/42_store/lcw/data2/Huawei/datasets
R=/mnt/42_store/lcw/data2/Huawei/MindPipe/results
O="--device auto --device_map auto --dtype bfloat16 --attn_implementation flash_attention_2 --sequence_length 2048 --calibration_samples 128 --data_path $D"

# 硬编码任务列表: 名称|命令
TASKS=(
"4b_base|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/baseline"
"4b_flap_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning flap --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/flap_s0.2"
"4b_flap_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning flap --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/flap_s0.4"
"4b_flap_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning flap --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/flap_s0.5"
"4b_wandasp_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning wanda_sp --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/wanda_sp_s0.2"
"4b_wandasp_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning wanda_sp --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/wanda_sp_s0.4"
"4b_wandasp_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning wanda_sp --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/wanda_sp_s0.5"
"4b_llmpruner_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning llm_pruner --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/llm_pruner_s0.2"
"4b_llmpruner_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning llm_pruner --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/llm_pruner_s0.4"
"4b_llmpruner_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning llm_pruner --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/llm_pruner_s0.5"
"4b_shortgpt_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning shortgpt --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/shortgpt_s0.2"
"4b_shortgpt_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning shortgpt --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/shortgpt_s0.4"
"4b_shortgpt_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning shortgpt --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/shortgpt_s0.5"
"4b_wanda_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning wanda --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/wanda_s0.2"
"4b_wanda_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning wanda --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/wanda_s0.4"
"4b_wanda_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning wanda --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/wanda_s0.5"
"4b_sgpt_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning sparsegpt --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/sparsegpt_s0.2"
"4b_sgpt_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning sparsegpt --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/sparsegpt_s0.4"
"4b_sgpt_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning sparsegpt --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/sparsegpt_s0.5"
"4b_alps_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning alps --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/alps_s0.2"
"4b_alps_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning alps --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/alps_s0.4"
"4b_alps_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-4B --pruning alps --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-4B/alps_s0.5"
"9b_base|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/baseline"
"9b_flap_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning flap --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/flap_s0.2"
"9b_flap_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning flap --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/flap_s0.4"
"9b_flap_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning flap --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/flap_s0.5"
"9b_wandasp_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning wanda_sp --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/wanda_sp_s0.2"
"9b_wandasp_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning wanda_sp --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/wanda_sp_s0.4"
"9b_wandasp_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning wanda_sp --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/wanda_sp_s0.5"
"9b_llmpruner_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning llm_pruner --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/llm_pruner_s0.2"
"9b_llmpruner_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning llm_pruner --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/llm_pruner_s0.4"
"9b_llmpruner_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning llm_pruner --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/llm_pruner_s0.5"
"9b_shortgpt_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning shortgpt --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/shortgpt_s0.2"
"9b_shortgpt_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning shortgpt --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/shortgpt_s0.4"
"9b_shortgpt_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning shortgpt --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/shortgpt_s0.5"
"9b_wanda_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning wanda --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/wanda_s0.2"
"9b_wanda_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning wanda --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/wanda_s0.4"
"9b_wanda_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning wanda --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/wanda_s0.5"
"9b_sgpt_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning sparsegpt --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/sparsegpt_s0.2"
"9b_sgpt_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning sparsegpt --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/sparsegpt_s0.4"
"9b_sgpt_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning sparsegpt --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/sparsegpt_s0.5"
"9b_alps_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning alps --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/alps_s0.2"
"9b_alps_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning alps --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/alps_s0.4"
"9b_alps_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.5-9B --pruning alps --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.5-9B/alps_s0.5"
"27b_base|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/baseline"
"27b_flap_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning flap --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/flap_s0.2"
"27b_flap_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning flap --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/flap_s0.4"
"27b_flap_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning flap --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/flap_s0.5"
"27b_wandasp_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning wanda_sp --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/wanda_sp_s0.2"
"27b_wandasp_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning wanda_sp --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/wanda_sp_s0.4"
"27b_wandasp_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning wanda_sp --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/wanda_sp_s0.5"
"27b_llmpruner_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning llm_pruner --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/llm_pruner_s0.2"
"27b_llmpruner_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning llm_pruner --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/llm_pruner_s0.4"
"27b_llmpruner_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning llm_pruner --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/llm_pruner_s0.5"
"27b_shortgpt_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning shortgpt --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/shortgpt_s0.2"
"27b_shortgpt_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning shortgpt --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/shortgpt_s0.4"
"27b_shortgpt_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning shortgpt --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/shortgpt_s0.5"
"27b_wanda_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning wanda --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/wanda_s0.2"
"27b_wanda_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning wanda --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/wanda_s0.4"
"27b_wanda_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning wanda --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/wanda_s0.5"
"27b_sgpt_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning sparsegpt --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/sparsegpt_s0.2"
"27b_sgpt_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning sparsegpt --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/sparsegpt_s0.4"
"27b_sgpt_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning sparsegpt --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/sparsegpt_s0.5"
"27b_alps_02|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning alps --sparsity_ratio 0.2 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/alps_s0.2"
"27b_alps_04|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning alps --sparsity_ratio 0.4 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/alps_s0.4"
"27b_alps_05|python $M --model_path /mnt/82_store/LLM-weights/Qwen3.6-27B --pruning alps --sparsity_ratio 0.5 --eval_ppl true --eval_zero_shot true $O --output_dir $R/Qwen3.6-27B/alps_s0.5"
)

# GPU卡对
PAIRS=("0,1" "2,3" "4,5" "6,7")

# 检测GPU对是否空闲（显存占用<500MiB）
gpu_idle() {
  local g1=${1%,*} g2=${1#*,}
  local m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g1 2>/dev/null | cut -d. -f1)
  local m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g2 2>/dev/null | cut -d. -f1)
  [ "${m1:-9999}" -lt 1000 ] && [ "${m2:-9999}" -lt 1000 ]
}

IDX=0
TOTAL=${#TASKS[@]}
echo "=== 调度器启动 $(date) 共 $TOTAL 个任务 ==="

while [ $IDX -lt $TOTAL ]; do
  for pair in "${PAIRS[@]}"; do
    [ $IDX -ge $TOTAL ] && break
    if gpu_idle "$pair"; then
      IFS='|' read -r name cmd <<< "${TASKS[$IDX]}"
      echo "[$(date '+%H:%M:%S')] 启动: $name → GPU $pair ($(($IDX+1))/$TOTAL)"
      ( CUDA_VISIBLE_DEVICES=$pair conda run -n mindpipe --live-stream $cmd > /tmp/sched_${name}.log 2>&1; echo "[$(date '+%H:%M:%S')] $([ $? -eq 0 ] && echo 完成 || echo 失败): $name"; ) &
      ((IDX++))
      sleep 10
    fi
  done
  sleep 30
done

echo "=== 全部派发完成，等待剩余任务... ==="
wait
echo "=== 全部完成 $(date) ==="
