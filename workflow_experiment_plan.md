# Workflow Experiment Plan

更新时间：2026-03-22

## 目标

- 在 `Qwen2.5-VL-7B-Instruct` 上真实完成 workflow 全矩阵实验。
- 覆盖两类顺序：
  - 先量化，再剪枝
  - 先剪枝，再量化
- 覆盖五类量化方法：
  - `gptq`
  - `awq`
  - `quarot`
  - `spinquant`
  - `flatquant`
- 覆盖三类剪枝方法：
  - `sparsegpt`
  - `wanda`
  - `flap`

## 基础配置

- 模型：`/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct`
- 评测集：`wikitext2`
- 序列长度：`512`
- `batch_size=1`
- `max_eval_chunks=64`
- dtype：`float16`
- 量化校准：
  - 数据集：`pileval`
  - 样本数：`4`
  - `weight_bits=4`
  - `group_size=128`
- 剪枝校准：
  - 数据集：`c4`
  - 样本数：`128`

## 方法级配置

- `gptq`
  - `activation_bits=16`
  - `weight_method=gptq`
- `awq`
  - `activation_bits=16`
  - `awq_search=false`
- `quarot`
  - `activation_bits=16`
  - `rotation_mode=hadamard`
  - `weight_method=gptq`
- `spinquant`
  - `activation_bits=16`
  - `rotation_mode=hadamard`
  - `weight_method=gptq`
- `flatquant`
  - `activation_bits=4`
  - `query_bits=16`
  - `key_bits=16`
  - `value_bits=16`
  - 该配置来自当前已验证的 `w4a4_q16k16v16` 稳定路径
- `sparsegpt`
  - `sparsity_ratio=0.5`
  - `structure_pattern=unstructured`
- `wanda`
  - `sparsity_ratio=0.5`
  - `structure_pattern=unstructured`
- `flap`
  - `sparsity_ratio=0.2`
  - `flap_metrics=WIFV`
  - `flap_remove_heads=8`
  - `pseudo_pruning=true`
  - 之所以不用 `0.5`，是因为当前单方法实测在 VL 模型上 PPL 明显失控

## 实验规模

- 组合数量：`5 * 3 * 2 = 30`
- 其中已完成并验证的旧子集：`gptq/awq x sparsegpt/wanda = 8`
- 本轮新增目标：`22` 组

## 执行方式

- 使用 `scripts/run_workflow_queue.py` 自动探测空闲 GPU。
- 本轮默认 GPU 池：`1,2,3,4,7`
- 默认最大并发：`2`
- 仅在显存和利用率都低于阈值时起新实验，避免挤占他人任务。
- `run_workflow_queue.py` 已从写死 `2x2` 改成：
  - 可指定量化算法列表
  - 可指定剪枝算法列表
  - 支持方法级参数覆盖

## 当前状态

- `workflow/builder.py` 已补齐：
  - `flatquant_*`
  - `flap_*`
  - `pseudo_pruning`
  - `query_bits/key_bits/value_bits`
- 批量队列已启动全矩阵补全任务。
- 状态快照文件：
  - `results/workflow_status.json`
- 监控日志：
  - `results/logs/workflow/monitor.log`

## 正式启动命令

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/scripts/run_workflow_queue.py \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/workflow \
  --log_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/logs/workflow \
  --gpu_pool 1,2,3,4,7 \
  --max_parallel 2
```

## 结果汇总命令

```bash
python3 /mnt/42_store/lcw/data2/Huawei/mindpipe/scripts/summarize_workflow_results.py \
  --results_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/workflow \
  --results_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/refactor_validation/workflow \
  --output_path /mnt/42_store/lcw/data2/Huawei/mindpipe/results/workflow_combined_summary.json
```
