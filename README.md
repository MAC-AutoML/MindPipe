# mindpipe

更新时间：2026-03-23

## 当前目标

- 在 `mindpipe` 内重建统一算法框架。
- 按 `a-ref` 原始源码目录迁移算法，方法级目录单独落盘。
- 当前阶段只保留伪量化 / 伪剪枝 / PPL 评测链路，不引入真实导出与部署代码。
- 统一外部命名、参数和入口，内部实现尽量保留各开源仓库原逻辑。

## 当前目录设计

```text
mindpipe/
├── main.py
├── README.md
├── algorithm/
│   ├── common/
│   ├── quantization/
│   │   ├── ptq/
│   │   │   ├── awq/
│   │   │   ├── gptq/
│   │   │   ├── quarot/
│   │   │   └── spinquant/
│   │   └── qat/
│   │       └── flatquant/
│   └── pruning/
│       ├── structured/
│       │   ├── flap/
│       │   └── wanda_sp/
│       └── unstructured/
│           ├── sparsegpt/
│           └── wanda/
├── workflow/
│   ├── schema.py
│   ├── builder.py
│   └── executor.py
├── evaluation/
│   ├── ppl.py
│   ├── lm_eval.py
│   └── runner.py
├── scripts/
│   ├── run_workflow_queue.py
│   ├── rerun_results_lm_eval.py
│   ├── summarize_workflow_results.py
│   └── monitor_workflow_status.py
└── results/
```

## 当前设计原则

- `algorithm/quantization/` 和 `algorithm/pruning/` 只保留方法实现，不直接承载编排逻辑。
- `workflow/` 只负责 workflow config 组装与执行：
  - 单独量化 = 1 个 quantization stage
  - 单独剪枝 = 1 个 pruning stage
  - 量化+剪枝 / 剪枝+量化 = 2 个 stage
- `evaluation/` 单独负责 PPL 和 lm-eval 两类评测。
- 多模态 benchmark 评测说明见 [vlmevalkit_usage.md](vlmevalkit_usage.md)。
- 根目录 `main.py` 是唯一 CLI 入口。

## 已完成内容

- 新建 `algorithm/`、`workflow/`、`evaluation/` 三层结构，根目录只保留一个 `main.py` 入口。
- 公共基础能力保留在 `algorithm/common/`：
  - `common/modeling.py`
  - `common/datasets.py`
  - `common/io.py`
  - `common/logging.py`
  - `common/runtime.py`
- 评测逻辑拆分到 `evaluation/`：
  - `evaluation/ppl.py`
  - `evaluation/lm_eval.py`
  - `evaluation/runner.py`
- 已迁移源码到独立方法目录：
  - `llm-awq-main/awq`
  - `gptq-main`
  - `QuaRot-main/fake_quant` + `quarot`
  - `SpinQuant-main/utils` + `eval_utils` + `train_utils` + `ptq.py` + `optimize_rotation.py`
  - `FlatQuant-main/flatquant` + `gptq_utils.py`
  - `wanda-main/lib`
  - `sparsegpt-master`
  - `FLAP-main/lib` + `models`
  - `structured/wanda_sp/source/lib`（从 FLAP 耦合实现中独立拆出）
- 已补齐并真实跑通的统一入口：
  - `awq`
  - `quarot`
  - `spinquant`
  - `flatquant`
  - `gptq`
  - `flap`
  - `wanda`
  - `wanda_sp`
  - `sparsegpt`

## 当前支持矩阵

| 方法 | 源码已迁移 | 统一入口已挂接 | 两个目标模型已真实执行 | 说明 |
|---|---:|---:|---:|---|
| AWQ | 是 | 是 | 是 | fake-quant 稳定 |
| GPTQ | 是 | 是 | 是 | 已补 CPU Hessian offload + damping 重试 |
| QuaRot | 是 | 是 | 是 | Qwen/Qwen2.5-VL 的 weight-only runtime 路径已接通；`w16a16` sanity 与 `w4a16` 均可真实评测 |
| SpinQuant | 是 | 是 | 是 | 已移除 proxy 兼容层，源码内部直接识别统一文本主干；Qwen/Qwen2.5-VL 默认自动生成 `identity-R2` 本地 checkpoint 走源码 `optimized_rotation_path` 分支；`w16a16` sanity 与 `w4a16` 均已于 2026-03-22 复跑验证 |
| FlatQuant | 是 | 是 | 是 | 已补 Qwen/Qwen2.5-VL wrapper 与稳定性修复 |
| Wanda | 是 | 是 | 是 | Qwen2.5-VL 文本主干已适配 |
| Wanda-SP | 是 | 是 | 是 | 已从 FLAP 源码解耦为独立结构化剪枝方法；保留本地 C4、Qwen GQA、VL decoder-root 适配 |
| SparseGPT | 是 | 是 | 是 | 已补 CPU Hessian offload |
| FLAP | 是 | 是 | 是 | 已补本地数据、Qwen GQA、VL decoder-root 适配 |

## 已验证命令

### AWQ fake-quant

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm awq \
  --no-awq_search \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:1 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 8 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --group_size 128 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm awq \
  --no-awq_search \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:2 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 8 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --group_size 128 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

### GPTQ fake-quant

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm gptq \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:3 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --group_size 128 \
  --damp_percent 0.05 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

### FlatQuant fake-quant

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm flatquant \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:1 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --activation_bits 4 \
  --query_bits 16 \
  --key_bits 4 \
  --value_bits 4 \
  --group_size 128 \
  --kv_group_size 128 \
  --weight_method rtn \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm flatquant \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:2 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --activation_bits 4 \
  --query_bits 16 \
  --key_bits 4 \
  --value_bits 4 \
  --group_size 128 \
  --kv_group_size 128 \
  --weight_method rtn \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm gptq \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:2 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --group_size 128 \
  --damp_percent 0.05 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

### QuaRot fake-quant

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm quarot \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --activation_bits 16 \
  --key_bits 16 \
  --value_bits 16 \
  --group_size -1 \
  --weight_group_size -1 \
  --activation_group_size -1 \
  --kv_group_size -1 \
  --weight_symmetric \
  --weight_method gptq \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm quarot \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --activation_bits 16 \
  --key_bits 16 \
  --value_bits 16 \
  --group_size -1 \
  --weight_group_size -1 \
  --activation_group_size -1 \
  --kv_group_size -1 \
  --weight_symmetric \
  --weight_method gptq \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

### SpinQuant fake-quant

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm spinquant \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --activation_bits 16 \
  --key_bits 16 \
  --value_bits 16 \
  --group_size 128 \
  --weight_method gptq \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  quantization \
  --algorithm spinquant \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 4 \
  --activation_bits 16 \
  --key_bits 16 \
  --value_bits 16 \
  --group_size 128 \
  --weight_method gptq \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/quantization
```

### Wanda pruning

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm wanda \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:5 \
  --dtype float16 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.5 \
  --structure_pattern unstructured \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm wanda \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:6 \
  --dtype float16 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.5 \
  --structure_pattern unstructured \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

### Wanda-SP pruning

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm wanda_sp \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:2 \
  --dtype float16 \
  --evaluation_dataset wikitext2 \
  --calibration_samples 128 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.2 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm wanda_sp \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:3 \
  --dtype float16 \
  --evaluation_dataset wikitext2 \
  --calibration_samples 128 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.2 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  workflow \
  --quantization_algorithm quarot \
  --pruning_algorithm wanda_sp \
  --execution_order quantization_then_pruning \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:4 \
  --dtype float16 \
  --evaluation_dataset wikitext2 \
  --quantization_calibration_dataset pileval \
  --pruning_calibration_dataset c4 \
  --quantization_calibration_samples 4 \
  --pruning_calibration_samples 128 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --weight_bits 16 \
  --activation_bits 16 \
  --key_bits 16 \
  --value_bits 16 \
  --group_size -1 \
  --weight_group_size -1 \
  --activation_group_size -1 \
  --kv_group_size -1 \
  --weight_method gptq \
  --sparsity_ratio 0.2 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/workflow
```

### SparseGPT pruning

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm sparsegpt \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:5 \
  --dtype float16 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.5 \
  --structure_pattern unstructured \
  --block_size 64 \
  --damp_percent 0.05 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

### FLAP pruning

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm flap \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-7B-Instruct \
  --device cuda:3 \
  --dtype float16 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.5 \
  --structure_pattern AL-AM \
  --flap_metrics WIFV \
  --pseudo_pruning \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm flap \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:4 \
  --dtype float16 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.5 \
  --structure_pattern AL-AM \
  --flap_metrics WIFV \
  --pseudo_pruning \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

```bash
conda run -n mindpipe python /mnt/42_store/lcw/data2/Huawei/mindpipe/main.py \
  pruning \
  --algorithm sparsegpt \
  --model_path /mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct \
  --device cuda:4 \
  --dtype float16 \
  --calibration_samples 4 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --sparsity_ratio 0.5 \
  --structure_pattern unstructured \
  --block_size 64 \
  --damp_percent 0.05 \
  --output_root /mnt/42_store/lcw/data2/Huawei/mindpipe/results/pruning
```

## 当前 PPL 结果

统一设置：

- 数据集：`wikitext2`
- `sequence_length=512`
- `max_eval_chunks=64`
- `batch_size=1`
- dtype：`float16`

基线 PPL 由共享评测 helper 单独测得：

- `Qwen2.5-7B-Instruct`: `9.423129808858079`
- `Qwen2.5-VL-7B-Instruct`: `10.961624306471439`

### 量化

推荐 weight-only 配置：

| 模型 | baseline | AWQ w4a16 | GPTQ w4a16 | QuaRot w4a16 | SpinQuant w4a16 | FlatQuant w4a4q16k4v4 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 9.4231 | 10.0488 | 10.1735 | 11.3231 | 10.3052 | 34077.5997 |
| Qwen2.5-VL-7B-Instruct | 10.9616 | 11.7213 | 11.4062 | 12.1375 | 11.4655 | 781492.1380 |

旋转 sanity 配置：

| 模型 | QuaRot w16a16 | SpinQuant w16a16 |
|---|---:|---:|
| Qwen2.5-7B-Instruct | 9.4218 | 9.4214 |
| Qwen2.5-VL-7B-Instruct | 10.9588 | 10.9570 |

### 剪枝

| 模型 | baseline | Wanda s=0.5 | Wanda-SP s=0.2 | SparseGPT s=0.5 | FLAP s=0.5 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 9.4231 | 12.1248 | 21.5992 | 12.7185 | 5549.2092 |
| Qwen2.5-VL-7B-Instruct | 10.9616 | 13.9875 | 22.6815 | 14.1716 | 6172.1969 |

## 结果文件

### 量化

- `results/quantization/Qwen2.5-7B-Instruct/awq/awq_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a4_seq512/metrics.json`
- `results/quantization/Qwen2.5-7B-Instruct/gptq/gptq_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-7B-Instruct/quarot/quarot_w16a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-7B-Instruct/quarot/quarot_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w16a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/awq/awq_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w4a4_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/gptq/gptq_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/quarot/quarot_w16a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/quarot/quarot_w4a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w16a16_seq512/metrics.json`
- `results/quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w4a16_seq512/metrics.json`
- `results/summary.json`

### 剪枝

- `results/pruning/Qwen2.5-7B-Instruct/flap/flap_s0.5_seq512/metrics.json`
- `results/pruning/Qwen2.5-7B-Instruct/wanda/wanda_s0.5_seq512/metrics.json`
- `results/pruning/Qwen2.5-7B-Instruct/wanda_sp/wanda_sp_s0.2_seq512/metrics.json`
- `results/pruning/Qwen2.5-7B-Instruct/sparsegpt/sparsegpt_s0.5_seq512/metrics.json`
- `results/pruning/Qwen2.5-VL-7B-Instruct/flap/flap_s0.5_seq512/metrics.json`
- `results/pruning/Qwen2.5-VL-7B-Instruct/wanda/wanda_s0.5_seq512/metrics.json`
- `results/pruning/Qwen2.5-VL-7B-Instruct/wanda_sp/wanda_sp_s0.2_seq512/metrics.json`
- `results/pruning/Qwen2.5-VL-7B-Instruct/sparsegpt/sparsegpt_s0.5_seq512/metrics.json`
- `results/workflow/Qwen2.5-7B-Instruct/quantization_then_pruning/quarot__wanda_sp/quarot_w16a16__wanda_sp_s0.2_seq512/metrics.json`

## 当前剩余问题

- `AWQ` 当前稳定使用的是 `--no-awq_search` 路径，完整 search 仍然更重。
- `QuaRot` 当前稳定的是 weight-only runtime 路径；低比特 activation / KV 配置还没有做 Qwen/Qwen2.5-VL 的进一步收敛。
- `SpinQuant` 当前默认是 `identity-R2` fallback，保证 `w16a16` 等价和 `w4a16` 可评测；源码原始的 learned rotation 训练链路仍然是 Llama-oriented，尚未在统一入口里做 Qwen/Qwen2.5-VL 版本。
- `FlatQuant` 的当前结果虽然可真实复现，但数值明显异常，后续需要继续查训练/重参数化路径。

## 本轮补充的关键工程修复

- `Qwen2.5-VL` 单层前向的 `position_embeddings / cache_position` 兼容。
- `FlatQuant` 的 Qwen attention 改成 `float32` 累积，修掉最后一层 `q @ k^T` 溢出导致的 `NaN`。
- `FlatQuant` 的 Qwen/Qwen2.5-VL wrapper、decoder-root、reparameterize 和 RTN/GPTQ 路径已统一。
- `QuaRot` 取消了 Qwen weight-only runtime 分支的 `force_eager`，`w16a16` 与 `w4a16` 已恢复到可用区间。
- `SpinQuant` 取消了 Qwen 加载时的 `force_eager`，并为 Qwen/Qwen2.5-VL 默认自动生成 `identity-R2` checkpoint，修复了无 checkpoint 时随机 `R2` 导致的严重失真。
- `SparseGPT` 大 Hessian CPU offload，避免 `down_proj` 显存爆炸。
- `GPTQ` 大 Hessian CPU offload + Cholesky damping 重试。
- `FLAP` 改成本地优先数据加载，并补了 Qwen GQA、VL decoder-root、通用 layer kwargs 与无 bias 补偿。
- `Wanda-SP` 已从 `FLAP` 源码目录彻底拆出为独立方法目录，保留本地 C4 fallback、Qwen GQA 与 VL decoder-root 兼容，并在 `mindpipe` 中完成独立挂接。
- 公共 calibration 前向与方法内部 block 前向补 `torch.no_grad()`，消除隐性显存累积。
