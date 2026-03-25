# Workflow Experiment Results

更新时间：2026-03-22

模型：`/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct`

baseline：

- `Qwen2.5-VL-7B-Instruct` PPL = `10.961624306471439`

## 8 组组合结果

### 先量化，再剪枝

| 组合 | PPL | 结果位置 |
|---|---:|---|
| `gptq -> sparsegpt` | `14.18987280840373` | `results/refactor_validation/workflow/.../quantization_then_pruning/gptq__sparsegpt/.../metrics.json` |
| `gptq -> wanda` | `13.883324080610485` | `results/workflow/.../quantization_then_pruning/gptq__wanda/.../metrics.json` |
| `awq -> sparsegpt` | `14.529758012554462` | `results/workflow/.../quantization_then_pruning/awq__sparsegpt/.../metrics.json` |
| `awq -> wanda` | `14.411407795565033` | `results/refactor_validation/workflow/.../quantization_then_pruning/awq__wanda/.../metrics.json` |

### 先剪枝，再量化

| 组合 | PPL | 结果位置 |
|---|---:|---|
| `sparsegpt -> gptq` | `13.684551389151803` | `results/workflow/.../pruning_then_quantization/gptq__sparsegpt/.../metrics.json` |
| `wanda -> gptq` | `14.05293132422766` | `results/workflow/.../pruning_then_quantization/gptq__wanda/.../metrics.json` |
| `sparsegpt -> awq` | `13.884618153735927` | `results/workflow/.../pruning_then_quantization/awq__sparsegpt/.../metrics.json` |
| `wanda -> awq` | `14.16503578635995` | `results/workflow/.../pruning_then_quantization/awq__wanda/.../metrics.json` |

## 当前判断

- 8 组都已真实跑通。
- 从当前结果看，`先剪枝，再量化` 整体略优于对应的 `先量化，再剪枝`。
- 当前最优组合是 `sparsegpt -> gptq`，PPL = `13.684551389151803`。
- 当前最差组合是 `awq -> sparsegpt`，PPL = `14.529758012554462`。
- 全部组合都显著高于 baseline，但仍处于同一量级，没有出现类似 `flatquant/flap` 那种失控爆炸。

## 理论显存收益

说明：

- 当前实验是 fake-quant + pseudo-prune，运行时仍然是 dense tensor，这里给的是“后续真正导出压缩格式后的理论权重显存”。
- `gptq/awq` 与 `sparsegpt/wanda` 当前都作用在 28 层 decoder 的 196 个线性层上。
- 这部分线性层共有 `6,525,288,448` 个权重，占整模型约 `78.69%`。
- 下表按保守口径估算：
  - int4 权重额外计入 group-wise `scale/zero` metadata
  - 稀疏率按 `50%`
  - 因为当前是非结构化稀疏，真正落地时还会受 sparse index / storage format 影响，所以组合收益应视为上界附近的理论值

### 被量化/剪枝的线性层权重

| 形态 | 理论显存 | 相对 fp16 baseline |
|---|---:|---:|
| fp16 dense | `12.1543 GiB` | `1.00x` |
| int4 only | `3.2285 GiB` | `3.76x` 压缩，`73.44%` 降低 |
| int4 + 50% sparse | `1.6142 GiB` | `7.53x` 压缩，`86.72%` 降低 |

### 整模型权重上界估算

| 形态 | 理论显存 | 相对整模型 fp16 baseline |
|---|---:|---:|
| fp16 dense | `15.4454 GiB` | `1.00x` |
| int4 only | `6.5196 GiB` | `2.37x` 压缩，`57.79%` 降低 |
| int4 + 50% sparse | `4.9053 GiB` | `3.15x` 压缩，`68.24%` 降低 |

### 组合相对单独量化的额外收益

- 在相同 `w4` 前提下，再叠加 `50%` 非结构化稀疏：
  - 对被处理线性层，理论上还能再降 `50%`
  - 对整模型权重，理论上还能再降约 `24.76%`

### 对 8 组组合的含义

- 这 8 组的理论最终显存收益是同一档的，因为最终配置都是：
  - `weight_bits=4`
  - `sparsity_ratio=0.5`
  - 同一批 decoder 线性层
- 区别主要体现在：
  - PPL 不同
  - 顺序不同导致的精度保持能力不同
- 不体现在：
  - 理论最终权重显存大小

## 汇总文件

- 组合总表：`results/workflow_combined_summary.json`
- 正式 workflow 状态快照：`results/workflow_status.json`
- 正式 workflow 监控日志：`results/logs/workflow/monitor.log`
