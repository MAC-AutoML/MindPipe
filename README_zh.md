<div align="center">

# 🧠 MindPipe

**大语言模型与视觉语言模型的统一压缩与评测框架**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-orange)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![NPU Ready](https://img.shields.io/badge/NPU-已支持-purple)]()

[English](README.md) | [中文](README_zh.md)

<p align="center">
  <em>一个入口，11 种量化方法，7 种剪枝方法，GPU & NPU 双后端，文本与视觉全覆盖。</em>
</p>

---

**量化** · **剪枝** · **评测** · **复现**

</div>

## ✨ 为什么选择 MindPipe？

> 大多数压缩工具只能处理单一技术、单一模型类型。  
> **MindPipe 将所有方法统一在一个可复现的流水线中。**

<table>
<tr>
<td width="50%">

### 🎯 一个入口，所有方法
单一 `main.py` 驱动量化、剪枝、组合压缩和评测 — 无需在多个脚本间切换。

### 🔀 GPU + NPU 双后端
同时支持 CUDA GPU 和昇腾 NPU，通过统一的设备抽象层管理。

### 📊 集成评测体系
PPL、lm-eval-harness zero-shot 和 VLMEvalKit 多模态评测 — 全部内置。

</td>
<td width="50%">

### 🧩 模块化 & 可扩展
基于注册表的清晰架构，新增算法简单直接。

### 🔬 可复现优先
JSON 产物、批量脚本和逐次运行 metrics，确保每个结果可追溯。

### 👁️ 视觉语言原生支持
VLM 不是附加功能 — 而是一等公民，配备专用的多模态评测通道。

</td>
</tr>
</table>

---

## 🚀 快速开始

```bash
# 1. 环境准备
conda activate mindpipe
git submodule update --init --recursive
pip install -r requirements.txt

# 2. 量化模型 (AWQ W4A16)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --quantization awq \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --calibration_dataset pileval \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --weight_bits 4 \
  --group_size 128 \
  --eval_ppl true \
  --output_dir ./results/awq

# 3. 剪枝模型 (Wanda 50% 稀疏)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --pruning wanda \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --calibration_dataset c4 \
  --calibration_samples 128 \
  --sparsity_ratio 0.5 \
  --eval_ppl true \
  --output_dir ./results/wanda
```

<details>
<summary><b>📋 更多示例（点击展开）</b></summary>

### 全精度评测

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --attn_implementation sdpa \
  --evaluation_dataset wikitext2 \
  --sequence_length 2048 \
  --batch_size 1 \
  --max_eval_chunks 64 \
  --eval_ppl true \
  --eval_zero_shot true \
  --zero_shot_tasks boolq piqa rte winogrande arc_easy arc_challenge openbookqa \
  --zero_shot_num_fewshot 0 \
  --zero_shot_batch_size 1 \
  --output_dir ./results/fp_eval
```

### GPTQ 量化

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --quantization gptq \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --attn_implementation sdpa \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --weight_bits 4 \
  --activation_bits 16 \
  --group_size 128 \
  --weight_group_size 128 \
  --eval_ppl true \
  --output_dir ./results/gptq
```

### 剪枝 + 量化组合流水线

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --pruning wanda_sp \
  --quantization gptq \
  --execution_order pruning_then_quantization \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --attn_implementation sdpa \
  --calibration_dataset c4 \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --sparsity_ratio 0.2 \
  --weight_bits 4 \
  --group_size 128 \
  --eval_ppl true \
  --output_dir ./results/workflow
```

### VLM 多模态评测

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_path /path/to/vlm \
  --device_map auto \
  --dtype float16 \
  --attn_implementation sdpa \
  --eval_ppl false \
  --eval_zero_shot false \
  --eval_vlm true \
  --vlm_datasets OCRBench TextVQA_VAL ChartQA_TEST InfoVQA_VAL \
  --vlm_mode all \
  --vlm_api_nproc 1 \
  --vlm_eval_kit_root /path/to/VLMEvalKit \
  --output_dir ./results/vlm_eval
```

</details>

---

## 📦 支持算法

请使用 CLI 列中的方法名作为命令行参数值。QA-LoRA、LLM-Pruner、Wanda-SP
等是展示名；实际 CLI 值分别是 `qalora`、`llm_pruner`、`wanda_sp`。

### 量化（11 种方法）

| 方法 | CLI | 类型 | 技术特点 | NPU |
|:----:|:---:|:----:|:--------:|:---:|
| **AWQ** | `awq` | PTQ | Activation-aware 权重量化 | ✅ |
| **GPTQ** | `gptq` | PTQ | GPTQ 权重量化 | ✅ |
| **MQuant** | `mquant` | PTQ | 多模态 GPTQ/AWQ 量化（语言+视觉分支） | ⏳ |
| **OmniQuant** | `omniquant` | PTQ | 可学习的权重与激活变换 | ✅ |
| **QuaRot** | `quarot` | PTQ | 基于旋转的 W/A/KV 量化 | ⏳ |
| **SmoothQuant** | `smoothquant` | PTQ | 面向 W/A 量化的激活平滑 | ✅ |
| **SpinQuant** | `spinquant` | PTQ | 基于 SpinQuant hook 的旋转量化 | ⏳ |
| **FlatQuant** | `flatquant` | QAT | 可训练变换 | ✅ |
| **QLoRA** | `qlora` | QAT | 低比特 fake-quant adapter 训练 | ✅ |
| **QA-LoRA** | `qalora` | QAT | Group-pooled adapter 训练 | 🔶 |
| **SplitQuant** | `splitquant` | QAT | SplitQuant 风格可训练变换 | ✅ |

### 剪枝（7 种方法）

| 方法 | CLI | 类型 | 校准集 | NPU |
|:----:|:---:|:----:|:------:|:---:|
| **ALPS** | `alps` | 非结构化 / n:m | `c4` | ✅ |
| **FLAP** | `flap` | 结构化 | `wikitext2` | ✅ |
| **LLM-Pruner** | `llm_pruner` | 结构化 | `c4` | ✅ |
| **ShortGPT** | `shortgpt` | 层剪枝 | `pg19` | ✅ |
| **SparseGPT** | `sparsegpt` | 非结构化 / n:m | `c4` | ✅ |
| **Wanda** | `wanda` | 非结构化 / n:m | `c4` | ✅ |
| **Wanda-SP** | `wanda_sp` | 结构化 | `c4` | ✅ |

> ✅ 已支持 &nbsp;|&nbsp; ⏳ 适配中 &nbsp;|&nbsp; 🔶 仅 CUDA

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      main.py (统一 CLI 入口)                     │
├─────────────────────────────────────────────────────────────────┤
│                workflow/ (配置构建 + 阶段执行器)                   │
├──────────────────┬──────────────────┬───────────────────────────┤
│       量化        │       剪枝       │           评测             │
│  ┌────────────┐  │  ┌────────────┐  │  ┌─────────────────────┐  │
│  │ PTQ  (7)   │  │  │ 结构化      │  │  │ PPL (wikitext2/c4)  │  │
│  │ QAT  (4)   │  │  │ 非结构化    │  │  │ Zero-shot (lm-eval) │  │
│  └────────────┘  │  │ 层剪枝      │  │  │ VLM (VLMEvalKit)    │  │
│                  │  └────────────┘  │  └─────────────────────┘  │
├──────────────────┴──────────────────┴───────────────────────────┤
│                algorithm/common/ (共享基础设施)                   │
│     模型加载 · 数据处理 · 设备管理 (GPU/NPU) · IO · Metrics         │
└─────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
MindPipe/
├── main.py                         # 统一 CLI 入口
├── algorithm/
│   ├── common/                     # 共享模型、数据、设备和 IO 工具
│   ├── quantization/
│   │   ├── ptq/                    # AWQ, GPTQ, MQuant, OmniQuant, QuaRot, SmoothQuant, SpinQuant
│   │   └── qat/                    # FlatQuant, QLoRA, QA-LoRA, SplitQuant
│   └── pruning/
│       ├── structured/             # FLAP, LLM-Pruner, ShortGPT, Wanda-SP
│       └── unstructured/           # ALPS, SparseGPT, Wanda
├── workflow/                       # CLI 配置构建与阶段执行器
├── evaluation/                     # PPL, lm-eval-harness, VLMEvalKit 评测入口
├── configs/                        # 通用配置和算法配置
├── scripts/                        # 批量运行与可复现实验脚本
└── third_party/                    # 可选外部评测工具
```

---

## 🤖 模型覆盖

<table>
<tr><td>

| 模型族 | 文本 | 视觉 |
|:------|:----:|:----:|
| LLaMA-2 / LLaMA-3 | ✅ | — |
| Qwen2.5 | ✅ | — |
| Qwen3 | ✅ | — |
| Qwen3.5 | ✅ | — |

</td><td>

| 模型族 | 文本 | 视觉 |
|:------|:----:|:----:|
| Qwen2-VL | ✅ | ✅ |
| Qwen2.5-VL | ✅ | ✅ |
| Qwen3-VL | ✅ | ✅ |
| MiniCPM-V | ✅ | ✅ |
| LLaVA / InternVL | ✅ | 🔶 |

</td></tr>
</table>

> **注意：** 模型支持情况与具体算法相关。请查看 `algorithm/quantization/*/*/method.py` 或 `algorithm/pruning/*/*/method.py` 确认准确覆盖范围。

---

## 📖 参数参考

<details>
<summary><b>⚙️ 通用参数</b></summary>

| 参数 | 默认值 | 说明 |
|:----|:------|:----|
| `--model_path` | 必填 | 本地模型路径或 Hugging Face 模型路径 |
| `--device` | `auto` | Runtime helper 使用的逻辑设备 |
| `--device_map` | `None` | 剪枝和量化必填，推荐 `auto` |
| `--dtype` | `bfloat16` | `auto`、`float16` 或 `bfloat16` |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2`、`sdpa` 或 `eager` |
| `--calibration_dataset` | 算法默认值 | `wikitext2`、`c4`、`pileval`、`pg19` 或 `bookcorpus` |
| `--evaluation_dataset` | `wikitext2` | PPL 评测数据集 |
| `--calibration_samples` | `128` | 校准样本数 |
| `--sequence_length` | `2048` | 校准和评测序列长度 |
| `--batch_size` | `1` | PPL batch size |
| `--max_eval_chunks` | `64` | PPL chunk 数量上限 |
| `--eval_ppl` | `false` | 是否开启 PPL 评测 |
| `--eval_zero_shot` | `false` | 是否开启 lm-eval-harness 任务 |
| `--eval_vlm` | `false` | 是否开启 VLMEvalKit 评测 |

</details>

<details>
<summary><b>🔢 量化参数</b></summary>

| 参数 | 默认值 | 说明 |
|:----|:------|:----|
| `--quantization` | `None` | 注册表中的量化方法名 |
| `--weight_bits` | `4` | 权重量化位宽 |
| `--activation_bits` | `16` | 激活量化位宽 |
| `--query_bits` | `16` | Query 激活位宽 |
| `--key_bits` | `16` | Key cache 位宽 |
| `--value_bits` | `16` | Value cache 位宽 |
| `--group_size` | `128` | 默认 group size |
| `--weight_group_size` | `None` | 覆盖 weight group size |
| `--activation_group_size` | `None` | 覆盖 activation group size |
| `--kv_group_size` | `None` | 覆盖 KV group size |
| `--weight_method` | `gptq` | 支持 GPTQ/RTN 的方法中使用的权重量化方法 |

</details>

<details>
<summary><b>✂️ 剪枝参数</b></summary>

| 参数 | 默认值 | 说明 |
|:----|:------|:----|
| `--pruning` | `None` | 注册表中的剪枝方法名 |
| `--sparsity_ratio` | `0.5` | 目标稀疏率 |
| `--structure_pattern` | `unstructured` | `unstructured`、`2:4` 或 `4:8` |
| `--block_size` | `128` | 支持该参数的方法中的 block size |
| `--damp_percent` | `0.01` | 二阶方法中的 Hessian damping ratio |

</details>

---

## 🔧 安装指南

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+（GPU）或 Ascend CANN（NPU）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-org/MindPipe.git
cd MindPipe

# 创建环境
conda create -n mindpipe python=3.10 -y
conda activate mindpipe

# 安装依赖
git submodule update --init --recursive
pip install -r requirements.txt
```

### 可选：VLMEvalKit

如果需要多模态评测，初始化 VLMEvalKit 子模块或设置环境变量：

```bash
git submodule update --init third_party/VLMEvalKit
# 或
export VLMEVALKIT_ROOT=/path/to/existing/VLMEvalKit
```

---

## 📈 可复现实验

`scripts/repro/` 目录提供开箱即用的 benchmark 启动脚本：

```bash
# 干跑模式（仅打印命令，不执行）
DRY_RUN=true bash scripts/repro/run_qlora_adapted_models_text_suite.sh

# 过滤特定模型
MODEL_FILTER=qwen3 bash scripts/repro/run_mquantpp_awq_vlm_serial_suite.sh
```

可用脚本包括：
- `run_qlora_adapted_models_text_suite.sh`
- `run_qalora_adapted_models_text_suite.sh`
- `run_mquantpp_awq_vlm_serial_suite.sh`
- `run_qwen2_5_vl_gptq_vlm_suite.sh`
- `run_qwen3_vl_2b_gptq_suite.sh`

---

## 📂 输出结构

```
results/
├── <model>/<algorithm>/<run_spec>/
│   ├── metrics.json        # 评测结果与运行元数据
│   └── artifacts.json      # 算法细节、校准设置、checkpoint 路径
└── <model>/<execution_order>/<algorithm1>__<algorithm2>/<run_spec>/
    └── metrics.json
```

---

## ⚠️ 已知限制

| 限制项 | 状态 |
|:------|:----:|
| QuaRot / SpinQuant 暂不支持 NPU | ⏳ |
| MQuant 仅支持 GPU | ⏳ |
| QA-LoRA 仅 CUDA，不导出 AutoGPTQ packed checkpoint | 🔶 |
| QLoRA W2/W3 在 NPU 上使用 fake-quant fallback | ℹ️ |
| 自定义 runtime wrapper 的 saved-model reload 取决于具体方法 | ℹ️ |

---

## 📜 引用与致谢

MindPipe 整合并适配了多个优秀的模型压缩研究成果，使用对应算法时请引用原始论文：

<details>
<summary><b>点击查看引用的工作</b></summary>

- **AWQ** — Activation-aware Weight Quantization
- **GPTQ** — Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **QuaRot** — Outlier-Free Quantization via Rotations
- **SpinQuant** — Rotation-Based Quantization
- **FlatQuant** — Flatness-Aware Quantization
- **SmoothQuant** — Accurate and Efficient Post-Training Quantization
- **OmniQuant** — Omnidirectionally Calibrated Quantization
- **SplitQuant** — Split Quantization
- **QLoRA** — Efficient Finetuning of Quantized LLMs
- **QA-LoRA** — Quantization-Aware Low-Rank Adaptation
- **Wanda** — Pruning by Weights and Activations
- **SparseGPT** — Massive Language Models Can Be Accurately Pruned in One-Shot
- **FLAP** — Fluctuation-based Adaptive Structured Pruning
- **ShortGPT** — Layers in LLMs are More Redundant Than You Expect
- **LLM-Pruner** — On the Structural Pruning of Large Language Models
- **ALPS** — Adaptive Layer-wise Pruning and Sparsification

</details>

---

<div align="center">

**⭐ 如果 MindPipe 对你有帮助，欢迎点个 Star！**

*为模型压缩社区用 ❤️ 打造*

</div>
