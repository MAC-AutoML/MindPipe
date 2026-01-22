# MindPipe 模型压缩框架

极致易用的大模型压缩框架，支持量化、剪枝等主流压缩技术，适配多种硬件后端。

## 框架架构

```
mindpipe/
├── auto_compressor.py              # 易用化工具（一键压缩入口）
│
├── workflows/                      # 工作流层
│   ├── quantization_workflow.py    # 量化工作流
│   ├── pruning_workflow.py         # 剪枝工作流
│   ├── hybrid_workflow.py          # 混合压缩工作流
│   └── evaluation_workflow.py      # 模型评估工作流
│
├── algorithm/                      # 算法库
│   ├── pruning/                    # 剪枝算法
│   │   ├── structured/             # 结构化剪枝 (FLAP, LLM-Pruner, SliceGPT)
│   │   ├── unstructured/           # 非结构化剪枝 (Wanda, SparseGPT)
│   │   └── semi_structured/        # 半结构化剪枝
│   └── quantization/               # 量化算法
│       ├── ptq/                    # PTQ (GPTQ, AWQ, SmoothQuant, LLM.int8, ...)
│       └── qat/                    # QAT (LLM-QAT, QLoRA)
│
├── adapters/                       # 适配层
│   ├── bitsandbytes.py             # NVIDIA bitsandbytes 适配
│   ├── compressed_tensor.py        # 压缩张量格式适配
│   └── modelslim.py                # 华为昇腾 ModelSlim 适配
│
└── core/                           # 基础底座层
    ├── device.py                   # 设备适配 (CANN/NPU, CUDA/GPU, MUSA/GPU)
    ├── distributed.py              # 分布式优化
    ├── model.py                    # 模型集成
    └── cluster.py                  # 集群优化
```

## 架构层级

```
┌─────────────────────────────────────────────────────────┐
│                    易用化工具                            │
│                 auto_compressor.py                      │
├─────────────────────────────────────────────────────────┤
│                      工作流                              │
│   量化工作流 │ 剪枝工作流 │ 混合压缩工作流 │ 模型评估工作流  │
├─────────────────────────────────────────────────────────┤
│                      算法库                              │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │      量化算法库      │  │       剪枝算法库         │   │
│  │  QAT: LLM-QAT,QLoRA │  │ Structured: FLAP,       │   │
│  │  PTQ: GPTQ,AWQ,     │  │   LLM-Pruner,SliceGPT   │   │
│  │   SmoothQuant,...   │  │ Unstructured: Wanda,    │   │
│  │                     │  │   SparseGPT             │   │
│  │                     │  │ Semi-structured         │   │
│  └─────────────────────┘  └─────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                      适配层                              │
│      bitsandbytes  │  compressed_tensor  │  modelslim   │
├─────────────────────────────────────────────────────────┤
│                     基础底座                             │
│    分布式优化  │  模型集成  │  设备适配  │  集群优化      │
├─────────────────────────────────────────────────────────┤
│                     硬件支持                             │
│           CANN/NPU  │  CUDA/GPU  │  MUSA/GPU            │
└─────────────────────────────────────────────────────────┘
```

## 支持的算法

### 剪枝算法

| 类型 | 算法 | 状态 |
|------|------|------|
| 结构化 | FLAP | ✅ 已实现 |
| 结构化 | LLM-Pruner | 待实现 |
| 结构化 | SliceGPT | 待实现 |
| 非结构化 | Wanda | ✅ 已实现 |
| 非结构化 | SparseGPT | ✅ 已实现 |
| 半结构化 | - | 待实现 |

### 量化算法

| 类型 | 算法 | 状态 |
|------|------|------|
| PTQ | GPTQ | 待实现 |
| PTQ | AWQ | 待实现 |
| PTQ | SmoothQuant | 待实现 |
| PTQ | LLM.int8 | 待实现 |
| QAT | LLM-QAT | 待实现 |
| QAT | QLoRA | 待实现 |

## 快速开始

### 环境要求

```bash
pip install torch transformers==4.28.0 accelerate datasets
```

### 使用方式

```python
# 方式1: 一键压缩
from mindpipe import AutoCompressor

compressor = AutoCompressor()
result = compressor.compress(
    model="path/to/model",
    method="pruning",
    algorithm="flap",
    pruning_ratio=0.2
)

# 方式2: 使用工作流
from mindpipe.workflows import PruningWorkflow

workflow = PruningWorkflow(
    model="path/to/model",
    algorithm="flap",
    pruning_ratio=0.2
)
result = workflow.run()

# 方式3: 直接使用算法（命令行）
# 详见 algorithm/README.md
```

### 命令行使用

```bash
# 剪枝
python -m mindpipe.algorithm.main --task pruning --algorithm flap --model <path> --pruning_ratio 0.2

# 量化（待实现）
python -m mindpipe.algorithm.main --task quantization --method gptq --model <path> --bits 4
```

## 硬件支持

| 硬件 | 后端 | 状态 |
|------|------|------|
| NVIDIA GPU | CUDA | ✅ 支持 |
| 华为昇腾 NPU | CANN | 待适配 |
| 摩尔线程 GPU | MUSA | 待适配 |
| CPU | - | ✅ 支持 |

## 目录说明

| 目录 | 说明 |
|------|------|
| `auto_compressor.py` | 一键压缩入口，自动选择工作流 |
| `workflows/` | 工作流实现，编排算法执行流程 |
| `algorithm/` | 算法库，包含剪枝和量化的具体实现 |
| `adapters/` | 第三方库适配层 |
| `core/` | 基础底座，设备管理、分布式等 |
