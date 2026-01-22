# LLM Compression Algorithm Framework

统一的大语言模型压缩算法框架，支持剪枝（Pruning）和量化（Quantization）两大类算法。

## 目录结构

```
algorithm/
├── __init__.py                      # 包初始化
├── main.py                          # 统一入口 (--task pruning/quantization)
├── README.md                        # 本文档
├── TODO.md                          # 开发计划
│
├── pruning/                         # 剪枝算法模块
│   ├── __init__.py
│   ├── main.py                      # 剪枝统一入口 (--algorithm wanda/flap)
│   │
│   ├── structured/                  # 结构化剪枝
│   │   └── flap/                    # FLAP算法
│   │       ├── prune.py             # 剪枝核心实现 (flap, wanda_sp, mag_sp)
│   │       ├── eval.py              # 评估函数
│   │       ├── data.py              # 数据加载
│   │       └── layerwrapper.py      # 层包装器
│   │
│   ├── unstructured/                # 非结构化剪枝
│   │   └── wanda/                   # Wanda算法
│   │       ├── prune.py             # 剪枝核心实现 (wanda, magnitude, sparsegpt)
│   │       ├── eval.py              # 评估函数
│   │       ├── data.py              # 数据加载
│   │       ├── layerwrapper.py      # 层包装器
│   │       ├── sparsegpt.py         # SparseGPT实现
│   │       └── ablate.py            # 消融实验
│   │
│   └── models/                      # 自定义模型
│       └── hf_llama/
│           └── modeling_llama.py    # FLAP专用LlamaForCausalLM (带bias)
│
├── quantization/                    # 量化算法模块
│   ├── __init__.py
│   ├── main.py                      # 量化统一入口 (--method gptq/awq/flatquant)
│   │
│   ├── ptq/                         # Post-Training Quantization (训练后量化)
│   │   ├── gptq/                    # GPTQ算法
│   │   │   ├── gptq.py              # GPTQ核心实现
│   │   │   ├── quant.py             # 量化器
│   │   │   ├── modelutils.py        # 模型工具函数
│   │   │   └── datautils.py         # 数据加载
│   │   │
│   │   └── awq/                     # AWQ算法
│   │       ├── quantize/            # 量化核心
│   │       │   ├── pre_quant.py     # AWQ搜索和应用
│   │       │   ├── quantizer.py     # 伪量化/真量化
│   │       │   ├── auto_scale.py    # 自动缩放
│   │       │   ├── auto_clip.py     # 自动裁剪
│   │       │   └── qmodule.py       # 量化模块
│   │       ├── utils/               # 工具函数
│   │       └── kernels/             # CUDA kernels (真量化推理)
│   │
│   └── qat/                         # Quantization-Aware Training (量化感知训练)
│       └── flatquant/               # FlatQuant算法
│           ├── model_utils.py       # 模型加载
│           ├── data_utils.py        # 数据加载
│           ├── train_utils.py       # 校准训练
│           ├── eval_utils.py        # 评估函数
│           ├── flat_utils.py        # FlatQuant工具
│           ├── quant_utils.py       # 量化工具
│           ├── gptq_utils.py        # GPTQ权重量化
│           ├── flat_linear.py       # FlatQuant线性层
│           ├── trans_utils.py       # 变换矩阵
│           ├── model_tools/         # 模型适配 (Llama, Llama3.1, Qwen)
│           └── deploy/              # 部署代码 (CUDA kernels)
│
├── datasets/                        # 本地数据集
│   ├── c4/
│   │   └── c4-train.00000-of-01024.json.gz
│   ├── wikitext2/
│   │   ├── wiki.train.raw
│   │   └── wiki.test.raw
│   └── pileval/
│       └── val.jsonl                # AWQ校准数据
│
├── output/                          # 输出目录
│
└── scripts/                         # 运行脚本
    ├── run_wanda.sh                 # 非结构化 Wanda
    ├── run_flap.sh                  # 结构化 FLAP
    ├── run_wanda_sp.sh              # 结构化 Wanda
    └── run_mag_sp.sh                # 结构化 Magnitude
```

## 快速开始

### 环境要求

```bash
# 基础环境
pip install torch transformers accelerate datasets

# FLAP 结构化剪枝需要 transformers 4.28.0
pip install transformers==4.28.0

# AWQ 真量化推理需要编译 CUDA kernel
cd quantization/ptq/awq/kernels
PATH=/usr/local/cuda-12.4/bin:$PATH python setup.py install
```

**注意**:
- FLAP 算法依赖自定义的 `LlamaForCausalLM`，仅兼容 transformers 4.28.0
- AWQ/GPTQ/FlatQuant 对 transformers 版本要求较宽松
- AWQ 真量化需要编译 CUDA kernel，需确保 CUDA 版本与 PyTorch 匹配

### 模型路径

当前服务器上可用的模型：

| 模型 | 路径 | 支持的算法 |
|------|------|------------|
| Llama-2-7b-hf | `/mnt/82_store/LLM-weights/Llama-2-7b-hf` | 全部 |

### 命令行调用

框架使用 `--task` 参数区分剪枝和量化任务：

```bash
# 剪枝任务
python main.py --task pruning [剪枝参数...]

# 量化任务
python main.py --task quantization [量化参数...]
```

---

## 剪枝 (Pruning)

### 剪枝示例

#### 1. 非结构化剪枝 (Wanda)

```bash
python main.py --task pruning \
    --algorithm wanda \
    --prune_method wanda \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --nsamples 128 \
    --save_model ./output/llama2-7b-wanda-0.5 \
    --generate \
    --prompts "What is artificial intelligence?"
```

#### 2. 结构化剪枝 (FLAP)

```bash
python main.py --task pruning \
    --algorithm flap \
    --prune_method flap \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --pruning_ratio 0.2 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 128 \
    --save_model ./output/llama2-7b-flap-0.2 \
    --eval \
    --generate \
    --prompts "What is artificial intelligence?"
```

### 剪枝算法

| 类型 | 算法 | --algorithm | --prune_method | 说明 |
|------|------|-------------|----------------|------|
| 非结构化 | Wanda | wanda | wanda | 基于权重和激活值的剪枝 |
| 非结构化 | Magnitude | wanda | magnitude | 基于权重大小的剪枝 |
| 非结构化 | SparseGPT | wanda | sparsegpt | SparseGPT算法 |
| 结构化 | FLAP | flap | flap | 基于波动的自适应剪枝 |
| 结构化 | Wanda-SP | flap | wanda_sp | Wanda结构化版本 |
| 结构化 | Magnitude-SP | flap | mag_sp | Magnitude结构化版本 |

### 剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --algorithm | str | 必填 | 算法族: wanda (非结构化) / flap (结构化) |
| --prune_method | str | wanda | 具体剪枝方法 |
| --sparsity_ratio | float | 0 | 稀疏度 (wanda系列) |
| --sparsity_type | str | unstructured | 稀疏类型: unstructured / 4:8 / 2:4 |
| --pruning_ratio | float | 0 | 剪枝比例 (flap系列) |
| --metrics | str | WIFV | FLAP指标: IFV / WIFV / WIFN |
| --structure | str | AL-AM | FLAP结构: UL-UM / UL-MM / AL-MM / AL-AM |
| --eval | flag | False | 运行PPL评估 |
| --eval_zero_shot | flag | False | 运行zero-shot评估 |
| --save_model | str | None | 保存剪枝后模型路径 |

### 剪枝输出

```
output/llama2-7b-wanda-0.5/
├── config.json                 # 模型配置
├── model-00001-of-00003.safetensors  # 模型权重
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── tokenizer.json              # 分词器
├── tokenizer_config.json
├── special_tokens_map.json
├── log_wanda.txt               # 剪枝日志 (稀疏度, PPL)
└── generation_results.txt      # 生成结果
```

---

## 量化 (Quantization)

### 量化示例

#### 1. GPTQ (W4 权重量化)

```bash
# 基础 W4 量化
python main.py --task quantization \
    --method gptq \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --wbits 4 \
    --dataset wikitext2 \
    --nsamples 128

# 高精度模式 (act-order + true-sequential)
python main.py --task quantization \
    --method gptq \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --wbits 4 \
    --act-order \
    --true-sequential \
    --save ./output/llama2-7b-gptq-w4.pt
```

#### 2. AWQ (W4G128 激活感知量化)

```bash
# 伪量化模式 (用于评估)
python main.py --task quantization \
    --method awq \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --w_bit 4 \
    --q_group_size 128 \
    --q_backend fake \
    --dump_fake ./output/llama2-7b-awq-w4-fake

# 真量化模式 (需要编译 CUDA kernel)
python main.py --task quantization \
    --method awq \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --w_bit 4 \
    --q_group_size 128 \
    --q_backend real \
    --dump_quant ./output/llama2-7b-awq-w4-real.pt

# 保存/加载 AWQ 搜索结果 (加速重复实验)
python main.py --task quantization \
    --method awq \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --dump_awq ./output/awq_results.pt   # 保存搜索结果

python main.py --task quantization \
    --method awq \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --load_awq ./output/awq_results.pt   # 加载搜索结果
```

#### 3. FlatQuant (W4A4KV4 全量化)

```bash
# 完整校准 (推荐)
python main.py --task quantization \
    --method flatquant \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --wbits 4 \
    --abits 4 \
    --kv_bits 4 \
    --cali_dataset wikitext2 \
    --nsamples 128 \
    --epochs 15 \
    --cali_trans \
    --add_diag \
    --lwc \
    --lac \
    --output_dir ./output

# 使用 GPTQ 进行权重量化 (更高精度)
python main.py --task quantization \
    --method flatquant \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --wbits 4 \
    --abits 4 \
    --kv_bits 4 \
    --use_gptq \
    --cali_trans \
    --add_diag \
    --lwc \
    --lac \
    --output_dir ./output

# 保存/加载变换矩阵 (加速重复实验)
python main.py --task quantization \
    --method flatquant \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --wbits 4 --abits 4 --kv_bits 4 \
    --cali_trans --add_diag --lwc --lac \
    --save_matrix \
    --output_dir ./output

python main.py --task quantization \
    --method flatquant \
    --model /mnt/82_store/LLM-weights/Llama-2-7b-hf \
    --wbits 4 --abits 4 --kv_bits 4 \
    --load_matrix ./output/Llama-2-7b-hf/w4a4/exp \
    --output_dir ./output
```

### 量化算法

| 类型 | 算法 | --method | 量化配置 | 说明 |
|------|------|----------|----------|------|
| PTQ | GPTQ | gptq | W2/W3/W4/W8 | 基于二阶信息的权重量化 |
| PTQ | AWQ | awq | W4G128 | 激活感知权重量化，支持伪量化/真量化 |
| QAT | FlatQuant | flatquant | W4A4KV4 | 全量化，带可学习变换矩阵 |

### 量化参数

#### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --method | str | gptq | 量化方法: gptq / awq / flatquant |
| --model | str | 必填 | 模型路径 |
| --seed | int | 0 | 随机种子 |
| --nsamples | int | 128 | 校准样本数 |

#### GPTQ 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --wbits | int | 16 | 权重量化位数: 2/3/4/8/16 |
| --groupsize | int | -1 | 分组大小，-1表示整行 |
| --dataset | str | wikitext2 | 校准数据集: wikitext2 / c4 |
| --sym | flag | False | 对称量化 |
| --act-order | flag | False | 激活值排序 (提高精度) |
| --true-sequential | flag | False | 真正的顺序量化 (提高精度) |
| --static-groups | flag | False | 静态分组 (配合 act-order) |
| --nearest | flag | False | RTN 基线 (最近舍入) |
| --save | str | '' | 保存量化模型路径 |

#### AWQ 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --w_bit | int | 4 | 权重量化位数 |
| --q_group_size | int | 128 | 量化分组大小 |
| --no_zero_point | flag | False | 禁用零点 |
| --q_backend | str | fake | 量化后端: fake (伪量化) / real (真量化) |
| --dump_awq | str | None | 保存 AWQ 搜索结果 |
| --load_awq | str | None | 加载 AWQ 搜索结果 |
| --dump_fake | str | None | 保存伪量化模型 |
| --dump_quant | str | None | 保存真量化模型 |

#### FlatQuant 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --wbits | int | 16 | 权重量化位数 |
| --abits | int | 16 | 激活值量化位数 |
| --kv_bits | int | 16 | KV cache 量化位数 |
| --cali_dataset | str | wikitext2 | 校准数据集: wikitext2 / c4 / pile |
| --epochs | int | 15 | 校准训练轮数 |
| --cali_bsz | int | 4 | 校准批大小 |
| --flat_lr | float | 1e-5 | 变换矩阵学习率 |
| --cali_trans | flag | False | 启用变换矩阵校准 |
| --add_diag | flag | False | 添加对角缩放 |
| --lwc | flag | False | 可学习权重裁剪 |
| --lac | flag | False | 可学习激活值裁剪 |
| --use_gptq | flag | False | 使用 GPTQ 进行权重量化 (默认 RTN) |
| --save_matrix | flag | False | 保存变换矩阵 |
| --load_matrix | str | None | 加载变换矩阵路径 |
| --output_dir | str | ./output | 输出目录 |

### 量化输出

#### GPTQ 输出

```
output/llama2-7b-gptq-w4.pt      # 量化后的模型权重 (state_dict)
```

#### AWQ 输出

```
# 伪量化模式
output/llama2-7b-awq-w4-fake/
├── config.json
├── model.safetensors
├── tokenizer.json
└── ...

# 真量化模式
output/llama2-7b-awq-w4-real.pt  # 量化后的模型权重

# AWQ 搜索结果
output/awq_results.pt            # 可复用的 scale/clip 参数
```

#### FlatQuant 输出

```
output/Llama-2-7b-hf/w4a4/exp/
├── flat_parameters.pth          # 变换矩阵参数 (如果 --save_matrix)
└── ...
```

### 量化模型兼容性

| 模型 | GPTQ | AWQ | FlatQuant |
|------|------|-----|-----------|
| Llama-2-7b-hf | 支持 | 支持 | 支持 |
| Llama-2-13b-hf | 支持 | 支持 | 支持 |
| Llama-3 系列 | 支持 | 支持 | 支持 |
| Qwen 系列 | 支持 | 支持 | 支持 |

---

## 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --task | str | 必填 | 任务类型: pruning / quantization |
| --model | str | 必填 | 模型路径 |
| --seed | int | 0 | 随机种子 |
| --nsamples | int | 128 | 校准样本数 |
| --cache_dir | str | llm_weights | 模型缓存目录 |

## 生成参数 (剪枝专用)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --generate | flag | False | 生成文本 |
| --prompts | str[] | None | 生成提示词 |
| --max_new_tokens | int | 128 | 最大生成token数 |
| --temperature | float | 0.7 | 采样温度 |
| --top_k | int | 50 | Top-k采样 |
| --top_p | float | 0.9 | Top-p采样 |
| --save_generation | str | None | 保存生成结果路径 |

---

## 架构设计

### 参数传递机制

框架使用 `parse_known_args()` 实现参数分层传递：

```
main.py (--task)
    │
    ├── pruning/main.py (--algorithm, --prune_method, ...)
    │
    └── quantization/main.py (--method, --wbits, ...)
```

1. 顶层 `main.py` 只解析 `--task` 参数
2. 其余参数通过 `remaining` 传递给子模块
3. 子模块有独立的 argparse，定义各自的参数

```python
# main.py 核心逻辑
args, remaining = parser.parse_known_args()  # 只解析 --task
if args.task == 'pruning':
    sys.argv = [sys.argv[0]] + remaining     # 传递剩余参数
    pruning_main()
elif args.task == 'quantization':
    sys.argv = [sys.argv[0]] + remaining
    quantization_main()
```

---

## 数据集

框架使用本地数据集，无需联网下载：

| 数据集 | 路径 | 用途 |
|--------|------|------|
| C4 | `datasets/c4/c4-train.00000-of-01024.json.gz` | Wanda/GPTQ/FlatQuant 校准 |
| WikiText-2 | `datasets/wikitext2/wiki.{train,test}.raw` | FLAP/GPTQ/FlatQuant 校准/评估 |
| PileVal | `datasets/pileval/val.jsonl` | AWQ 校准 |

---

## 参考文献

### 剪枝
- [Wanda](https://github.com/locuslab/wanda): Pruning by Weights and Activations
- [FLAP](https://github.com/CASIA-IVA-Lab/FLAP): Fluctuation-based Adaptive Structured Pruning
- [SparseGPT](https://github.com/IST-DASLab/sparsegpt): Massive Language Models Can Be Accurately Pruned in One-Shot

### 量化
- [GPTQ](https://github.com/IST-DASLab/gptq): Accurate Post-Training Quantization for Generative Pre-trained Transformers
- [AWQ](https://github.com/mit-han-lab/llm-awq): Activation-aware Weight Quantization
- [FlatQuant](https://github.com/ruikangliu/FlatQuant): Flatness Matters for LLM Quantization
