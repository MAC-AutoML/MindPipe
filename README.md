# MindPipe

大语言模型压缩算法统一评测框架。集成 5 种量化算法和 4 种剪枝算法，提供统一的 CLI 入口、评测链路和 GPU/NPU 双后端支持。

## 目录结构

```text
MindPipe/
├── main.py                   # 唯一 CLI 入口
├── algorithm/
│   ├── common/               # 公共基础设施
│   │   ├── device.py         # GPU/NPU 设备抽象（resolve_device, empty_cache, synchronize...）
│   │   ├── hadamard.py       # Hadamard 变换 dispatch（CUDA kernel / PyTorch butterfly fallback）
│   │   ├── modeling.py       # 模型加载、文本主干提取、block 输入捕获
│   │   ├── datasets.py       # 校准数据集和评测数据集加载
│   │   ├── reproducibility.py # 全局种子设定
│   │   ├── io.py             # 路径和 JSON 工具
│   │   ├── runtime.py        # sys.path 注入
│   │   └── logging.py        # 日志配置
│   ├── quantization/
│   │   ├── base.py           # BaseQuantizationMethod + NPU fail-fast
│   │   ├── registry.py       # 量化算法注册表
│   │   ├── config.py         # 参数归一化
│   │   ├── ptq/
│   │   │   ├── awq/          # AWQ（method.py + source/）
│   │   │   ├── gptq/         # GPTQ
│   │   │   ├── quarot/       # QuaRot
│   │   │   └── spinquant/    # SpinQuant
│   │   └── qat/
│   │       └── flatquant/    # FlatQuant
│   └── pruning/
│       ├── base.py           # BasePruningMethod + NPU fail-fast
│       ├── registry.py       # 剪枝算法注册表
│       ├── structured/
│       │   ├── flap/         # FLAP
│       │   ├── shortgpt/    # ShortGPT
│       │   └── wanda_sp/     # Wanda-SP
│       └── unstructured/
│           ├── wanda/        # Wanda
│           └── sparsegpt/    # SparseGPT
├── workflow/
│   ├── schema.py             # WorkflowStage, WorkflowConfig 数据结构
│   ├── builder.py            # CLI 参数解析 → WorkflowConfig
│   └── executor.py           # 多阶段编排执行
├── evaluation/
│   ├── ppl.py                # PPL 评测（wikitext2, c4）
│   ├── lm_eval.py            # Zero-shot 评测（lm-eval-harness）
│   ├── vlm_eval.py           # 多模态评测（VLMEvalKit）
│   └── runner.py             # 评测路由（自动根据参数选择评测类型）
└── scripts/                  # 批量运行、结果汇总等辅助脚本
```

## 支持的算法

### 量化（PTQ / QAT）

| 算法 | 类型 | 权重 | 激活 | KV Cache | NPU 就绪 |
|------|------|------|------|----------|---------|
| AWQ | PTQ | W4 | - | - | 是 |
| GPTQ | PTQ | W4 | - | - | 是 |
| QuaRot | PTQ | W4 | A4 | K4 V4 | 否（Hadamard 需验证） |
| SpinQuant | PTQ | W4 | A4 | K4 V4 | 否（Hadamard 需验证） |
| FlatQuant | QAT | W4 | A4 | K4 V4 | 是 |

### 剪枝

| 算法 | 结构 | NPU 就绪 | 推荐校准数据集 |
|------|------|---------|--------------|
| Wanda | 非结构化 | 是 | c4 |
| SparseGPT | 非结构化 | 是 | c4 |
| Wanda-SP | 结构化 | 是 | c4 |
| FLAP | 结构化 | 是 | wikitext2 |
| ShortGPT | 结构化（层剪枝） | 是 | pg19 |
| ALPS | 非结构化 | 是 | c4 |

所有剪枝方法均通过 `--calibration_dataset` 参数统一选择校准数据集（`wikitext2` / `c4` / `pileval` / `pg19`），推荐使用上表中的数据集以获得最佳效果。

### 已验证模型

- Qwen2.5-7B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-7B-Instruct（FlatQuant / NPU）

## 快速开始

### 环境准备

```bash
conda activate mindpipe
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

### 量化

```bash
python main.py \
  --quantization awq \
  --model_path /path/to/model \
  --device cuda:0 \
  --dtype float16 \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --weight_bits 4 \
  --group_size 128 \
  --output_dir ./results/quantization
```

### 剪枝

```bash
python main.py \
  --pruning wanda \
  --model_path /path/to/model \
  --device cuda:0 \
  --dtype float16 \
  --calibration_dataset c4 \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --sparsity_ratio 0.5 \
  --output_dir ./results/pruning
```

### 剪枝 + 量化

```bash
python main.py \
  --pruning wanda_sp \
  --quantization gptq \
  --execution_order pruning_then_quantization \
  --model_path /path/to/model \
  --device cuda:0 \
  --dtype float16 \
  --weight_bits 4 \
  --sparsity_ratio 0.2 \
  --output_dir ./results/workflow
```

### 仅评测

```bash
python main.py \
  --model_path ./results/pruning/qwen2.5-7b/wanda/.../saved_model \
  --device cuda:0 \
  --eval_ppl true \
  --eval_zero_shot true \
  --output_dir ./results/evaluate
```

## GPU / NPU 设备适配

MindPipe 通过 `algorithm/common/device.py` 提供 GPU/NPU 统一抽象层，无需任何 monkey-patch：

- **`resolve_device(device)`** — 将字符串（`"cuda:0"`、`"npu:0"`、`"auto"`）解析为 `torch.device`
- **`backend_module(device)`** — 返回 `torch.cuda` 或 `torch.npu`
- **`empty_cache(device)`** / **`synchronize(device)`** — 自动选择后端
- **`manual_seed_all(seed, device)`** — 自动选择后端

使用方式：

```bash
# GPU
python main.py --quantization gptq --model_path /path/to/model --device cuda:0 ...

# NPU（需安装 torch_npu）
python main.py --quantization gptq --model_path /path/to/model --device npu:0 ...

# 自动检测
python main.py --quantization gptq --model_path /path/to/model --device auto ...
```

### NPU fail-fast

未在 NPU 上验证的算法（QuaRot、SpinQuant）设有 `npu_ready = False` 标记。在 NPU 上调用时会直接报错，避免静默出错：

```
RuntimeError: Algorithm 'quarot' is not yet NPU-ready. Please use --device cuda:0 to run.
```

### Hadamard 变换 dispatch

`algorithm/common/hadamard.py` 提供统一的 `hadamard_transform()` 函数：

- **CUDA** + `fast_hadamard_transform` 已安装 → 使用 CUDA kernel（快）
- **其他情况** → 使用纯 PyTorch 蝴蝶算法 fallback（正确，慢 2-5x）

各算法的 `method.py` 通过注入机制将 dispatch 函数替换到 source 模块中，source 代码本身无需修改。

## 架构设计

```
CLI (main.py)  — 统一入口，无子命令
  │  python main.py --pruning wanda ...
  │  python main.py --quantization gptq ...
  │  python main.py --pruning wanda --quantization gptq ...
  │  python main.py --model_path <saved> --eval_ppl true ...
  ▼
workflow/builder.py → WorkflowConfig
  │  根据 --pruning / --quantization 是否传入动态构建 stages
  ▼
workflow/executor.py
  │  加载模型 → 逐阶段执行 → 评测 → 保存结果
  ▼
algorithm/*/method.py
  │  各算法的统一入口（apply_pruning / apply_fake_quantization）
  ▼
evaluation/runner.py
  │  PPL / Zero-shot / VLM 评测（通过 --eval_ppl / --eval_zero_shot / --eval_vlm 控制）
  ▼
results/<model>/<algorithm>/metrics.json
```

**设计原则**：

1. **三层分离** — `algorithm/` 只管算法实现，`workflow/` 只管编排，`evaluation/` 只管评测
2. **source/ 不动** — 第三方源码 vendored 进来，适配逻辑全在 `method.py` 包装层
3. **device 从上层传入** — `args.device` 从 CLI 流入各层，不依赖全局 monkey-patch
4. **缺失算子直接报错** — 不静默 fallback 到 CPU，确保问题可发现

## 常用参数

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `auto` | `cuda:0` / `npu:0` / `auto` |
| `--dtype` | `bfloat16` | `float16` / `bfloat16` |
| `--sequence_length` | 512 | 序列长度 |
| `--calibration_samples` | 128 | 校准样本数 |
| `--seed` | 0 | 随机种子 |
| `--evaluation_dataset` | `wikitext2` | 评测数据集（`wikitext2` / `c4`） |

### 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--quantization` | None | `awq` / `gptq` / `quarot` / `spinquant` / `flatquant` |
| `--weight_bits` | 4 | 权重比特数 |
| `--activation_bits` | 16 | 激活比特数 |
| `--key_bits` / `--value_bits` | 16 | KV Cache 比特数 |
| `--group_size` | 128 | 量化分组大小 |
| `--weight_method` | `gptq` | 权重量化方法（`gptq` / `rtn`） |
| `--weight_symmetric` | True | 权重对称量化 |

### 剪枝参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pruning` | None | `wanda` / `sparsegpt` / `wanda_sp` / `flap` / `shortgpt` / `alps` |
| `--sparsity_ratio` | 0.5 | 稀疏率 |
| `--structure_pattern` | `unstructured` | 剪枝结构模式 |
| `--damp_percent` | 0.01 | Hessian 阻尼系数 |

> **校准数据集选择**：各算法均支持 `wikitext2` / `c4` / `pileval` / `pg19` 四种校准数据集。上表"推荐校准数据集"列为各算法原始论文使用的默认数据集，效果最好。ShortGPT 强烈推荐 `pg19`（长文本书籍，Block Influence 统计更稳定），使用其他数据集不会报错但可能影响精度。`--calibration_samples` 控制采样窗口数，ShortGPT 适当增大（如 256 或 512）可提升重要性排序的稳定性。

### 组合参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--execution_order` | `pruning_then_quantization` | 两阶段执行顺序 |

### 保存 / 评测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_model` | False | 保存压缩后的模型 |
| `--eval_ppl` | True | PPL 评测 |
| `--eval_zero_shot` | False | Zero-shot 评测 |
| `--eval_vlm` | False | 多模态评测 |

## 评测

评测与压缩是平级功能，通过参数控制：

```bash
# 压缩 + 评测（默认）
python main.py --model_path /path/to/model --pruning wanda --sparsity_ratio 0.5

# 仅压缩，跳过评测
python main.py --model_path /path/to/model --pruning wanda --sparsity_ratio 0.5 --eval_ppl false

# 仅评测已保存的模型（不传压缩方法）
python main.py \
  --model_path ./results/pruning/.../saved_model \
  --eval_ppl true \
  --eval_zero_shot true \
  --output_dir ./results/evaluate

# 评测原始未压缩模型
python main.py \
  --model_path /path/to/original/model \
  --eval_ppl true \
  --eval_zero_shot true \
  --output_dir ./results/evaluate_baseline
```

> **TODO**:
> 1. QuaRot / SpinQuant 等带 ActQuantWrapper 和 Hadamard 变换的量化方法暂不支持单独评测，
>    因为 `AutoModelForCausalLM.from_pretrained()` 无法恢复自定义 wrapper 结构。
>    后续需为各量化方法实现 `prepare_structure()` 接口，在加载时重建 wrapper 再灌入 state_dict。
> 2. `--save_model` 时自动写入 `method_info.json` 记录压缩方法、参数等信息，
>    以便加载时自动识别并选择正确的加载逻辑（标准 HF 加载 vs 重建 wrapper）。
>    目前仅对伪剪枝 / AWQ / GPTQ 有效（标准结构，可直接 `from_pretrained`）。
>
>    方案：save 时自动写入元数据文件：
>
>    ```
>    saved_model/
>    ├── config.json
>    ├── model.safetensors
>    ├── method_info.json        ← 新增，记录压缩方法
>    └── ...
>    ```
>
>    `method_info.json` 内容示例：
>
>    ```json
>    {
>      "method": "quarot",
>      "stages": [
>        {"type": "quantization", "algorithm": "quarot", "weight_bits": 4, ...}
>      ]
>    }
>    ```
>
>    load 时：
>    - 读到 `method_info.json` → 按记录的方法重建 wrapper → 灌入 state_dict
>    - 没有 `method_info.json` → 当标准 HF 模型加载（剪枝 / AWQ / GPTQ 的情况）
>
>    这样用户不用手动指定，路径一传就自动识别。

## 结果输出

每次运行在 `output_dir` 下生成目录结构：

```text
results/
├── quantization/<model>/<algorithm>/<run_spec>/metrics.json
├── pruning/<model>/<algorithm>/<run_spec>/metrics.json
└── workflow/<model>/<order>/<algo1>__<algo2>/<run_spec>/metrics.json
```

`metrics.json` 包含 PPL 结果、算法配置、量化/剪枝层详情等完整信息。

## 当前已知限制

- **FlatQuant** 已支持 NPU 运行；若出现 PPL 偏高，优先排查训练/重参数化路径与数值稳定性
- **QuaRot / SpinQuant** 的低比特 activation 配置尚未在 Qwen/Qwen2.5-VL 上做精细收敛
- **SpinQuant** 默认使用 `identity-R2` fallback，learned rotation 训练链路尚未适配 Qwen
