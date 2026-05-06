# MindPipe

[English](README.md) | [中文](README_zh.md)

MindPipe 是一个面向大语言模型和视觉语言模型的统一压缩与评测框架。它提供统一的 `main.py`
入口，覆盖后训练量化、量化感知训练、剪枝、PPL 评测、zero-shot 评测，以及基于 VLMEvalKit 的多模态评测。

框架目标是让 GPU/NPU 后端下的模型加载、数据集处理、设备管理、结果保存和可复现实验流程尽量统一。

## 主要特性

- 使用统一的 `main.py` 入口运行量化、剪枝、组合压缩流程和纯评测流程。
- 当前注册了 11 个量化算法，覆盖 PTQ 和 QAT 风格方法。
- 当前注册了 7 个剪枝算法，覆盖非结构化、半结构化和结构化剪枝。
- 通过共享模型适配层支持纯文本模型和视觉语言模型。
- 提供 GPU/NPU 抽象，统一 cache 清理、同步、随机种子和 dtype 策略。
- 每次运行都会保存 JSON 格式的 metrics 和 artifacts，便于后续汇总。
- 提供常用文本与多模态 benchmark 的可复现实验脚本。

## 目录结构

```text
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

## 支持算法

### 量化

| 方法 | 类型 | 主要覆盖范围 | NPU 状态 |
| --- | --- | --- | --- |
| `awq` | PTQ | Activation-aware scaling 的权重量化 | 已支持 |
| `gptq` | PTQ | GPTQ 权重量化 | 已支持 |
| `mquant` | PTQ | 面向语言分支和视觉分支的多模态 GPTQ/AWQ 风格量化 | 暂未支持 |
| `omniquant` | PTQ | 可学习的权重和激活变换 | 已支持 |
| `quarot` | PTQ | 基于旋转的 W/A/KV 量化 | 暂未支持 |
| `smoothquant` | PTQ | 面向 W/A 量化的激活平滑 | 已支持 |
| `spinquant` | PTQ | 基于 SpinQuant hook 的 W/A/KV 旋转量化 | 暂未支持 |
| `flatquant` | QAT | FlatQuant 风格可训练变换 | 已支持 |
| `qlora` | QAT | QLoRA 与低比特 fake-quant adapter 训练 | 已支持，NPU 使用实验性 fake-quant fallback |
| `qalora` | QAT | 基础 QA-LoRA group-pooled adapter 训练 | 仅 CUDA |
| `splitquant` | QAT | SplitQuant 风格可训练变换 | 已支持 |

### 剪枝

| 方法 | 类型 | 默认校准集 | NPU 状态 |
| --- | --- | --- | --- |
| `alps` | 非结构化与 n:m 半结构化 | `c4` | 已支持 |
| `flap` | 结构化 | `wikitext2` | 已支持 |
| `llm_pruner` | 结构化 | `c4` | 已支持 |
| `shortgpt` | 层剪枝 | `pg19` | 已支持 |
| `sparsegpt` | 非结构化与 n:m 半结构化 | `c4` | 已支持 |
| `wanda` | 非结构化与 n:m 半结构化 | `c4` | 已支持 |
| `wanda_sp` | 结构化 | `c4` | 已支持 |

## 模型覆盖

MindPipe 已适配多个纯文本和多模态模型族，包括：

- LLaMA-family 纯文本模型，包括 LLaMA-2 和 LLaMA-3 风格 checkpoint。
- Qwen2.5 纯文本模型。
- Qwen3 纯文本模型。
- Qwen3.5 文本或 language-only 路径。
- Qwen2-VL、Qwen2.5-VL 和 Qwen3-VL 多模态路径。
- MiniCPM-V 的部分语言和多模态量化路径。
- 在本地 Transformers 环境支持时，提供 LLaVA 和 InternVL 的加载兼容路径。

模型支持情况与具体算法相关。最可靠的检查方式是查看 `algorithm/quantization/*/*/method.py`、
`algorithm/pruning/*/*/method.py`，以及 `configs/algorithms/` 下的模型配置。

## 适配进展

### 2026-04-18

- 已在 GPU 上完成 Qwen3、Qwen3-VL、Qwen3.5、Qwen2-VL 和 LLaVA-1.5 的 AWQ W4A16 验证。文本侧 PPL 能正常跑通，结果无明显异常。
- Qwen2-VL 和 Qwen3-VL 已在验证过的多模态数据集上完成 VLMEvalKit 评测。AWQ W4A16 相比 FP16 的精度下降在可接受范围内。
- 当时评测框架尚不支持 Qwen3.5 和 LLaVA-1.5 的多模态评测，因此这两个模型只完成了文本侧验证。

### 2026-04-19

- 已在 NPU 上完成 Qwen3、Qwen3-VL、Qwen3.5、Qwen2-VL 和 LLaVA-1.5 的 AWQ W4A16 验证。文本侧 PPL 能正常跑通，结果无明显异常。

### 2026-04-20

- 已在 GPU 上完成 Qwen3-VL、Qwen2-VL 和 Qwen2.5-VL 的 MQuant 适配与验证。文本侧 PPL 能正常跑通，结果无明显异常。
- Qwen3-VL、Qwen2-VL 和 Qwen2.5-VL 已在验证过的多模态数据集上完成 VLMEvalKit 评测。视觉 W8A8 加语言 W4A8 相比 FP16 的精度下降在可接受范围内。

### 2026-04-21

- 已在 GPU 上完成 Qwen3、Qwen3.5、Qwen3-VL 和 Qwen2-VL 的 QuaRot 与 SpinQuant 适配验证。文本侧 PPL 能正常跑通，结果无明显异常。
- Qwen3-VL、Qwen2-VL 和 Qwen2.5-VL 已在验证过的多模态数据集上完成 VLMEvalKit 评测。W4A8 配置相比 FP16 的精度下降在可接受范围内。

## 安装

```bash
conda activate mindpipe
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

如果需要运行 VLMEvalKit 评测，请初始化 VLMEvalKit submodule，或将 `VLMEVALKIT_ROOT` 指向已有的 VLMEvalKit checkout。

## 设备加载策略

量化和剪枝运行都要求指定 `--device_map`。这一要求同时适用于单 GPU 和多 GPU。推荐模式如下：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --quantization awq \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --attn_implementation sdpa \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --weight_bits 4 \
  --group_size 128 \
  --eval_ppl true \
  --output_dir ./results/awq
```

该策略让模型放置统一由 Hugging Face Accelerate dispatch hook 管理。使用 `device_map` 加载后，不要再手动对压缩模型执行 `.to(device)`。

## 快速开始

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

### 量化

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

### 剪枝

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --pruning wanda \
  --model_path /path/to/model \
  --device_map auto \
  --dtype float16 \
  --attn_implementation sdpa \
  --calibration_dataset c4 \
  --calibration_samples 128 \
  --sequence_length 2048 \
  --sparsity_ratio 0.5 \
  --eval_ppl true \
  --output_dir ./results/wanda
```

### 剪枝后量化

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

## 多模态评测

MindPipe 通过 `evaluation/vlm_eval.py` 集成 VLMEvalKit。典型命令如下：

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

可以使用 `--num_samples` 做 smoke test，使用 `--vlm_resume true` 复用已有的单数据集产物。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | 必填 | 本地模型路径或 Hugging Face 模型路径 |
| `--device` | `auto` | runtime helper 使用的逻辑设备 |
| `--device_map` | `None` | 剪枝和量化必填，推荐 `auto` |
| `--dtype` | `bfloat16` | `auto`、`float16` 或 `bfloat16` |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2`、`sdpa` 或 `eager` |
| `--calibration_dataset` | 算法默认值 | `wikitext2`、`c4`、`pileval`、`pg19` 或 `bookcorpus` |
| `--evaluation_dataset` | `wikitext2` | PPL 评测数据集 |
| `--calibration_samples` | `128` | 校准样本数 |
| `--sequence_length` | 多数脚本中为 `2048` | 校准和评测序列长度 |
| `--batch_size` | `1` | PPL batch size |
| `--max_eval_chunks` | `64` | PPL chunk 数量上限 |
| `--eval_ppl` | `false` | 是否开启 PPL 评测 |
| `--eval_zero_shot` | `false` | 是否开启 lm-eval-harness 任务 |
| `--eval_vlm` | `false` | 是否开启 VLMEvalKit 评测 |

## 量化参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--quantization` | `None` | 注册表中的量化方法名 |
| `--weight_bits` | `4` | 权重量化 bit |
| `--activation_bits` | `16` | 激活量化 bit |
| `--query_bits` | `16` | 支持该参数的方法中的 query bit |
| `--key_bits` | `16` | 支持该参数的方法中的 key cache bit |
| `--value_bits` | `16` | 支持该参数的方法中的 value cache bit |
| `--group_size` | `128` | 默认 group size |
| `--weight_group_size` | `None` | 覆盖 weight group size |
| `--activation_group_size` | `None` | 覆盖 activation group size |
| `--kv_group_size` | `None` | 覆盖 KV group size |
| `--weight_method` | `gptq` | 支持 GPTQ/RTN 的方法中使用的权重量化方法 |

## 剪枝参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--pruning` | `None` | 注册表中的剪枝方法名 |
| `--sparsity_ratio` | `0.5` | 目标稀疏率 |
| `--structure_pattern` | `unstructured` | 支持时可设为 `unstructured`、`2:4` 或 `4:8` |
| `--block_size` | `128` | 支持该参数的方法中的 block size |
| `--damp_percent` | `0.01` | 二阶方法中的 Hessian damping ratio |

## 可复现实验脚本

`scripts/repro/` 目录包含面向适配模型族和算法路径的串行 benchmark launcher，例如：

- `scripts/repro/run_qlora_adapted_models_text_suite.sh`
- `scripts/repro/run_qalora_adapted_models_text_suite.sh`
- `scripts/repro/run_mquantpp_awq_vlm_serial_suite.sh`
- `scripts/repro/run_qwen2_5_vl_gptq_vlm_suite.sh`
- `scripts/repro/run_qwen3_vl_2b_gptq_suite.sh`

可以使用 `DRY_RUN=true` 打印命令但不执行；脚本支持时，可以使用 `MODEL_FILTER=<model_key>` 选择部分模型。

## 输出

每次运行都会在解析后的输出目录中写入 metrics 和 artifacts。

```text
results/
├── <model>/<algorithm>/<run_spec>/metrics.json
├── <model>/<algorithm>/<run_spec>/artifacts.json
└── <model>/<workflow>/<run_spec>/metrics.json
```

`metrics.json` 保存评测结果和运行元数据；`artifacts.json` 保存算法相关信息，例如量化层、adapter 路径、校准设置和生成的 checkpoint 路径。

## 已知限制

- QuaRot 和 SpinQuant 当前未标记为 NPU-ready。
- MQuant 当前主要面向 GPU 路径，未标记为 NPU-ready。
- QA-LoRA 是基础 CUDA-only 实现，不导出 AutoGPTQ packed checkpoint。
- QLoRA 在 CUDA W4 上优先使用 bitsandbytes；W2/W3 和 NPU 路径使用框架内 fake-quant fallback。
- 插入自定义 runtime wrapper 的方法，其 saved-model reload 支持取决于具体方法。

## 引用与致谢

MindPipe vendored 或适配了多个模型压缩项目的思路和实现组件，包括 AWQ、GPTQ、QuaRot、SpinQuant、FlatQuant、SmoothQuant、OmniQuant、SplitQuant、QLoRA、QA-LoRA、Wanda、SparseGPT、FLAP、ShortGPT、LLM-Pruner 和 ALPS。使用对应算法时，请引用原始方法论文。
