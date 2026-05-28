<div align="center">

# 🧠 MindPipe

**A Unified Compression & Evaluation Framework for LLMs and VLMs**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-orange)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![NPU Ready](https://img.shields.io/badge/NPU-Ready-purple)]()

[English](README.md) | [中文](README_zh.md)

<p align="center">
  <em>One CLI. 11 quantization methods. 7 pruning methods. GPU & NPU. Text & Vision.</em>
</p>

---

**Quantize** · **Prune** · **Evaluate** · **Reproduce**

</div>

## ✨ Why MindPipe?

> Most compression tools only handle one technique on one type of model.  
> **MindPipe unifies them all under a single, reproducible pipeline.**

<table>
<tr>
<td width="50%">

### 🎯 One Entrypoint, All Methods
A single `main.py` drives quantization, pruning, combined workflows, and evaluation — no juggling scripts.

### 🔀 GPU + NPU
First-class support for both CUDA GPUs and Ascend NPUs with shared device abstraction.

### 📊 Integrated Evaluation
PPL, lm-eval-harness zero-shot, and VLMEvalKit multimodal benchmarks — all built in.

</td>
<td width="50%">

### 🧩 Modular & Extensible
Clean registry-based architecture makes adding new algorithms straightforward.

### 🔬 Reproducibility First
JSON artifacts, batch scripts, and per-run metrics ensure every result is traceable.

### 👁️ Vision-Language Native
Not an afterthought — VLMs are first-class citizens with dedicated multimodal eval.

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# 1. Setup
conda activate mindpipe
git submodule update --init --recursive
pip install -r requirements.txt

# 2. Quantize a model (AWQ W4A16)
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

# 3. Prune a model (Wanda 50% sparsity)
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

# 4. Recover a FlatQuant + pruning workflow with Compression LoRA
MODEL_PATH=/path/to/model \
GPU_ID=0,1 \
bash scripts/finetuning/flatquant_lora_auto_gpu.sh
```

<details>
<summary><b>📋 More Examples (Click to Expand)</b></summary>

### Full-Precision Evaluation

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

### GPTQ Quantization

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

### Pruning + Quantization Pipeline

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

### VLM Multimodal Evaluation

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

## 📦 Supported Algorithms

Use the method identifiers in the CLI column as command-line values. Display
names such as QA-LoRA, LLM-Pruner, and Wanda-SP are descriptive; the actual CLI
values are `qalora`, `llm_pruner`, and `wanda_sp`.

### Quantization (11 Methods)

| Method | CLI | Family | Technique | NPU |
|:------:|:---:|:------:|:---------:|:---:|
| **AWQ** | `awq` | PTQ | Weight-only with activation-aware scaling | ✅ |
| **GPTQ** | `gptq` | PTQ | Weight-only GPTQ quantization | ✅ |
| **MQuant** | `mquant` | PTQ | Multimodal GPTQ/AWQ for language & visual branches | ⏳ |
| **OmniQuant** | `omniquant` | PTQ | Learnable weight & activation transformation | ✅ |
| **QuaRot** | `quarot` | PTQ | Rotation-based W/A/KV quantization | ⏳ |
| **SmoothQuant** | `smoothquant` | PTQ | Activation smoothing for W/A quantization | ✅ |
| **SpinQuant** | `spinquant` | PTQ | Rotation-based W/A/KV with SpinQuant hooks | ⏳ |
| **FlatQuant** | `flatquant` | QAT | Trainable transformations | ✅ |
| **QLoRA** | `qlora` | QAT | Low-bit fake-quant adapter training | ✅ |
| **QA-LoRA** | `qalora` | QAT | Group-pooled adapter training | 🔶 |
| **SplitQuant** | `splitquant` | QAT | SplitQuant-style trainable transformations | ✅ |

### Pruning (7 Methods)

| Method | CLI | Type | Calibration | NPU |
|:------:|:---:|:----:|:-----------:|:---:|
| **ALPS** | `alps` | Unstructured / n:m | `c4` | ✅ |
| **FLAP** | `flap` | Structured | `wikitext2` | ✅ |
| **LLM-Pruner** | `llm_pruner` | Structured | `c4` | ✅ |
| **ShortGPT** | `shortgpt` | Layer pruning | `pg19` | ✅ |
| **SparseGPT** | `sparsegpt` | Unstructured / n:m | `c4` | ✅ |
| **Wanda** | `wanda` | Unstructured / n:m | `c4` | ✅ |
| **Wanda-SP** | `wanda_sp` | Structured | `c4` | ✅ |

### Finetuning

| Method | CLI | Scope | NPU |
|:------:|:---:|:------|:---:|
| **Compression LoRA** | `compression_lora` | FlatQuant + fixed-mask pruning recovery | ⏳ |

Compression LoRA currently supports FlatQuant combined with fixed-shape pruning
masks, such as Wanda, SparseGPT, and ALPS. Structured pruning requires
pseudo-pruning mode.

> ✅ Ready &nbsp;|&nbsp; ⏳ In Progress &nbsp;|&nbsp; 🔶 CUDA Only

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
├─────────────────────────────────────────────────────────────────┤
│                    workflow/ (Config + Executor)                │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ Quantization │   Pruning    │  Finetuning  │     Evaluation     │
│ ┌──────────┐ │ ┌──────────┐ │ ┌──────────┐ │ ┌────────────────┐ │
│ │ PTQ (7)  │ │ │Structured│ │ │Compress. │ │ │ PPL            │ │
│ │ QAT (4)  │ │ │Unstruct. │ │ │LoRA      │ │ │ Zero-shot      │ │
│ └──────────┘ │ │LayerPrune│ │ └──────────┘ │ │ VLM Eval       │ │
│              │ └──────────┘ │              │ └────────────────┘ │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│              algorithm/common/ (Shared Infrastructure)          │
│     Model Loading · Data · Device (GPU/NPU) · IO · Metrics      │
└─────────────────────────────────────────────────────────────────┘
```

### Repository Layout

```
MindPipe/
├── main.py                         # Unified CLI entrypoint
├── algorithm/
│   ├── common/                     # Shared model, data, device, IO utilities
│   ├── quantization/
│   │   ├── ptq/                    # AWQ, GPTQ, MQuant, OmniQuant, QuaRot, SmoothQuant, SpinQuant
│   │   └── qat/                    # FlatQuant, QLoRA, QA-LoRA, SplitQuant
│   ├── finetuning/                 # Compression LoRA recovery finetuning
│   └── pruning/
│       ├── structured/             # FLAP, LLM-Pruner, ShortGPT, Wanda-SP
│       └── unstructured/           # ALPS, SparseGPT, Wanda
├── workflow/                       # CLI config builder and stage executor
├── evaluation/                     # PPL, lm-eval-harness, and VLMEvalKit runners
├── configs/                        # Shared and algorithm-specific configs
├── scripts/                        # Batch and reproducibility scripts
└── third_party/                    # Optional external evaluation tools
```

---

## 🤖 Model Coverage

<table>
<tr><td>

| Model Family | Text | Vision |
|:------------|:----:|:------:|
| LLaMA-2 / LLaMA-3 | ✅ | — |
| Qwen2.5 | ✅ | — |
| Qwen3 | ✅ | — |
| Qwen3.5 / Qwen3.6 dense | ✅ | ✅ |

</td><td>

| Model Family | Text | Vision |
|:------------|:----:|:------:|
| Qwen2-VL | ✅ | ✅ |
| Qwen2.5-VL | ✅ | ✅ |
| Qwen3-VL | ✅ | ✅ |
| MiniCPM-V | ✅ | ✅ |
| LLaVA / InternVL | ✅ | 🔶 |

</td></tr>
</table>

> **Note:** Model support is algorithm-dependent. Check `algorithm/quantization/*/*/method.py` or `algorithm/pruning/*/*/method.py` for exact coverage.
> MoE variants are method-dependent and are not covered by Compression LoRA yet.

---

## 📖 Configuration Reference

<details>
<summary><b>⚙️ Common Arguments</b></summary>

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--model_path` | Required | Local or Hugging Face model path |
| `--device` | `auto` | Logical device used by runtime helpers |
| `--device_map` | `None` | Required for pruning/quantization (`auto` recommended) |
| `--dtype` | `bfloat16` | `auto`, `float16`, or `bfloat16` |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2`, `sdpa`, or `eager` |
| `--calibration_dataset` | Method default | `wikitext2`, `c4`, `pileval`, `pg19`, or `bookcorpus` |
| `--evaluation_dataset` | `wikitext2` | Dataset used for PPL evaluation |
| `--calibration_samples` | `128` | Number of calibration samples |
| `--sequence_length` | `2048` | Sequence length for calibration and evaluation |
| `--batch_size` | `1` | PPL batch size |
| `--max_eval_chunks` | `64` | Optional cap for PPL chunks |
| `--eval_ppl` | `false` | Enable perplexity evaluation |
| `--eval_zero_shot` | `false` | Enable lm-eval-harness tasks |
| `--eval_vlm` | `false` | Enable VLMEvalKit evaluation |

</details>

<details>
<summary><b>🔢 Quantization Arguments</b></summary>

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--quantization` | `None` | One of the registered quantization methods |
| `--weight_bits` | `4` | Weight quantization bit width |
| `--activation_bits` | `16` | Activation quantization bit width |
| `--query_bits` | `16` | Query activation bit width |
| `--key_bits` | `16` | Key cache bit width |
| `--value_bits` | `16` | Value cache bit width |
| `--group_size` | `128` | Default group size |
| `--weight_group_size` | `None` | Overrides weight group size |
| `--activation_group_size` | `None` | Overrides activation group size |
| `--kv_group_size` | `None` | Overrides KV group size |
| `--weight_method` | `gptq` | Weight method for methods supporting GPTQ/RTN |

</details>

<details>
<summary><b>✂️ Pruning Arguments</b></summary>

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--pruning` | `None` | One of the registered pruning methods |
| `--sparsity_ratio` | `0.5` | Target sparsity ratio |
| `--structure_pattern` | `unstructured` | `unstructured`, `2:4`, or `4:8` |
| `--block_size` | `128` | Block size for supported pruning methods |
| `--damp_percent` | `0.01` | Hessian damping ratio for second-order methods |

</details>

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU) or Ascend CANN (for NPU)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/MindPipe.git
cd MindPipe

# Create environment
conda create -n mindpipe python=3.10 -y
conda activate mindpipe

# Install dependencies
git submodule update --init --recursive
pip install -r requirements.txt
```

### Optional: VLMEvalKit

For multimodal evaluation, initialize the VLMEvalKit submodule or set `VLMEVALKIT_ROOT`:

```bash
git submodule update --init third_party/VLMEvalKit
# or
export VLMEVALKIT_ROOT=/path/to/existing/VLMEvalKit
```

---

## 📈 Reproducibility

The `scripts/repro/` directory contains ready-to-use benchmark launchers:

```bash
# Dry run (print commands without executing)
DRY_RUN=true bash scripts/repro/run_qlora_adapted_models_text_suite.sh

# Filter specific models
MODEL_FILTER=qwen3 bash scripts/repro/run_mquantpp_awq_vlm_serial_suite.sh
```

Available scripts include:
- `run_qlora_adapted_models_text_suite.sh`
- `run_qalora_adapted_models_text_suite.sh`
- `run_mquantpp_awq_vlm_serial_suite.sh`
- `run_qwen2_5_vl_gptq_vlm_suite.sh`
- `run_qwen3_vl_2b_gptq_suite.sh`

Compression LoRA finetuning launchers are grouped under `scripts/finetuning/`:

- `flatquant_lora_auto_gpu.sh` automatically dispatches by model config.
- `llm/flatquant_lora_llm_gpu.sh` is for text-only LLMs.
- `vlm/flatquant_lora_vlm_gpu.sh` is for MiniCPM-V, Qwen2.5-VL, and Qwen3-VL.
- `qwen3_5/flatquant_lora_qwen3_5_gpu.sh` is for dense Qwen3.5/Qwen3.6 VLMs.

---

## 📂 Output Structure

```
results/
├── <model>/<algorithm>/<run_spec>/
│   ├── metrics.json        # Evaluation results & run metadata
│   └── artifacts.json      # Algorithm details, calibration settings, checkpoints
└── <model>/<execution_order>/<algorithm1>__<algorithm2>/<run_spec>/
    └── metrics.json
```

---

## ⚠️ Known Limitations

| Limitation | Status |
|:-----------|:------:|
| QuaRot / SpinQuant not NPU-ready | ⏳ |
| MQuant GPU-only | ⏳ |
| QA-LoRA CUDA-only, no AutoGPTQ export | 🔶 |
| QLoRA W2/W3 use fake-quant fallback on NPU | ℹ️ |
| Compression LoRA currently requires FlatQuant + fixed-mask pruning | ℹ️ |
| Qwen3.5-MoE / Qwen3.6-35B-A3B Compression LoRA is not supported yet | ⏳ |
| Custom runtime wrapper reload is method-dependent | ℹ️ |

---

## 📜 Citation & Acknowledgements

MindPipe builds upon the following outstanding research. Please cite the original papers when using their methods:

<details>
<summary><b>Click to see referenced works</b></summary>

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

**⭐ If you find MindPipe useful, please consider giving it a star!**

*Built with ❤️ for the model compression community*

</div>
