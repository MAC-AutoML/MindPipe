# MindPipe

[English](README.md) | [中文](README_zh.md)

MindPipe is a unified compression and evaluation framework for large language
models and vision-language models. It provides one CLI entrypoint for
post-training quantization, quantization-aware training, pruning, perplexity
evaluation, zero-shot evaluation, and VLMEvalKit-based multimodal evaluation.

The framework is designed for reproducible research across GPU and NPU
backends, with shared model loading, dataset handling, device management, and
result serialization.

## Highlights

- Unified `main.py` entrypoint for quantization, pruning, compression pipelines,
  and evaluation-only runs.
- 11 quantization methods registered in-tree, including PTQ and QAT-style
  methods.
- 7 pruning methods registered in-tree, covering unstructured, semi-structured,
  and structured pruning.
- Text and vision-language model support through a shared model adapter layer.
- GPU and NPU device abstraction for cache management, synchronization, seeds,
  and dtype policy.
- Per-run artifacts and metrics written as JSON for downstream aggregation.
- Reproducibility scripts for common text and multimodal benchmark suites.

## Repository Layout

```text
MindPipe/
├── main.py                         # Unified CLI entrypoint
├── algorithm/
│   ├── common/                     # Shared model, data, device, IO utilities
│   ├── quantization/
│   │   ├── ptq/                    # AWQ, GPTQ, MQuant, OmniQuant, QuaRot, SmoothQuant, SpinQuant
│   │   └── qat/                    # FlatQuant, QLoRA, QA-LoRA, SplitQuant
│   └── pruning/
│       ├── structured/             # FLAP, LLM-Pruner, ShortGPT, Wanda-SP
│       └── unstructured/           # ALPS, SparseGPT, Wanda
├── workflow/                       # CLI config builder and stage executor
├── evaluation/                     # PPL, lm-eval-harness, and VLMEvalKit runners
├── configs/                        # Shared and algorithm-specific configs
├── scripts/                        # Batch and reproducibility scripts
└── third_party/                    # Optional external evaluation tools
```

## Supported Algorithms

### Quantization

| Method | Family | Main Coverage | NPU Status |
| --- | --- | --- | --- |
| `awq` | PTQ | Weight-only quantization with activation-aware scaling | Ready |
| `gptq` | PTQ | Weight-only GPTQ quantization | Ready |
| `mquant` | PTQ | Multimodal GPTQ/AWQ-style quantization for language and visual branches | Not ready |
| `omniquant` | PTQ | Learnable weight and activation transformation | Ready |
| `quarot` | PTQ | Rotation-based W/A/KV quantization | Not ready |
| `smoothquant` | PTQ | Activation smoothing for W/A quantization | Ready |
| `spinquant` | PTQ | Rotation-based W/A/KV quantization with SpinQuant-style hooks | Not ready |
| `flatquant` | QAT | FlatQuant-style trainable transformations | Ready |
| `qlora` | QAT | QLoRA and low-bit fake-quant adapter training | Ready, with experimental NPU fake-quant fallback |
| `qalora` | QAT | Basic QA-LoRA group-pooled adapter training | CUDA only |
| `splitquant` | QAT | SplitQuant-style trainable transformations | Ready |

### Pruning

| Method | Type | Default Calibration Dataset | NPU Status |
| --- | --- | --- | --- |
| `alps` | Unstructured and n:m semi-structured | `c4` | Ready |
| `flap` | Structured | `wikitext2` | Ready |
| `llm_pruner` | Structured | `c4` | Ready |
| `shortgpt` | Layer pruning | `pg19` | Ready |
| `sparsegpt` | Unstructured and n:m semi-structured | `c4` | Ready |
| `wanda` | Unstructured and n:m semi-structured | `c4` | Ready |
| `wanda_sp` | Structured | `c4` | Ready |

## Model Coverage

MindPipe has been adapted across text-only and multimodal model families,
including:

- LLaMA-family text models, including LLaMA-2 and LLaMA-3 style checkpoints.
- Qwen2.5 text models.
- Qwen3 text models.
- Qwen3.5 text/language-only paths.
- Qwen2-VL, Qwen2.5-VL, and Qwen3-VL multimodal paths.
- MiniCPM-V language and multimodal paths for selected quantization flows.
- LLaVA and InternVL loader compatibility paths where supported by the local
  Transformers environment.

Model support is algorithm-dependent. The most reliable way to check current
support is to inspect each method under `algorithm/quantization/*/*/method.py`
or `algorithm/pruning/*/*/method.py`, and the model-specific configs under
`configs/algorithms/`.

## Adaptation Progress

### 2026-04-18

- Completed GPU validation for AWQ W4A16 on Qwen3, Qwen3-VL, Qwen3.5,
  Qwen2-VL, and LLaVA-1.5. Text-side PPL runs completed successfully with no
  obvious anomalies.
- Qwen2-VL and Qwen3-VL completed VLMEvalKit evaluation on the validated
  multimodal datasets. AWQ W4A16 showed acceptable accuracy degradation compared
  with FP16.
- The evaluation framework did not yet support Qwen3.5 and LLaVA-1.5 multimodal
  evaluation at that time, so only text-side validation was completed for those
  models.

### 2026-04-19

- Completed NPU validation for AWQ W4A16 on Qwen3, Qwen3-VL, Qwen3.5, Qwen2-VL,
  and LLaVA-1.5. Text-side PPL runs completed successfully with no obvious
  anomalies.

### 2026-04-20

- Completed GPU adaptation and validation for MQuant on Qwen3-VL, Qwen2-VL, and
  Qwen2.5-VL. Text-side PPL runs completed successfully with no obvious
  anomalies.
- Qwen3-VL, Qwen2-VL, and Qwen2.5-VL completed VLMEvalKit evaluation on the
  validated multimodal datasets. The visual W8A8 plus language W4A8 setting
  showed acceptable accuracy degradation compared with FP16.

### 2026-04-21

- Completed GPU adaptation and validation for QuaRot and SpinQuant on Qwen3,
  Qwen3.5, Qwen3-VL, and Qwen2-VL. Text-side PPL runs completed successfully
  with no obvious anomalies.
- Qwen3-VL, Qwen2-VL, and Qwen2.5-VL completed VLMEvalKit evaluation on the
  validated multimodal datasets. The W4A8 setting showed acceptable accuracy
  degradation compared with FP16.

## Installation

```bash
conda activate mindpipe
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

If VLMEvalKit evaluation is required, initialize the VLMEvalKit submodule or set
`VLMEVALKIT_ROOT` to an existing checkout.

## Device Loading Policy

Quantization and pruning runs require `--device_map`. This applies to single-GPU
runs as well as multi-GPU runs. The recommended pattern is:

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

This policy keeps model placement under the Hugging Face Accelerate dispatch
hooks. Avoid manually moving compressed models with `.to(device)` after loading
with `device_map`.

## Quick Start

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

### Quantization

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

### Pruning

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

### Pruning Followed by Quantization

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

## Multimodal Evaluation

MindPipe integrates VLMEvalKit through `evaluation/vlm_eval.py`. A typical VLM
evaluation command is:

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

Use `--num_samples` for smoke tests and `--vlm_resume true` to reuse existing
per-dataset artifacts when available.

## Common Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model_path` | Required | Local or Hugging Face model path |
| `--device` | `auto` | Logical device used by runtime helpers |
| `--device_map` | `None` | Required for pruning and quantization, recommended value: `auto` |
| `--dtype` | `bfloat16` | `auto`, `float16`, or `bfloat16` |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2`, `sdpa`, or `eager` |
| `--calibration_dataset` | Method default | `wikitext2`, `c4`, `pileval`, `pg19`, or `bookcorpus` |
| `--evaluation_dataset` | `wikitext2` | Dataset used for PPL evaluation |
| `--calibration_samples` | `128` | Number of calibration samples |
| `--sequence_length` | `2048` in many scripts | Sequence length for calibration and evaluation |
| `--batch_size` | `1` | PPL batch size |
| `--max_eval_chunks` | `64` | Optional cap for PPL chunks |
| `--eval_ppl` | `false` | Enable perplexity evaluation |
| `--eval_zero_shot` | `false` | Enable lm-eval-harness tasks |
| `--eval_vlm` | `false` | Enable VLMEvalKit evaluation |

## Quantization Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--quantization` | `None` | One of the registered quantization methods |
| `--weight_bits` | `4` | Weight quantization bit width |
| `--activation_bits` | `16` | Activation quantization bit width |
| `--query_bits` | `16` | Query activation bit width for supported methods |
| `--key_bits` | `16` | Key cache bit width for supported methods |
| `--value_bits` | `16` | Value cache bit width for supported methods |
| `--group_size` | `128` | Default group size |
| `--weight_group_size` | `None` | Overrides weight group size |
| `--activation_group_size` | `None` | Overrides activation group size |
| `--kv_group_size` | `None` | Overrides KV group size |
| `--weight_method` | `gptq` | Weight method for methods that support GPTQ or RTN |

## Pruning Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--pruning` | `None` | One of the registered pruning methods |
| `--sparsity_ratio` | `0.5` | Target sparsity ratio |
| `--structure_pattern` | `unstructured` | `unstructured`, `2:4`, or `4:8` where supported |
| `--block_size` | `128` | Block size for supported pruning methods |
| `--damp_percent` | `0.01` | Hessian damping ratio for second-order methods |

## Reproducibility Scripts

The `scripts/repro/` directory contains serial benchmark launchers for adapted
model families and algorithm paths. Examples include:

- `scripts/repro/run_qlora_adapted_models_text_suite.sh`
- `scripts/repro/run_qalora_adapted_models_text_suite.sh`
- `scripts/repro/run_mquantpp_awq_vlm_serial_suite.sh`
- `scripts/repro/run_qwen2_5_vl_gptq_vlm_suite.sh`
- `scripts/repro/run_qwen3_vl_2b_gptq_suite.sh`

Use `DRY_RUN=true` to print commands without executing them, and use
`MODEL_FILTER=<model_key>` when the script supports model-level filtering.

## Outputs

Each run writes metrics and artifacts under the resolved output directory.

```text
results/
├── <model>/<algorithm>/<run_spec>/metrics.json
├── <model>/<algorithm>/<run_spec>/artifacts.json
└── <model>/<workflow>/<run_spec>/metrics.json
```

`metrics.json` stores evaluation results and run metadata. `artifacts.json`
stores algorithm-specific details such as quantized layers, adapter paths,
calibration settings, and generated checkpoint locations.

## Known Limitations

- QuaRot and SpinQuant are not marked NPU-ready in the current registry.
- MQuant is currently GPU-oriented and not marked NPU-ready.
- QA-LoRA is a basic CUDA-only implementation and does not export an AutoGPTQ
  packed checkpoint.
- QLoRA uses bitsandbytes for CUDA W4 when available; W2/W3 and NPU paths use
  the in-tree fake-quant fallback.
- Support for saved-model reload after methods that insert custom runtime
  wrappers is method-dependent.

## Citation and Acknowledgements

MindPipe vendors or adapts ideas and implementation components from several
model compression projects, including AWQ, GPTQ, QuaRot, SpinQuant, FlatQuant,
SmoothQuant, OmniQuant, SplitQuant, QLoRA, QA-LoRA, Wanda, SparseGPT, FLAP,
ShortGPT, LLM-Pruner, and ALPS. Please cite the original method papers when
using the corresponding algorithms.
