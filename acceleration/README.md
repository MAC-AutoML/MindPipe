# Four-model Ascend acceleration

This directory delivers the tested runtime changes from MindPipe without
forking or publishing either vLLM repository. The installer applies one
consolidated patch to vLLM and one to vLLM-Ascend, so all four model paths are
installed as a single compatible unit.

Supported source line:

- vLLM 0.11.0 release-candidate sources
- vLLM-Ascend 0.11.0rc0 release-candidate sources

Compatibility is checked from generated version text when available and from
required source/API anchors. Commit identity is not required or recorded.

Install and verify:

```bash
python acceleration/install_runtime_patch.py \
  --vllm-root /path/to/vllm \
  --vllm-ascend-root /path/to/vllm-ascend

python acceleration/verify_runtime.py \
  --vllm-root /path/to/vllm \
  --vllm-ascend-root /path/to/vllm-ascend
```

The install is idempotent. Both patches are dry-run before either runtime is
changed, and a partial install is rolled back on error. Remove the changes with:

```bash
python acceleration/uninstall_runtime_patch.py \
  --vllm-root /path/to/vllm \
  --vllm-ascend-root /path/to/vllm-ascend
```

The Qwen3-MoE path keeps one expert-parameter allocation. The removed
`REPLICATED_LOCAL_EXPERTS` and `REPLICATED_SINGLE_PASS` mechanisms are neither
packaged nor accepted by the runtime verifier.

The Qwen2.5-VL-72B candidate also avoids per-layer FIA output allocation and
device-to-host cumulative-length conversion. Its acceptance runner first
checks the preallocated FIA result element-for-element against the original
call at Qwen2.5-VL attention shapes.

Run the fixed acceptance campaigns from the repository root. All four runners
require `VLLM_ROOT` and `VLLM_ASCEND_ROOT`; model directories must contain a
`config.json` file.

Qwen2.5-VL-7B also needs a fixed image set and the GCC 11 cross-toolchain bin
directory used to compile its custom operator:

```bash
VLLM_ROOT=/path/to/vllm VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
FP16_MODEL=/path/to/qwen25-vl-7b-fp16 \
W8A8_MODEL=/path/to/qwen25-vl-7b-w8a8 \
IMAGE_DIR=/path/to/benchmark-images TOOLCHAIN=/path/to/gcc-11/bin \
  scripts/repro/run_qwen25_vl_7b_1p5x.sh
```

Qwen2.5-VL-72B:

```bash
VLLM_ROOT=/path/to/vllm VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
FP16_MODEL=/path/to/qwen25-vl-72b-fp16 \
W8A8_MODEL=/path/to/qwen25-vl-72b-w8a8 \
IMAGE_DIR=/path/to/benchmark-images \
  scripts/repro/run_qwen25_vl_72b_1p5x.sh
```

Qwen3-MoE:

```bash
VLLM_ROOT=/path/to/vllm VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
FP16_MODEL=/path/to/qwen3-moe-fp16 W8A8_MODEL=/path/to/qwen3-moe-w8a8 \
  scripts/repro/run_qwen3_moe_1p5x.sh
```

Mixtral:

```bash
VLLM_ROOT=/path/to/vllm VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
BF16_MODEL=/path/to/mixtral-bf16 W8A8_MODEL=/path/to/mixtral-w8a8 \
  scripts/repro/run_mixtral_1p5x.sh
```

`PYTHON`, `OUT`, and device/port settings can be overridden as documented at
the top of each runner. Every campaign uses fresh services, an alternating run
order, fixed workload settings, and an unrounded `>= 1.5`
ratio-of-mean-throughputs gate.
