# Algorithm-v1 Results Summary

Generated from saved `metrics.json` artifacts on `2026-03-22` (UTC).

## Notes

- This file supersedes the stale values in `results/summary.json` and the old result tables in `README.md`.
- `results/quantization` and `results/quantization_full_eval` use different evaluation setups. Do not compare them directly in one row.
- For `FLAP`, `s0.2` is the only configuration that currently looks usable on `Qwen2.5`; `s0.5` is kept here as a real measured result, not as a recommended configuration.

## Standard Single-Method Results

These are the main pseudo-quant / pseudo-prune results under the shared helper evaluation flow.

| Category | Method | Config | Qwen2.5-7B-Instruct | Qwen2.5-VL-7B-Instruct | Note |
|---|---|---|---:|---:|---|
| Baseline | Shared helper | baseline | 9.4231 | 10.9616 | README/shared helper baseline |
| Quantization | AWQ | `w4a16 seq512` | 10.0488 | 11.7213 | current stable path |
| Quantization | GPTQ | `w4a16 seq512` | 10.1735 | 11.4062 | current stable path |
| Quantization | QuaRot | `w4a16 seq512` | 11.3231 | 12.1375 | weight-only runtime path |
| Quantization | SpinQuant | `w4a16 seq512` | 10.3052 | 11.7240 | identity-R2 fallback |
| Pruning | Wanda | `s0.5 seq512` | 12.1248 | 13.9875 | observed sparsity ~= 0.5 |
| Pruning | SparseGPT | `s0.5 seq512` | 12.7185 | 14.1716 | observed sparsity ~= 0.5 |
| Pruning | FLAP | `s0.2 seq256 WIFN AL-AM` | 12.9172 | 13.8726 | recommended FLAP row on Qwen2.5 |
| Pruning | FLAP | `s0.5 seq512 WIFV AL-AM h8` | 136.0999 | 372.8084 | real result, not recommended |

## FlatQuant Full Eval

These are the trusted `FlatQuant` results after the checkpoint-structure fix. They come from `results/quantization_full_eval`.

| Method | Config | Qwen2.5-7B-Instruct | Qwen2.5-VL-7B-Instruct | Artifact root |
|---|---|---:|---:|---|
| FlatQuant baseline | `w16a16 q16k16v16 seq2048` | 7.3588 | 8.8652 | `results/quantization_full_eval/<model>/flatquant/flatquant_w16a16_q16k16v16_seq2048/` |
| FlatQuant | `w4a4 q16k4v4 seq2048` | 8.4739 | 12.5210 | `results/quantization_full_eval/<model>/flatquant/flatquant_w4a4_q16k4v4_seq2048/` |

## Quantization Artifacts

### Qwen2.5-7B-Instruct

| Method | Config | PPL | Artifact |
|---|---|---:|---|
| AWQ | `w4a16 seq512` | 10.0488 | `results/quantization/Qwen2.5-7B-Instruct/awq/awq_w4a16_seq512/metrics.json` |
| GPTQ | `w4a16 seq512` | 10.1735 | `results/quantization/Qwen2.5-7B-Instruct/gptq/gptq_w4a16_seq512/metrics.json` |
| QuaRot | `w16a16 seq512` | 9.4218 | `results/quantization/Qwen2.5-7B-Instruct/quarot/quarot_w16a16_seq512/metrics.json` |
| QuaRot | `w4a16 seq512` | 11.3231 | `results/quantization/Qwen2.5-7B-Instruct/quarot/quarot_w4a16_seq512/metrics.json` |
| SpinQuant | `w16a16 seq512` | 9.4214 | `results/quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w16a16_seq512/metrics.json` |
| SpinQuant | `w4a16 seq512` | 10.3052 | `results/quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w4a16_seq512/metrics.json` |
| SpinQuant | `w4a4 seq512` | 161157.0967 | `results/quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w4a4_seq512/metrics.json` |
| FlatQuant | `w16a16 seq512` | 9.4231 | `results/quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w16a16_seq512/metrics.json` |
| FlatQuant | `w16a16 q16k16v16 seq2048` | 7.0847 | `results/quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w16a16_q16k16v16_seq2048/metrics.json` |
| FlatQuant | `w4a4 q16k16v16 seq512` | 18.6998 | `results/quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a4_q16k16v16_seq512/metrics.json` |
| FlatQuant | `w4a4 q16k4v4 seq2048` | 8.1497 | `results/quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a4_q16k4v4_seq2048/metrics.json` |

### Qwen2.5-VL-7B-Instruct

| Method | Config | PPL | Artifact |
|---|---|---:|---|
| AWQ | `w4a16 seq512` | 11.7213 | `results/quantization/Qwen2.5-VL-7B-Instruct/awq/awq_w4a16_seq512/metrics.json` |
| GPTQ | `w4a16 seq512` | 11.4062 | `results/quantization/Qwen2.5-VL-7B-Instruct/gptq/gptq_w4a16_seq512/metrics.json` |
| QuaRot | `w16a16 seq512` | 10.9588 | `results/quantization/Qwen2.5-VL-7B-Instruct/quarot/quarot_w16a16_seq512/metrics.json` |
| QuaRot | `w4a16 seq512` | 12.1375 | `results/quantization/Qwen2.5-VL-7B-Instruct/quarot/quarot_w4a16_seq512/metrics.json` |
| SpinQuant | `w16a16 seq512` | 10.9601 | `results/quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w16a16_seq512/metrics.json` |
| SpinQuant | `w4a16 seq512` | 11.7240 | `results/quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w4a16_seq512/metrics.json` |
| SpinQuant | `w4a4 seq512` | 152780.4795 | `results/quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w4a4_seq512/metrics.json` |
| FlatQuant | `w16a16 seq512` | 10.9616 | `results/quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w16a16_seq512/metrics.json` |
| FlatQuant | `w16a16 q16k16v16 seq2048` | 8.5313 | `results/quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w16a16_q16k16v16_seq2048/metrics.json` |
| FlatQuant | `w4a4 q16k16v16 seq512` | 22.2019 | `results/quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w4a4_q16k16v16_seq512/metrics.json` |
| FlatQuant | `w4a4 q16k4v4 seq2048` | 9.2920 | `results/quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w4a4_q16k4v4_seq2048/metrics.json` |

## FlatQuant Full-Eval Artifacts

### Qwen2.5-7B-Instruct

| Method | Config | PPL | Artifact |
|---|---|---:|---|
| FlatQuant | `w16a16 q16k16v16 seq2048` | 7.3588 | `results/quantization_full_eval/Qwen2.5-7B-Instruct/flatquant/flatquant_w16a16_q16k16v16_seq2048/metrics.json` |
| FlatQuant | `w4a4 q16k4v4 seq2048` | 8.4739 | `results/quantization_full_eval/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a4_q16k4v4_seq2048/metrics.json` |

### Qwen2.5-VL-7B-Instruct

| Method | Config | PPL | Artifact |
|---|---|---:|---|
| FlatQuant | `w16a16 q16k16v16 seq2048` | 8.8652 | `results/quantization_full_eval/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w16a16_q16k16v16_seq2048/metrics.json` |
| FlatQuant | `w4a4 q16k4v4 seq2048` | 12.5210 | `results/quantization_full_eval/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w4a4_q16k4v4_seq2048/metrics.json` |

## Pruning Artifacts

### Qwen2.5-7B-Instruct

| Method | Config | PPL | Observed sparsity | Artifact |
|---|---|---:|---:|---|
| Wanda | `s0.5 seq512` | 12.1248 | 0.5000 | `results/pruning/Qwen2.5-7B-Instruct/wanda/wanda_s0.5_seq512/metrics.json` |
| SparseGPT | `s0.5 seq512` | 12.7185 | 0.5000 | `results/pruning/Qwen2.5-7B-Instruct/sparsegpt/sparsegpt_s0.5_seq512/metrics.json` |
| FLAP | `s0.2 seq128 WIFV AL-AM` | 19.3275 | 0.2000 | `results/pruning/Qwen2.5-7B-Instruct/flap/flap_s0.2_seq128/metrics.json` |
| FLAP | `s0.2 seq256 WIFN AL-AM` | 12.9172 | 0.2000 | `results/pruning/Qwen2.5-7B-Instruct/flap/flap_s0.2_seq256/metrics.json` |
| FLAP | `s0.5 seq128 WIFV AL-AM` | 321.9448 | 0.5000 | `results/pruning/Qwen2.5-7B-Instruct/flap/flap_s0.5_seq128/metrics.json` |
| FLAP | `s0.5 seq512 WIFV AL-AM h8` | 136.0999 | 0.5000 | `results/pruning/Qwen2.5-7B-Instruct/flap/flap_s0.5_seq512/metrics.json` |

### Qwen2.5-VL-7B-Instruct

| Method | Config | PPL | Observed sparsity | Artifact |
|---|---|---:|---:|---|
| Wanda | `s0.5 seq512` | 13.9875 | 0.5000 | `results/pruning/Qwen2.5-VL-7B-Instruct/wanda/wanda_s0.5_seq512/metrics.json` |
| SparseGPT | `s0.5 seq512` | 14.1716 | 0.5000 | `results/pruning/Qwen2.5-VL-7B-Instruct/sparsegpt/sparsegpt_s0.5_seq512/metrics.json` |
| FLAP | `s0.2 seq256 WIFN AL-AM` | 13.8726 | 0.2000 | `results/pruning/Qwen2.5-VL-7B-Instruct/flap/flap_s0.2_seq256/metrics.json` |
| FLAP | `s0.5 seq512 WIFV AL-AM h8` | 372.8084 | 0.5000 | `results/pruning/Qwen2.5-VL-7B-Instruct/flap/flap_s0.5_seq512/metrics.json` |
