# GPU vs NPU Compare

- Updated at: `2026-03-25 15:16:21`
- GPU official metrics: `31`
- NPU official metrics: `31`
- Matched paths: `31`
- Missing on NPU: `0`
- Comparable finite pairs: `29`
- Non-finite pairs: `2`
- Extra on NPU: `0`

| Status | Path | GPU PPL | NPU PPL | Delta | Delta % |
| --- | --- | ---: | ---: | ---: | ---: |
| nonfinite_pair | `quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a16_seq128/metrics.json` | NaN | 16.286047 |  |  |
| nonfinite_pair | `quantization/Qwen2.5-7B-Instruct/quarot/quarot_w4a4_seq512/metrics.json` | NaN | 158691.993800 |  |  |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w4a4_seq512/metrics.json` | 781492.137985 | 9266.674015 | -772225.463970 | -98.81% |
| compared | `quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w4a4_seq512/metrics.json` | 161157.096700 | 4779.217006 | -156377.879694 | -97.03% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w4a4_seq512/metrics.json` | 152780.479508 | 9832.385908 | -142948.093600 | -93.56% |
| compared | `pruning/Qwen2.5-7B-Instruct/flap/flap_s0.5_seq128/metrics.json` | 12105.366460 | 297.583736 | -11807.782724 | -97.54% |
| compared | `quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a4_seq512/metrics.json` | 2801.141886 | 4238.604178 | 1437.462292 | 51.32% |
| compared | `pruning/Qwen2.5-VL-7B-Instruct/flap/flap_s0.5_seq512/metrics.json` | 399.113353 | 189.530566 | -209.582787 | -52.51% |
| compared | `pruning/Qwen2.5-7B-Instruct/flap/flap_s0.5_seq512/metrics.json` | 125.718071 | 314.564511 | 188.846440 | 150.21% |
| compared | `pruning/Qwen2.5-7B-Instruct/flap/flap_s0.2_seq256/metrics.json` | 14.219840 | 39.214227 | 24.994387 | 175.77% |
| compared | `quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w4a4_q16k16v16_seq512/metrics.json` | 18.699820 | 20.827152 | 2.127332 | 11.38% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w4a4_q16k16v16_seq512/metrics.json` | 22.201861 | 23.870543 | 1.668682 | 7.52% |
| compared | `pruning/Qwen2.5-VL-7B-Instruct/flap/flap_s0.2_seq256/metrics.json` | 17.383944 | 15.956601 | -1.427343 | -8.21% |
| compared | `pruning/Qwen2.5-VL-7B-Instruct/wanda/wanda_s0.5_seq512/metrics.json` | 13.987538 | 14.725858 | 0.738321 | 5.28% |
| compared | `pruning/Qwen2.5-7B-Instruct/sparsegpt/sparsegpt_s0.5_seq512/metrics.json` | 12.718453 | 13.277350 | 0.558897 | 4.39% |
| compared | `pruning/Qwen2.5-7B-Instruct/wanda/wanda_s0.5_seq512/metrics.json` | 12.124774 | 12.681858 | 0.557084 | 4.59% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/gptq/gptq_w4a16_seq512/metrics.json` | 11.406152 | 11.961609 | 0.555457 | 4.87% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/quarot/quarot_w16a16_seq512/metrics.json` | 10.958785 | 11.389605 | 0.430820 | 3.93% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w16a16_seq512/metrics.json` | 10.960102 | 11.389867 | 0.429765 | 3.92% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/flatquant/flatquant_w16a16_seq512/metrics.json` | 10.961624 | 11.390682 | 0.429057 | 3.91% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/awq/awq_w4a16_seq512/metrics.json` | 11.721283 | 12.109663 | 0.388380 | 3.31% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/quarot/quarot_w4a16_seq512/metrics.json` | 12.137550 | 12.506063 | 0.368513 | 3.04% |
| compared | `quantization/Qwen2.5-VL-7B-Instruct/spinquant/spinquant_w4a16_seq512/metrics.json` | 11.723958 | 12.076701 | 0.352743 | 3.01% |
| compared | `quantization/Qwen2.5-7B-Instruct/awq/awq_w4a16_seq512/metrics.json` | 10.048791 | 10.297856 | 0.249065 | 2.48% |
| compared | `quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w4a16_seq512/metrics.json` | 10.305184 | 10.531210 | 0.226026 | 2.19% |
| compared | `quantization/Qwen2.5-7B-Instruct/flatquant/flatquant_w16a16_seq512/metrics.json` | 9.423130 | 9.642051 | 0.218921 | 2.32% |
| compared | `quantization/Qwen2.5-7B-Instruct/quarot/quarot_w16a16_seq512/metrics.json` | 9.421810 | 9.639587 | 0.217777 | 2.31% |
| compared | `quantization/Qwen2.5-7B-Instruct/spinquant/spinquant_w16a16_seq512/metrics.json` | 9.421448 | 9.638601 | 0.217154 | 2.30% |
| compared | `pruning/Qwen2.5-VL-7B-Instruct/sparsegpt/sparsegpt_s0.5_seq512/metrics.json` | 14.171643 | 14.367203 | 0.195560 | 1.38% |
| compared | `quantization/Qwen2.5-7B-Instruct/gptq/gptq_w4a16_seq512/metrics.json` | 10.173475 | 10.274930 | 0.101455 | 1.00% |
| compared | `quantization/Qwen2.5-7B-Instruct/quarot/quarot_w4a16_seq512/metrics.json` | 11.323114 | 11.375393 | 0.052279 | 0.46% |
