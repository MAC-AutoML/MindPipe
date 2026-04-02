# VLMEvalKit 评测说明

## 做了哪些处理

为了把 `VLMEvalKit` 接入 `MindPipe`，主要做了下面几类处理：

- 在 `evaluation/` 中新增了多模态评测入口，把 `VLMEvalKit` 评测流程接入统一评测链路。
- 在 `workflow` 和 `runner` 中补充了 `VLM` 评测参数，使其可以跟随原有 `quantization`、`pruning`、`workflow` 流程一起执行。
- 为 `Qwen2.5-VL` 和 `MiniCPM-V` 增加了适配 wrapper，使 `MindPipe` 内部模型可以直接按 `VLMEvalKit` 的接口进行推理。
- 对 `MiniCPM-V` 补了与新版 `transformers` 的兼容逻辑，包括生成接口、cache 接口和推理 dtype 相关处理。
- 环境中安装了 `VLMEvalKit` 及所需依赖，并清理了会造成冲突但当前链路不需要的包。

## 如何使用

当前有两种用法。

### 1. 跟随 `main.py` 主流程一起评测

`VLM` 评测已经接入统一 CLI。只要在原有命令上增加：

- `--eval_vlm`
- `--vlm_datasets ...`

如果希望使用本地规则完成评分，建议加：

- `--vlm_judge exact_matching`

示例：

```bash
python main.py \
  quantization \
  --algorithm awq \
  --model_path <your_model_path> \
  --device cuda:0 \
  --dtype auto \
  --calibration_dataset pileval \
  --evaluation_dataset wikitext2 \
  --calibration_samples 128 \
  --sequence_length 512 \
  --batch_size 1 \
  --max_eval_chunks 4 \
  --weight_bits 4 \
  --group_size 128 \
  --eval_vlm \
  --vlm_datasets OCRBench MMStar \
  --vlm_judge exact_matching
```

常用参数：

- `--eval_vlm`: 打开多模态 benchmark 评测。
- `--vlm_datasets`: 指定 benchmark 名称，可传多个。
- `--vlm_mode`: `all / infer / eval`。
- `--vlm_judge`: 指定打分方式。
- `--vlm_work_dir`: 单独指定 VLMEvalKit 输出目录。
- `--vlm_eval_kit_root`: 指定 `VLMEvalKit` 源码目录。

### 2. 直接调用 `evaluate_vlm`

如果只是做模型连通性验证，直接调 Python 会更轻量，不必走量化或剪枝阶段。

示例：

```python
from algorithm.common.modeling import load_model_and_tokenizer
from evaluation.vlm_eval import evaluate_vlm

model_path = "<your_model_path>"
model, tokenizer_bundle = load_model_and_tokenizer(model_path, dtype="auto")

results = evaluate_vlm(
    model,
    tokenizer_bundle,
    {
        "model_path": model_path,
        "device": "cuda:0",
        "vlm_datasets": ["OCRBench"],
        "vlm_eval_kit_root": "<your_vlmevalkit_root>",
        "vlm_work_dir": "<your_output_dir>",
        "vlm_mode": "all",
        "vlm_judge": "exact_matching",
        "vlm_api_nproc": 4,
        "vlm_pred_format": "xlsx",
    },
)

print(results)
```

### 输出结果

评测完成后，通常会生成：

- 模型预测结果文件
- benchmark 分数文件
- 中间判分结果文件
