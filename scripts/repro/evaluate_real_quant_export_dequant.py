#!/usr/bin/env python3
"""Evaluate a vLLM compressed-tensors W4A16 export by dequantizing into HF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.common.io import ensure_dir
from algorithm.common.io import write_json
from algorithm.common.modeling import load_model_and_tokenizer
from evaluation.runner import run_evaluations


def _bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _unpack_int4(packed: torch.Tensor, original_shape: tuple[int, int]) -> torch.Tensor:
    if packed.dtype != torch.int32:
        raise TypeError(f"Expected int32 packed tensor, got {packed.dtype}.")
    rows, cols = original_shape
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * 8), dtype=torch.int16)
    for offset in range(8):
        unpacked[:, offset::8] = ((packed >> (4 * offset)) & 0xF).to(torch.int16)
    return (unpacked[:, :cols] - 8).to(torch.int8).reshape(rows, cols)


def _dequantize_weight(
    *,
    packed: torch.Tensor,
    scale: torch.Tensor,
    weight_shape: torch.Tensor,
) -> torch.Tensor:
    original_shape = tuple(int(value) for value in weight_shape.tolist())
    if len(original_shape) != 2:
        raise ValueError(f"Expected 2D weight_shape, got {original_shape}.")
    int_weight = _unpack_int4(packed.cpu(), original_shape).float()
    if scale.ndim != 2:
        raise ValueError(f"Expected 2D scale tensor, got shape {tuple(scale.shape)}.")
    if original_shape[1] % scale.shape[1] != 0:
        raise ValueError(
            f"Cannot infer group size for shape={original_shape}, scale={tuple(scale.shape)}."
        )
    group_size = original_shape[1] // int(scale.shape[1])
    expanded_scale = scale.float().repeat_interleave(group_size, dim=1)
    return int_weight * expanded_scale[:, : original_shape[1]]


@torch.no_grad()
def load_dequantized_export(model, export_dir: Path) -> dict[str, int]:
    safetensors_files = sorted(export_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found under {export_dir}.")

    modules = dict(model.named_modules())
    loaded = 0
    missing_modules = 0
    for path in safetensors_files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            packed_names = sorted(
                name for name in handle.keys() if name.endswith(".weight_packed")
            )
            for packed_name in packed_names:
                module_name = packed_name.removesuffix(".weight_packed")
                module = modules.get(module_name)
                if module is None or not isinstance(module, nn.Linear):
                    missing_modules += 1
                    continue
                packed = handle.get_tensor(packed_name)
                scale = handle.get_tensor(f"{module_name}.weight_scale")
                weight_shape = handle.get_tensor(f"{module_name}.weight_shape")
                dequantized = _dequantize_weight(
                    packed=packed,
                    scale=scale,
                    weight_shape=weight_shape,
                )
                if tuple(module.weight.shape) != tuple(dequantized.shape):
                    raise ValueError(
                        f"{module_name} shape mismatch: model={tuple(module.weight.shape)}, "
                        f"export={tuple(dequantized.shape)}."
                    )
                module.weight.data.copy_(
                    dequantized.to(
                        device=module.weight.device, dtype=module.weight.dtype
                    )
                )
                loaded += 1
    return {"dequantized_linear_count": loaded, "missing_module_count": missing_modules}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"]
    )
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=None)
    parser.add_argument("--eval_ppl", type=_bool_flag, default=True)
    parser.add_argument("--eval_zero_shot", type=_bool_flag, default=False)
    parser.add_argument(
        "--zero_shot_tasks",
        nargs="+",
        default=[
            "boolq",
            "rte",
            "winogrande",
            "arc_easy",
            "arc_challenge",
            "openbookqa",
        ],
    )
    parser.add_argument("--zero_shot_num_fewshot", type=int, default=0)
    parser.add_argument("--zero_shot_batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    model, tokenizer_bundle = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    export_stats = load_dequantized_export(model, Path(args.export_dir))
    common_args = vars(args).copy()
    common_args["evaluation_output_dir"] = str(output_dir)
    metrics = run_evaluations(
        model=model,
        tokenizer_bundle=tokenizer_bundle,
        common_args=common_args,
    )
    metrics.update(
        {
            "model_path": args.model_path,
            "export_dir": args.export_dir,
            "device": args.device,
            "dtype": args.dtype,
            "dequantized_export": export_stats,
        }
    )
    metrics_path = write_json(output_dir / "metrics.json", metrics)
    print(json.dumps({"metrics_path": str(metrics_path), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
