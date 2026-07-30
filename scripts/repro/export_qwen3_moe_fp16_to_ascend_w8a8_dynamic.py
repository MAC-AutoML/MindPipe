#!/usr/bin/env python3
"""Export Qwen3-30B-A3B FP16/BF16 weights to Ascend W8A8_DYNAMIC.

The conversion is deliberately model-specific. Qwen3 MoE experts are packed
by the vLLM Ascend loader, so every source expert projection must retain its
original name and have a matching per-channel scale and offset tensor.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPECTED_MODEL_TYPE = "qwen3_moe"
EXPECTED_ARCHITECTURE = "Qwen3MoeForCausalLM"
QUANT_TYPE = "W8A8_DYNAMIC"
FLOAT_TYPE = "FLOAT"
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
INFERENCE_FILE_NAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source Hugging Face checkpoint")
    parser.add_argument("--output", required=True, help="output W8A8_DYNAMIC checkpoint")
    parser.add_argument(
        "--source-revision",
        default="unknown",
        help="immutable source revision recorded in conversion_summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate a previously exported checkpoint without rewriting it",
    )
    return parser.parse_args()


def validate_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source checkpoint directory does not exist: {source}")
    if source == output:
        raise ValueError("--source and --output must be different directories")
    if source in output.parents or output in source.parents:
        raise ValueError("--source and --output must not contain one another")


def read_config(source: Path) -> dict[str, Any]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"Expected model_type={EXPECTED_MODEL_TYPE!r}, got "
            f"{config.get('model_type')!r}."
        )
    if EXPECTED_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Expected architectures to contain {EXPECTED_ARCHITECTURE!r}, got "
            f"{architectures!r}."
        )
    return config


def _validate_shard_name(file_name: object, index_path: Path) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"Invalid shard name in {index_path}: {file_name!r}")
    if Path(file_name).name != file_name:
        raise ValueError(f"Unsafe shard name in {index_path}: {file_name!r}")
    return file_name


def get_weight_index(source: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")
        by_file: dict[str, list[str]] = {}
        normalized_map: dict[str, str] = {}
        for tensor_name, raw_file_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError(f"Invalid tensor name in {index_path}: {tensor_name!r}")
            file_name = _validate_shard_name(raw_file_name, index_path)
            by_file.setdefault(file_name, []).append(tensor_name)
            normalized_map[tensor_name] = file_name
        for file_name in by_file:
            if not (source / file_name).is_file():
                raise FileNotFoundError(f"Source shard missing: {source / file_name}")
        return by_file, normalized_map

    single = source / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint found in {source}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        names = list(handle.keys())
    return {single.name: names}, {name: single.name for name in names}


def expected_weight_names(config: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    layers = int(config["num_hidden_layers"])
    experts = int(config["num_experts"])
    attention: set[str] = set()
    expert_weights: set[str] = set()
    routers: set[str] = set()
    for layer_idx in range(layers):
        layer_prefix = f"model.layers.{layer_idx}"
        attention.update(
            f"{layer_prefix}.self_attn.{projection}.weight"
            for projection in ATTENTION_PROJECTIONS
        )
        routers.add(f"{layer_prefix}.mlp.gate.weight")
        for expert_idx in range(experts):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert_idx}"
            expert_weights.update(
                f"{expert_prefix}.{projection}.weight"
                for projection in EXPERT_PROJECTIONS
            )
    return attention, expert_weights, routers


def validate_source_layout(
    config: dict[str, Any], source_weight_map: dict[str, str]
) -> tuple[set[str], set[str], set[str]]:
    attention, expert_weights, routers = expected_weight_names(config)
    required = attention | expert_weights | routers
    missing = sorted(required - set(source_weight_map))
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"Source checkpoint is missing {len(missing)} required Qwen3 MoE "
            f"weights; first entries: {preview}"
        )
    expected_expert_count = (
        int(config["num_hidden_layers"])
        * int(config["num_experts"])
        * len(EXPERT_PROJECTIONS)
    )
    if len(expert_weights) != expected_expert_count:
        raise AssertionError(
            f"Expected {expected_expert_count} expert projections, found "
            f"{len(expert_weights)}."
        )
    return attention, expert_weights, routers


def copy_inference_files(source: Path, output: Path, config: dict[str, Any]) -> None:
    for name in INFERENCE_FILE_NAMES:
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / name)
    clean_config = dict(config)
    clean_config.pop("compression_config", None)
    clean_config.pop("quantization_config", None)
    (output / "config.json").write_text(
        json.dumps(clean_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def quantize_per_output_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a two-dimensional Linear weight, got {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"Expected a floating-point source weight, got {weight.dtype}")
    fp32 = weight.to(torch.float32)
    scale = fp32.abs().amax(dim=1, keepdim=True) / 127.0
    if not torch.isfinite(scale).all():
        raise ValueError("Source weight contains non-finite values.")
    scale = torch.clamp(scale, min=1e-8)
    quantized = torch.round(fp32 / scale).clamp(-127, 127).to(torch.int8)
    offset = torch.zeros_like(scale, dtype=weight.dtype)
    return (
        quantized.contiguous(),
        scale.to(weight.dtype).contiguous(),
        offset.contiguous(),
    )


def write_quant_description(
    output: Path,
    config: dict[str, Any],
    attention: set[str],
    expert_weights: set[str],
    routers: set[str],
) -> dict[str, str]:
    description: dict[str, str] = {
        "model.embed_tokens.weight": FLOAT_TYPE,
        "lm_head.weight": FLOAT_TYPE,
    }
    for name in sorted(attention | expert_weights):
        description[name] = QUANT_TYPE
    for name in sorted(routers):
        description[name] = FLOAT_TYPE

    for layer_idx in range(int(config["num_hidden_layers"])):
        attn_prefix = f"model.layers.{layer_idx}.self_attn"
        description[f"{attn_prefix}.qkv_proj.weight"] = QUANT_TYPE

    expected_experts = len(expert_weights)
    actual_experts = sum(
        value == QUANT_TYPE and ".mlp.experts." in name
        for name, value in description.items()
    )
    if actual_experts != expected_experts:
        raise AssertionError(
            f"Description contains {actual_experts} quantized expert weights, "
            f"expected {expected_experts}."
        )
    (output / "quant_model_description.json").write_text(
        json.dumps(description, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return description


def convert_shard(
    source_file: Path,
    tensor_names: list[str],
    output_file: Path,
    quantized_weights: set[str],
) -> tuple[dict[str, str], Counter[str], int]:
    tensors: dict[str, torch.Tensor] = {}
    output_weight_map: dict[str, str] = {}
    precision_counts: Counter[str] = Counter()
    total_bytes = 0
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        for name in sorted(tensor_names):
            tensor = handle.get_tensor(name)
            if name in quantized_weights:
                quantized, scale, offset = quantize_per_output_channel(tensor)
                output_tensors = {
                    name: quantized,
                    f"{name}_scale": scale,
                    f"{name}_offset": offset,
                }
                precision_counts["W8A8_DYNAMIC"] += 1
            else:
                output_tensors = {name: tensor}
                precision_counts[str(tensor.dtype)] += 1
            for output_name, output_tensor in output_tensors.items():
                tensors[output_name] = output_tensor
                output_weight_map[output_name] = output_file.name
                total_bytes += output_tensor.numel() * output_tensor.element_size()
    save_file(tensors, output_file)
    del tensors
    gc.collect()
    return output_weight_map, precision_counts, total_bytes


def _safe_dtype(handle: Any, name: str) -> str:
    return str(handle.get_slice(name).get_dtype()).upper()


def _safe_shape(handle: Any, name: str) -> tuple[int, ...]:
    return tuple(handle.get_slice(name).get_shape())


def _scan_output_weight_map(output: Path) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    shard_paths = sorted(output.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No output safetensors checkpoint found in {output}")
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in weight_map:
                    raise ValueError(f"Duplicate output tensor across shards: {name}")
                weight_map[name] = shard_path.name
    return weight_map


def _source_quantized_dtypes(
    source: Path,
    source_by_file: dict[str, list[str]],
    quantized_weights: set[str],
) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for shard_name, tensor_names in sorted(source_by_file.items()):
        expected = quantized_weights.intersection(tensor_names)
        if not expected:
            continue
        shard_path = source / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Source shard missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(expected - available)
            if missing:
                raise ValueError(
                    f"Source shard {shard_name} misses {len(missing)} quantized tensors; "
                    f"first: {', '.join(missing[:8])}"
                )
            for name in expected:
                dtype = _safe_dtype(handle, name)
                if dtype not in FLOAT_DTYPES:
                    raise ValueError(
                        f"Source weight {name} has dtype {dtype}, expected floating point."
                    )
                dtypes[name] = dtype
    missing_dtypes = sorted(quantized_weights - set(dtypes))
    if missing_dtypes:
        raise ValueError(
            f"Could not resolve source dtype for {len(missing_dtypes)} quantized tensors; "
            f"first: {', '.join(missing_dtypes[:8])}"
        )
    return dtypes


def validate_output(
    output: Path,
    source: Path,
    source_by_file: dict[str, list[str]],
    config: dict[str, Any],
    attention: set[str],
    expert_weights: set[str],
    routers: set[str],
) -> dict[str, Any]:
    if not output.is_dir():
        raise FileNotFoundError(f"Output checkpoint directory does not exist: {output}")
    index_path = output / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing output index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    raw_weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError(f"Invalid output weight_map in {index_path}")
    indexed_weight_map: dict[str, str] = {}
    for tensor_name, raw_file_name in raw_weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"Invalid tensor name in {index_path}: {tensor_name!r}")
        indexed_weight_map[tensor_name] = _validate_shard_name(
            raw_file_name, index_path
        )
    physical_weight_map = _scan_output_weight_map(output)

    source_names = {name for names in source_by_file.values() for name in names}
    expected_output_names = set(source_names)
    for name in attention | expert_weights:
        expected_output_names.add(f"{name}_scale")
        expected_output_names.add(f"{name}_offset")
    missing_index_entries = sorted(expected_output_names - set(indexed_weight_map))
    if missing_index_entries:
        raise ValueError(
            f"Output index misses {len(missing_index_entries)} tensors; first: "
            f"{', '.join(missing_index_entries[:8])}"
        )
    unexpected_index_entries = sorted(set(indexed_weight_map) - expected_output_names)
    if unexpected_index_entries:
        raise ValueError(
            f"Output index contains {len(unexpected_index_entries)} unexpected tensors; "
            f"first: {', '.join(unexpected_index_entries[:8])}"
        )
    missing_physical_tensors = sorted(expected_output_names - set(physical_weight_map))
    if missing_physical_tensors:
        raise ValueError(
            f"Physical checkpoint misses {len(missing_physical_tensors)} indexed tensors; "
            f"first: {', '.join(missing_physical_tensors[:8])}"
        )
    unexpected_physical_tensors = sorted(
        set(physical_weight_map) - expected_output_names
    )
    if unexpected_physical_tensors:
        raise ValueError(
            f"Physical checkpoint contains {len(unexpected_physical_tensors)} "
            f"unexpected tensors; first: {', '.join(unexpected_physical_tensors[:8])}"
        )
    mismatched_index_entries = sorted(
        name
        for name in expected_output_names
        if indexed_weight_map[name] != physical_weight_map[name]
    )
    if mismatched_index_entries:
        raise ValueError(
            f"Output index maps {len(mismatched_index_entries)} tensors to the "
            f"wrong shard; first: {', '.join(mismatched_index_entries[:8])}"
        )

    description_path = output / "quant_model_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    expected_quantized = attention | expert_weights
    bad_description = sorted(
        name for name in expected_quantized if description.get(name) != QUANT_TYPE
    )
    if bad_description:
        raise ValueError(
            f"Description has {len(bad_description)} invalid dynamic entries; first: "
            f"{', '.join(bad_description[:8])}"
        )
    bad_float = sorted(name for name in routers if description.get(name) != FLOAT_TYPE)
    if bad_float:
        raise ValueError(
            f"Description has {len(bad_float)} invalid router FLOAT entries; first: "
            f"{', '.join(bad_float[:8])}"
        )

    expected_expert_count = (
        int(config["num_hidden_layers"])
        * int(config["num_experts"])
        * len(EXPERT_PROJECTIONS)
    )
    if len(expert_weights) != expected_expert_count:
        raise AssertionError("Unexpected expert projection count during validation.")

    grouped_expected: dict[str, set[str]] = {}
    for name in expected_quantized:
        shard = physical_weight_map[name]
        grouped_expected.setdefault(shard, set()).add(name)
    source_dtypes = _source_quantized_dtypes(
        source, source_by_file, expected_quantized
    )

    checked_experts = 0
    checked_attention = 0
    for shard_name, names in sorted(grouped_expected.items()):
        shard_path = output / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Output shard missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = names - available
            if missing:
                raise ValueError(
                    f"Shard {shard_name} misses {len(missing)} quantized tensors; "
                    f"first: {', '.join(sorted(missing)[:8])}"
                )
            for name in names:
                weight_dtype = _safe_dtype(handle, name)
                if weight_dtype not in {"I8", "INT8"}:
                    raise ValueError(f"{name} has dtype {weight_dtype}, expected int8.")
                weight_shape = _safe_shape(handle, name)
                if len(weight_shape) != 2:
                    raise ValueError(f"{name} has invalid shape {weight_shape}.")
                for suffix in ("_scale", "_offset"):
                    parameter_name = f"{name}{suffix}"
                    if parameter_name not in available:
                        raise ValueError(f"{name} is missing {parameter_name}.")
                    parameter_dtype = _safe_dtype(handle, parameter_name)
                    expected_dtype = source_dtypes[name]
                    if parameter_dtype != expected_dtype:
                        raise ValueError(
                            f"{parameter_name} has dtype {parameter_dtype}, "
                            f"expected source weight dtype {expected_dtype}."
                        )
                    parameter_shape = _safe_shape(handle, parameter_name)
                    if parameter_shape != (weight_shape[0], 1):
                        raise ValueError(
                            f"{parameter_name} has shape {parameter_shape}, expected "
                            f"({weight_shape[0]}, 1)."
                        )
                if name in expert_weights:
                    checked_experts += 1
                else:
                    checked_attention += 1

    if checked_experts != expected_expert_count:
        raise AssertionError(
            f"Validated {checked_experts} expert projections, expected "
            f"{expected_expert_count}."
        )
    return {
        "validated_expert_projection_weights": checked_experts,
        "validated_attention_projection_weights": checked_attention,
        "validated_scale_offset_pairs": checked_experts + checked_attention,
        "all_expert_weights_int8": True,
        "all_expert_scale_offset_shapes_valid": True,
        "all_scale_offset_dtypes_match_source": True,
    }


def write_index(output: Path, weight_map: dict[str, str], total_bytes: int) -> None:
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_bytes},
                "weight_map": dict(sorted(weight_map.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    validate_paths(source, output)
    config = read_config(source)
    source_by_file, source_weight_map = get_weight_index(source)
    attention, expert_weights, routers = validate_source_layout(config, source_weight_map)

    if args.verify_only:
        report = validate_output(
            output, source, source_by_file, config, attention, expert_weights, routers
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    copy_inference_files(source, output, config)
    description = write_quant_description(
        output, config, attention, expert_weights, routers
    )

    quantized_weights = attention | expert_weights
    output_weight_map: dict[str, str] = {}
    precision_counts: Counter[str] = Counter()
    total_bytes = 0
    for file_name, tensor_names in sorted(source_by_file.items()):
        source_file = source / file_name
        if not source_file.is_file():
            raise FileNotFoundError(f"Source shard missing: {source_file}")
        shard_weight_map, shard_counts, shard_size = convert_shard(
            source_file,
            tensor_names,
            output / file_name,
            quantized_weights,
        )
        output_weight_map.update(shard_weight_map)
        precision_counts.update(shard_counts)
        total_bytes += shard_size

    write_index(output, output_weight_map, total_bytes)
    validation = validate_output(
        output, source, source_by_file, config, attention, expert_weights, routers
    )
    summary = {
        "source": str(source),
        "output": str(output),
        "source_revision": args.source_revision,
        "format": "ascend_w8a8_dynamic_qwen3_moe_from_fp16",
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_experts": int(config["num_experts"]),
        "expert_projection_weights": len(expert_weights),
        "attention_projection_weights": len(attention),
        "router_weights_left_float": len(routers),
        "quant_description_entries": len(description),
        "output_tensor_count": len(output_weight_map),
        "output_total_bytes": total_bytes,
        "precision_tensor_counts": dict(sorted(precision_counts.items())),
        "quantized_weight_type": QUANT_TYPE,
        "float_modules": ["router", "embed_tokens", "lm_head", "norm"],
        "validation": validation,
    }
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
