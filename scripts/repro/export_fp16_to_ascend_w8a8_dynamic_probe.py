#!/usr/bin/env python3
"""Export a dense Qwen2/Qwen2.5 checkpoint to Ascend W8A8_DYNAMIC."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPECTED_MODEL_TYPE = "qwen2"
EXPECTED_ARCHITECTURE = "Qwen2ForCausalLM"
QUANT_TYPE = "W8A8_DYNAMIC"
FLOAT_TYPE = "FLOAT"
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
INFERENCE_FILE_NAMES = (
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="FP16/BF16 Hugging Face checkpoint directory"
    )
    parser.add_argument(
        "--output", required=True, help="output Ascend W8A8_DYNAMIC directory"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate an existing export without rewriting it",
    )
    return parser.parse_args()


def resolve_paths(source_arg: str, output_arg: str) -> tuple[Path, Path]:
    source = Path(source_arg).expanduser().resolve()
    output = Path(output_arg).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source checkpoint directory does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "--source and --output must be separate, non-nested directories: "
            f"source={source}, output={output}"
        )
    return source, output


def read_config(source: Path) -> dict[str, Any]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid JSON object in {config_path}")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"Expected dense Qwen2/Qwen2.5 model_type={EXPECTED_MODEL_TYPE!r}, "
            f"got {config.get('model_type')!r}."
        )
    architectures = config.get("architectures") or []
    if EXPECTED_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Expected architectures to contain {EXPECTED_ARCHITECTURE!r}, got "
            f"{architectures!r}."
        )
    try:
        num_layers = int(config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("config.json has no valid num_hidden_layers") from exc
    if num_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
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
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
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


def expected_quantized_weights(config: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for layer_idx in range(int(config["num_hidden_layers"])):
        base = f"model.layers.{layer_idx}"
        names.update(f"{base}.{suffix}.weight" for suffix in LINEAR_SUFFIXES)
    return names


def validate_source_layout(
    config: dict[str, Any], source_weight_map: dict[str, str]
) -> set[str]:
    quantized_weights = expected_quantized_weights(config)
    missing = sorted(quantized_weights - set(source_weight_map))
    if missing:
        raise ValueError(
            f"Source checkpoint is missing {len(missing)} required dense Qwen2 "
            f"Linear weights; first entries: {', '.join(missing[:8])}"
        )
    return quantized_weights


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


def quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional Linear weight, got {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise ValueError(f"Expected a floating-point source weight, got {weight.dtype}")
    weight_fp32 = weight.to(torch.float32)
    if not torch.isfinite(weight_fp32).all():
        raise ValueError("Source weight contains non-finite values")
    scale = weight_fp32.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-8)
    quantized = torch.round(weight_fp32 / scale).clamp(-127, 127).to(torch.int8)
    offset = torch.zeros_like(scale, dtype=weight.dtype)
    return (
        quantized.contiguous(),
        scale.to(weight.dtype).contiguous(),
        offset.contiguous(),
    )


def write_quant_description(
    config: dict[str, Any], output: Path, quantized_weights: set[str]
) -> dict[str, str]:
    description: dict[str, str] = {
        "model.embed_tokens.weight": FLOAT_TYPE,
        "lm_head.weight": FLOAT_TYPE,
    }
    for name in sorted(quantized_weights):
        description[name] = QUANT_TYPE
    for layer_idx in range(int(config["num_hidden_layers"])):
        base = f"model.layers.{layer_idx}"
        description[f"{base}.self_attn.qkv_proj.weight"] = QUANT_TYPE
        description[f"{base}.mlp.gate_up_proj.weight"] = QUANT_TYPE
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
) -> tuple[dict[str, str], int, int, int]:
    tensors: dict[str, torch.Tensor] = {}
    output_weight_map: dict[str, str] = {}
    converted = 0
    total_bytes = 0
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = sorted(set(tensor_names) - available)
        if missing:
            raise ValueError(
                f"{source_file.name} misses {len(missing)} indexed tensors; first: "
                f"{', '.join(missing[:8])}"
            )
        for name in sorted(tensor_names):
            tensor = handle.get_tensor(name)
            if name in quantized_weights:
                quantized, scale, offset = quantize_weight_per_channel(tensor)
                prefix = name.removesuffix(".weight")
                output_tensors = {
                    name: quantized,
                    f"{prefix}.weight_scale": scale,
                    f"{prefix}.weight_offset": offset,
                }
                converted += 1
            else:
                output_tensors = {name: tensor}
            for output_name, output_tensor in output_tensors.items():
                tensors[output_name] = output_tensor
                output_weight_map[output_name] = output_file.name
                total_bytes += output_tensor.numel() * output_tensor.element_size()
    save_file(tensors, output_file)
    output_tensor_count = len(tensors)
    del tensors
    gc.collect()
    return output_weight_map, converted, output_tensor_count, total_bytes


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


def get_output_weight_map(output: Path) -> tuple[dict[str, str], bool]:
    index_path = output / "model.safetensors.index.json"
    if not index_path.is_file():
        return _scan_output_weight_map(output), False
    index = json.loads(index_path.read_text(encoding="utf-8"))
    raw_weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError(f"Invalid output weight_map in {index_path}")
    weight_map: dict[str, str] = {}
    for tensor_name, raw_file_name in raw_weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"Invalid tensor name in {index_path}: {tensor_name!r}")
        weight_map[tensor_name] = _validate_shard_name(raw_file_name, index_path)
    return weight_map, True


def _safe_dtype(handle: Any, name: str) -> str:
    return str(handle.get_slice(name).get_dtype()).upper()


def _safe_shape(handle: Any, name: str) -> tuple[int, ...]:
    return tuple(handle.get_slice(name).get_shape())


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
                        f"Source weight {name} has dtype {dtype}, expected floating point"
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
    quantized_weights: set[str],
) -> dict[str, Any]:
    if not output.is_dir():
        raise FileNotFoundError(f"Output checkpoint directory does not exist: {output}")
    indexed_weight_map, has_index = get_output_weight_map(output)
    weight_map = _scan_output_weight_map(output) if has_index else indexed_weight_map
    source_names = {name for names in source_by_file.values() for name in names}
    expected_output_names = set(source_names)
    for name in quantized_weights:
        prefix = name.removesuffix(".weight")
        expected_output_names.add(f"{prefix}.weight_scale")
        expected_output_names.add(f"{prefix}.weight_offset")
    missing = sorted(expected_output_names - set(weight_map))
    if missing:
        raise ValueError(
            f"Output misses {len(missing)} required tensors; first: "
            f"{', '.join(missing[:8])}"
        )
    if has_index:
        missing_index_entries = sorted(expected_output_names - set(indexed_weight_map))
        if missing_index_entries:
            raise ValueError(
                f"Output index misses {len(missing_index_entries)} required tensors; "
                f"first: {', '.join(missing_index_entries[:8])}"
            )
        mismatched_index_entries = sorted(
            name
            for name in expected_output_names
            if indexed_weight_map[name] != weight_map[name]
        )
        if mismatched_index_entries:
            raise ValueError(
                f"Output index maps {len(mismatched_index_entries)} tensors to the "
                f"wrong shard; first: {', '.join(mismatched_index_entries[:8])}"
            )

    description_path = output / "quant_model_description.json"
    if not description_path.is_file():
        raise FileNotFoundError(f"Missing quantization description: {description_path}")
    description = json.loads(description_path.read_text(encoding="utf-8"))
    if not isinstance(description, dict):
        raise ValueError(f"Invalid quantization description: {description_path}")
    bad_description = sorted(
        name for name in quantized_weights if description.get(name) != QUANT_TYPE
    )
    if bad_description:
        raise ValueError(
            f"Description has {len(bad_description)} invalid W8A8_DYNAMIC entries; "
            f"first: {', '.join(bad_description[:8])}"
        )
    float_terminal_weights = {
        "model.embed_tokens.weight",
        "lm_head.weight",
    } & source_names
    bad_float_description = sorted(
        name for name in float_terminal_weights if description.get(name) != FLOAT_TYPE
    )
    if bad_float_description:
        raise ValueError(
            "Description does not keep terminal weights FLOAT: "
            + ", ".join(bad_float_description)
        )

    grouped: dict[str, set[str]] = {}
    for name in quantized_weights:
        shard_name = weight_map[name]
        grouped.setdefault(shard_name, set()).add(name)
    source_dtypes = _source_quantized_dtypes(
        source, source_by_file, quantized_weights
    )

    checked = 0
    for shard_name, names in sorted(grouped.items()):
        shard_path = output / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Output shard missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in sorted(names):
                if name not in available:
                    raise ValueError(f"Output shard {shard_name} is missing {name}")
                dtype = _safe_dtype(handle, name)
                if dtype not in {"I8", "INT8"}:
                    raise ValueError(f"{name} has dtype {dtype}, expected int8")
                weight_shape = _safe_shape(handle, name)
                if len(weight_shape) != 2:
                    raise ValueError(f"{name} has invalid shape {weight_shape}")
                prefix = name.removesuffix(".weight")
                for suffix in ("weight_scale", "weight_offset"):
                    parameter_name = f"{prefix}.{suffix}"
                    if parameter_name not in available:
                        raise ValueError(f"{name} is missing {parameter_name}")
                    parameter_dtype = _safe_dtype(handle, parameter_name)
                    expected_dtype = source_dtypes[name]
                    if parameter_dtype != expected_dtype:
                        raise ValueError(
                            f"{parameter_name} has dtype {parameter_dtype}, "
                            f"expected source weight dtype {expected_dtype}"
                        )
                    parameter_shape = _safe_shape(handle, parameter_name)
                    if parameter_shape != (weight_shape[0], 1):
                        raise ValueError(
                            f"{parameter_name} has shape {parameter_shape}, expected "
                            f"({weight_shape[0]}, 1)"
                        )
                checked += 1

    for name in source_names:
        shard_name = weight_map.get(name)
        if shard_name is None or not (output / shard_name).is_file():
            raise ValueError(f"Output source tensor or shard is missing: {name}")
    if checked != len(quantized_weights):
        raise AssertionError(
            f"Validated {checked} quantized weights, expected {len(quantized_weights)}"
        )
    for name in sorted(float_terminal_weights):
        shard_path = output / weight_map[name]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            dtype = _safe_dtype(handle, name)
            if dtype not in {"BF16", "F16", "F32", "F64"}:
                raise ValueError(f"{name} has dtype {dtype}, expected floating point")
    return {
        "validated_linear_weights": checked,
        "validated_scale_offset_pairs": checked,
        "validated_float_terminal_weights": len(float_terminal_weights),
        "all_linear_weights_int8": True,
        "all_scale_offset_shapes_valid": True,
        "all_scale_offset_dtypes_match_source": True,
        "output_index_present": has_index,
    }


def main() -> int:
    args = parse_args()
    source, output = resolve_paths(args.source, args.output)
    config = read_config(source)
    source_by_file, source_weight_map = get_weight_index(source)
    quantized_weights = validate_source_layout(config, source_weight_map)

    if args.verify_only:
        report = validate_output(output, source, source_by_file, quantized_weights)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    copy_inference_files(source, output, config)
    description = write_quant_description(config, output, quantized_weights)

    output_weight_map: dict[str, str] = {}
    converted_modules = 0
    output_tensors = 0
    total_bytes = 0
    for file_name, tensor_names in sorted(source_by_file.items()):
        shard_weight_map, converted, tensor_count, shard_bytes = convert_shard(
            source / file_name,
            tensor_names,
            output / file_name,
            quantized_weights,
        )
        output_weight_map.update(shard_weight_map)
        converted_modules += converted
        output_tensors += tensor_count
        total_bytes += shard_bytes
    write_index(output, output_weight_map, total_bytes)
    validation = validate_output(output, source, source_by_file, quantized_weights)

    summary = {
        "source": str(source),
        "output": str(output),
        "model_type": config["model_type"],
        "architecture": EXPECTED_ARCHITECTURE,
        "converted_linear_modules": converted_modules,
        "output_tensors": output_tensors,
        "quant_description_entries": len(description),
        "format": "ascend_w8a8_dynamic_probe_from_fp16",
        "validation": validation,
        "notes": (
            "Dense transformer block Linear weights are W8A8_DYNAMIC; "
            "embed_tokens and lm_head remain FLOAT."
        ),
    }
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
