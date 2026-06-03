"""vLLM compressed-tensors exporter for real quantized weights."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .schema import RealQuantLinearArtifact


_INFERENCE_FILE_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.*",
    "merges.txt",
    "*.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template*.jinja",
    "generation_config*.json",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
    "feature_extractor_config.json",
    "configuration.json",
    "configuration*.py",
    "modeling*.py",
    "processing*.py",
    "tokenization*.py",
    "image_processing*.py",
    "video_processing*.py",
)

_WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
    "*.onnx",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class VllmPackedInt4Linear(nn.Module):
    """Export-only module carrying packed int4 weights and scales."""

    def __init__(
        self,
        *,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_packed", packed_weight.detach().cpu().contiguous())
        self.register_buffer("weight_scale", scale.detach().cpu().to(torch.float16).contiguous())
        self.register_buffer(
            "weight_shape",
            torch.tensor([out_features, in_features], dtype=torch.int64),
        )
        if bias is not None:
            self.register_buffer("bias", bias.detach().cpu().contiguous())
        else:
            self.bias = None
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    def forward(self, _x):  # pragma: no cover - export-only module
        raise NotImplementedError("VllmPackedInt4Linear is only used for model export.")


def pack_int4_for_vllm(int_weight: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 weights into int32 along the input-feature dimension."""

    if int_weight.ndim != 2:
        raise ValueError(f"Expected 2D int_weight, got shape {tuple(int_weight.shape)}.")
    if int_weight.numel() and (int_weight.min() < -8 or int_weight.max() > 7):
        raise ValueError("int4 weights must be in signed range [-8, 7].")

    weight = (int_weight.to(torch.int16) + 8).to(torch.uint8).cpu().numpy().astype(np.uint32)
    pack_factor = 8
    packed_cols = math.ceil(weight.shape[1] / pack_factor)
    padding = packed_cols * pack_factor - weight.shape[1]
    if padding:
        weight = np.pad(weight, pad_width=[(0, 0), (0, padding)], constant_values=0)

    packed = np.zeros((weight.shape[0], packed_cols), dtype=np.uint32)
    for offset in range(pack_factor):
        packed |= weight[:, offset::pack_factor] << (4 * offset)
    return torch.from_numpy(np.ascontiguousarray(packed).view(np.int32))


def _get_child(parent: nn.Module, name: str) -> nn.Module:
    if name in parent._modules:
        return parent._modules[name]
    return getattr(parent, name)


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if name in parent._modules:
        parent._modules[name] = child
    else:
        setattr(parent, name, child)


def _resolve_parent_module(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    if len(parts) < 1:
        raise ValueError("Empty module name.")
    parent = model
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    return parent, parts[-1]


def _build_compression_config(
    *,
    group_size: int,
    ignored_layers: list[str],
    quant_method: str = "compressed-tensors",
) -> dict[str, Any]:
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "weights": {
                    "dynamic": False,
                    "group_size": int(group_size),
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ignored_layers,
        "quant_method": quant_method,
    }


def _update_config_json(export_dir: Path, *, group_size: int, ignored_layers: list[str]) -> None:
    config_path = export_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.pop("quantization_config", None)
    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None and config.get("rope_theta") is None:
            config["rope_theta"] = rope_theta
    elif config.get("rope_theta") is not None:
        config["rope_parameters"] = {
            "rope_theta": config["rope_theta"],
            "rope_type": "default",
        }
    config["compression_config"] = _build_compression_config(
        group_size=group_size,
        ignored_layers=ignored_layers,
    )
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(Path(name).match(pattern) for pattern in patterns)


def _copy_inference_files_from_source(tokenizer_bundle, export_path: Path) -> list[str]:
    source_path = getattr(tokenizer_bundle, "source_path", None)
    if not source_path:
        return []

    source_dir = Path(source_path)
    if not source_dir.is_dir():
        return []

    copied: list[str] = []
    for source_file in sorted(source_dir.iterdir()):
        if not source_file.is_file():
            continue
        name = source_file.name
        if name == "config.json" or _matches_any(name, _WEIGHT_FILE_PATTERNS):
            continue
        if not _matches_any(name, _INFERENCE_FILE_PATTERNS):
            continue

        target_file = export_path / name
        if source_file.resolve() == target_file.resolve():
            continue
        shutil.copy2(source_file, target_file)
        copied.append(name)
    return copied


def export_vllm_gptq_w4a16(
    *,
    model: nn.Module,
    tokenizer_bundle,
    artifacts: dict[str, RealQuantLinearArtifact],
    export_dir: str | Path,
    group_size: int,
) -> dict[str, Any]:
    """Export collected GPTQ W4A16 artifacts in vLLM compressed-tensors format."""

    if not artifacts:
        raise ValueError("No real-quant Linear artifacts were provided for vLLM export.")

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    named_modules = dict(model.named_modules())
    quantized_names = set(artifacts)
    ignored_layers = sorted(
        name for name, module in named_modules.items()
        if isinstance(module, nn.Linear) and name not in quantized_names
    )

    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    copied_inference_files: list[str] = []
    try:
        for name, artifact in artifacts.items():
            module = named_modules.get(name)
            if module is None:
                raise KeyError(f"Cannot find quantized module {name!r} in model.")
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Module {name!r} is {type(module)!r}, expected nn.Linear.")
            if artifact.bits != 4 or not artifact.symmetric:
                raise ValueError(f"Only symmetric W4 artifacts are supported, got {artifact!r}.")
            if int(artifact.group_size) != int(group_size):
                raise ValueError(
                    f"Artifact {name!r} group_size={artifact.group_size}, expected {group_size}."
                )
            if tuple(artifact.original_shape) != tuple(module.weight.shape):
                raise ValueError(
                    f"Artifact {name!r} original_shape={artifact.original_shape}, "
                    f"but module weight shape is {tuple(module.weight.shape)}."
                )
            if tuple(artifact.int_weight.shape) != tuple(module.weight.shape):
                raise ValueError(
                    f"Artifact {name!r} int_weight shape={tuple(artifact.int_weight.shape)}, "
                    f"expected {tuple(module.weight.shape)}."
                )
            if module.in_features % int(group_size) != 0:
                raise ValueError(
                    f"vLLM compressed-tensors W4A16 requires in_features divisible by group_size; "
                    f"{name!r} has in_features={module.in_features}, group_size={group_size}."
                )
            expected_scale_shape = (module.out_features, module.in_features // int(group_size))
            if tuple(artifact.scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"Artifact {name!r} scale shape={tuple(artifact.scale.shape)}, "
                    f"expected {expected_scale_shape}."
                )

            parent, child_name = _resolve_parent_module(model, name)
            packed = pack_int4_for_vllm(artifact.int_weight)
            replacement = VllmPackedInt4Linear(
                packed_weight=packed,
                scale=artifact.scale,
                bias=module.bias.data if module.bias is not None else None,
                in_features=module.in_features,
                out_features=module.out_features,
            )
            replacements.append((parent, child_name, module))
            _set_child(parent, child_name, replacement)

        model.save_pretrained(export_path, safe_serialization=True)
        tokenizer_bundle.save_pretrained(str(export_path))
        copied_inference_files = _copy_inference_files_from_source(tokenizer_bundle, export_path)
        _update_config_json(export_path, group_size=group_size, ignored_layers=ignored_layers)
    finally:
        for parent, child_name, original in reversed(replacements):
            _set_child(parent, child_name, original)

    return {
        "backend": "vllm",
        "format": "compressed-tensors",
        "precision": "W4A16",
        "path": str(export_path),
        "quantized_linear_count": len(artifacts),
        "ignored_linear_count": len(ignored_layers),
        "ignored_layers": ignored_layers,
        "copied_inference_files": copied_inference_files,
    }
