"""Fixed-mask compression-aware LoRA for transform-based quantizers."""

from __future__ import annotations

import logging
import math
import os
import random
import errno
import types
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from algorithm.common.datasets import get_calibration_and_evaluation_data
from algorithm.common.device import empty_cache
from algorithm.common.io import ensure_dir
from algorithm.common.io import model_slug
from algorithm.common.io import write_json
from algorithm.common.modeling import load_model_and_tokenizer
from algorithm.common.runtime import prepend_python_path
from algorithm.finetuning.base import BaseFinetuningMethod
from algorithm.quantization.qat.flatquant.method import FlatQuantMethod
from algorithm.quantization.qat.flatquant.method import _infer_direct_inv_from_checkpoint
from algorithm.quantization.qat.flatquant.method import _purge_conflicting_modules
from algorithm.quantization.qat.splitquant.method import SplitQuantMethod
from .collators import RawTextCPTCollator
from .collators import MiniCPMVImageTextSFTCollator
from .collators import QwenVLImageTextSFTCollator
from .collators import TextSFTCollator
from .data import build_raw_text_jsonl_dataset
from .flatquant_linear import CompressionLoRAFlatQuantLinear
from .flatquant_linear import LoRAConfig
from .splitquant_linear import CompressionLoRASplitQuantLinear
from .packed_moe import PackedCompressionLoRAExperts
from .packed_moe import apply_packed_moe_masks
from .mask_utils import load_masks
from .mask_utils import mask_sparsity
from .mask_utils import apply_masks_to_model
from .mask_utils import validate_masks
from .run_spec import compression_lora_run_spec
from .run_spec import parse_compression_lora_train_plan
from .sft_data import build_alpaca_sft_dataset
from .sft_data import build_llava_sft_dataset


LOGGER = logging.getLogger(__name__)


def _is_flatquant_linear(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "FlatQuantizedLinear" and hasattr(module, "linear")


def _is_splitquant_linear(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "SplitQuantizedLinear" and hasattr(module, "linear")


_COMPRESSION_LORA_LINEAR_TYPES = (
    CompressionLoRAFlatQuantLinear,
    CompressionLoRASplitQuantLinear,
    PackedCompressionLoRAExperts,
)


def _is_compression_lora_linear(module: torch.nn.Module) -> bool:
    return isinstance(module, _COMPRESSION_LORA_LINEAR_TYPES)


def _matches_lora_target(name: str, target_modules) -> bool:
    targets = set(target_modules or [])
    if not targets:
        return True
    # Keep the public target names shared by Qwen/Llama recipes while allowing
    # Mixtral's native expert projections to use their w1/w3/w2 names.
    aliases = {
        "w1": "gate_proj",
        "w3": "up_proj",
        "w2": "down_proj",
    }
    for target in targets:
        if name == target or name.endswith(f".{target}"):
            return True
        suffix = name.rsplit(".", 1)[-1]
        if ".experts." in name and aliases.get(suffix) == target:
            return True
    return False


def _set_child_module(root: torch.nn.Module, qualified_name: str, replacement: torch.nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _collect_adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_compression_lora_linear(module):
            continue
        if isinstance(module, PackedCompressionLoRAExperts):
            for param_name in module.adapter_parameter_names():
                state[f"{name}.{param_name}"] = getattr(module, param_name).detach().cpu()
            continue
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
        if not isinstance(module, PackedCompressionLoRAExperts) and module.adapter_type == "dora":
            state[f"{name}.dora_magnitude"] = module.dora_magnitude.detach().cpu()
    return state


def _assert_finite_adapter_state(adapter_state: dict[str, torch.Tensor]) -> None:
    nonfinite = [
        name
        for name, tensor in adapter_state.items()
        if not torch.isfinite(tensor).all().item()
    ]
    if nonfinite:
        preview = ", ".join(nonfinite[:8])
        raise RuntimeError(
            "Compression LoRA produced non-finite adapter parameters. "
            f"Bad tensors: {preview}"
        )


def _load_adapter_state(model: torch.nn.Module, adapter_state: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    loaded_keys = set(adapter_state)
    for key, tensor in adapter_state.items():
        module_name, param_name = key.rsplit(".", maxsplit=1)
        module = module_map.get(module_name)
        if not _is_compression_lora_linear(module):
            raise KeyError(f"Adapter target {module_name!r} is not a compression LoRA wrapper.")
        if isinstance(module, PackedCompressionLoRAExperts):
            if param_name not in module.adapter_parameter_names():
                raise KeyError(f"Unknown packed expert adapter parameter {key!r}.")
        elif param_name == "dora_magnitude" and module.adapter_type != "dora":
            raise ValueError(f"Adapter contains DoRA magnitude for non-DoRA module {module_name!r}.")
        param = getattr(module, param_name)
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
    missing = []
    for name, module in model.named_modules():
        if not _is_compression_lora_linear(module):
            continue
        parameter_names = module.adapter_parameter_names() if isinstance(
            module, PackedCompressionLoRAExperts
        ) else ("lora_A", "lora_B")
        for param_name in parameter_names:
            key = f"{name}.{param_name}"
            if key not in loaded_keys:
                missing.append(key)
        if not isinstance(module, PackedCompressionLoRAExperts) and module.adapter_type == "dora":
            key = f"{name}.dora_magnitude"
            if key not in loaded_keys:
                missing.append(key)
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"Adapter checkpoint is missing required parameter(s): {preview}")


def _iter_adapter_parameters(module: torch.nn.Module):
    if isinstance(module, PackedCompressionLoRAExperts):
        yield from module.adapter_parameters()
        return
    yield module.lora_A
    yield module.lora_B
    if module.adapter_type == "dora":
        yield module.dora_magnitude


def _named_adapter_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    named: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if not _is_compression_lora_linear(module):
            continue
        if isinstance(module, PackedCompressionLoRAExperts):
            parameter_names = module.adapter_parameter_names()
        else:
            parameter_names = ["lora_A", "lora_B"]
            if module.adapter_type == "dora":
                parameter_names.append("dora_magnitude")
        for parameter_name in parameter_names:
            parameter = getattr(module, parameter_name)
            if parameter.requires_grad:
                prefix = f"{module_name}." if module_name else ""
                named.append((f"{prefix}{parameter_name}", parameter))
    return named


def _parse_train_plan(plan: str | None) -> list[str]:
    return parse_compression_lora_train_plan(plan)


def _save_adapter(
    path: Path,
    model: torch.nn.Module,
    args,
    masks_path: str,
    stage: str,
    train_metrics: dict[str, Any],
) -> Path:
    adapter_state = _collect_adapter_state(model)
    _assert_finite_adapter_state(adapter_state)
    torch.save(
        {
            "metadata": {
                "rank": int(args.compression_lora_rank),
                "alpha": float(args.compression_lora_alpha),
                "dropout": float(args.compression_lora_dropout),
                "init": args.compression_lora_init,
                "weight_checkpointing": bool(getattr(args, "compression_lora_weight_checkpointing", False)),
                "adapter_type": getattr(args, "compression_lora_adapter_type", "lora"),
                "dora_simple": bool(getattr(args, "compression_lora_dora_simple", True)),
                "dora_eps": float(getattr(args, "compression_lora_dora_eps", 1e-6)),
                "lr_scheduler_type": getattr(args, "compression_lora_lr_scheduler_type", "cosine"),
                "warmup_ratio": float(getattr(args, "compression_lora_warmup_ratio", 0.03)),
                "resume_from": getattr(args, "compression_lora_resume_from", None),
                "sft_min_response_tokens": int(getattr(args, "compression_lora_sft_min_response_tokens", 8)),
                "quantization": args.quantization,
                "pruning": args.pruning,
                "masks_path": str(masks_path),
                "target_modules": list(args.compression_lora_target_modules),
                "stage": stage,
                "train_metrics": train_metrics,
            },
            "state_dict": adapter_state,
        },
        path,
    )
    return path


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Compression LoRA checkpoint CUDA device count mismatch: "
                f"checkpoint={len(cuda_states)} current={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rotate_training_checkpoints(checkpoint_dir: Path, save_total_limit: int) -> None:
    limit = int(save_total_limit)
    if limit <= 0:
        return
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*.pth"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
    )
    for stale in checkpoints[:-limit]:
        stale.unlink()


def _save_training_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    stage_name: str,
    global_step: int,
    micro_step: int,
    epoch: int,
    sample_offset: int,
    dataset_size: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
    run_start_step: int,
    run_steps: int,
    data_seed: int,
    parameter_names: list[str],
    completed: bool,
    metadata: dict[str, Any],
) -> Path:
    _atomic_torch_save(
        {
            "format": "mindpipe.compression_lora.training_checkpoint.v1",
            "metadata": dict(metadata),
            "adapter_state_dict": _collect_adapter_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
            "training_state": {
                "stage": stage_name,
                "global_step": int(global_step),
                "micro_step": int(micro_step),
                "epoch": int(epoch),
                "sample_offset": int(sample_offset),
                "dataset_size": int(dataset_size),
                "batch_size": int(batch_size),
                "grad_accum": int(grad_accum),
                "max_steps": int(max_steps),
                "run_start_step": int(run_start_step),
                "run_steps": int(run_steps),
                "data_seed": int(data_seed),
                "parameter_names": list(parameter_names),
                "completed": bool(completed),
            },
        },
        path,
    )
    return path


def _load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    retryable_errors = {errno.EBUSY, getattr(errno, "ESTALE", 116)}
    for attempt in range(6):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            break
        except OSError as exc:
            if exc.errno not in retryable_errors or attempt == 5:
                raise
            time.sleep(2**attempt)
    if not isinstance(payload, dict) or payload.get("format") != "mindpipe.compression_lora.training_checkpoint.v1":
        raise ValueError(
            "compression_lora_resume_from must point to a MindPipe compression LoRA training checkpoint; "
            "the final adapter-only file cannot restore optimizer and data state."
        )
    required = {"adapter_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict", "training_state"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"Compression LoRA training checkpoint is missing fields: {missing}")
    return payload


def _patch_minicpm_v_vision_embedding_dtype(model: torch.nn.Module) -> None:
    source_model = getattr(model, "_source_model", model)
    if getattr(source_model, "_mindpipe_patched_vision_embedding_dtype", False):
        return
    if not all(hasattr(source_model, name) for name in ("vpm", "resampler", "get_vision_embedding")):
        return

    def get_vision_embedding_patched(self, pixel_values):
        res = []
        vpm_param = next(self.vpm.parameters())
        if hasattr(self.resampler, "kv_proj") and hasattr(self.resampler.kv_proj, "weight"):
            resampler_param = self.resampler.kv_proj.weight
        else:
            resampler_param = next(self.resampler.parameters())
        text_embed_weight = self.llm.model.embed_tokens.weight
        self.resampler.to(device=resampler_param.device, dtype=resampler_param.dtype)
        for pixel_value in pixel_values:
            with torch.no_grad():
                pixel_value = pixel_value.to(device=vpm_param.device, dtype=vpm_param.dtype)
                vision_embedding = self.vpm.forward_features(pixel_value.unsqueeze(0))
                if hasattr(self.vpm, "num_prefix_tokens") and self.vpm.num_prefix_tokens > 0:
                    vision_embedding = vision_embedding[:, self.vpm.num_prefix_tokens :]
                vision_embedding = vision_embedding.to(device=resampler_param.device, dtype=resampler_param.dtype)
                vision_hidden = self.resampler(vision_embedding)
                vision_hidden = vision_hidden.to(device=text_embed_weight.device, dtype=text_embed_weight.dtype)
            res.append(vision_hidden.detach())
        return torch.vstack(res)

    source_model.get_vision_embedding = types.MethodType(get_vision_embedding_patched, source_model)
    source_model._mindpipe_patched_vision_embedding_dtype = True


def _build_sft_dataset_and_collator(model, tokenizer_bundle, args):
    tokenizer = tokenizer_bundle.tokenizer
    sft_format = str(args.compression_lora_sft_format)
    if sft_format == "alpaca":
        dataset = build_alpaca_sft_dataset(
            train_file=args.compression_lora_sft_train_file,
            sample_count=int(args.compression_lora_sft_samples),
            seed=int(args.seed),
            tokenizer=tokenizer,
            max_length=int(args.sequence_length),
            min_response_tokens=int(getattr(args, "compression_lora_sft_min_response_tokens", 8)),
            sample_start=int(getattr(args, "compression_lora_sft_sample_start", 0)),
        )
        return dataset, TextSFTCollator(
            tokenizer,
            max_length=int(args.sequence_length),
            min_response_tokens=int(getattr(args, "compression_lora_sft_min_response_tokens", 8)),
        ), _default_causal_lm_loss

    if sft_format == "llava":
        dataset = build_llava_sft_dataset(
            train_file=args.compression_lora_sft_train_file,
            sample_count=int(args.compression_lora_sft_samples),
            seed=int(args.seed),
        )
        model_type = getattr(model.config, "model_type", None)
        if model_type in {"minicpm", "minicpmv"}:
            _patch_minicpm_v_vision_embedding_dtype(model)
            return dataset, MiniCPMVImageTextSFTCollator(
                tokenizer=tokenizer,
                model=model,
                max_length=int(args.sequence_length),
            ), _minicpm_v_sft_loss
        if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
            return dataset, QwenVLImageTextSFTCollator(
                processor=tokenizer_bundle.processor,
                tokenizer=tokenizer,
                max_length=int(args.sequence_length),
                model_type=str(model_type),
                image_max_pixels=getattr(args, "compression_lora_vlm_image_max_pixels", 262144),
            ), _default_causal_lm_loss
        raise NotImplementedError(
            f"compression_lora llava SFT currently supports MiniCPM-V and Qwen-VL only; "
            f"got model_type={model_type!r}."
        )

    raise ValueError(f"Unsupported compression_lora_sft_format: {sft_format}")


def _default_causal_lm_loss(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    outputs = model(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("Compression LoRA model forward did not return loss.")
    return loss


def _minicpm_v_sft_loss(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    source_model = getattr(model, "_source_model", None)
    if source_model is None:
        raise AttributeError("MiniCPM-V SFT requires TextModelAdapter._source_model.")
    labels = batch.pop("labels", None)
    if labels is None:
        raise RuntimeError("MiniCPM-V SFT batch does not contain labels.")
    outputs = source_model(data=batch, use_cache=False)
    logits = outputs.logits
    vocab_size = int(getattr(source_model.config, "vocab_size", logits.shape[-1]))
    labels = labels.to(logits.device).long()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size).contiguous(),
        labels.view(-1).contiguous(),
    )


def _raise_nonfinite_loss(model: torch.nn.Module, batch: dict[str, Any], loss: torch.Tensor) -> None:
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    batch_bits = []
    if torch.is_tensor(input_ids):
        batch_bits.append(
            f"input_ids_shape={tuple(input_ids.shape)} input_min={int(input_ids.min())} input_max={int(input_ids.max())}"
        )
    if torch.is_tensor(labels):
        valid = labels.ne(-100)
        batch_bits.append(f"valid_labels={int(valid.sum())}")
    if torch.is_tensor(attention_mask):
        batch_bits.append(f"attention_tokens={int(attention_mask.sum())}")
    raise RuntimeError(
        "Compression LoRA produced non-finite loss before backward: "
        f"{loss.item()}. {'; '.join(batch_bits)}."
    )


def _first_parameter_device(module: torch.nn.Module | None) -> torch.device | None:
    if module is None:
        return None
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _first_floating_dtype(module: torch.nn.Module | None) -> torch.dtype | None:
    if module is None:
        return None
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers(recurse=True):
        if torch.is_tensor(buffer) and buffer.is_floating_point():
            return buffer.dtype
    return None


def _get_submodule_or_none(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    try:
        return model.get_submodule(name)
    except AttributeError:
        current = model
        for part in name.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current
    except Exception:
        return None


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        device = _first_parameter_device(model.get_input_embeddings())
        if device is not None:
            return device
    for name in ("model.embed_tokens", "llm.model.embed_tokens", "language_model.model.embed_tokens"):
        device = _first_parameter_device(_get_submodule_or_none(model, name))
        if device is not None:
            return device
    return next(model.parameters()).device


def _infer_visual_device(model: torch.nn.Module) -> torch.device | None:
    candidates = [model]
    source_model = getattr(model, "_source_model", None)
    if source_model is not None:
        candidates.insert(0, source_model)
    for candidate in candidates:
        for name in ("vpm", "visual", "vision_tower", "vision_model", "model.visual", "model.vision_tower"):
            device = _first_parameter_device(_get_submodule_or_none(candidate, name))
            if device is not None:
                return device
    return None


def _infer_visual_dtype(model: torch.nn.Module) -> torch.dtype | None:
    candidates = [model]
    source_model = getattr(model, "_source_model", None)
    if source_model is not None:
        candidates.insert(0, source_model)
    for candidate in candidates:
        for name in ("vpm", "visual", "vision_tower", "vision_model", "model.visual", "model.vision_tower"):
            dtype = _first_floating_dtype(_get_submodule_or_none(candidate, name))
            if dtype is not None:
                return dtype
    return _first_floating_dtype(model)


def _trainable_param_devices(params: list[torch.nn.Parameter]) -> list[torch.device]:
    devices = []
    seen = set()
    for param in params:
        device = param.device
        key = str(device)
        if key in seen:
            continue
        seen.add(key)
        devices.append(device)
    return devices


def _summarize_hf_device_map(model: torch.nn.Module) -> dict[str, int]:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return {}
    summary: dict[str, int] = {}
    for device in device_map.values():
        key = str(device)
        summary[key] = summary.get(key, 0) + 1
    return summary


def _move_to_device(value: Any, device: torch.device | str, dtype: torch.dtype | None = None):
    if torch.is_tensor(value):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(device=device, dtype=target_dtype, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, dtype=dtype) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device, dtype=dtype) for key, item in value.items()}
    return value


def _move_batch_for_model(batch: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    input_device = _infer_input_device(model)
    visual_device = _infer_visual_device(model)
    visual_dtype = _infer_visual_dtype(model)
    visual_keys = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key in visual_keys and visual_device is not None:
            moved[key] = _move_to_device(value, visual_device, dtype=visual_dtype)
        else:
            moved[key] = _move_to_device(value, input_device)
    return moved


def _batch_example_count(batch: dict[str, Any], fallback: int) -> int:
    for key in ("input_ids", "labels", "attention_mask"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    return int(fallback)


def _train_lora_manually(
    model: torch.nn.Module,
    train_dataset,
    collate_fn,
    *,
    loss_fn,
    stage_name: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    grad_accum: int,
    num_train_epochs: float,
    max_steps: int,
    logging_steps: int,
    gradient_checkpointing: bool,
    lr_scheduler_type: str,
    warmup_ratio: float,
    data_seed: int,
    checkpoint_dir: Path,
    save_steps: int,
    save_total_limit: int,
    resume_payload: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    resume_mode: str = "strict",
) -> dict[str, Any]:
    named_trainable_params = _named_adapter_parameters(model)
    parameter_names = [name for name, _ in named_trainable_params]
    trainable_params = [parameter for _, parameter in named_trainable_params]
    if not trainable_params:
        raise RuntimeError("No trainable compression LoRA parameters found.")

    dataset_size = len(train_dataset)
    batch_size = max(1, int(batch_size))
    grad_accum = max(1, int(grad_accum))
    batches_per_epoch = math.ceil(dataset_size / batch_size)
    if batches_per_epoch == 0:
        raise RuntimeError("Compression LoRA train dataset is empty.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    planned_steps = int(max_steps)
    if planned_steps <= 0:
        planned_steps = max(1, math.ceil(batches_per_epoch * float(num_train_epochs) / grad_accum))
    resume_mode = str(resume_mode)
    if resume_mode not in {"strict", "extend"}:
        raise ValueError(f"Unsupported compression LoRA resume mode: {resume_mode!r}.")
    resume_state = resume_payload["training_state"] if resume_payload is not None else None
    extending_completed_run = bool(
        resume_state is not None and resume_mode == "extend" and resume_state.get("completed", False)
    )
    resuming_extension = bool(
        resume_state is not None
        and not resume_state.get("completed", False)
        and "run_start_step" in resume_state
    )
    if extending_completed_run:
        run_start_step = int(resume_state["global_step"])
        run_steps = planned_steps
        target_global_step = run_start_step + run_steps
    elif resuming_extension:
        run_start_step = int(resume_state["run_start_step"])
        run_steps = int(resume_state["run_steps"])
        target_global_step = int(resume_state["max_steps"])
    else:
        run_start_step = 0
        run_steps = planned_steps
        target_global_step = planned_steps
    from transformers import get_scheduler

    warmup_steps = int(run_steps * max(0.0, float(warmup_ratio)))
    lr_scheduler = get_scheduler(
        name=str(lr_scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=run_steps,
    )
    logging_steps = max(1, int(logging_steps))
    save_steps = max(0, int(save_steps))
    input_device = _infer_input_device(model)
    visual_device = _infer_visual_device(model)
    trainable_devices = _trainable_param_devices(trainable_params)
    hf_device_summary = _summarize_hf_device_map(model)
    LOGGER.info(
        "compression_lora %s training: samples=%s batch_size=%s grad_accum=%s max_steps=%s lr_scheduler=%s warmup_steps=%s gradient_checkpointing=%s",
        stage_name,
        len(train_dataset),
        batch_size,
        grad_accum,
        target_global_step,
        str(lr_scheduler_type),
        warmup_steps,
        bool(gradient_checkpointing),
    )
    LOGGER.info(
        "compression_lora %s devices: input_device=%s visual_device=%s trainable_devices=%s hf_device_map=%s",
        stage_name,
        input_device,
        visual_device,
        [str(device) for device in trainable_devices],
        hf_device_summary or None,
    )
    start_time = time.perf_counter()
    step_start_time = start_time
    global_step = 0
    micro_step = 0
    epoch = 0
    sample_offset = 0
    resumed_from = None
    loss_sum = 0.0
    loss_count = 0
    log_loss_sum = 0.0
    log_loss_count = 0
    loss_history: list[dict[str, Any]] = []
    last_saved_step: int | None = None
    optimizer.zero_grad(set_to_none=True)

    if resume_payload is not None:
        state = resume_payload["training_state"]
        expected = {
            "stage": stage_name,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "data_seed": int(data_seed),
            "parameter_names": parameter_names,
        }
        if not extending_completed_run:
            expected["dataset_size"] = dataset_size
            expected["run_steps"] = run_steps
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if (state.get("max_steps") if key == "run_steps" and "run_steps" not in state else state.get(key)) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={old!r} current={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(f"Compression LoRA resume configuration mismatch: {details}")
        global_step = int(state["global_step"])
        micro_step = int(state["micro_step"])
        if extending_completed_run:
            epoch = 0
            sample_offset = 0
        else:
            epoch = int(state["epoch"])
            sample_offset = int(state["sample_offset"])
        if global_step > target_global_step:
            raise ValueError(
                f"Resume global_step={global_step} exceeds current max_steps={target_global_step}."
            )
        if micro_step % grad_accum != 0:
            raise ValueError("Compression LoRA checkpoints must be saved at an optimizer-step boundary.")
        if not 0 <= sample_offset <= dataset_size:
            raise ValueError(
                f"Invalid resume sample_offset={sample_offset} for dataset_size={dataset_size}."
            )
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if extending_completed_run:
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(learning_rate)
                param_group["initial_lr"] = float(learning_rate)
        else:
            lr_scheduler.load_state_dict(resume_payload["lr_scheduler_state_dict"])
        _restore_rng_state(resume_payload.get("rng_state"))
        resumed_from = str(resume_payload.get("_checkpoint_path", "<checkpoint>"))
        LOGGER.info(
            "Resumed compression_lora %s from %s at step=%d epoch=%d sample_offset=%d",
            stage_name,
            resumed_from,
            global_step,
            epoch,
            sample_offset,
        )

    while global_step < target_global_step:
        if sample_offset >= dataset_size:
            epoch += 1
            sample_offset = 0
        generator = torch.Generator()
        generator.manual_seed(int(data_seed) + epoch)
        epoch_indices = torch.randperm(dataset_size, generator=generator).tolist()
        remaining_indices = epoch_indices[sample_offset:]
        dataloader = DataLoader(
            Subset(train_dataset, remaining_indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        for batch in dataloader:
            batch_examples = _batch_example_count(batch, batch_size)
            batch = _move_batch_for_model(batch, model)
            loss = loss_fn(model, batch)
            if not torch.isfinite(loss).item():
                _raise_nonfinite_loss(model, batch, loss)
            (loss / grad_accum).backward()
            loss_value = float(loss.detach().cpu())
            loss_sum += loss_value
            loss_count += 1
            log_loss_sum += loss_value
            log_loss_count += 1
            micro_step += 1
            sample_offset += batch_examples

            if micro_step % grad_accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0,
                error_if_nonfinite=True,
                foreach=True,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            if torch.cuda.is_available():
                for sync_device in trainable_devices:
                    if sync_device.type == "cuda":
                        torch.cuda.synchronize(sync_device)
            now = time.perf_counter()
            step_elapsed = now - step_start_time
            total_elapsed = now - start_time
            step_start_time = now

            if global_step == 1 or global_step % logging_steps == 0 or global_step >= target_global_step:
                avg_loss = loss_sum / max(1, loss_count)
                interval_loss = log_loss_sum / max(1, log_loss_count)
                grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                loss_history.append(
                    {
                        "stage": stage_name,
                        "step": global_step,
                        "max_steps": target_global_step,
                        "loss": avg_loss,
                        "interval_loss": interval_loss,
                        "grad_norm": grad_norm_value,
                        "learning_rate": current_lr,
                        "step_time": step_elapsed,
                        "elapsed": total_elapsed,
                    }
                )
                LOGGER.info(
                    "compression_lora %s step %s/%s interval_loss %.6f avg_loss %.6f "
                    "grad_norm %.6f lr %.6g time %.3fs elapsed %.3fs",
                    stage_name,
                    global_step,
                    target_global_step,
                    interval_loss,
                    avg_loss,
                    grad_norm_value,
                    current_lr,
                    step_elapsed,
                    total_elapsed,
                )
                log_loss_sum = 0.0
                log_loss_count = 0
            if save_steps and global_step % save_steps == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}.pth"
                _save_training_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    stage_name=stage_name,
                    global_step=global_step,
                    micro_step=micro_step,
                    epoch=epoch,
                    sample_offset=sample_offset,
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    grad_accum=grad_accum,
                    max_steps=target_global_step,
                    run_start_step=run_start_step,
                    run_steps=run_steps,
                    data_seed=data_seed,
                    parameter_names=parameter_names,
                    completed=global_step >= target_global_step,
                    metadata=checkpoint_metadata or {},
                )
                _rotate_training_checkpoints(checkpoint_dir, save_total_limit)
                LOGGER.info("Saved compression_lora training checkpoint to %s", checkpoint_path)
                last_saved_step = global_step
            if global_step >= target_global_step:
                break

    final_checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}.pth"
    if last_saved_step != global_step:
        _save_training_checkpoint(
            path=final_checkpoint_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            stage_name=stage_name,
            global_step=global_step,
            micro_step=micro_step,
            epoch=epoch,
            sample_offset=sample_offset,
            dataset_size=dataset_size,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_steps=target_global_step,
            run_start_step=run_start_step,
            run_steps=run_steps,
            data_seed=data_seed,
            parameter_names=parameter_names,
            completed=True,
            metadata=checkpoint_metadata or {},
        )
        _rotate_training_checkpoints(checkpoint_dir, save_total_limit)

    elapsed = time.perf_counter() - start_time
    return {
        "train_runtime": elapsed,
        "train_samples_per_second": len(train_dataset) / elapsed if elapsed > 0 else 0.0,
        "train_steps_per_second": global_step / elapsed if elapsed > 0 else 0.0,
        "train_loss": loss_sum / max(1, loss_count),
        "global_step": global_step,
        "initial_global_step": int(resume_payload["training_state"]["global_step"]) if resume_payload else 0,
        "resume_mode": resume_mode,
        "run_start_step": run_start_step,
        "run_steps": run_steps,
        "resumed_from": resumed_from,
        "training_checkpoint_path": str(final_checkpoint_path),
        "train_examples": len(train_dataset),
        "stage": stage_name,
        "learning_rate": float(learning_rate),
        "lr_scheduler_type": str(lr_scheduler_type),
        "warmup_ratio": float(warmup_ratio),
        "warmup_steps": warmup_steps,
        "num_train_epochs": float(num_train_epochs),
        "trainer": "manual_pytorch",
        "loss_history": loss_history,
    }


def _write_stage_loss_history(output_dir: Path, stage: str, metrics: dict[str, Any]) -> Path | None:
    history = metrics.get("loss_history")
    if not history:
        return None
    path = write_json(
        output_dir / f"compression_lora_{stage}_loss_history.json",
        {
            "stage": stage,
            "learning_rate": metrics.get("learning_rate"),
            "global_step": metrics.get("global_step"),
            "train_loss": metrics.get("train_loss"),
            "history": history,
        },
    )
    metrics["loss_history_path"] = str(path)
    return path


def _replace_quantized_linears_with_lora(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
    config: LoRAConfig,
    quantization: str,
    target_modules=None,
) -> dict[str, dict[str, Any]]:
    module_map = dict(model.named_modules())
    replaced: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        if not _matches_lora_target(name, target_modules):
            continue
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Compression LoRA mask target not found: {name}")
        if _is_compression_lora_linear(module):
            continue
        if quantization == "flatquant":
            is_supported = _is_flatquant_linear(module)
            wrapper_type = CompressionLoRAFlatQuantLinear
        elif quantization == "splitquant":
            is_supported = _is_splitquant_linear(module)
            wrapper_type = CompressionLoRASplitQuantLinear
        else:
            raise ValueError(f"Unsupported compression LoRA quantization: {quantization!r}.")
        if not is_supported:
            raise TypeError(
                f"Compression LoRA expected a {quantization} quantized Linear; "
                f"target {name} is {module.__class__.__name__}."
            )
        wrapper = wrapper_type(module, mask, config)
        _set_child_module(model, name, wrapper)
        replaced[name] = {
            "in_features": int(wrapper.in_features),
            "out_features": int(wrapper.out_features),
            "rank": int(config.rank),
            "alpha": float(config.alpha),
            "dropout": float(config.dropout),
            "adapter_type": config.adapter_type,
        }
    return replaced


def _replace_packed_moe_experts_with_lora(model, masks, config, quantization, target_modules):
    targets = set(target_modules or [])
    expert_projections = {"gate_proj", "up_proj", "down_proj"}
    expert_targets = expert_projections if not targets else expert_projections & targets
    if not expert_targets:
        return {}, set()
    if expert_targets != expert_projections:
        raise ValueError(
            "Packed Qwen3-MoE compression LoRA currently requires gate_proj, up_proj, "
            "and down_proj together to preserve the exact packed objective."
        )
    replaced, consumed = {}, set()
    for name, block in list(model.named_modules()):
        expected_classes = (
            {"FlatQuantQwen3MoeSparseMoeBlock", "FlatQuantMixtralSparseMoeBlock"}
            if quantization == "flatquant"
            else {"SplitQuantQwen3MoeSparseMoeBlock", "SplitQuantMixtralSparseMoeBlock"}
        )
        if block.__class__.__name__ not in expected_classes or not getattr(block, "experts_are_packed", False):
            continue
        prefix = name
        block_masks = {}
        for expert_index in range(int(block.experts.num_experts)):
            # Qwen3 masks use gate/up/down names. Mixtral masks may originate
            # from its native w1/w3/w2 expert layout, even when quantization
            # repacks the experts into gate_up_proj/down_proj tensors.
            projection_aliases = {
                "gate_proj": ("gate_proj", "w1"),
                "up_proj": ("up_proj", "w3"),
                "down_proj": ("down_proj", "w2"),
            }
            for projection, aliases in projection_aliases.items():
                found_key = next(
                    (
                        f"{prefix}.experts.{expert_index}.{alias}"
                        for alias in aliases
                        if f"{prefix}.experts.{expert_index}.{alias}" in masks
                    ),
                    None,
                )
                if found_key is None:
                    raise KeyError(
                        f"Missing packed expert pruning mask for {prefix}.experts.{expert_index}.{projection}"
                    )
                block_masks[f"{prefix}.experts.{expert_index}.{projection}"] = masks[found_key]
                consumed.add(found_key)
        wrapper = PackedCompressionLoRAExperts(
            block.experts,
            block_masks,
            prefix,
            config,
            quantization,
        )
        block.experts = wrapper
        replaced[f"{prefix}.experts"] = {
            "num_experts": wrapper.num_experts,
            "rank": wrapper.rank,
            "alpha": wrapper.alpha,
            "packed": True,
        }
    return replaced, consumed


@torch.no_grad()
def _unwrap_lora_wrappers(model: torch.nn.Module) -> list[str]:
    merged: list[str] = []
    for name, module in list(model.named_modules()):
        if not _is_compression_lora_linear(module):
            continue
        base = module.merge_into_base()
        _set_child_module(model, name, base)
        merged.append(name)
    return merged


def _prepare_lora_training(model: torch.nn.Module, args) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if _is_compression_lora_linear(module):
            if isinstance(module, PackedCompressionLoRAExperts):
                for param in module.adapter_parameters():
                    param.requires_grad = True
                continue
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if module.adapter_type == "dora":
                module.dora_magnitude.requires_grad = True
    if bool(args.compression_lora_gradient_checkpointing):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()


def _run_lora_train_plan(
    *,
    model: torch.nn.Module,
    tokenizer_bundle,
    tokenizer,
    args,
    output_dir: Path,
    masks_path: str,
    cpt_adapter_path: Path,
    resume_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    _prepare_lora_training(model, args)
    train_plan = _parse_train_plan(getattr(args, "compression_lora_train_plan", None))
    resume_stage = resume_payload["training_state"]["stage"] if resume_payload else None
    if resume_stage is not None and resume_stage not in train_plan:
        raise ValueError(
            f"Resume checkpoint stage={resume_stage!r} is not present in train_plan={train_plan}."
        )
    resume_stage_index = train_plan.index(resume_stage) if resume_stage is not None else 0
    checkpoint_metadata = {
        "model_path": str(args.model_path),
        "quantization": str(args.quantization),
        "pruning": str(args.pruning),
        "masks_path": str(masks_path),
        "quantization_from": str(
            getattr(args, "compression_lora_flatquant_from", None)
            or getattr(args, "compression_lora_splitquant_from", None)
            or getattr(args, "flatquant_resume_from", None)
            or getattr(args, "splitquant_resume_from", None)
        ),
        "rank": int(args.compression_lora_rank),
        "alpha": float(args.compression_lora_alpha),
        "dropout": float(args.compression_lora_dropout),
        "init": str(args.compression_lora_init),
        "adapter_type": str(args.compression_lora_adapter_type),
        "target_modules": list(args.compression_lora_target_modules),
        "sequence_length": int(args.sequence_length),
        "weight_decay": float(args.compression_lora_weight_decay),
        "lr_scheduler_type": str(args.compression_lora_lr_scheduler_type),
        "warmup_ratio": float(args.compression_lora_warmup_ratio),
        "cpt_learning_rate": float(args.compression_lora_cpt_learning_rate),
        "sft_learning_rate": float(args.compression_lora_sft_learning_rate),
        "cpt_train_file": getattr(args, "compression_lora_cpt_train_file", None),
        "cpt_samples": int(args.compression_lora_cpt_samples),
        "sft_train_file": getattr(args, "compression_lora_sft_train_file", None),
        "sft_samples": int(args.compression_lora_sft_samples),
        "sft_sample_start": int(getattr(args, "compression_lora_sft_sample_start", 0)),
        "sft_format": str(args.compression_lora_sft_format),
    }
    checkpoint_root = (
        Path(args.compression_lora_checkpoint_dir)
        if getattr(args, "compression_lora_checkpoint_dir", None)
        else output_dir / "compression_lora_checkpoints"
    )
    stage_metrics: dict[str, Any] = {}
    for stage_index, stage in enumerate(train_plan):
        if resume_payload is not None and stage_index < resume_stage_index:
            stage_metrics[stage] = {"skipped_due_to_resume": True}
            continue
        stage_resume_payload = resume_payload if stage == resume_stage else None
        if stage == "cpt":
            if not getattr(args, "compression_lora_cpt_train_file", None):
                raise ValueError("compression_lora cpt stage requires --compression_lora_cpt_train_file.")
            train_dataset = build_raw_text_jsonl_dataset(
                tokenizer=tokenizer,
                train_file=args.compression_lora_cpt_train_file,
                sequence_length=int(args.sequence_length),
                sample_count=int(args.compression_lora_cpt_samples),
                seed=int(args.seed),
            )
            stage_metrics[stage] = _train_lora_manually(
                model=model,
                train_dataset=train_dataset,
                collate_fn=RawTextCPTCollator(tokenizer),
                loss_fn=_default_causal_lm_loss,
                stage_name=stage,
                learning_rate=float(args.compression_lora_cpt_learning_rate),
                weight_decay=float(args.compression_lora_weight_decay),
                batch_size=int(args.compression_lora_per_device_train_batch_size),
                grad_accum=int(args.compression_lora_gradient_accumulation_steps),
                num_train_epochs=float(args.compression_lora_cpt_num_train_epochs),
                max_steps=int(args.compression_lora_cpt_max_steps),
                logging_steps=int(args.compression_lora_logging_steps),
                gradient_checkpointing=bool(args.compression_lora_gradient_checkpointing),
                lr_scheduler_type=str(args.compression_lora_lr_scheduler_type),
                warmup_ratio=float(args.compression_lora_warmup_ratio),
                data_seed=int(args.seed) + 1000,
                checkpoint_dir=checkpoint_root / stage,
                save_steps=int(args.compression_lora_save_steps),
                save_total_limit=int(args.compression_lora_save_total_limit),
                resume_payload=stage_resume_payload,
                checkpoint_metadata=checkpoint_metadata,
                resume_mode=str(getattr(args, "compression_lora_resume_mode", "strict")),
            )
            _write_stage_loss_history(output_dir, stage, stage_metrics[stage])
            if bool(getattr(args, "compression_lora_save_cpt_adapter", True)):
                _save_adapter(cpt_adapter_path, model, args, str(masks_path), stage, stage_metrics[stage])
            continue

        if stage == "sft":
            if not getattr(args, "compression_lora_sft_train_file", None):
                raise ValueError("compression_lora sft stage requires --compression_lora_sft_train_file.")
            train_dataset, sft_collator, sft_loss_fn = _build_sft_dataset_and_collator(model, tokenizer_bundle, args)
            stage_metrics[stage] = _train_lora_manually(
                model=model,
                train_dataset=train_dataset,
                collate_fn=sft_collator,
                loss_fn=sft_loss_fn,
                stage_name=stage,
                learning_rate=float(args.compression_lora_sft_learning_rate),
                weight_decay=float(args.compression_lora_weight_decay),
                batch_size=int(args.compression_lora_per_device_train_batch_size),
                grad_accum=int(args.compression_lora_gradient_accumulation_steps),
                num_train_epochs=float(args.compression_lora_sft_num_train_epochs),
                max_steps=int(args.compression_lora_sft_max_steps),
                logging_steps=int(args.compression_lora_logging_steps),
                gradient_checkpointing=bool(args.compression_lora_gradient_checkpointing),
                lr_scheduler_type=str(args.compression_lora_lr_scheduler_type),
                warmup_ratio=float(args.compression_lora_warmup_ratio),
                data_seed=int(args.seed) + 2000,
                checkpoint_dir=checkpoint_root / stage,
                save_steps=int(args.compression_lora_save_steps),
                save_total_limit=int(args.compression_lora_save_total_limit),
                resume_payload=stage_resume_payload,
                checkpoint_metadata=checkpoint_metadata,
                resume_mode=str(getattr(args, "compression_lora_resume_mode", "strict")),
            )
            _write_stage_loss_history(output_dir, stage, stage_metrics[stage])
            continue

    train_metrics = {
        "train_plan": train_plan,
        "stages": stage_metrics,
        "cpt_train_file": getattr(args, "compression_lora_cpt_train_file", None),
        "sft_train_file": getattr(args, "compression_lora_sft_train_file", None),
        "sft_format": getattr(args, "compression_lora_sft_format", None),
        "sft_min_response_tokens": int(getattr(args, "compression_lora_sft_min_response_tokens", 8)),
    }
    return train_plan, stage_metrics, train_metrics


def _select_flatquant_apply_wrapper(model, source_root: Path):
    from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
    from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
    from flatquant.model_tools.minicpm_utils import apply_flatquant_to_minicpm
    from flatquant.model_tools.qwen3_utils import apply_flatquant_to_qwen3
    from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
    from flatquant.model_tools.mixtral_utils import apply_flatquant_to_mixtral

    model_type = getattr(model.config, "model_type", None)
    rope_scaling = getattr(model.config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None
    if model_type == "llama":
        return apply_flatquant_to_llama_31 if rope_type == "llama3" else apply_flatquant_to_llama
    if model_type in {"minicpm", "minicpmv"}:
        return apply_flatquant_to_minicpm
    if model_type in {"qwen2", "qwen2_5_vl"}:
        return apply_flatquant_to_qwen
    if model_type in {"qwen3", "qwen3_vl"}:
        return apply_flatquant_to_qwen3
    if model_type == "qwen3_moe":
        from flatquant.model_tools.qwen3_utils import apply_flatquant_to_qwen3_moe

        return apply_flatquant_to_qwen3_moe
    if model_type == "mixtral":
        return apply_flatquant_to_mixtral
    if model_type == "qwen3_5":
        from flatquant.model_tools.qwen3_5_utils import apply_flatquant_to_qwen3_5

        return apply_flatquant_to_qwen3_5
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from flatquant.model_tools.qwen3_5_utils import apply_flatquant_to_qwen3_5_moe

        return apply_flatquant_to_qwen3_5_moe
    raise NotImplementedError(f"FlatQuant compression_lora does not support model_type={model_type!r}.")


def _select_splitquant_apply_wrapper(model):
    from splitquant.model_tools.llama31_utils import apply_splitquant_to_llama_31
    from splitquant.model_tools.llama_utils import apply_splitquant_to_llama
    from splitquant.model_tools.minicpm_split_utils import apply_splitquant_to_minicpm
    from splitquant.model_tools.qwen3_split_utils import apply_splitquant_to_qwen3
    from splitquant.model_tools.qwen_split_utils import apply_splitquant_to_qwen
    from splitquant.model_tools.mixtral_split_utils import apply_splitquant_to_mixtral

    model_type = getattr(model.config, "model_type", None)
    rope_scaling = getattr(model.config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None
    if model_type == "llama":
        return apply_splitquant_to_llama_31 if rope_type == "llama3" else apply_splitquant_to_llama
    if model_type in {"minicpm", "minicpmv"}:
        return apply_splitquant_to_minicpm
    if model_type in {"qwen2", "qwen2_5_vl"}:
        return apply_splitquant_to_qwen
    if model_type in {"qwen3", "qwen3_vl"}:
        return apply_splitquant_to_qwen3
    if model_type == "qwen3_moe":
        from splitquant.model_tools.qwen3_split_utils import apply_splitquant_to_qwen3_moe

        return apply_splitquant_to_qwen3_moe
    if model_type == "mixtral":
        return apply_splitquant_to_mixtral
    if model_type == "qwen3_5":
        from splitquant.model_tools.qwen3_5_split_utils import apply_splitquant_to_qwen3_5

        return apply_splitquant_to_qwen3_5
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from splitquant.model_tools.qwen3_5_split_utils import apply_splitquant_to_qwen3_5_moe

        return apply_splitquant_to_qwen3_5_moe
    raise NotImplementedError(f"SplitQuant compression_lora does not support model_type={model_type!r}.")


class CompressionLoRAMethod(BaseFinetuningMethod):
    name = "compression_lora"
    npu_ready = False

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = compression_lora_run_spec(args)
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, args) -> None:
        if getattr(args, "quantization", None) not in {"flatquant", "splitquant"}:
            raise ValueError("compression_lora supports --quantization flatquant or splitquant.")
        if getattr(args, "pruning", None) is None:
            raise ValueError("compression_lora requires a pruning stage to provide fixed masks.")
        if getattr(args, "pruning", None) == "flap" and not bool(getattr(args, "pseudo_pruning", True)):
            raise ValueError("compression_lora requires FLAP pseudo_pruning=true because masks must preserve shape.")
        if bool(getattr(args, "compression_lora_gradient_checkpointing", False)):
            raise ValueError(
                "compression_lora_gradient_checkpointing is not supported for dynamic fake-quant LoRA training. "
                "Set --compression_lora_gradient_checkpointing false."
            )

    def _load_fp_flatquant_model(self, args, flatquant_from: str):
        dtype = getattr(args, "dtype", "auto")
        model, tokenizer_bundle = load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            attn_implementation=getattr(args, "attn_implementation", None),
            device_map=getattr(args, "device_map", None),
            max_memory=getattr(args, "max_memory", None),
            offload_folder=getattr(args, "offload_folder", None),
            offload_state_dict=getattr(args, "offload_state_dict", None),
            no_split_module_classes=getattr(args, "no_split_module_classes", None),
        )
        model.seqlen = int(args.sequence_length)

        source_root = Path(__file__).resolve().parents[2] / "quantization" / "qat" / "flatquant" / "source"
        output_dir = ensure_dir(Path(args._workflow_output_dir))
        flatquant_method = FlatQuantMethod()
        source_args = flatquant_method._build_source_args(args, output_dir)
        source_args.resume = True
        source_args.exp_dir = str(output_dir)
        source_args.output_dir = str(output_dir)
        inferred_direct_inv = _infer_direct_inv_from_checkpoint(flatquant_from, "flat_parameters.pth")
        if inferred_direct_inv is not None:
            source_args.direct_inv = inferred_direct_inv

        with prepend_python_path(source_root):
            import importlib

            importlib.invalidate_caches()
            _purge_conflicting_modules("flatquant", source_root / "flatquant")
            apply_wrapper = _select_flatquant_apply_wrapper(model, source_root)
            from flatquant.flat_utils import load_flat_parameters

            original_layer_devices = []
            from flatquant.backbone_utils import get_decoder_layers

            for layer in get_decoder_layers(model):
                original_layer_devices.append(next(layer.parameters()).device)
            model = apply_wrapper(source_args, model)
            for layer, original_device in zip(get_decoder_layers(model), original_layer_devices):
                layer.to(original_device)
            load_flat_parameters(source_args, model, path=flatquant_from)
        return model, tokenizer_bundle, source_args, source_root

    def _load_fp_splitquant_model(self, args, splitquant_from: str):
        dtype = getattr(args, "dtype", "auto")
        model, tokenizer_bundle = load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            attn_implementation=getattr(args, "attn_implementation", None),
            device_map=getattr(args, "device_map", None),
            max_memory=getattr(args, "max_memory", None),
            offload_folder=getattr(args, "offload_folder", None),
            offload_state_dict=getattr(args, "offload_state_dict", None),
            no_split_module_classes=getattr(args, "no_split_module_classes", None),
        )
        model.seqlen = int(args.sequence_length)

        source_root = Path(__file__).resolve().parents[2] / "quantization" / "qat" / "splitquant" / "source"
        output_dir = ensure_dir(Path(args._workflow_output_dir))
        splitquant_method = SplitQuantMethod()
        source_args = splitquant_method._build_source_args(args, output_dir)
        source_args.resume = True
        source_args.exp_dir = str(output_dir)
        source_args.output_dir = str(output_dir)

        with prepend_python_path(source_root):
            import importlib

            importlib.invalidate_caches()
            _purge_conflicting_modules("splitquant", source_root / "splitquant")
            apply_wrapper = _select_splitquant_apply_wrapper(model)
            from splitquant.backbone_utils import get_decoder_layers
            from splitquant.split_utils import load_splitquant_parameters

            original_layer_devices = [next(layer.parameters()).device for layer in get_decoder_layers(model)]
            model = apply_wrapper(source_args, model)
            for layer, original_device in zip(get_decoder_layers(model), original_layer_devices):
                layer.to(original_device)
            load_splitquant_parameters(source_args, model, path=splitquant_from)
        return model, tokenizer_bundle, source_args, source_root

    def apply_finetuning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(args)
        del model, tokenizer_bundle
        empty_cache(args.device)

        output_dir = ensure_dir(Path(args._workflow_output_dir))
        adapter_path = output_dir / "compression_lora_adapter.pth"
        cpt_adapter_path = output_dir / "compression_lora_cpt_adapter.pth"
        config_path = output_dir / "compression_lora_config.json"

        resume_from = getattr(args, "compression_lora_resume_from", None)
        resume_payload = None
        resume_metadata: dict[str, Any] = {}
        if resume_from:
            resume_payload = _load_training_checkpoint(resume_from)
            resume_payload["_checkpoint_path"] = str(resume_from)
            resume_metadata = dict(resume_payload.get("metadata", {}))

        quantization = str(args.quantization)
        if quantization == "flatquant":
            quantization_from = (
                getattr(args, "compression_lora_flatquant_from", None)
                or getattr(args, "flatquant_resume_from", None)
                or resume_metadata.get("quantization_from")
            )
            checkpoint_name = "flat_parameters.pth"
            load_quantized_model = self._load_fp_flatquant_model
        else:
            quantization_from = (
                getattr(args, "compression_lora_splitquant_from", None)
                or getattr(args, "splitquant_resume_from", None)
                or resume_metadata.get("quantization_from")
            )
            checkpoint_name = "splitquant_parameters.pth"
            load_quantized_model = self._load_fp_splitquant_model
        if not quantization_from:
            raise ValueError(
                f"compression_lora requires {quantization} parameters. Provide the corresponding "
                "--compression_lora_*_from option or run finetuning after its quantization stage."
            )
        if not (Path(quantization_from) / checkpoint_name).exists():
            raise FileNotFoundError(f"Missing {checkpoint_name} under {quantization_from}")

        model, tokenizer_bundle, source_args, source_root = load_quantized_model(args, quantization_from)
        tokenizer = tokenizer_bundle.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must define eos_token or pad_token for compression_lora.")
            tokenizer.pad_token = tokenizer.eos_token

        masks_path = getattr(args, "compression_lora_masks_from", None) or resume_metadata.get("masks_path")
        if not masks_path:
            raise ValueError(
                "compression_lora requires --compression_lora_masks_from or an earlier pruning stage "
                "that generated pruning_masks.pth."
            )
        masks, mask_metadata = load_masks(masks_path)
        lora_config = LoRAConfig(
            rank=int(args.compression_lora_rank),
            alpha=float(args.compression_lora_alpha),
            dropout=float(args.compression_lora_dropout),
            init=str(args.compression_lora_init),
            weight_checkpointing=bool(getattr(args, "compression_lora_weight_checkpointing", False)),
            adapter_type=str(getattr(args, "compression_lora_adapter_type", "lora")),
            dora_simple=bool(getattr(args, "compression_lora_dora_simple", True)),
            dora_eps=float(getattr(args, "compression_lora_dora_eps", 1e-6)),
        )
        packed_layers, consumed_masks = _replace_packed_moe_experts_with_lora(
            model,
            masks,
            lora_config,
            quantization,
            getattr(args, "compression_lora_target_modules", None),
        )
        # Qwen3 packed experts consume their per-expert masks in the packed
        # wrapper. Mixtral keeps experts as ordinary w1/w2/w3 linears, so those
        # masks must remain in the regular replacement path.
        linear_masks = {name: mask for name, mask in masks.items() if name not in consumed_masks}
        selected_linear_masks = {
            name: mask
            for name, mask in linear_masks.items()
            if _matches_lora_target(name, getattr(args, "compression_lora_target_modules", None))
        }
        validate_masks(model, selected_linear_masks)
        adapter_layers = _replace_quantized_linears_with_lora(
            model, selected_linear_masks, lora_config, quantization,
            target_modules=getattr(args, "compression_lora_target_modules", None),
        )
        adapter_layers.update(packed_layers)
        if not adapter_layers:
            raise RuntimeError(f"No {quantization} Linear layers were wrapped for compression_lora.")

        if resume_from:
            checkpoint_metadata = resume_metadata
            resume_mode = str(getattr(args, "compression_lora_resume_mode", "strict"))
            expected_metadata = {
                "model_path": str(args.model_path),
                "quantization": quantization,
                "pruning": str(args.pruning),
                "masks_path": str(masks_path),
                "quantization_from": str(quantization_from),
                "rank": int(args.compression_lora_rank),
                "alpha": float(args.compression_lora_alpha),
                "dropout": float(args.compression_lora_dropout),
                "init": str(args.compression_lora_init),
                "adapter_type": str(args.compression_lora_adapter_type),
                "target_modules": list(args.compression_lora_target_modules),
                "sequence_length": int(args.sequence_length),
                "weight_decay": float(args.compression_lora_weight_decay),
                "lr_scheduler_type": str(args.compression_lora_lr_scheduler_type),
                "warmup_ratio": float(args.compression_lora_warmup_ratio),
                "cpt_learning_rate": float(args.compression_lora_cpt_learning_rate),
                "sft_learning_rate": float(args.compression_lora_sft_learning_rate),
                "cpt_train_file": getattr(args, "compression_lora_cpt_train_file", None),
                "cpt_samples": int(args.compression_lora_cpt_samples),
                "sft_train_file": getattr(args, "compression_lora_sft_train_file", None),
                "sft_samples": int(args.compression_lora_sft_samples),
                "sft_sample_start": int(getattr(args, "compression_lora_sft_sample_start", 0)),
                "sft_format": str(args.compression_lora_sft_format),
            }
            if resume_mode == "extend":
                resume_stage = str(resume_payload["training_state"].get("stage"))
                if not bool(resume_payload["training_state"].get("completed", False)):
                    raise ValueError("compression_lora resume_mode=extend requires a completed source checkpoint.")
                if resume_stage == "sft":
                    previous_samples = int(checkpoint_metadata.get("sft_samples", 0))
                    current_start = int(getattr(args, "compression_lora_sft_sample_start", 0))
                    if current_start != previous_samples:
                        raise ValueError(
                            "compression_lora extend must start immediately after the source SFT slice: "
                            f"expected sample_start={previous_samples}, got {current_start}."
                        )
                    expected_metadata.pop("sft_samples", None)
                    expected_metadata.pop("sft_sample_start", None)
            mismatches = {
                key: (checkpoint_metadata.get(key), value)
                for key, value in expected_metadata.items()
                if (0 if key == "sft_sample_start" and key not in checkpoint_metadata else checkpoint_metadata.get(key)) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{key}: checkpoint={old!r} current={new!r}"
                    for key, (old, new) in mismatches.items()
                )
                raise ValueError(f"Compression LoRA resume metadata mismatch: {details}")
            _load_adapter_state(model, resume_payload["adapter_state_dict"])

        train_plan, _stage_metrics, train_metrics = _run_lora_train_plan(
            model=model,
            tokenizer_bundle=tokenizer_bundle,
            tokenizer=tokenizer,
            args=args,
            output_dir=output_dir,
            masks_path=str(masks_path),
            cpt_adapter_path=cpt_adapter_path,
            resume_payload=resume_payload,
        )
        _save_adapter(adapter_path, model, args, str(masks_path), "final", train_metrics)

        merged_layers = _unwrap_lora_wrappers(model)
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )
        with prepend_python_path(source_root):
            import importlib

            importlib.invalidate_caches()
            _purge_conflicting_modules(quantization, source_root / quantization)
            _purge_conflicting_modules("gptq_utils", source_root)
            import gptq_utils
            if quantization == "flatquant":
                from flatquant.flat_utils import reparameterize_model

                reparameterize_model(model)
            else:
                from splitquant.split_utils import reparameterize_splitquant_model

                reparameterize_splitquant_model(model)
            if source_args.w_bits < 16:
                if source_args.gptq:
                    gptq_utils.gptq_fwrd(model, calibration_batches, args.device, source_args)
                else:
                    gptq_utils.rtn_fwrd(model, args.device, source_args)
        from .mask_utils import apply_masks_to_model

        apply_masks_to_model(model, linear_masks)
        apply_packed_moe_masks(model, masks)
        model.config.use_cache = True
        model.eval()
        model.seqlen = int(args.sequence_length)

        config_payload = {
            "mode": "quant_then_prune_fixed_mask",
            "adapter_layers": adapter_layers,
            "merged_layers": merged_layers,
            "masks_path": str(masks_path),
            "quantization": quantization,
            "quantization_from": str(quantization_from),
            "mask_metadata": mask_metadata,
            "mask_sparsity": mask_sparsity(masks),
            "train_metrics": train_metrics,
            "train_plan": train_plan,
            "save_merged_model": bool(args.compression_lora_save_merged_model),
            "weight_checkpointing": bool(getattr(args, "compression_lora_weight_checkpointing", False)),
            "adapter_type": getattr(args, "compression_lora_adapter_type", "lora"),
            "dora_simple": bool(getattr(args, "compression_lora_dora_simple", True)),
            "dora_eps": float(getattr(args, "compression_lora_dora_eps", 1e-6)),
            "lr_scheduler_type": getattr(args, "compression_lora_lr_scheduler_type", "cosine"),
            "warmup_ratio": float(getattr(args, "compression_lora_warmup_ratio", 0.03)),
            "resume_from": resume_from,
            "sft_min_response_tokens": int(getattr(args, "compression_lora_sft_min_response_tokens", 8)),
        }
        config_path = write_json(config_path, config_payload)

        merged_model_dir = None
        if bool(args.compression_lora_save_merged_model):
            merged_model_dir = ensure_dir(output_dir / "compression_lora_merged_model")
            model.save_pretrained(merged_model_dir)
            tokenizer_bundle.save_pretrained(str(merged_model_dir))

        artifacts: dict[str, object] = {
            "compression_lora_config_path": str(config_path),
            "compression_lora_masks_path": str(masks_path),
            "compression_lora_adapter_path": str(adapter_path) if adapter_path.exists() else None,
            "compression_lora_cpt_adapter_path": str(cpt_adapter_path) if cpt_adapter_path.exists() else None,
            "compression_lora_wrapped_layer_count": len(adapter_layers),
            "compression_lora_merged_layers": merged_layers,
            "train_metrics": train_metrics,
            "_updated_model": model,
            "_updated_tokenizer_bundle": tokenizer_bundle,
        }
        latest_training_checkpoint = None
        for stage in reversed(train_plan):
            stage_state = train_metrics["stages"].get(stage)
            if isinstance(stage_state, dict) and stage_state.get("training_checkpoint_path"):
                latest_training_checkpoint = stage_state["training_checkpoint_path"]
                break
        if latest_training_checkpoint is not None:
            artifacts["compression_lora_training_checkpoint_path"] = latest_training_checkpoint
        if merged_model_dir is not None:
            artifacts["compression_lora_merged_model_dir"] = str(merged_model_dir)
        return artifacts
