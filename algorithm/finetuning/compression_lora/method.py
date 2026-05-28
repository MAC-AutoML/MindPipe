"""FlatQuant fixed-mask compression-aware LoRA."""

from __future__ import annotations

import logging
import math
import types
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

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
from .collators import RawTextCPTCollator
from .collators import MiniCPMVImageTextSFTCollator
from .collators import QwenVLImageTextSFTCollator
from .collators import TextSFTCollator
from .data import build_raw_text_jsonl_dataset
from .flatquant_linear import CompressionLoRAFlatQuantLinear
from .flatquant_linear import LoRAConfig
from .mask_utils import load_masks
from .mask_utils import mask_sparsity
from .mask_utils import validate_masks
from .sft_data import build_alpaca_sft_dataset
from .sft_data import build_llava_sft_dataset


LOGGER = logging.getLogger(__name__)


def _is_flatquant_linear(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "FlatQuantizedLinear" and hasattr(module, "linear")


def _set_child_module(root: torch.nn.Module, qualified_name: str, replacement: torch.nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _collect_adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, CompressionLoRAFlatQuantLinear):
            continue
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
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
    for key, tensor in adapter_state.items():
        module_name, param_name = key.rsplit(".", maxsplit=1)
        module = module_map.get(module_name)
        if not isinstance(module, CompressionLoRAFlatQuantLinear):
            raise KeyError(f"Adapter target {module_name!r} is not a compression LoRA wrapper.")
        param = getattr(module, param_name)
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _parse_train_plan(plan: str | None) -> list[str]:
    if not plan:
        return ["cpt", "sft"]
    stages = [stage.strip().lower() for stage in plan.split(",") if stage.strip()]
    if not stages:
        return ["cpt", "sft"]
    allowed = {"cpt", "sft"}
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise ValueError(f"Unsupported compression_lora train stage(s): {unknown}. Allowed: {sorted(allowed)}")
    return stages


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
        )
        return dataset, TextSFTCollator(tokenizer, max_length=int(args.sequence_length)), _default_causal_lm_loss

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
) -> dict[str, Any]:
    trainable_params = [
        param
        for module in model.modules()
        if isinstance(module, CompressionLoRAFlatQuantLinear)
        for param in (module.lora_A, module.lora_B)
        if param.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("No trainable compression LoRA parameters found.")

    grad_accum = max(1, int(grad_accum))
    dataloader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_fn,
    )
    if len(dataloader) == 0:
        raise RuntimeError("Compression LoRA train dataset is empty.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    max_steps = int(max_steps)
    if max_steps <= 0:
        max_steps = max(1, math.ceil(len(dataloader) * float(num_train_epochs) / grad_accum))
    logging_steps = max(1, int(logging_steps))
    input_device = _infer_input_device(model)
    visual_device = _infer_visual_device(model)
    trainable_devices = _trainable_param_devices(trainable_params)
    hf_device_summary = _summarize_hf_device_map(model)
    LOGGER.info(
        "compression_lora %s training: samples=%s batch_size=%s grad_accum=%s max_steps=%s gradient_checkpointing=%s",
        stage_name,
        len(train_dataset),
        batch_size,
        grad_accum,
        max_steps,
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
    loss_sum = 0.0
    loss_count = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        for batch in dataloader:
            batch = _move_batch_for_model(batch, model)
            loss = loss_fn(model, batch)
            if not torch.isfinite(loss).item():
                _raise_nonfinite_loss(model, batch, loss)
            (loss / grad_accum).backward()
            loss_sum += float(loss.detach().cpu())
            loss_count += 1
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            bad_grads = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad
                and param.grad is not None
                and not torch.isfinite(param.grad).all().item()
            ]
            if bad_grads:
                preview = ", ".join(bad_grads[:8])
                raise RuntimeError(f"Compression LoRA produced non-finite gradients: {preview}")

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if torch.cuda.is_available():
                for sync_device in trainable_devices:
                    if sync_device.type == "cuda":
                        torch.cuda.synchronize(sync_device)
            now = time.perf_counter()
            step_elapsed = now - step_start_time
            total_elapsed = now - start_time
            step_start_time = now

            adapter_state = _collect_adapter_state(model)
            _assert_finite_adapter_state(adapter_state)

            if global_step == 1 or global_step % logging_steps == 0:
                avg_loss = loss_sum / max(1, loss_count)
                LOGGER.info(
                    "compression_lora %s step %s/%s loss %.6f grad_norm %.6f lr %.6g time %.3fs elapsed %.3fs",
                    stage_name,
                    global_step,
                    max_steps,
                    avg_loss,
                    float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    float(learning_rate),
                    step_elapsed,
                    total_elapsed,
                )
            if global_step >= max_steps:
                break

    elapsed = time.perf_counter() - start_time
    return {
        "train_runtime": elapsed,
        "train_samples_per_second": len(train_dataset) / elapsed if elapsed > 0 else 0.0,
        "train_steps_per_second": global_step / elapsed if elapsed > 0 else 0.0,
        "train_loss": loss_sum / max(1, loss_count),
        "global_step": global_step,
        "train_examples": len(train_dataset),
        "stage": stage_name,
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "trainer": "manual_pytorch",
    }


def _replace_flatquant_linears_with_lora(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
    config: LoRAConfig,
) -> dict[str, dict[str, Any]]:
    module_map = dict(model.named_modules())
    replaced: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Compression LoRA mask target not found: {name}")
        if isinstance(module, CompressionLoRAFlatQuantLinear):
            continue
        if not _is_flatquant_linear(module):
            raise TypeError(
                f"Compression LoRA currently supports FlatQuantizedLinear only; "
                f"target {name} is {module.__class__.__name__}."
            )
        wrapper = CompressionLoRAFlatQuantLinear(module, mask, config)
        _set_child_module(model, name, wrapper)
        replaced[name] = {
            "in_features": int(wrapper.in_features),
            "out_features": int(wrapper.out_features),
            "rank": int(config.rank),
            "alpha": float(config.alpha),
            "dropout": float(config.dropout),
        }
    return replaced


@torch.no_grad()
def _unwrap_lora_wrappers(model: torch.nn.Module) -> list[str]:
    merged: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, CompressionLoRAFlatQuantLinear):
            continue
        base = module.merge_into_base()
        _set_child_module(model, name, base)
        merged.append(name)
    return merged


def _select_flatquant_apply_wrapper(model, source_root: Path):
    from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
    from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
    from flatquant.model_tools.minicpm_utils import apply_flatquant_to_minicpm
    from flatquant.model_tools.qwen3_utils import apply_flatquant_to_qwen3
    from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen

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
    if model_type == "qwen3_5":
        from flatquant.model_tools.qwen3_5_utils import apply_flatquant_to_qwen3_5

        return apply_flatquant_to_qwen3_5
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from flatquant.model_tools.qwen3_5_utils import apply_flatquant_to_qwen3_5_moe

        return apply_flatquant_to_qwen3_5_moe
    raise NotImplementedError(f"FlatQuant compression_lora does not support model_type={model_type!r}.")


class CompressionLoRAMethod(BaseFinetuningMethod):
    name = "compression_lora"
    npu_ready = False

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = (
            f"{self.name}_r{args.compression_lora_rank}"
            f"_lr{args.compression_lora_learning_rate:g}"
            f"_seq{args.sequence_length}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, args) -> None:
        if getattr(args, "quantization", None) != "flatquant":
            raise ValueError("compression_lora v1 only supports --quantization flatquant.")
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

    def apply_finetuning(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(args)
        del model, tokenizer_bundle
        empty_cache(args.device)

        output_dir = ensure_dir(Path(args._workflow_output_dir))
        adapter_path = output_dir / "compression_lora_adapter.pth"
        cpt_adapter_path = output_dir / "compression_lora_cpt_adapter.pth"
        config_path = output_dir / "compression_lora_config.json"

        flatquant_from = getattr(args, "compression_lora_flatquant_from", None) or getattr(args, "flatquant_resume_from", None)
        if not flatquant_from:
            raise ValueError(
                "compression_lora requires FlatQuant parameters. Provide --compression_lora_flatquant_from "
                "or run it after a flatquant stage that produced flat_parameters.pth."
            )
        if not (Path(flatquant_from) / "flat_parameters.pth").exists():
            raise FileNotFoundError(f"Missing flat_parameters.pth under {flatquant_from}")

        model, tokenizer_bundle, source_args, source_root = self._load_fp_flatquant_model(args, flatquant_from)
        tokenizer = tokenizer_bundle.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must define eos_token or pad_token for compression_lora.")
            tokenizer.pad_token = tokenizer.eos_token

        masks_path = getattr(args, "compression_lora_masks_from", None)
        if not masks_path:
            raise ValueError(
                "compression_lora requires --compression_lora_masks_from or an earlier pruning stage "
                "that generated pruning_masks.pth."
            )
        masks, mask_metadata = load_masks(masks_path)
        validate_masks(model, masks)
        lora_config = LoRAConfig(
            rank=int(args.compression_lora_rank),
            alpha=float(args.compression_lora_alpha),
            dropout=float(args.compression_lora_dropout),
            init=str(args.compression_lora_init),
        )
        adapter_layers = _replace_flatquant_linears_with_lora(model, masks, lora_config)
        if not adapter_layers:
            raise RuntimeError("No FlatQuantizedLinear layers were wrapped for compression_lora.")

        adapter_from = getattr(args, "compression_lora_adapter_from", None)
        if adapter_from:
            payload = torch.load(adapter_from, map_location="cpu")
            _load_adapter_state(model, payload["state_dict"] if "state_dict" in payload else payload)
            train_metrics = {"skipped_training": True, "adapter_from": adapter_from}
            train_plan = ["adapter_from"]
        else:
            for param in model.parameters():
                param.requires_grad = False
            for module in model.modules():
                if isinstance(module, CompressionLoRAFlatQuantLinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
            if bool(args.compression_lora_gradient_checkpointing):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
            model.config.use_cache = False
            model.train()

            train_plan = _parse_train_plan(getattr(args, "compression_lora_train_plan", None))
            stage_metrics: dict[str, Any] = {}
            for stage in train_plan:
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
                    )
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
                    )
                    continue

            train_metrics = {
                "train_plan": train_plan,
                "stages": stage_metrics,
                "cpt_train_file": getattr(args, "compression_lora_cpt_train_file", None),
                "sft_train_file": getattr(args, "compression_lora_sft_train_file", None),
                "sft_format": getattr(args, "compression_lora_sft_format", None),
            }
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
            _purge_conflicting_modules("flatquant", source_root / "flatquant")
            _purge_conflicting_modules("gptq_utils", source_root)
            import gptq_utils
            from flatquant.flat_utils import reparameterize_model

            reparameterize_model(model)
            if source_args.w_bits < 16:
                if source_args.gptq:
                    gptq_utils.gptq_fwrd(model, calibration_batches, args.device, source_args)
                else:
                    gptq_utils.rtn_fwrd(model, args.device, source_args)
        from .mask_utils import apply_masks_to_model

        apply_masks_to_model(model, masks)
        model.config.use_cache = True
        model.eval()
        model.seqlen = int(args.sequence_length)

        config_payload = {
            "mode": "quant_then_prune_fixed_mask",
            "adapter_layers": adapter_layers,
            "merged_layers": merged_layers,
            "masks_path": str(masks_path),
            "flatquant_from": str(flatquant_from),
            "mask_metadata": mask_metadata,
            "mask_sparsity": mask_sparsity(masks),
            "train_metrics": train_metrics,
            "train_plan": train_plan,
            "save_merged_model": bool(args.compression_lora_save_merged_model),
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
            "compression_lora_adapter_path": str(adapter_path) if adapter_path.exists() else adapter_from,
            "compression_lora_cpt_adapter_path": str(cpt_adapter_path) if cpt_adapter_path.exists() else None,
            "compression_lora_wrapped_layer_count": len(adapter_layers),
            "compression_lora_merged_layers": merged_layers,
            "train_metrics": train_metrics,
            "_updated_model": model,
            "_updated_tokenizer_bundle": tokenizer_bundle,
        }
        if merged_model_dir is not None:
            artifacts["compression_lora_merged_model_dir"] = str(merged_model_dir)
        return artifacts
