"""MQuant adapter for multimodal VLM quantization."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import math
import os
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ....common.device import resolve_device
from ....common.modeling import MiniCPMTokenizerAdapter
from ....common.modeling import move_tensors_to_device
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


def _ensure_vlmeval_transformers_compat() -> None:
    """Patch removed/renamed HF symbols expected by older VLMEvalKit revisions."""
    try:
        import transformers
    except Exception:
        return

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        fallback = getattr(transformers, "AutoModelForImageTextToText", None)
        if fallback is not None:
            transformers.AutoModelForVision2Seq = fallback


def _ensure_image_url(image: str) -> str:
    prefixes = ("http://", "https://", "file://", "data:image;")
    if image.startswith(prefixes):
        return image
    return Path(image).expanduser().resolve().as_uri()


def _build_qwen2_messages(message, dataset=None):
    conversation = []
    for item in message:
        item_type = item.get("type")
        if item_type == "text":
            conversation.append({"type": "text", "text": item["value"]})
            continue
        if item_type == "image":
            content_item = {"type": "image", "image": _ensure_image_url(str(item["value"]))}
            if dataset == "OCRBench":
                content_item["min_pixels"] = 10 * 10 * 28 * 28
            conversation.append(content_item)
            continue
        if item_type == "video":
            raise NotImplementedError("MQuant GPTQ adapter does not support video prompts.")
        raise ValueError(f"Unsupported message item: {item}")
    return [{"role": "user", "content": conversation}]


def _open_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _build_internvl_transform(input_size: int):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except Exception as err:
        raise RuntimeError(
            "InternVL MQuant GPTQ wrapper requires torchvision for image preprocessing."
        ) from err

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _find_internvl_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess_internvl_image(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                blocks = i * j
                if min_num <= blocks <= max_num:
                    target_ratios.add((i, j))
    sorted_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    target_aspect_ratio = _find_internvl_closest_aspect_ratio(
        aspect_ratio,
        sorted_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height))

    processed_images: list[Image.Image] = []
    grid_width = target_width // image_size
    for block_idx in range(blocks):
        box = (
            (block_idx % grid_width) * image_size,
            (block_idx // grid_width) * image_size,
            ((block_idx % grid_width) + 1) * image_size,
            ((block_idx // grid_width) + 1) * image_size,
        )
        processed_images.append(resized.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_internvl_image(
    image_path: str,
    *,
    input_size: int,
    max_num: int,
    use_thumbnail: bool,
    transform=None,
) -> tuple[torch.Tensor, int]:
    image = _open_rgb_image(image_path)
    transform = transform or _build_internvl_transform(input_size)
    processed_images = _dynamic_preprocess_internvl_image(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    pixel_values = torch.stack([transform(item) for item in processed_images])
    return pixel_values, pixel_values.size(0)


def _internvl_prompt_from_message(message) -> tuple[str, int]:
    image_num = len([item for item in message if item["type"] == "image"])
    if image_num == 1:
        prompt = "<image>\n" + "\n".join(
            item["value"] for item in message if item["type"] == "text"
        )
    else:
        prompt = ""
        image_idx = 1
        for item in message:
            item_type = item.get("type")
            if item_type == "text":
                prompt += item["value"]
            elif item_type == "image":
                prompt += f"<image-{image_idx}>"
                image_idx += 1
            elif item_type == "video":
                raise NotImplementedError("MQuant GPTQ adapter does not support video prompts.")
            else:
                raise ValueError(f"Unsupported message item: {item}")
        if image_num > 1:
            prompt = (
                " ".join(f"<image-{index + 1}>: <image>" for index in range(image_num))
                + "\n"
                + prompt
            )
    if image_num == 0:
        prompt = "\n".join(item["value"] for item in message if item["type"] == "text")
    return prompt.strip(), image_num


def _internvl_default_max_num(dataset: str | None) -> int:
    if dataset in {"ChartQA_TEST", "MMMU_DEV_VAL"}:
        return 12
    if dataset in {"DocVQA_VAL", "DocVQA_TEST"}:
        return 18
    if dataset in {"InfoVQA_VAL", "InfoVQA_TEST", "OCRBench", "HRBench4K", "HRBench8K"}:
        return 24
    return 6


class _MindPipeQwen2VLGPTQWrapper:
    """Minimal wrapper to satisfy MQuant GPTQ dataset-driven collection."""

    def __init__(
        self,
        *,
        model_root,
        source_model,
        processor,
        tokenizer,
        target_device,
        max_new_tokens: int,
    ):
        self.model = model_root
        self._source_model = source_model
        self._processor = processor
        self._tokenizer = tokenizer
        self._target_device = resolve_device(target_device)
        self._max_new_tokens = int(max_new_tokens)
        # Keep compatibility with upstream MQuant calibration helpers which mutate this dict.
        self.generate_kwargs = {"max_new_tokens": int(max_new_tokens)}
        self._model_prepared = False

    @staticmethod
    def _maybe_to_device(inputs, device):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if isinstance(inputs, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs

    @staticmethod
    def _model_input_device(model, fallback):
        try:
            return next(model.parameters()).device
        except Exception:
            return fallback

    def use_custom_prompt(self, dataset):
        return False

    def build_prompt(self, line, dataset):
        raise NotImplementedError("MindPipe MQuant wrapper relies on dataset prompts.")

    def _ensure_model_ready(self):
        if self._model_prepared:
            return
        if not getattr(self._source_model, "hf_device_map", None):
            self._source_model.to(self._target_device)
        if hasattr(self._source_model, "config") and hasattr(self._source_model.config, "use_cache"):
            self._source_model.config.use_cache = False
        if hasattr(self._source_model, "llm"):
            if hasattr(self._source_model.llm, "config") and hasattr(self._source_model.llm.config, "use_cache"):
                self._source_model.llm.config.use_cache = False
            if getattr(self._source_model.llm, "generation_config", None) is not None:
                self._source_model.llm.generation_config.use_cache = False
        self._source_model.eval()
        self._model_prepared = True

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError(
                "qwen_vl_utils is required for MQuant GPTQ multimodal calibration. "
                "Please install `qwen-vl-utils`."
            ) from err

        self._ensure_model_ready()
        messages = _build_qwen2_messages(message, dataset)
        prompt = self._processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info([messages])
        inputs = self._processor(
            text=prompt,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        model_device = self._model_input_device(self._source_model, self._target_device)
        inputs = self._maybe_to_device(inputs, model_device)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated_ids = self._source_model.generate(
            **inputs,
            max_new_tokens=int(self.generate_kwargs.get("max_new_tokens", self._max_new_tokens)),
            do_sample=False,
        )
        trimmed_ids = [
            output_ids[len(input_row):]
            for input_row, output_ids in zip(input_ids, generated_ids)
        ]
        responses = self._tokenizer.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return responses[0].strip()

    def generate(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)


def _open_minicpmv_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


class _MindPipeInternVLGPTQWrapper:
    """InternVL2 wrapper matching MQuant GPTQ dataset-driven collection hooks."""

    def __init__(
        self,
        *,
        source_model,
        tokenizer,
        target_device,
        max_new_tokens: int,
        use_cache: bool = False,
    ):
        self.model = source_model
        self.tokenizer = tokenizer
        self._target_device = resolve_device(target_device)
        self._max_new_tokens = int(max_new_tokens)
        self._use_cache = bool(use_cache)
        self.generate_kwargs = {"max_new_tokens": int(max_new_tokens)}
        self._model_prepared = False
        self._input_size = int(
            getattr(
                getattr(getattr(source_model, "config", None), "vision_config", None),
                "image_size",
                448,
            )
        )
        self._use_thumbnail = True
        self._transform = _build_internvl_transform(self._input_size)
        try:
            self._chat_signature = inspect.signature(source_model.chat)
        except (TypeError, ValueError):
            self._chat_signature = None

    @staticmethod
    def _model_input_device(model, fallback):
        try:
            return next(model.parameters()).device
        except Exception:
            return fallback

    @staticmethod
    def _model_dtype(model):
        model_dtype = getattr(model, "dtype", None)
        if isinstance(model_dtype, torch.dtype) and model_dtype.is_floating_point:
            return model_dtype
        try:
            parameter_dtype = next(model.parameters()).dtype
            if parameter_dtype.is_floating_point:
                return parameter_dtype
        except Exception:
            pass
        return torch.bfloat16

    def use_custom_prompt(self, dataset):
        return False

    def build_prompt(self, line, dataset):
        raise NotImplementedError("MindPipe InternVL GPTQ wrapper relies on dataset prompts.")

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.generate_kwargs

    @kwargs.setter
    def kwargs(self, value: dict[str, Any] | None) -> None:
        self.generate_kwargs = dict(value or {})

    def _ensure_model_ready(self):
        if self._model_prepared:
            return
        if not getattr(self.model, "hf_device_map", None):
            self.model.to(self._target_device)
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self._use_cache
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            if hasattr(language_model, "config") and hasattr(language_model.config, "use_cache"):
                language_model.config.use_cache = self._use_cache
            if getattr(language_model, "generation_config", None) is not None:
                language_model.generation_config.use_cache = self._use_cache
        self.model.eval()
        self._model_prepared = True

    def generate_inner(self, message, dataset=None):
        self._ensure_model_ready()
        prompt, image_num = _internvl_prompt_from_message(message)
        image_paths = [item["value"] for item in message if item["type"] == "image"]
        max_num = _internvl_default_max_num(dataset)
        model_device = self._model_input_device(self.model, self._target_device)
        model_dtype = self._model_dtype(self.model)

        pixel_values = None
        num_patches_list: list[int] = []
        if image_paths:
            pixel_values_list = []
            for image_path in image_paths:
                current_pixels, num_patches = _load_internvl_image(
                    image_path,
                    input_size=self._input_size,
                    max_num=max_num,
                    use_thumbnail=self._use_thumbnail,
                    transform=self._transform,
                )
                pixel_values_list.append(current_pixels)
                num_patches_list.append(num_patches)
            pixel_values = torch.cat(pixel_values_list, dim=0).to(
                device=model_device,
                dtype=model_dtype,
            )

        generation_config = {
            "max_new_tokens": int(self.generate_kwargs.get("max_new_tokens", self._max_new_tokens)),
            "do_sample": False,
            "num_beams": 1,
        }
        chat_kwargs: dict[str, Any] = {
            "tokenizer": self.tokenizer,
            "pixel_values": pixel_values,
            "question": prompt,
            "generation_config": generation_config,
        }
        if image_num > 0 and self._chat_signature is not None and "num_patches_list" in self._chat_signature.parameters:
            chat_kwargs["num_patches_list"] = num_patches_list
        if self._chat_signature is not None and "verbose" in self._chat_signature.parameters:
            chat_kwargs["verbose"] = False

        with torch.no_grad():
            response = self.model.chat(**chat_kwargs)
        if isinstance(response, tuple) and response:
            response = response[0]
        return str(response).strip()

    def generate(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)


class _MindPipeMiniCPMVGPTQWrapper:
    """MiniCPM-V wrapper matching MQuant GPTQ dataset-driven collection hooks."""

    def __init__(
        self,
        *,
        source_model,
        tokenizer,
        target_device,
        max_new_tokens: int,
        use_cache: bool = False,
    ):
        self.model = source_model
        if not isinstance(tokenizer, MiniCPMTokenizerAdapter):
            tokenizer = MiniCPMTokenizerAdapter(tokenizer)
        self.tokenizer = tokenizer
        self._target_device = resolve_device(target_device)
        self._max_new_tokens = int(max_new_tokens)
        self._use_cache = bool(use_cache)
        self.generate_kwargs = {"max_new_tokens": int(max_new_tokens)}
        self._model_prepared = False
        try:
            self._chat_signature = inspect.signature(source_model.chat)
        except (TypeError, ValueError):
            self._chat_signature = None

    def use_custom_prompt(self, dataset):
        return False

    def build_prompt(self, line, dataset):
        raise NotImplementedError("MindPipe MiniCPM-V GPTQ wrapper relies on dataset prompts.")

    def _ensure_model_ready(self):
        if self._model_prepared:
            return
        if not getattr(self.model, "hf_device_map", None):
            self.model.to(self._target_device)
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self._use_cache
        if hasattr(self.model, "llm"):
            if hasattr(self.model.llm, "config") and hasattr(self.model.llm.config, "use_cache"):
                self.model.llm.config.use_cache = self._use_cache
            if getattr(self.model.llm, "generation_config", None) is not None:
                self.model.llm.generation_config.use_cache = self._use_cache
        self.model.eval()
        self._model_prepared = True

    @staticmethod
    def _unwrap_response(response):
        if isinstance(response, tuple) and len(response) > 0:
            return response[0]
        return response

    def _generation_context(self):
        model_device = resolve_device(self._target_device)
        try:
            model_dtype = next(self.model.parameters()).dtype
        except Exception:
            model_dtype = torch.bfloat16
        if model_device.type in {"cuda", "npu"}:
            return torch.autocast(device_type=model_device.type, dtype=model_dtype)
        return contextlib.nullcontext()

    def _generate_legacy(self, prompt: str, images: list[Image.Image], max_new_tokens: int):
        with self._generation_context():
            return self._unwrap_response(
                self.model.chat(
                    image=images[0] if images else None,
                    msgs=[{"role": "user", "content": prompt}],
                    context=None,
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_new_tokens,
                    sampling=False,
                    num_beams=1,
                    use_cache=self._use_cache,
                )
            )

    def _generate_interleaved(self, message, max_new_tokens: int):
        content = []
        for item in message:
            item_type = item.get("type")
            if item_type == "text":
                content.append(item["value"])
            elif item_type == "image":
                content.append(_open_minicpmv_image(item["value"]))
            elif item_type == "video":
                raise NotImplementedError("MindPipe MiniCPM-V GPTQ wrapper does not support video prompts.")
            else:
                raise ValueError(f"Unsupported message item: {item}")
        if not content:
            content = [""]
        with self._generation_context():
            return self._unwrap_response(
                self.model.chat(
                    msgs=[{"role": "user", "content": content}],
                    context=None,
                    image=None,
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_new_tokens,
                    sampling=False,
                    num_beams=1,
                    use_cache=self._use_cache,
                )
            )

    def generate_inner(self, message, dataset=None):
        self._ensure_model_ready()
        max_new_tokens = int(self.generate_kwargs.get("max_new_tokens", self._max_new_tokens))
        prompt = "\n".join(item["value"] for item in message if item.get("type") == "text")
        images = [_open_minicpmv_image(item["value"]) for item in message if item.get("type") == "image"]
        if self._chat_signature is not None and "image" in self._chat_signature.parameters:
            return self._generate_legacy(prompt=prompt, images=images, max_new_tokens=max_new_tokens)
        try:
            return self._generate_legacy(prompt=prompt, images=images, max_new_tokens=max_new_tokens)
        except TypeError:
            return self._generate_interleaved(message=message, max_new_tokens=max_new_tokens)

    def generate(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)


class _Qwen2VLCompatRoot(torch.nn.Module):
    """Module container matching legacy MQuant expected attribute layout."""

    def __init__(self, *, visual, text_root, lm_head, config):
        super().__init__()
        self.visual = visual
        self.model = text_root
        self.lm_head = lm_head
        self.config = config


class _Qwen3VLTextCompatInner(torch.nn.Module):
    """Expose the text stack under `.model` for legacy Qwen2-VL helpers."""

    def __init__(self, *, layers, norm):
        super().__init__()
        self.layers = layers
        self.norm = norm


class _Qwen3VLTextCompatRoot(torch.nn.Module):
    """Expose both `.layers` and `.model.layers` for legacy MQuant code paths."""

    def __init__(self, text_root):
        super().__init__()
        self.embed_tokens = text_root.embed_tokens
        self.model = _Qwen3VLTextCompatInner(
            layers=text_root.layers,
            norm=text_root.norm,
        )
        self.config = text_root.config

    @property
    def layers(self):
        return self.model.layers

    @property
    def norm(self):
        return self.model.norm


def _unwrap_module(module):
    seen: set[int] = set()
    current = module
    while current is not None and hasattr(current, "module"):
        next_module = getattr(current, "module")
        if next_module is None or id(next_module) == id(current) or id(next_module) in seen:
            break
        seen.add(id(current))
        current = next_module
    return current


def _record_mquant_quantizer(quantizers: dict[str, Any] | None, name: str, quantizer: Any) -> None:
    if quantizers is None:
        return
    quantizers[name] = quantizer.cpu() if hasattr(quantizer, "cpu") else quantizer


def _mindpipe_internvl_visual_clip_rtn(
    model,
    dev,
    args,
    quant_utils,
    quantizers: dict[str, Any] | None = None,
) -> None:
    patch_embedding = getattr(model.vision_model.embeddings, "patch_embedding", None)
    patch_leaf = _unwrap_module(patch_embedding)
    if patch_leaf is None or not hasattr(patch_leaf, "weight"):
        raise AttributeError("InternVL patch_embedding does not expose a quantizable weight.")

    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.visual_w_bits,
        perchannel=True,
        sym=not bool(args.w_asym),
        mse=bool(args.visual_w_clip),
    )
    weight = patch_leaf.weight.data
    quantizer.find_params(weight)
    patch_leaf.weight.data = quantizer.quantize(weight).to(patch_leaf.weight.dtype)
    _record_mquant_quantizer(
        quantizers,
        "model.vision_model.embeddings.patch_embedding",
        quantizer,
    )

    layers = model.vision_model.encoder.layers
    for layer_idx in range(len(layers)):
        # device_map 模式下不手动移动 layer
        layer = layers[layer_idx]
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for name, linear in subset.items():
            if any(skip_name in name for skip_name in args.skip_names) or "L1" in name:
                continue
            layer_quantizer = quant_utils.WeightQuantizer()
            layer_quantizer.configure(
                args.visual_w_bits,
                perchannel=True,
                sym=not bool(args.w_asym),
                mse=bool(args.visual_w_clip),
            )
            weight = linear.weight.data
            layer_quantizer.find_params(weight)
            linear.weight.data = layer_quantizer.quantize(weight).to(weight.dtype)
            _record_mquant_quantizer(
                quantizers,
                f"model.vision_model.encoder.layers.{layer_idx}.{name}",
                layer_quantizer,
            )


def _mindpipe_internvl_visual_cross_attention_rtn(
    model,
    dev,
    args,
    quant_utils,
    quantizers: dict[str, Any] | None = None,
) -> None:
    subset = quant_utils.find_qlayers(model.mlp1, layers=[torch.nn.Linear])
    for name, linear in subset.items():
        if any(skip_name in name for skip_name in args.skip_names) or "L1" in name:
            continue
        layer_quantizer = quant_utils.WeightQuantizer()
        layer_quantizer.configure(
            args.visual_w_bits,
            perchannel=True,
            sym=not bool(args.w_asym),
            mse=bool(args.visual_w_clip),
        )
        weight = linear.weight.data
        layer_quantizer.find_params(weight)
        linear.weight.data = layer_quantizer.quantize(weight).to(weight.dtype)
        _record_mquant_quantizer(
            quantizers,
            f"model.resampler.{name}",
            layer_quantizer,
        )


@torch.no_grad()
def _mindpipe_gptq_internvl_fwrd_visual_clip_conv1(
    internvl_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
    quantizers,
):
    """Collect InternVL2 conv1 inputs from either positional args or `pixel_values=...` kwargs."""
    use_cache = model.model.config.llm_config.use_cache
    model.model.config.llm_config.use_cache = False
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *call_args, **kwargs):
            captured = call_args[0] if len(call_args) > 0 else kwargs.get("pixel_values")
            if captured is None:
                captured = kwargs.get("inputs_embeds")
            inps[cache["i"]] = captured
            cache["i"] += 1
            raise ValueError

    model.model.vision_model.embeddings = Catcher(model.model.vision_model.embeddings)
    try:
        lt = len(dataset.data)
        for i in range(lt):
            if cache["i"] >= args.nsamples:
                break
            struct = dataset.build_prompt(dataset.data.iloc[i])
            try:
                model.generate(message=struct, dataset=args.dataset_name)
            except ValueError:
                pass
    finally:
        model.model.vision_model.embeddings = model.model.vision_model.embeddings.module

    nsamples = cache["i"]
    if nsamples == 0:
        raise RuntimeError("InternVL GPTQ conv1 capture collected zero samples.")
    if any(inp is None for inp in inps[:nsamples]):
        raise RuntimeError("InternVL GPTQ conv1 capture observed empty `pixel_values` inputs.")

    layer_weight_bits = args.visual_w_bits
    layer_weight_sym = not bool(args.w_asym)
    conv1_module = _unwrap_module(model.model.vision_model.embeddings.patch_embedding)
    if conv1_module is None:
        raise AttributeError("InternVL GPTQ conv1 could not resolve patch embedding leaf module.")
    conv1_gptq = internvl_gptq_plus.GPTQConv(conv1_module)
    conv1_gptq.quantizer = internvl_gptq_plus.quant_utils.WeightQuantizer()
    conv1_gptq.quantizer.configure(
        layer_weight_bits,
        perchannel=True,
        sym=layer_weight_sym,
        mse=args.visual_w_clip,
    )

    def add_batch():
        def tmp(_, inp, out):
            conv1_gptq.add_batch(inp[0].data, out.data)

        return tmp

    handles = [conv1_module.register_forward_hook(add_batch())]
    try:
        for j in range(nsamples):
            model.model.vision_model.embeddings.patch_embedding(inps[j])
    finally:
        for handle in handles:
            handle.remove()
    layer_w_groupsize = args.w_groupsize
    conv1_gptq.fasterquant(
        percdamp=args.percdamp,
        groupsize=layer_w_groupsize,
        actorder=args.act_order,
        static_groups=False,
    )
    quantizers["model.vision_model.embeddings.patch_embedding"] = conv1_gptq.quantizer
    conv1_gptq.free()
    del conv1_gptq
    model.model.config.llm_config.use_cache = use_cache
    internvl_gptq_plus.utils.cleanup_memory(verbos=True)
    print("-----GPTQ Quantization visual clip conv1 Done-----")


@torch.no_grad()
def _mindpipe_gptq_internvl_fwrd_visual_clip_resblocks(
    internvl_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
    quantizers,
):
    """Collect InternVL2 visual block inputs from positional args or `hidden_states=...` kwargs."""
    use_cache = model.model.config.llm_config.use_cache
    model.model.config.llm_config.use_cache = False
    layers = model.model.vision_model.encoder.layers
    # device_map 模式下不手动移动 layers[0]
    layer0_device = next(layers[0].parameters()).device
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *call_args, **kwargs):
            captured = call_args[0] if len(call_args) > 0 else kwargs.get("hidden_states")
            inps[cache["i"]] = captured
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        lt = len(dataset.data)
        for i in range(lt):
            if cache["i"] >= args.nsamples:
                break
            struct = dataset.build_prompt(dataset.data.iloc[i])
            try:
                model.generate(message=struct, dataset=args.dataset_name)
            except ValueError:
                pass
    finally:
        layers[0] = layers[0].module

    nsamples = cache["i"]
    if nsamples == 0:
        raise RuntimeError("InternVL GPTQ visual block capture collected zero samples.")
    if any(inp is None for inp in inps[:nsamples]):
        raise RuntimeError("InternVL GPTQ visual block capture observed empty hidden states.")

    outs = [None] * nsamples
    sequential = [["attn.qkv.module"], ["attn.proj.module"], ["mlp.fc1.module"]]
    if args.visual_split:
        sequential.append(["mlp.fc2.L2"])
    else:
        sequential.append(["mlp.fc2.module"])

    for layer_idx in range(len(layers)):
        print(f"\nLayer {layer_idx}:", flush=True, end=" ")
        # device_map 模式下不手动移动 layer
        layer = layers[layer_idx]
        # 将输入数据移到当前层设备
        layer_dev = next(layer.parameters()).device
        inps = [move_tensors_to_device(inp, layer_dev) if inp is not None else inp for inp in inps]
        outs = [move_tensors_to_device(o, layer_dev) if o is not None else o for o in outs]
        full = internvl_gptq_plus.quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            if any(skip_name in name for skip_name in args.skip_names for name in names):
                continue
            subset = {name: full[name] for name in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.visual_w_bits
                layer_weight_sym = not bool(args.w_asym)
                gptq[name] = internvl_gptq_plus.GPTQ(subset[name])
                gptq[name].quantizer = internvl_gptq_plus.quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.visual_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(inps[j])
            for handle in handles:
                handle.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers[f"model.vision_model.encoder.layers.{layer_idx}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j])

        layers[layer_idx] = layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.model.config.llm_config.use_cache = use_cache
    internvl_gptq_plus.utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip resblocks Done----")


@torch.no_grad()
def _mindpipe_gptq_internvl_fwrd_visual_clip_cross_attention(
    internvl_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
    quantizers,
):
    """Collect InternVL2 connector inputs from positional args or keyword-only hidden states."""
    print("-----GPTQ Quantization visual clip cross attention-----")
    layer = model.model.mlp1
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *call_args, **kwargs):
            captured = call_args[0] if len(call_args) > 0 else kwargs.get("hidden_states")
            inps[cache["i"]] = captured
            cache["i"] += 1
            raise ValueError

    model.model.mlp1 = Catcher(model.model.mlp1)
    try:
        lt = len(dataset.data)
        for i in range(lt):
            if cache["i"] >= args.nsamples:
                break
            struct = dataset.build_prompt(dataset.data.iloc[i])
            try:
                model.generate(message=struct, dataset=args.dataset_name)
            except ValueError:
                pass
    finally:
        model.model.mlp1 = model.model.mlp1.module

    nsamples = cache["i"]
    if nsamples == 0:
        raise RuntimeError("InternVL GPTQ cross-attention capture collected zero samples.")
    if any(inp is None for inp in inps[:nsamples]):
        raise RuntimeError("InternVL GPTQ cross-attention capture observed empty hidden states.")

    sequential = [["1.module"], ["3.module"]]
    full = internvl_gptq_plus.quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
    for names in sequential:
        subset = {name: full[name] for name in names}
        gptq = {}
        for name in subset:
            print(f"{name}", end="  ", flush=True)
            layer_weight_bits = args.visual_w_bits
            layer_weight_sym = not bool(args.w_asym)
            gptq[name] = internvl_gptq_plus.GPTQ(subset[name])
            gptq[name].quantizer = internvl_gptq_plus.quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=layer_weight_sym,
                mse=args.visual_w_clip,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            layer(inps[j])
        for handle in handles:
            handle.remove()

        for name in subset:
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.w_groupsize,
                actorder=args.act_order,
                static_groups=False,
            )
            quantizers[f"model.resampler.{name}"] = gptq[name].quantizer
            gptq[name].free()

    model.model.mlp1 = layer
    del gptq
    torch.cuda.empty_cache()
    internvl_gptq_plus.utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip cross attention Done-----")


@torch.no_grad()
def _mindpipe_gptq_internvl_fwrd_llm(
    internvl_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
    quantizers,
):
    """Collect InternVL2 language hidden states from positional args or `hidden_states=...` kwargs."""
    print("-----GPTQ Quantization LLM-----")
    layers = model.model.language_model.model.layers
    # device_map 模式下不手动移动 layers[0]
    layer0_device = next(layers[0].parameters()).device
    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    position_ids = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *call_args, **kwargs):
            captured = call_args[0] if len(call_args) > 0 else kwargs.get("hidden_states")
            inps[cache["i"]] = captured
            attention_masks[cache["i"]] = kwargs.get("attention_mask")
            position_ids[cache["i"]] = kwargs.get("position_ids")
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        lt = len(dataset.data)
        for i in range(lt):
            if cache["i"] >= args.nsamples:
                break
            if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(dataset.data.iloc[i])
            try:
                model.generate(message=struct, dataset=args.dataset_name)
            except ValueError:
                pass
    finally:
        layers[0] = layers[0].module

    nsamples = cache["i"]
    if nsamples == 0:
        raise RuntimeError("InternVL GPTQ LLM capture collected zero samples.")
    if any(inp is None for inp in inps[:nsamples]):
        raise RuntimeError("InternVL GPTQ LLM capture observed empty hidden states.")

    outs = [None] * nsamples
    sequential = [["attention.wqkv.module"], ["attention.wo.module"], ["feed_forward.w1.module", "feed_forward.w3.module"]]
    if args.llm_split:
        sequential.append(["feed_forward.w2.L2"])
    else:
        sequential.append(["feed_forward.w2.module"])

    for layer_idx in range(len(layers)):
        print(f"\nLayer {layer_idx}:", flush=True, end=" ")
        # device_map 模式下不手动移动 layer
        layer = layers[layer_idx]
        # 将输入数据移到当前层设备
        layer_dev = next(layer.parameters()).device
        inps = [move_tensors_to_device(inp, layer_dev) if inp is not None else inp for inp in inps]
        outs = [move_tensors_to_device(o, layer_dev) if o is not None else o for o in outs]
        attention_masks = [move_tensors_to_device(m, layer_dev) if m is not None else m for m in attention_masks]
        position_ids = [move_tensors_to_device(p, layer_dev) if p is not None else p for p in position_ids]
        full = internvl_gptq_plus.quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            if any(skip_name in name for skip_name in args.skip_names for name in names):
                continue
            subset = {name: full[name] for name in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.llm_w_bits
                layer_weight_sym = not bool(args.w_asym)
                gptq[name] = internvl_gptq_plus.GPTQ(subset[name])
                gptq[name].quantizer = internvl_gptq_plus.quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.llm_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_masks[j],
                    position_ids=position_ids[j],
                )[0]
            for handle in handles:
                handle.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers[f"model.language_model.model.layers.{layer_idx}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_masks[j],
                position_ids=position_ids[j],
            )[0]

        layers[layer_idx] = layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    internvl_gptq_plus.utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization LLM Done-----")
    return quantizers


class _MiniCPMVLeafModuleProxy(nn.Module):
    """Forward to the wrapped module but keep `.module.weight` access stable for MQuant."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    @property
    def weight(self):
        leaf = _unwrap_module(self.module)
        if leaf is None or not hasattr(leaf, "weight"):
            raise AttributeError(f"{type(leaf).__name__} does not expose `weight`.")
        return leaf.weight

    @property
    def bias(self):
        leaf = _unwrap_module(self.module)
        if leaf is None:
            return None
        return getattr(leaf, "bias", None)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _wrap_leaf_module_for_mquant(module: nn.Module | None):
    if module is None or hasattr(module, "module"):
        return module
    leaf = _unwrap_module(module)
    if not isinstance(leaf, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return module
    return _MiniCPMVLeafModuleProxy(module)


def _replace_module_attr_for_mquant(owner: nn.Module | None, attr_name: str) -> bool:
    if owner is None or not hasattr(owner, attr_name):
        return False
    current = getattr(owner, attr_name)
    wrapped = _wrap_leaf_module_for_mquant(current)
    if wrapped is current:
        return False
    setattr(owner, attr_name, wrapped)
    return True


def _ensure_internvl_gptq_mquant_compat(source_model) -> None:
    if getattr(source_model, "_mindpipe_internvl_gptq_mquant_compatible", False):
        return

    vision_model = getattr(source_model, "vision_model", None)
    embeddings = getattr(vision_model, "embeddings", None)
    _replace_module_attr_for_mquant(embeddings, "patch_embedding")

    encoder_layers = getattr(getattr(vision_model, "encoder", None), "layers", [])
    for layer in encoder_layers:
        attn = getattr(layer, "attn", None)
        _replace_module_attr_for_mquant(attn, "qkv")
        _replace_module_attr_for_mquant(attn, "proj")
        mlp = getattr(layer, "mlp", None)
        _replace_module_attr_for_mquant(mlp, "fc1")
        _replace_module_attr_for_mquant(mlp, "fc2")

    mlp1 = getattr(source_model, "mlp1", None)
    if isinstance(mlp1, nn.Sequential):
        for index in (1, 3):
            if index >= len(mlp1):
                continue
            wrapped = _wrap_leaf_module_for_mquant(mlp1[index])
            if wrapped is not mlp1[index]:
                mlp1[index] = wrapped

    llm_layers = getattr(getattr(getattr(source_model, "language_model", None), "model", None), "layers", [])
    for layer in llm_layers:
        attention = getattr(layer, "attention", None)
        _replace_module_attr_for_mquant(attention, "wqkv")
        _replace_module_attr_for_mquant(attention, "wo")
        feed_forward = getattr(layer, "feed_forward", None)
        _replace_module_attr_for_mquant(feed_forward, "w1")
        _replace_module_attr_for_mquant(feed_forward, "w2")
        _replace_module_attr_for_mquant(feed_forward, "w3")

    source_model._mindpipe_internvl_gptq_mquant_compatible = True


class _MiniCPMVPositionEmbeddingAdapter(nn.Module):
    """Expose `vpm.pos_embed` under the legacy `position_embedding.weight` path."""

    def __init__(self, owner: nn.Module):
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    @property
    def weight(self):
        return self._owner.pos_embed


class _MiniCPMVTimmEmbeddingsAdapter(nn.Module):
    """Expose `patch_embedding` / `position_embedding` like HF CLIP vision models."""

    def __init__(self, owner: nn.Module):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.position_embedding = _MiniCPMVPositionEmbeddingAdapter(owner)

    def __getattr__(self, name: str):
        if name == "patch_embedding":
            return self._owner.patch_embed.proj
        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
        if name == "patch_embedding":
            self._owner.patch_embed.proj = value
            return
        super().__setattr__(name, value)

    @property
    def embed_dim(self):
        return int(self._owner.embed_dim)

    def named_children(self):
        yield "patch_embedding", self._owner.patch_embed.proj
        yield "position_embedding", self.position_embedding


class _MiniCPMVTimmEncoderAdapter(nn.Module):
    """Expose the timm block stack under `.encoder.layers` for upstream MQuant."""

    def __init__(self, owner: nn.Module):
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    @property
    def layers(self):
        return self._owner.blocks

    def named_children(self):
        yield "layers", self._owner.blocks


def _maybe_add_mask(attn: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
    if attn_mask is None:
        return attn
    if attn_mask.dtype == torch.bool:
        return attn.masked_fill(attn_mask, torch.finfo(attn.dtype).min)
    return attn + attn_mask.to(dtype=attn.dtype, device=attn.device)


class _MiniCPMVTimmSplitAttention(nn.Module):
    """Split timm fused `qkv` attention into q/k/v linears expected by MQuant."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        qkv = attn.qkv
        device = qkv.weight.device
        dtype = qkv.weight.dtype
        bias = qkv.bias is not None
        out_features = qkv.out_features // 3
        in_features = qkv.in_features

        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.attn_dim = getattr(attn, "attn_dim", out_features)
        self.scale = getattr(attn, "scale", self.head_dim**-0.5)
        self.fused_attn = bool(getattr(attn, "fused_attn", False))
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.norm = attn.norm
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.q_proj = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

        q_weight, k_weight, v_weight = qkv.weight.data.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q_weight)
        self.k_proj.weight.data.copy_(k_weight)
        self.v_proj.weight.data.copy_(v_weight)
        if bias:
            q_bias, k_bias, v_bias = qkv.bias.data.chunk(3, dim=0)
            self.q_proj.bias.data.copy_(q_bias)
            self.k_proj.bias.data.copy_(k_bias)
            self.v_proj.bias.data.copy_(v_bias)

    @property
    def out_proj(self):
        return self.proj

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = _maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _MiniCPMVSplitMultiheadAttention(nn.Module):
    """Split `nn.MultiheadAttention` into explicit q/k/v linears expected by MQuant."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        device = attn.out_proj.weight.device
        dtype = attn.out_proj.weight.dtype
        bias = attn.in_proj_bias is not None
        out_bias = attn.out_proj.bias is not None

        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=out_bias,
            device=device,
            dtype=dtype,
        )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype)

        w_q, w_k, w_v = attn.in_proj_weight.data.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)
        if bias:
            b_q, b_k, b_v = attn.in_proj_bias.data.chunk(3, dim=0)
            self.q_proj.bias.data.copy_(b_q)
            self.k_proj.bias.data.copy_(b_k)
            self.v_proj.bias.data.copy_(b_v)
        self.out_proj.weight.data.copy_(attn.out_proj.weight.data)
        if out_bias:
            self.out_proj.bias.data.copy_(attn.out_proj.bias.data)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        tgt_len, batch_size, _ = query.shape
        src_len = key.shape[0]

        q = self.q_proj(query).view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(key).view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value).view(value.shape[0], batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                padding_mask = (
                    key_padding_mask.view(batch_size, 1, 1, src_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(batch_size * self.num_heads, 1, src_len)
                )
                attn = attn.masked_fill(padding_mask, torch.finfo(attn.dtype).min)
            else:
                padding_mask = (
                    key_padding_mask.view(batch_size, 1, 1, src_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(batch_size * self.num_heads, 1, src_len)
                )
                attn = attn + padding_mask.to(dtype=attn.dtype, device=attn.device)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask.to(device=attn.device), torch.finfo(attn.dtype).min)
            else:
                attn = attn + attn_mask.to(dtype=attn.dtype, device=attn.device)

        attn = attn.softmax(dim=-1)
        attn_output = attn @ v
        attn_output = attn_output.permute(1, 0, 2).reshape(tgt_len, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


def _resize_minicpmv_abs_pos(
    abs_pos: torch.Tensor,
    tgt_size: int | tuple[int, int] | None,
) -> torch.Tensor:
    """Resize old MiniCPM-V absolute position embeddings to the target patch grid."""
    if tgt_size is None:
        return abs_pos

    if isinstance(tgt_size, tuple):
        tgt_h, tgt_w = int(tgt_size[0]), int(tgt_size[1])
    else:
        tgt_tokens = int(tgt_size)
        tgt_edge = int(math.sqrt(tgt_tokens))
        if tgt_edge * tgt_edge != tgt_tokens:
            raise ValueError(
                f"MiniCPM-V absolute position interpolation expects a square token grid, got {tgt_tokens}."
            )
        tgt_h = tgt_w = tgt_edge

    src_tokens = int(abs_pos.size(0))
    src_edge = int(math.sqrt(src_tokens))
    if src_edge * src_edge != src_tokens:
        raise ValueError(
            f"MiniCPM-V absolute position interpolation expects square source embeddings, got {src_tokens}."
        )
    if src_edge == tgt_h and src_edge == tgt_w:
        return abs_pos

    dtype = abs_pos.dtype
    device = abs_pos.device
    return (
        F.interpolate(
            abs_pos.float().reshape(1, src_edge, src_edge, -1).permute(0, 3, 1, 2),
            size=(tgt_h, tgt_w),
            mode="bicubic",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .reshape(tgt_h * tgt_w, -1)
        .to(device=device, dtype=dtype)
    )


def _minicpmv_layer_output_tensor(output):
    if isinstance(output, tuple) and len(output) > 0:
        return output[0]
    return output


def _run_minicpmv_optional_forward(module: nn.Module, hidden_states, maybe_aux):
    if maybe_aux is None:
        return module(hidden_states)
    return module(hidden_states, maybe_aux)


def _should_skip_minicpmv_gptq_block(block_names: list[str], skip_names: list[str]) -> bool:
    return any(any(skip_name in block_name for skip_name in skip_names) for block_name in block_names)


def _resolve_minicpmv_gptq_name_group(
    full: dict[str, nn.Module],
    candidate_groups: list[list[str]],
) -> list[str]:
    for names in candidate_groups:
        variants = [list(names)]
        stripped = [name[: -len(".module")] if name.endswith(".module") else name for name in names]
        if stripped != list(names):
            variants.append(stripped)
        for variant in variants:
            if all(name in full for name in variant):
                return list(variant)
    available = ", ".join(sorted(full.keys()))
    candidates = " | ".join(", ".join(names) for names in candidate_groups)
    raise KeyError(
        "Failed to resolve MiniCPM-V GPTQ linear group. "
        f"Tried: {candidates}. Available: {available}"
    )


def _minicpmv_legacy_resampler_forward(self, x, attn_mask=None):
    q_proj_leaf = _unwrap_module(getattr(self.attn, "q_proj", None))
    attn_dtype = getattr(getattr(q_proj_leaf, "weight", None), "dtype", x.dtype)
    kv_proj_leaf = _unwrap_module(getattr(self, "kv_proj", None))
    kv_proj_dtype = getattr(getattr(kv_proj_leaf, "weight", None), "dtype", x.dtype)
    pos_embed = _resize_minicpmv_abs_pos(self.pos_embed, x.size(1)).to(dtype=attn_dtype, device=x.device)

    x = self.kv_proj(x.to(dtype=kv_proj_dtype))
    x = self.ln_kv(x).to(dtype=attn_dtype).permute(1, 0, 2)

    batch_size = x.shape[1]
    q = self.ln_q(self.query).to(dtype=attn_dtype, device=x.device)
    out = self.attn(
        self._repeat(q, batch_size) + self.pos_embed.unsqueeze(1).to(dtype=attn_dtype, device=x.device),
        x + pos_embed.unsqueeze(1),
        x,
        attn_mask=attn_mask,
    )[0]
    x = out.permute(1, 0, 2)

    proj_fc = getattr(self, "proj_fc", None)
    proj_leaf = _unwrap_module(proj_fc)
    proj_dtype = getattr(getattr(proj_leaf, "weight", None), "dtype", x.dtype)
    x = self.ln_post(x).to(dtype=proj_dtype)
    if hasattr(self, "proj_fc"):
        x = self.proj_fc(x)
    elif hasattr(self, "proj"):
        x = x @ self.proj
    else:
        raise AttributeError("MiniCPM-V resampler exposes neither `proj_fc` nor `proj`.")
    return x


def _ensure_minicpmv_timm_block_properties(block: nn.Module) -> None:
    block_cls = type(block)
    if not hasattr(block_cls, "layer_norm1"):
        block_cls.layer_norm1 = property(lambda self: self.norm1)
    if not hasattr(block_cls, "layer_norm2"):
        block_cls.layer_norm2 = property(lambda self: self.norm2)
    if not hasattr(block_cls, "self_attn"):
        block_cls.self_attn = property(lambda self: self.attn)


def _ensure_minicpmv_visual_mquant_compat(source_model) -> None:
    vpm = getattr(source_model, "vpm", None)
    if vpm is None or not hasattr(vpm, "blocks") or len(vpm.blocks) == 0:
        return
    if getattr(source_model, "_mindpipe_minicpmv_visual_mquant_compatible", False):
        return

    if not hasattr(vpm, "_mindpipe_mquant_embeddings"):
        vpm._mindpipe_mquant_embeddings = _MiniCPMVTimmEmbeddingsAdapter(vpm)
    if not hasattr(vpm, "_mindpipe_mquant_encoder"):
        vpm._mindpipe_mquant_encoder = _MiniCPMVTimmEncoderAdapter(vpm)

    vpm_cls = type(vpm)
    if not hasattr(vpm_cls, "embeddings"):
        vpm_cls.embeddings = property(lambda self: self._mindpipe_mquant_embeddings)
    if not hasattr(vpm_cls, "encoder"):
        vpm_cls.encoder = property(lambda self: self._mindpipe_mquant_encoder)
    if not hasattr(vpm_cls, "post_layernorm"):
        vpm_cls.post_layernorm = property(lambda self: self.norm)

    for block in vpm.blocks:
        _ensure_minicpmv_timm_block_properties(block)
        if not isinstance(block.attn, _MiniCPMVTimmSplitAttention):
            block.attn = _MiniCPMVTimmSplitAttention(block.attn)

    config = getattr(source_model, "config", None)
    if config is not None and not hasattr(config, "vision_config"):
        hidden_size = getattr(vpm, "embed_dim", None)
        intermediate_size = getattr(getattr(vpm.blocks[0], "mlp", None), "fc1", None)
        config.vision_config = SimpleNamespace(
            hidden_size=int(hidden_size) if hidden_size is not None else None,
            intermediate_size=int(intermediate_size.out_features) if intermediate_size is not None else None,
            need_pad=False,
        )

    source_model._mindpipe_minicpmv_visual_mquant_compatible = True


def _ensure_minicpmv_resampler_mquant_compat(source_model) -> None:
    resampler = getattr(source_model, "resampler", None)
    if resampler is None or getattr(source_model, "_mindpipe_minicpmv_resampler_mquant_compatible", False):
        return

    if hasattr(resampler, "proj") and not hasattr(resampler, "proj_fc"):
        proj_fc_weight = resampler.proj.data.clone().T.contiguous()
        proj_fc = nn.Linear(
            proj_fc_weight.size(1),
            proj_fc_weight.size(0),
            bias=True,
            device=proj_fc_weight.device,
            dtype=proj_fc_weight.dtype,
        )
        proj_fc.weight.data.copy_(proj_fc_weight)
        proj_fc.bias.data.zero_()
        resampler.proj_fc = proj_fc
        delattr(resampler, "proj")

    if isinstance(resampler.attn, nn.MultiheadAttention):
        resampler.attn = _MiniCPMVSplitMultiheadAttention(resampler.attn)
    if not isinstance(getattr(resampler, "forward", None), MethodType) or getattr(
        getattr(resampler, "forward", None), "__func__", None
    ) is not _minicpmv_legacy_resampler_forward:
        resampler.forward = MethodType(_minicpmv_legacy_resampler_forward, resampler)

    source_model._mindpipe_minicpmv_resampler_mquant_compatible = True


def _ensure_minicpmv_mquant_compat(source_model) -> None:
    if getattr(getattr(source_model, "config", None), "model_type", None) != "minicpmv":
        return
    _ensure_minicpmv_visual_mquant_compat(source_model)
    _ensure_minicpmv_resampler_mquant_compat(source_model)


@torch.no_grad()
def _minicpmv_gptq_fwrd_visual_clip_resblocks(
    minicpmv_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
    quantizers,
):
    """Run MiniCPM-V visual block GPTQ with old timm-block signatures."""
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    layers = model.model.vpm.encoder.layers
    # device_map 模式下不手动移动 layers[0]
    layer0_device = next(layers[0].parameters()).device
    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *call_args, **kwargs):
            inps[cache["i"]] = call_args[0]
            attention_masks[cache["i"]] = (
                call_args[1] if len(call_args) > 1 else kwargs.get("attention_mask")
            )
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    lt = len(dataset.data)
    for i in range(lt):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])
        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None] * args.nsamples
    for layer_idx in range(len(layers)):
        print(f"\nLayer {layer_idx}:", flush=True, end=" ")
        # device_map 模式下不手动移动 layer
        layer = layers[layer_idx]
        # 将输入数据移到当前层设备
        layer_dev = next(layer.parameters()).device
        inps = [move_tensors_to_device(inp, layer_dev) if inp is not None else inp for inp in inps]
        outs = [move_tensors_to_device(o, layer_dev) if o is not None else o for o in outs]
        attention_masks = [move_tensors_to_device(m, layer_dev) if m is not None else m for m in attention_masks]
        full = minicpmv_gptq_plus.quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        sequential = [
            _resolve_minicpmv_gptq_name_group(
                full,
                [
                    [
                        "self_attn.k_proj.module",
                        "self_attn.v_proj.module",
                        "self_attn.q_proj.module",
                    ],
                    [
                        "attn.k_proj.module",
                        "attn.v_proj.module",
                        "attn.q_proj.module",
                    ],
                ],
            ),
            _resolve_minicpmv_gptq_name_group(
                full,
                [
                    ["self_attn.out_proj.module"],
                    ["attn.proj.module"],
                ],
            ),
            _resolve_minicpmv_gptq_name_group(
                full,
                [
                    ["mlp.fc1.module"],
                ],
            ),
        ]
        if args.visual_split:
            sequential.append(
                _resolve_minicpmv_gptq_name_group(
                    full,
                    [
                        ["mlp.fc2.L2"],
                        ["mlp.fc2.module"],
                    ],
                )
            )
        else:
            sequential.append(
                _resolve_minicpmv_gptq_name_group(
                    full,
                    [
                        ["mlp.fc2.module"],
                        ["mlp.fc2.L2"],
                    ],
                )
            )
        for names in sequential:
            if _should_skip_minicpmv_gptq_block(list(names), list(args.skip_names)):
                continue
            subset = {name: full[name] for name in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                gptq[name] = minicpmv_gptq_plus.GPTQ(subset[name])
                gptq[name].quantizer = minicpmv_gptq_plus.quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    args.visual_w_bits,
                    perchannel=True,
                    sym=not args.w_asym,
                    mse=args.visual_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for sample_idx in range(args.nsamples):
                outs[sample_idx] = _minicpmv_layer_output_tensor(
                    _run_minicpmv_optional_forward(
                        layer,
                        inps[sample_idx],
                        attention_masks[sample_idx],
                    )
                )
            for handle in handles:
                handle.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers[f"model.vpm.encoder.layers.{layer_idx}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for sample_idx in range(args.nsamples):
            outs[sample_idx] = _minicpmv_layer_output_tensor(
                _run_minicpmv_optional_forward(
                    layer,
                    inps[sample_idx],
                    attention_masks[sample_idx],
                )
            )

        layers[layer_idx] = layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.model.config.use_cache = use_cache
    minicpmv_gptq_plus.utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip resblocks Done----")
    return quantizers


@torch.no_grad()
def _minicpmv_gptq_fwrd_visual_clip_cross_attention(
    minicpmv_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
    quantizers,
):
    """Run MiniCPM-V resampler GPTQ with old resampler forward signatures."""
    print("-----GPTQ Quantization visual clip cross attention-----")
    layer = model.model.resampler
    inps = [None] * args.nsamples
    aux_inputs = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *call_args, **kwargs):
            inps[cache["i"]] = call_args[0]
            aux_inputs[cache["i"]] = (
                call_args[1] if len(call_args) > 1 else kwargs.get("attn_mask")
            )
            cache["i"] += 1
            raise ValueError

    model.model.resampler = Catcher(model.model.resampler)
    lt = len(dataset.data)
    for i in range(lt):
        if cache["i"] >= args.nsamples:
            break
        struct = dataset.build_prompt(dataset.data.iloc[i])
        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
    model.model.resampler = model.model.resampler.module

    full = minicpmv_gptq_plus.quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
    sequential = [
        _resolve_minicpmv_gptq_name_group(
            full,
            [
                ["kv_proj.module"],
                ["kv_proj"],
            ],
        ),
        _resolve_minicpmv_gptq_name_group(
            full,
            [
                [
                    "attn.k_proj.module",
                    "attn.v_proj.module",
                    "attn.q_proj.module",
                ],
                [
                    "attn.k_proj",
                    "attn.v_proj",
                    "attn.q_proj",
                ],
            ],
        ),
        _resolve_minicpmv_gptq_name_group(
            full,
            [
                ["attn.out_proj.module"],
                ["attn.out_proj"],
            ],
        ),
        _resolve_minicpmv_gptq_name_group(
            full,
            [
                ["proj_fc.module"],
                ["proj_fc"],
            ],
        ),
    ]
    for names in sequential:
        if _should_skip_minicpmv_gptq_block(list(names), list(args.skip_names)):
            continue
        subset = {name: full[name] for name in names}

        gptq = {}
        for name in subset:
            print(f"{name}", end="  ", flush=True)
            gptq[name] = minicpmv_gptq_plus.GPTQ(subset[name])
            gptq[name].quantizer = minicpmv_gptq_plus.quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                args.visual_w_bits,
                perchannel=True,
                sym=not args.w_asym,
                mse=args.visual_w_clip,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
        for sample_idx in range(args.nsamples):
            _run_minicpmv_optional_forward(layer, inps[sample_idx], aux_inputs[sample_idx])
        for handle in handles:
            handle.remove()

        for name in subset:
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.w_groupsize,
                actorder=args.act_order,
                static_groups=False,
            )
            quantizers[f"model.resampler.{name}"] = gptq[name].quantizer
            gptq[name].free()

    model.model.resampler = layer
    del gptq
    torch.cuda.empty_cache()
    minicpmv_gptq_plus.utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization visual clip cross attention Done-----")
    return quantizers


@torch.no_grad()
def _minicpmv_rtn_gptq_fwrd_plus(
    minicpmv_gptq_plus,
    model,
    dataset,
    dev,
    dataset_name,
    args,
):
    """MiniCPM-V GPTQ/RTN dispatcher with legacy visual tower compatibility."""
    logging.info("-----RTN Or GPTQ Quantization-----")

    quantizers = {}
    if args.quant_visual_clip:
        if args.visual_w_rtn:
            minicpmv_gptq_plus.minicpmv_visual_clip_rtn(model.model, dev, args)
        else:
            minicpmv_gptq_plus.gptq_minicpmv_fwrd_visual_clip_conv1(
                model,
                dataset,
                dev,
                dataset_name,
                args,
                quantizers,
            )
            _minicpmv_gptq_fwrd_visual_clip_resblocks(
                minicpmv_gptq_plus,
                model,
                dataset,
                dev,
                dataset_name,
                args,
                quantizers,
            )

    if args.quant_cross_attention:
        if args.visual_w_rtn:
            minicpmv_gptq_plus.minicpmv_visual_cross_attention_rtn(model.model, dev, args)
        else:
            _minicpmv_gptq_fwrd_visual_clip_cross_attention(
                minicpmv_gptq_plus,
                model,
                dataset,
                dev,
                dataset_name,
                args,
                quantizers,
            )

    if args.quant_llm:
        if args.llm_w_rtn:
            minicpmv_gptq_plus.minicpmv_llm_rtn(model.model, dev, args, quantizers)
        else:
            minicpmv_gptq_plus.gptq_minicpmv_fwrd_llm(
                model,
                dataset,
                dev,
                dataset_name,
                args,
                quantizers,
            )

    return quantizers


def _copy_namespace(namespace: SimpleNamespace, **updates: Any) -> SimpleNamespace:
    data = vars(namespace).copy()
    data.update(updates)
    return SimpleNamespace(**data)


def _fuse_qwen2_5_vl_visual_layer_norms(model_root, rotation_utils) -> None:
    """Fuse Qwen2.5-VL visual RMSNorm scale into the adjacent visual linears."""
    logging.info("Fusing Qwen2.5-VL visual RMSNorm weights into visual linears.")
    for layer in model_root.visual.blocks:
        rotation_utils.fuse_ln_linear(layer.norm1, [layer.attn.qkv])
        rotation_utils.fuse_ln_linear(
            layer.norm2,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )


def _rotate_qwen2_5_vl_mlp_input(layer, q: torch.Tensor) -> None:
    """Rotate Qwen2.5-VL visual MLP input projections."""
    for linear in (layer.mlp.gate_proj, layer.mlp.up_proj):
        dtype = linear.weight.dtype
        weight = linear.weight.data.to(dtype=torch.float64)
        linear.weight.data = torch.matmul(weight, q.to(weight.device)).to(dtype=dtype)


def _rotate_qwen2_5_vl_mlp_output(
    layer,
    q: torch.Tensor,
    *,
    online_hadamard: bool,
    apply_exact_had_to_linear,
) -> None:
    """Rotate Qwen2.5-VL visual MLP output projection."""
    out_layer = layer.mlp.down_proj
    dtype = out_layer.weight.dtype
    q = q.to(out_layer.weight.device)
    weight = out_layer.weight.data.to(dtype=torch.float64)
    out_layer.weight.data = torch.matmul(q.T, weight).to(dtype=dtype)

    if online_hadamard:
        apply_exact_had_to_linear(out_layer, had_dim=-1, output=False)

    if out_layer.bias is not None:
        bias = out_layer.bias.data.to(dtype=torch.float64)
        out_layer.bias.data = torch.matmul(q.T, bias).to(dtype=dtype)


def _rotate_qwen2_5_vl_visual(
    model_root,
    mquant_args: SimpleNamespace,
    *,
    rotation_utils,
    qwen2vl_rotation,
    mquant_utils,
    apply_exact_had_to_linear,
) -> None:
    """Rotate the Qwen2.5-VL visual branch using the Qwen2-VL-compatible adapter root."""
    logging.info("Rotating Qwen2.5-VL visual branch.")
    visual = model_root.visual
    num_heads = visual.blocks[0].attn.num_heads
    hidden_size = visual.blocks[0].attn.qkv.in_features
    head_dim = hidden_size // num_heads
    visual_device = visual.patch_embed.proj.weight.device
    q_visual = rotation_utils.get_orthogonal_matrix(
        hidden_size,
        mquant_args.rotate_mode,
        device=visual_device,
    )

    rotation_utils.rotate_conv(visual.patch_embed.proj, q_visual, hidden_size)

    for layer in visual.blocks:
        layer_q = q_visual.to(next(layer.parameters()).device)
        qwen2vl_rotation.rotate_qwen2vl_attention_inputs(layer, layer_q, is_visual=True)
        qwen2vl_rotation.rotate_qwen2vl_attention_output(layer, layer_q, is_visual=True)
        _rotate_qwen2_5_vl_mlp_input(layer, layer_q)
        _rotate_qwen2_5_vl_mlp_output(
            layer,
            layer_q,
            online_hadamard=bool(mquant_args.online_visual_hadamard),
            apply_exact_had_to_linear=apply_exact_had_to_linear,
        )
        qwen2vl_rotation.rotate_qwen2vl_ov_proj(
            layer,
            num_heads,
            head_dim,
            is_visual=True,
        )

    qwen2vl_rotation.rotate_visual_merger(
        model_root,
        q_visual.to(visual.merger.mlp[0].weight.device),
    )
    mquant_utils.cleanup_memory()


def _qwen3_vl_add_extra_act_quant_wrappers(
    quant_utils,
    *,
    model_root,
    mquant_args: SimpleNamespace,
) -> None:
    """Wrap Qwen3-VL deepstack mergers for activation quantization."""
    if not bool(mquant_args.quant_cross_attention):
        return
    quant_utils.add_actquant(
        model_root.visual.deepstack_merger_list,
        bool(mquant_args.act_per_tensor),
    )


def _qwen3_vl_quantize_deepstack_mergers_rtn(
    quant_utils,
    *,
    model_root,
    mquant_args: SimpleNamespace,
    quantizers: dict[str, Any],
) -> None:
    """Quantize Qwen3-VL deepstack merger weights with the RTN recipe."""
    if not bool(mquant_args.quant_cross_attention):
        return
    for idx, merger in enumerate(model_root.visual.deepstack_merger_list):
        subset = quant_utils.find_qlayers(merger, layers=[torch.nn.Linear])
        for name, module in subset.items():
            if any(pattern in name for pattern in mquant_args.skip_names) or "L1" in name:
                continue
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                int(mquant_args.visual_w_bits),
                perchannel=True,
                sym=not bool(mquant_args.w_asym),
                mse=bool(mquant_args.visual_w_clip),
            )
            weight = module.weight.data
            dtype = weight.dtype
            quantizer.find_params(weight)
            module.weight.data = quantizer.quantize(weight).to(dtype)
            quantizers[f"model.visual.deepstack_merger_list.{idx}.{name}"] = quantizer.cpu()


@torch.no_grad()
def _qwen3_vl_gptq_llm(
    qwen2vl_gptq_plus,
    quant_utils,
    *,
    proxy,
    dataset,
    dev,
    dataset_name: str,
    args: SimpleNamespace,
    quantizers: dict[str, Any],
) -> dict[str, Any]:
    """Run a DeepStack-aware GPTQ replay for the Qwen3-VL language stack."""
    logging.info("-----GPTQ Quantization Qwen3-VL LLM-----")
    source_model = proxy._source_model
    source_text_model = source_model.model.language_model
    use_cache = bool(getattr(source_text_model.config, "use_cache", False))
    source_text_model.config.use_cache = False
    layers = source_text_model.layers

    inps = [None] * int(args.nsamples)
    attention_masks = [None] * int(args.nsamples)
    position_ids = [None] * int(args.nsamples)
    position_embeddings = [None] * int(args.nsamples)
    visual_pos_masks = [None] * int(args.nsamples)
    deepstack_visual_embeds = [None] * int(args.nsamples)
    cache = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            attention_masks[cache["i"]] = kwargs.get("attention_mask", None)
            position_ids[cache["i"]] = kwargs.get("position_ids", None)
            position_embeddings[cache["i"]] = kwargs.get("position_embeddings", None)
            cache["i"] += 1
            raise ValueError

    original_first_layer = layers[0]
    original_text_forward = source_text_model.forward

    def _patched_text_forward(self, *forward_args, **forward_kwargs):
        slot = int(cache["i"])
        if slot < int(args.nsamples):
            visual_pos_masks[slot] = forward_kwargs.get("visual_pos_masks", None)
            visual_embeds = forward_kwargs.get("deepstack_visual_embeds", None)
            deepstack_visual_embeds[slot] = list(visual_embeds) if visual_embeds is not None else None
        return original_text_forward(*forward_args, **forward_kwargs)

    layers[0] = Catcher(layers[0])
    source_text_model.forward = MethodType(_patched_text_forward, source_text_model)

    try:
        lt = len(dataset.data)
        for i in range(lt):
            if cache["i"] >= int(args.nsamples):
                break
            if hasattr(proxy, "use_custom_prompt") and proxy.use_custom_prompt(dataset_name):
                struct = proxy.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(dataset.data.iloc[i])
            try:
                proxy.generate(message=struct, dataset=args.dataset_name)
            except ValueError:
                pass
    finally:
        source_text_model.forward = original_text_forward
        layers[0] = original_first_layer

    captured = int(cache["i"])
    if captured <= 0:
        raise RuntimeError(
            "No language samples were captured for Qwen3-VL GPTQ calibration. "
            "Please verify calibration prompts are valid."
        )
    if captured < int(args.nsamples):
        logging.warning(
            "Qwen3-VL LLM captured samples %d < requested nsamples %d. Proceeding with captured samples.",
            captured,
            int(args.nsamples),
        )

    inps = inps[:captured]
    attention_masks = attention_masks[:captured]
    position_ids = position_ids[:captured]
    position_embeddings = position_embeddings[:captured]
    visual_pos_masks = visual_pos_masks[:captured]
    deepstack_visual_embeds = deepstack_visual_embeds[:captured]
    outs = [None] * captured

    sequential = [
        [
            "self_attn.q_proj.module",
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    for layer_idx in range(len(layers)):
        print(f"\nLayer {layer_idx}:", flush=True, end=" ")
        layer = layers[layer_idx]
        # 将输入数据移到当前层设备
        layer_dev = next(layer.parameters()).device
        inps = [move_tensors_to_device(inp, layer_dev) if inp is not None else inp for inp in inps]
        outs = [move_tensors_to_device(o, layer_dev) if o is not None else o for o in outs]
        attention_masks = [move_tensors_to_device(m, layer_dev) if m is not None else m for m in attention_masks]
        position_ids = [move_tensors_to_device(p, layer_dev) if p is not None else p for p in position_ids]
        position_embeddings = [move_tensors_to_device(pe, layer_dev) if pe is not None else pe for pe in position_embeddings]
        visual_pos_masks = [move_tensors_to_device(m, layer_dev) if m is not None else m for m in visual_pos_masks]
        deepstack_visual_embeds = [
            [move_tensors_to_device(e, layer_dev) if e is not None else e for e in embeds] if embeds is not None else None
            for embeds in deepstack_visual_embeds
        ]
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for names in sequential:
            if any(any(skip_name in name for skip_name in args.skip_names) for name in names):
                continue
            subset = {name: full[name] for name in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                gptq[name] = qwen2vl_gptq_plus.GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    int(args.llm_w_bits),
                    perchannel=True,
                    sym=not bool(args.w_asym),
                    mse=bool(args.llm_w_clip),
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for sample_id in range(captured):
                layer_out = layer(
                    inps[sample_id],
                    attention_mask=attention_masks[sample_id],
                    position_ids=position_ids[sample_id],
                    position_embeddings=position_embeddings[sample_id],
                )
                hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
                deepstack_embeds = deepstack_visual_embeds[sample_id]
                if deepstack_embeds is not None and layer_idx < len(deepstack_embeds):
                    hidden_states = source_text_model._deepstack_process(
                        hidden_states,
                        visual_pos_masks[sample_id],
                        deepstack_embeds[layer_idx],
                    )
                outs[sample_id] = hidden_states

            for handle in handles:
                handle.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=float(args.percdamp),
                    groupsize=int(args.w_groupsize),
                    actorder=bool(args.act_order),
                    static_groups=False,
                )
                quantizers[f"model.model.layers.{layer_idx}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for sample_id in range(captured):
            layer_out = layer(
                inps[sample_id],
                attention_mask=attention_masks[sample_id],
                position_ids=position_ids[sample_id],
                position_embeddings=position_embeddings[sample_id],
            )
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            deepstack_embeds = deepstack_visual_embeds[sample_id]
            if deepstack_embeds is not None and layer_idx < len(deepstack_embeds):
                hidden_states = source_text_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks[sample_id],
                    deepstack_embeds[layer_idx],
                )
            outs[sample_id] = hidden_states

        torch.cuda.empty_cache()
        inps, outs = outs, inps

    source_text_model.config.use_cache = use_cache
    qwen2vl_gptq_plus.utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization Qwen3-VL LLM Done-----")
    return quantizers


@torch.no_grad()
def _qwen3_vl_rtn_gptq_fwrd_plus(
    qwen2vl_gptq_plus,
    quant_utils,
    *,
    proxy,
    dataset,
    dev,
    dataset_name: str,
    args: SimpleNamespace,
) -> dict[str, Any]:
    """Conservative Qwen3-VL mixed path: visual RTN + language GPTQ."""
    logging.info("-----Qwen3-VL Mixed RTN/GPTQ Quantization-----")
    if not bool(args.visual_w_rtn):
        raise NotImplementedError(
            "Qwen3-VL visual GPTQ is not enabled yet. "
            "Use the conservative mixed path with visual RTN."
        )
    if bool(args.llm_w_rtn):
        raise ValueError("Qwen3-VL mixed GPTQ path expects language GPTQ (`llm_w_rtn=False`).")

    quantizers: dict[str, Any] = {}

    if bool(args.quant_visual_clip):
        qwen2vl_gptq_plus.qwen2vl_visual_clip_rtn(proxy.model, dev, args, quantizers)

    if bool(args.quant_cross_attention):
        qwen2vl_gptq_plus.qwen2vl_visual_cross_attention_rtn(proxy.model, dev, args, quantizers)
        _qwen3_vl_quantize_deepstack_mergers_rtn(
            quant_utils,
            model_root=proxy.model,
            mquant_args=args,
            quantizers=quantizers,
        )

    if bool(args.quant_llm):
        _qwen3_vl_gptq_llm(
            qwen2vl_gptq_plus,
            quant_utils,
            proxy=proxy,
            dataset=dataset,
            dev=dev,
            dataset_name=dataset_name,
            args=args,
            quantizers=quantizers,
        )

    return quantizers


class MQuantMethod(BaseQuantizationMethod):
    """Run MQuant-style RTN quantization on supported multimodal VLM backbones."""

    name = "mquant"
    npu_ready = False
    default_calibration_dataset = "pileval"

    _DEFAULT_MQUANT_ROOT = Path("/mnt/42_store/zy/HUAWEI/work1/MQuant")
    _SUPPORTED_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "internvl_chat", "minicpmv"}

    @staticmethod
    def _resolve_source_model(model):
        return getattr(model, "_source_model", model)

    @staticmethod
    def _resolve_model_type(model) -> str:
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", "")
        return str(model_type)

    def _resolve_family(self, source_model) -> str:
        model_type = self._resolve_model_type(source_model)
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return "qwen2vl"
        if model_type == "qwen3_vl":
            return "qwen3vl"
        if model_type == "internvl_chat":
            return "internvl"
        if model_type == "minicpmv":
            return "minicpmv"
        raise NotImplementedError(
            f"MQuant currently supports {sorted(self._SUPPORTED_MODEL_TYPES)}, got model_type={model_type!r}."
        )

    @staticmethod
    def _build_qwen2vl_compat_root(source_model):
        """Build a lightweight view matching the module layout expected by MQuant scripts."""
        multimodal_root = getattr(source_model, "model", source_model)
        visual = getattr(multimodal_root, "visual", None)
        text_root = (
            getattr(multimodal_root, "language_model", None)
            or getattr(multimodal_root, "model", None)
            or getattr(source_model, "language_model", None)
            or getattr(source_model, "model", None)
        )
        lm_head = getattr(source_model, "lm_head", None) or getattr(multimodal_root, "lm_head", None)
        text_config = getattr(text_root, "config", None) if text_root is not None else None
        model_config = text_config or getattr(source_model, "config", None)

        if visual is None or text_root is None or lm_head is None or model_config is None:
            raise AttributeError(
                "Failed to build Qwen2-VL compatibility root. "
                "Expected visual/text/lm_head/config are missing."
            )
        if getattr(text_config, "model_type", "") == "qwen3_vl_text":
            text_root = _Qwen3VLTextCompatRoot(text_root)
        return _Qwen2VLCompatRoot(
            visual=visual,
            text_root=text_root,
            lm_head=lm_head,
            config=model_config,
        )

    @staticmethod
    def _infer_module_device(module: Any) -> torch.device:
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    @classmethod
    def _resolve_mquant_root(cls) -> Path:
        root = Path(os.environ.get("MQUANT_ROOT", str(cls._DEFAULT_MQUANT_ROOT))).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"MQuant repo root does not exist: {root}. "
                "Set `MQUANT_ROOT` or prepare `/mnt/42_store/zy/HUAWEI/work1/MQuant`."
            )
        return root

    @staticmethod
    def _resolve_mquant_dataset_name(args) -> str | None:
        explicit = getattr(args, "mquant_dataset_name", None)
        if explicit:
            return str(explicit)
        env_name = os.environ.get("MQUANT_DATASET_NAME", "").strip()
        return env_name or None

    @staticmethod
    def _resolve_mquant_calib_num(args) -> int | None:
        explicit = getattr(args, "mquant_calib_num", None)
        if explicit is not None:
            return int(explicit)
        env_value = os.environ.get("MQUANT_CALIB_NUM", "").strip()
        if env_value:
            return int(env_value)
        return None

    @staticmethod
    def _resolve_optional_int_arg(args, arg_name: str, env_name: str) -> int | None:
        explicit = getattr(args, arg_name, None)
        if explicit is not None:
            return int(explicit)
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return int(env_value)
        return None

    @staticmethod
    def _resolve_optional_bool_arg(args, arg_name: str, env_name: str) -> bool | None:
        explicit = getattr(args, arg_name, None)
        if explicit is not None:
            return bool(explicit)
        env_value = os.environ.get(env_name, "").strip().lower()
        if not env_value:
            return None
        if env_value in {"1", "true", "yes", "y", "on"}:
            return True
        if env_value in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for {env_name}: {env_value}")

    @staticmethod
    def _resolve_optional_list_arg(args, arg_name: str, env_name: str) -> list[str] | None:
        explicit = getattr(args, arg_name, None)
        if explicit is not None:
            return [str(item) for item in explicit]
        env_value = os.environ.get(env_name, "").strip()
        if not env_value:
            return None
        return [item.strip() for item in env_value.split(",") if item.strip()]

    @classmethod
    def _apply_mquant_overrides(cls, mquant_args: SimpleNamespace, args) -> SimpleNamespace:
        int_overrides = {
            "visual_w_bits": ("mquant_visual_w_bits", "MQUANT_VISUAL_W_BITS"),
            "visual_a_bits": ("mquant_visual_a_bits", "MQUANT_VISUAL_A_BITS"),
            "llm_w_bits": ("mquant_llm_w_bits", "MQUANT_LLM_W_BITS"),
            "llm_a_bits": ("mquant_llm_a_bits", "MQUANT_LLM_A_BITS"),
            "w_groupsize": ("mquant_weight_group_size", "MQUANT_WEIGHT_GROUP_SIZE"),
            "a_groupsize": ("mquant_activation_group_size", "MQUANT_ACTIVATION_GROUP_SIZE"),
        }
        for attr, (arg_name, env_name) in int_overrides.items():
            value = cls._resolve_optional_int_arg(args, arg_name, env_name)
            if value is not None:
                setattr(mquant_args, attr, int(value))

        mquant_args.visual_w_clip = bool(int(mquant_args.visual_w_bits) <= 4)
        mquant_args.llm_w_clip = bool(int(mquant_args.llm_w_bits) <= 4)

        bool_overrides = {
            "visual_w_clip": ("mquant_visual_w_clip", "MQUANT_VISUAL_W_CLIP"),
            "llm_w_clip": ("mquant_llm_w_clip", "MQUANT_LLM_W_CLIP"),
            "visual_static": ("mquant_visual_static", "MQUANT_VISUAL_STATIC"),
            "llm_static": ("mquant_llm_static", "MQUANT_LLM_STATIC"),
            "quant_llm": ("mquant_quant_llm", "MQUANT_QUANT_LLM"),
            "quant_visual_clip": ("mquant_quant_visual_clip", "MQUANT_QUANT_VISUAL_CLIP"),
            "quant_cross_attention": ("mquant_quant_cross_attention", "MQUANT_QUANT_CROSS_ATTENTION"),
            "not_fuse_layer_norms": ("mquant_not_fuse_layer_norms", "MQUANT_NOT_FUSE_LAYER_NORMS"),
            "no_fuse_visual_clip": ("mquant_no_fuse_visual_clip", "MQUANT_NO_FUSE_VISUAL_CLIP"),
            "no_fuse_visual_cross_attn": (
                "mquant_no_fuse_visual_cross_attn",
                "MQUANT_NO_FUSE_VISUAL_CROSS_ATTN",
            ),
            "no_fuse_llm": ("mquant_no_fuse_llm", "MQUANT_NO_FUSE_LLM"),
            "rotate": ("mquant_rotate", "MQUANT_ROTATE"),
            "rotate_visual_clip": ("mquant_rotate_visual_clip", "MQUANT_ROTATE_VISUAL_CLIP"),
            "rotate_visual_cross_attn": (
                "mquant_rotate_visual_cross_attn",
                "MQUANT_ROTATE_VISUAL_CROSS_ATTN",
            ),
            "rotate_llm": ("mquant_rotate_llm", "MQUANT_ROTATE_LLM"),
            "act_per_tensor": ("mquant_act_per_tensor", "MQUANT_ACT_PER_TENSOR"),
            "online_llm_hadamard": ("mquant_online_llm_hadamard", "MQUANT_ONLINE_LLM_HADAMARD"),
            "online_visual_hadamard": (
                "mquant_online_visual_hadamard",
                "MQUANT_ONLINE_VISUAL_HADAMARD",
            ),
            "fp32_had": ("mquant_fp32_had", "MQUANT_FP32_HAD"),
            "llm_split": ("mquant_llm_split", "MQUANT_LLM_SPLIT"),
            "visual_split": ("mquant_visual_split", "MQUANT_VISUAL_SPLIT"),
        }
        for attr, (arg_name, env_name) in bool_overrides.items():
            value = cls._resolve_optional_bool_arg(args, arg_name, env_name)
            if value is not None:
                setattr(mquant_args, attr, bool(value))

        skip_names = cls._resolve_optional_list_arg(args, "mquant_skip_names", "MQUANT_SKIP_NAMES")
        if skip_names is not None:
            mquant_args.skip_names = skip_names
        act_skip_names = cls._resolve_optional_list_arg(
            args,
            "mquant_act_skip_names",
            "MQUANT_ACT_SKIP_NAMES",
        )
        if act_skip_names is not None:
            mquant_args.act_skip_names = act_skip_names
        return mquant_args

    @staticmethod
    def _build_mquant_args(args) -> SimpleNamespace:
        activation_group_size = int(args.activation_group_size)
        return SimpleNamespace(
            # Core run flags
            seed=int(args.seed),
            quant=True,
            quant_llm=True,
            quant_visual_clip=True,
            quant_cross_attention=True,
            # Fuse/rotation
            not_fuse_layer_norms=False,
            no_fuse_visual_clip=False,
            no_fuse_visual_cross_attn=False,
            no_fuse_llm=False,
            rotate=True,
            rotate_visual_clip=True,
            rotate_visual_cross_attn=True,
            rotate_llm=True,
            rotate_mode=str(args.rotation_mode),
            # Weight quantization
            visual_w_bits=int(args.weight_bits),
            llm_w_bits=int(args.weight_bits),
            w_groupsize=int(args.weight_group_size),
            w_asym=not bool(args.weight_symmetric),
            visual_w_rtn=True,
            llm_w_rtn=True,
            visual_w_clip=bool(args.weight_bits <= 4),
            llm_w_clip=bool(args.weight_bits <= 4),
            percdamp=float(args.damp_percent),
            act_order=bool(args.use_activation_order),
            # Activation quantization
            visual_a_bits=int(args.activation_bits),
            llm_a_bits=int(args.activation_bits),
            a_groupsize=activation_group_size,
            a_asym=not bool(args.activation_symmetric),
            a_clip_ratio=1.0,
            visual_static=False,
            llm_static=False,
            act_per_tensor=False,
            # Online hadamard (defer to follow-up tuning; keep disabled by default)
            online_llm_hadamard=False,
            online_visual_hadamard=False,
            fp32_had=False,
            llm_split=False,
            visual_split=False,
            # Calibration placeholders
            dataset_name="",
            nsamples=int(args.calibration_samples),
            calib_num=int(args.calibration_samples),
            skip_names=[],
            act_skip_names=[],
        )

    @staticmethod
    def _adapt_mquant_args_for_model(mquant_args: SimpleNamespace, source_model) -> SimpleNamespace:
        """Patch MQuant flags for model-specific architectural differences."""
        model_type = getattr(getattr(source_model, "config", None), "model_type", "")
        if str(model_type) == "qwen2_5_vl" and bool(getattr(mquant_args, "online_visual_hadamard", False)):
            logging.warning(
                "Disabling MQuant visual online Hadamard for Qwen2.5-VL: "
                "vision down_proj input dim is not supported by upstream Hadamard kernels."
            )
            mquant_args.online_visual_hadamard = False
            mquant_args.visual_split = False
        if str(model_type) == "qwen3_vl":
            if bool(getattr(mquant_args, "rotate", False)) or not bool(
                getattr(mquant_args, "not_fuse_layer_norms", False)
            ):
                logging.warning(
                    "Applying conservative Qwen3-VL MQuant safety patch: "
                    "disable fuse/rotate/hadamard/split and keep only the conservative quantization path."
                )
            mquant_args.not_fuse_layer_norms = True
            mquant_args.rotate = False
            mquant_args.rotate_visual_clip = False
            mquant_args.rotate_visual_cross_attn = False
            mquant_args.rotate_llm = False
            mquant_args.online_visual_hadamard = False
            mquant_args.online_llm_hadamard = False
            mquant_args.visual_split = False
            mquant_args.llm_split = False
        return mquant_args

    @staticmethod
    def _configure_activation_quantizers(
        quant_utils,
        modules: list,
        *,
        bits: int,
        groupsize: int,
        symmetric: bool,
        clip_ratio: float,
        static: bool,
        act_per_tensor: bool,
        skip_names: list[str],
    ) -> int:
        configured = 0
        for module in modules:
            qlayers = quant_utils.find_qlayers(module, layers=[quant_utils.ActQuantWrapper])
            for name, layer in qlayers.items():
                if any(pattern in name for pattern in skip_names):
                    continue
                layer_groupsize = groupsize
                wrapped_module = _unwrap_module(getattr(layer, "module", None))
                # Conv wrappers (e.g. Qwen2/Qwen2.5-VL patch_embed.proj) receive 5D inputs.
                # Group-wise activation quantization in MQuant assumes the last dim can be
                # split by groupsize; for patch inputs like (..., 14), groupsize=128 fails.
                # Fall back to non-grouped activation quantization for conv wrappers.
                if isinstance(wrapped_module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                    layer_groupsize = -1
                elif (
                    isinstance(wrapped_module, torch.nn.Linear)
                    and layer_groupsize is not None
                    and int(layer_groupsize) > 0
                ):
                    in_features = getattr(wrapped_module, "in_features", None)
                    if in_features is not None and int(in_features) % int(layer_groupsize) != 0:
                        layer_groupsize = -1
                layer.quantizer.configure(
                    bits=bits,
                    groupsize=layer_groupsize,
                    sym=symmetric,
                    clip_ratio=clip_ratio,
                    act_per_tensor=bool(act_per_tensor),
                    static=bool(static),
                    observer_type="minmax",
                )
                configured += 1
        return configured

    @staticmethod
    def _configure_qwen2_online_hadamard_wrappers(
        quant_utils,
        hadamard_get_k,
        mquant_utils,
        *,
        model_root,
        mquant_args: SimpleNamespace,
    ) -> None:
        if bool(mquant_args.online_llm_hadamard) and bool(mquant_args.rotate_llm):
            qlayers = quant_utils.find_qlayers(
                model_root.model,
                layers=[quant_utils.ActQuantWrapper],
            )
            for name, layer in qlayers.items():
                if "mlp.down_proj" not in name:
                    continue
                had_k, k = hadamard_get_k(model_root.config.intermediate_size)
                layer.online_full_had = True
                layer.had_K = had_k
                layer.K = k
                layer.fp32_had = bool(mquant_args.fp32_had)
                layer.split = bool(mquant_args.llm_split)
                if bool(mquant_args.llm_split):
                    layer.split_weights()
                if getattr(model_root.config, "need_pad", False):
                    hook = partial(
                        mquant_utils.revise_down_input,
                        new_size=model_root.config.intermediate_size,
                    )
                    layer.register_forward_pre_hook(hook)

        if bool(mquant_args.online_visual_hadamard) and bool(mquant_args.rotate_visual_clip):
            qlayers = quant_utils.find_qlayers(
                model_root.visual,
                layers=[quant_utils.ActQuantWrapper],
            )
            for name, layer in qlayers.items():
                if "mlp.fc2" not in name and "mlp.down_proj" not in name:
                    continue
                module = _unwrap_module(getattr(layer, "module", None))
                in_features = getattr(module, "in_features", None)
                if in_features is None:
                    continue
                had_k, k = hadamard_get_k(int(in_features))
                layer.online_full_had = True
                layer.had_K = had_k
                layer.K = k
                layer.fp32_had = bool(mquant_args.fp32_had)
                layer.split = bool(mquant_args.visual_split)
                if bool(mquant_args.visual_split):
                    layer.split_weights()

    @staticmethod
    def _configure_minicpmv_online_hadamard_wrappers(
        quant_utils,
        hadamard_get_k,
        mquant_utils,
        *,
        source_model,
        mquant_args: SimpleNamespace,
    ) -> None:
        if bool(mquant_args.online_llm_hadamard) and bool(mquant_args.rotate_llm):
            qlayers = quant_utils.find_qlayers(
                source_model,
                layers=[quant_utils.ActQuantWrapper],
            )
            for name, layer in qlayers.items():
                if "mlp.down_proj" not in name:
                    continue
                had_k, k = hadamard_get_k(source_model.config.intermediate_size)
                layer.online_full_had = True
                layer.had_K = had_k
                layer.K = k
                layer.fp32_had = bool(mquant_args.fp32_had)
                layer.split = bool(mquant_args.llm_split)
                if bool(mquant_args.llm_split):
                    layer.split_weights()
                if getattr(source_model.config, "need_pad", False):
                    hook = partial(
                        mquant_utils.revise_down_input,
                        new_size=source_model.config.intermediate_size,
                    )
                    layer.register_forward_pre_hook(hook)

        if bool(mquant_args.online_visual_hadamard) and bool(mquant_args.rotate_visual_clip):
            qlayers = quant_utils.find_qlayers(
                source_model.vpm,
                layers=[quant_utils.ActQuantWrapper],
            )
            for name, layer in qlayers.items():
                if "mlp.fc2" not in name:
                    continue
                module = _unwrap_module(getattr(layer, "module", None))
                in_features = getattr(module, "in_features", None)
                if in_features is None:
                    continue
                had_k, k = hadamard_get_k(int(in_features))
                layer.online_full_had = True
                layer.had_K = had_k
                layer.K = k
                layer.fp32_had = bool(mquant_args.fp32_had)
                layer.split = bool(mquant_args.visual_split)
                if bool(mquant_args.visual_split):
                    layer.split_weights()
                vision_config = getattr(source_model.config, "vision_config", None)
                if vision_config is not None and getattr(vision_config, "need_pad", False):
                    hook = partial(
                        mquant_utils.revise_down_input,
                        new_size=vision_config.intermediate_size,
                    )
                    layer.register_forward_pre_hook(hook)

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, Any]:
        source_model = self._resolve_source_model(model)
        family = self._resolve_family(source_model)
        model_type = self._resolve_model_type(source_model)
        if family in {"minicpmv", "internvl"}:
            _ensure_minicpmv_mquant_compat(source_model)
            if not getattr(source_model, "hf_device_map", None):
                source_model.to(resolve_device(args.device))
            source_model.eval()
        mquant_root = self._resolve_mquant_root()
        use_gptq = str(args.weight_method).lower() == "gptq"
        if use_gptq and family not in {"qwen2vl", "qwen3vl", "minicpmv", "internvl"}:
            raise NotImplementedError(
                "MindPipe MQuant GPTQ path currently supports Qwen2/Qwen2.5-VL, "
                "InternVL2, MiniCPM-V, and the conservative Qwen3-VL mixed path only."
            )
        mquant_dataset_name = self._resolve_mquant_dataset_name(args)
        mquant_calib_num = self._resolve_mquant_calib_num(args)
        if use_gptq and not mquant_dataset_name:
            raise ValueError(
                "MQuant GPTQ requires a VLM calibration dataset name. "
                "Set `--mquant_dataset_name` (e.g. OCRBench/TextVQA_VAL/DocVQA_VAL/MME)."
            )
        mquant_args = self._build_mquant_args(args)
        mquant_args = self._apply_mquant_overrides(mquant_args, args)
        mquant_args = self._adapt_mquant_args_for_model(mquant_args, source_model)
        if bool(mquant_args.online_llm_hadamard) and bool(mquant_args.rotate_llm):
            mquant_args.quant_llm = True
        if bool(mquant_args.online_visual_hadamard) and bool(mquant_args.rotate_visual_clip):
            mquant_args.quant_visual_clip = True
        mquant_args.visual_w_rtn = not use_gptq
        mquant_args.llm_w_rtn = not use_gptq
        if str(model_type) == "qwen3_vl" and use_gptq:
            logging.warning(
                "Qwen3-VL MQuant GPTQ currently uses a conservative mixed path: "
                "visual RTN + language GPTQ."
            )
            mquant_args.visual_w_rtn = True
            mquant_args.llm_w_rtn = False
        mquant_args.dataset_name = mquant_dataset_name or ""
        if mquant_calib_num is not None:
            mquant_args.calib_num = int(mquant_calib_num)
        if family in {"qwen2vl", "qwen3vl"}:
            mquant_root_model = self._build_qwen2vl_compat_root(source_model)
        else:
            mquant_root_model = source_model
        proxy = SimpleNamespace(model=mquant_root_model)
        mquant_dataset = None

        quantizer_map: dict[str, Any] = {}
        configured_act_quantizers = 0

        with prepend_python_path(mquant_root):
            try:
                from fake_quant import gptq as mquant_gptq
            except ModuleNotFoundError as exc:
                if exc.name == "unfoldNd":
                    raise ModuleNotFoundError(
                        "Missing MQuant dependency `unfoldNd`. "
                        "Install it in the current environment, e.g. `pip install unfoldNd`."
                    ) from exc
                raise
            if (
                use_gptq
                or family in {"qwen2vl", "qwen3vl"}
                or (
                    family == "internvl"
                    and bool(mquant_args.dataset_name)
                    and (bool(mquant_args.visual_static) or bool(mquant_args.llm_static))
                )
            ):
                _ensure_vlmeval_transformers_compat()
                from vlmeval.dataset import build_dataset as mquant_build_dataset
            from fake_quant.gptq import minicpmv_gptq_plus as mquant_minicpmv_gptq_plus
            from fake_quant.gptq import qwen2vl_gptq_plus as mquant_qwen2vl_gptq_plus
            mquant_internvl_gptq_plus = importlib.import_module("fake_quant.gptq.internvl_gptq_plus")
            from fake_quant import quant_utils as mquant_quant_utils
            from fake_quant import rotation_utils as mquant_rotation_utils
            from fake_quant import qwen2vl_rotation as mquant_qwen2vl_rotation
            from fake_quant import internvl_rotation as mquant_internvl_rotation
            from fake_quant import utils as mquant_utils
            from fake_quant.hadamard_utils import apply_exact_had_to_linear
            from fake_quant.hadamard_utils import get_hadK as mquant_get_hadK
            from fake_quant.internvl_rotation import fuse_internvl_layer_norms
            from fake_quant.internvl_rotation import rotate_internvl2_model
            from fake_quant.minicpmv_rotation import fuse_minicpmv_layer_norms
            from fake_quant.minicpmv_rotation import rotate_minicpmv_model
            from fake_quant.qwen2vl_rotation import fuse_qwen2vl_layer_norms
            from fake_quant.qwen2vl_rotation import rotate_qwen2vl_model

            if not hasattr(mquant_internvl_rotation, "utils"):
                mquant_internvl_rotation.utils = mquant_utils
            internvl_visual_quantizers: dict[str, Any] = {}
            mquant_internvl_gptq_plus.internvl_visual_clip_rtn = (
                lambda model, dev, args: _mindpipe_internvl_visual_clip_rtn(
                    model,
                    dev,
                    args,
                    mquant_internvl_gptq_plus.quant_utils,
                    internvl_visual_quantizers,
                )
            )
            mquant_internvl_gptq_plus.internvl_visual_cross_attention_rtn = (
                lambda model, dev, args: _mindpipe_internvl_visual_cross_attention_rtn(
                    model,
                    dev,
                    args,
                    mquant_internvl_gptq_plus.quant_utils,
                    internvl_visual_quantizers,
                )
            )
            mquant_internvl_gptq_plus.gptq_internvl_fwrd_visual_clip_conv1 = (
                lambda model, dataset, dev, dataset_name, args, quantizers:
                _mindpipe_gptq_internvl_fwrd_visual_clip_conv1(
                    mquant_internvl_gptq_plus,
                    model,
                    dataset,
                    dev,
                    dataset_name,
                    args,
                    quantizers,
                )
            )
            mquant_internvl_gptq_plus.gptq_internvl_fwrd_visual_clip_resblocks = (
                lambda model, dataset, dev, dataset_name, args, quantizers:
                _mindpipe_gptq_internvl_fwrd_visual_clip_resblocks(
                    mquant_internvl_gptq_plus,
                    model,
                    dataset,
                    dev,
                    dataset_name,
                    args,
                    quantizers,
                )
            )
            mquant_internvl_gptq_plus.gptq_internvl_fwrd_visual_clip_cross_attention = (
                lambda model, dataset, dev, dataset_name, args, quantizers:
                _mindpipe_gptq_internvl_fwrd_visual_clip_cross_attention(
                    mquant_internvl_gptq_plus,
                    model,
                    dataset,
                    dev,
                    dataset_name,
                    args,
                    quantizers,
                )
            )
            mquant_internvl_gptq_plus.gptq_internvl_fwrd_llm = (
                lambda model, dataset, dev, dataset_name, args, quantizers:
                _mindpipe_gptq_internvl_fwrd_llm(
                    mquant_internvl_gptq_plus,
                    model,
                    dataset,
                    dev,
                    dataset_name,
                    args,
                    quantizers,
                )
            )

            if family in {"qwen2vl", "qwen3vl"}:
                need_static_activation_calib = (
                    int(mquant_args.visual_a_bits) < 16 and bool(mquant_args.visual_static)
                ) or (
                    int(mquant_args.llm_a_bits) < 16 and bool(mquant_args.llm_static)
                )
                need_act_quant_wrappers = (
                    use_gptq
                    or int(mquant_args.visual_a_bits) < 16
                    or int(mquant_args.llm_a_bits) < 16
                    or bool(mquant_args.visual_static)
                    or bool(mquant_args.llm_static)
                    or bool(mquant_args.online_visual_hadamard)
                    or bool(mquant_args.online_llm_hadamard)
                )
                need_visual_runtime_wrappers = (
                    not need_act_quant_wrappers
                    and bool(mquant_args.visual_w_rtn)
                    and (bool(mquant_args.quant_visual_clip) or bool(mquant_args.quant_cross_attention))
                )
                if use_gptq or need_static_activation_calib:
                    processor = getattr(tokenizer_bundle, "processor", None)
                    tokenizer = getattr(tokenizer_bundle, "tokenizer", None)
                    if processor is None or tokenizer is None:
                        raise ValueError(
                            "MQuant Qwen2/Qwen2.5-VL path requires both tokenizer and processor in TokenizerBundle."
                        )
                    if mquant_args.dataset_name:
                        mquant_dataset = mquant_build_dataset(mquant_args.dataset_name)
                        if mquant_dataset is None:
                            raise ValueError(f"Failed to build VLM dataset: {mquant_args.dataset_name}")
                        if mquant_calib_num is not None and hasattr(mquant_dataset, "data"):
                            mquant_dataset.data = mquant_dataset.data.head(int(mquant_calib_num))
                    elif use_gptq:
                        raise ValueError(
                            "MQuant GPTQ for Qwen2/Qwen2.5-VL requires `--mquant_dataset_name`."
                        )
                    proxy = _MindPipeQwen2VLGPTQWrapper(
                        model_root=mquant_root_model,
                        source_model=source_model,
                        processor=processor,
                        tokenizer=tokenizer,
                        target_device=args.device,
                        max_new_tokens=int(getattr(args, "mquant_max_new_tokens", 20)),
                    )

                fuse_args = mquant_args
                if not mquant_args.not_fuse_layer_norms:
                    if str(model_type) == "qwen2_5_vl" and not bool(mquant_args.no_fuse_visual_clip):
                        _fuse_qwen2_5_vl_visual_layer_norms(
                            mquant_root_model,
                            mquant_rotation_utils,
                        )
                        fuse_args = _copy_namespace(mquant_args, no_fuse_visual_clip=True)
                    fuse_qwen2vl_layer_norms(proxy, fuse_args)
                if mquant_args.rotate:
                    original_get_orthogonal_matrix = mquant_qwen2vl_rotation.get_orthogonal_matrix

                    def _device_aware_get_orthogonal_matrix(size, mode):
                        visual_device = self._infer_module_device(mquant_root_model.visual.patch_embed.proj)
                        return mquant_rotation_utils.get_orthogonal_matrix(
                            size, mode, device=visual_device
                        )

                    mquant_qwen2vl_rotation.get_orthogonal_matrix = _device_aware_get_orthogonal_matrix
                    original_named_modules = None
                    try:
                        rotate_args = mquant_args
                        if str(model_type) == "qwen2_5_vl" and bool(mquant_args.rotate_visual_clip):
                            _rotate_qwen2_5_vl_visual(
                                mquant_root_model,
                                mquant_args,
                                rotation_utils=mquant_rotation_utils,
                                qwen2vl_rotation=mquant_qwen2vl_rotation,
                                mquant_utils=mquant_utils,
                                apply_exact_had_to_linear=apply_exact_had_to_linear,
                            )
                            rotate_args = _copy_namespace(mquant_args, rotate_visual_clip=False)
                        if str(model_type) == "qwen2_5_vl" and bool(rotate_args.online_llm_hadamard):
                            original_named_modules = mquant_root_model.named_modules

                            def _named_modules_without_visual(self, *named_args, **named_kwargs):
                                for name, module in original_named_modules(*named_args, **named_kwargs):
                                    if name == "visual" or name.startswith("visual."):
                                        continue
                                    yield name, module

                            mquant_root_model.named_modules = MethodType(
                                _named_modules_without_visual,
                                mquant_root_model,
                            )
                        rotate_qwen2vl_model(mquant_root_model, rotate_args)
                    finally:
                        if original_named_modules is not None:
                            mquant_root_model.named_modules = original_named_modules
                        mquant_qwen2vl_rotation.get_orthogonal_matrix = original_get_orthogonal_matrix
                if need_act_quant_wrappers or need_visual_runtime_wrappers:
                    mquant_quant_utils.qwen2vl_add_act_qaunt(proxy, mquant_args)
                    if str(model_type) == "qwen3_vl":
                        _qwen3_vl_add_extra_act_quant_wrappers(
                            mquant_quant_utils,
                            model_root=mquant_root_model,
                            mquant_args=mquant_args,
                        )
                    self._configure_qwen2_online_hadamard_wrappers(
                        mquant_quant_utils,
                        mquant_get_hadK,
                        mquant_utils,
                        model_root=mquant_root_model,
                        mquant_args=mquant_args,
                    )
                if str(model_type) == "qwen3_vl" and use_gptq:
                    quantizer_map = _qwen3_vl_rtn_gptq_fwrd_plus(
                        mquant_qwen2vl_gptq_plus,
                        mquant_quant_utils,
                        proxy=proxy,
                        dataset=mquant_dataset,
                        dev=args.device,
                        dataset_name=mquant_args.dataset_name,
                        args=mquant_args,
                    )
                else:
                    quantizer_map = mquant_gptq.qwen2vl_rtn_gptq_fwrd_plus(
                        proxy,
                        dataset=mquant_dataset,
                        dev=args.device,
                        dataset_name=mquant_args.dataset_name,
                        args=mquant_args,
                    )
                if str(model_type) == "qwen3_vl" and mquant_args.visual_w_rtn and not use_gptq:
                    _qwen3_vl_quantize_deepstack_mergers_rtn(
                        mquant_quant_utils,
                        model_root=mquant_root_model,
                        mquant_args=mquant_args,
                        quantizers=quantizer_map,
                    )
                if (
                    int(mquant_args.visual_a_bits) < 16
                    or int(mquant_args.llm_a_bits) < 16
                    or bool(mquant_args.visual_static)
                    or bool(mquant_args.llm_static)
                ):
                    if (bool(mquant_args.visual_static) or bool(mquant_args.llm_static)) and mquant_dataset is None:
                        logging.warning(
                            "Static activation quantization requested but no VLM calibration dataset is available; "
                            "forcing dynamic activation quantization for this run."
                        )
                        mquant_args.visual_static = False
                        mquant_args.llm_static = False
                    if int(mquant_args.visual_a_bits) < 16 or bool(mquant_args.visual_static):
                        act_skip_names = list(mquant_args.skip_names) + list(
                            getattr(mquant_args, "act_skip_names", [])
                        )
                        configured_act_quantizers += self._configure_activation_quantizers(
                            mquant_quant_utils,
                            modules=[mquant_root_model.visual],
                            bits=int(mquant_args.visual_a_bits),
                            groupsize=int(mquant_args.a_groupsize),
                            symmetric=bool(args.activation_symmetric),
                            clip_ratio=float(mquant_args.a_clip_ratio),
                            static=bool(mquant_args.visual_static),
                            act_per_tensor=bool(mquant_args.act_per_tensor),
                            skip_names=act_skip_names,
                        )
                    if int(mquant_args.llm_a_bits) < 16 or bool(mquant_args.llm_static):
                        act_skip_names = list(mquant_args.skip_names) + list(
                            getattr(mquant_args, "act_skip_names", [])
                        )
                        configured_act_quantizers += self._configure_activation_quantizers(
                            mquant_quant_utils,
                            modules=[mquant_root_model.model],
                            bits=int(mquant_args.llm_a_bits),
                            groupsize=int(mquant_args.a_groupsize),
                            symmetric=bool(args.activation_symmetric),
                            clip_ratio=float(mquant_args.a_clip_ratio),
                            static=bool(mquant_args.llm_static),
                            act_per_tensor=bool(mquant_args.act_per_tensor),
                            skip_names=act_skip_names,
                        )
                    if (bool(mquant_args.visual_static) or bool(mquant_args.llm_static)) and mquant_dataset is not None:
                        mquant_quant_utils.calib_qwen2vl_plus(
                            proxy,
                            mquant_args,
                            mquant_dataset,
                            int(mquant_args.calib_num),
                        )

            elif family == "internvl":
                need_static_activation_calib = bool(mquant_args.visual_static) or bool(mquant_args.llm_static)
                if use_gptq or (need_static_activation_calib and bool(mquant_args.dataset_name)):
                    tokenizer = getattr(tokenizer_bundle, "tokenizer", None)
                    if tokenizer is None:
                        raise ValueError(
                            "MQuant InternVL GPTQ/static activation path requires a tokenizer in TokenizerBundle."
                        )
                    if mquant_args.dataset_name:
                        mquant_dataset = mquant_build_dataset(mquant_args.dataset_name)
                        if mquant_dataset is None:
                            raise ValueError(f"Failed to build VLM dataset: {mquant_args.dataset_name}")
                        if mquant_calib_num is not None and hasattr(mquant_dataset, "data"):
                            mquant_dataset.data = mquant_dataset.data.head(int(mquant_calib_num))
                    else:
                        raise ValueError("MQuant InternVL GPTQ requires `--mquant_dataset_name`.")
                    proxy = _MindPipeInternVLGPTQWrapper(
                        source_model=source_model,
                        tokenizer=tokenizer,
                        target_device=args.device,
                        max_new_tokens=int(getattr(args, "mquant_max_new_tokens", 20)),
                        use_cache=False,
                    )
                mquant_quant_utils.fuse_internvl(proxy)
                if not mquant_args.not_fuse_layer_norms:
                    fuse_internvl_layer_norms(proxy, mquant_args)
                if mquant_args.rotate:
                    rotate_internvl2_model(source_model, mquant_args)
                if (
                    int(mquant_args.visual_a_bits) < 16
                    or int(mquant_args.llm_a_bits) < 16
                    or bool(mquant_args.visual_static)
                    or bool(mquant_args.llm_static)
                ):
                    mquant_quant_utils.internvl_add_act_qaunt(proxy, mquant_args)
                if use_gptq:
                    _ensure_internvl_gptq_mquant_compat(source_model)
                quantizer_map = mquant_gptq.internvl_rtn_gptq_fwrd_plus(
                    proxy,
                    dataset=mquant_dataset,
                    dev=args.device,
                    dataset_name=mquant_args.dataset_name,
                    args=mquant_args,
                )
                quantizer_map.update(internvl_visual_quantizers)
                if (bool(mquant_args.visual_static) or bool(mquant_args.llm_static)) and mquant_dataset is None:
                    logging.warning(
                        "Static activation quantization requested but no VLM calibration dataset is available; "
                        "forcing dynamic activation quantization for this run."
                    )
                    mquant_args.visual_static = False
                    mquant_args.llm_static = False
                if int(mquant_args.visual_a_bits) < 16 or bool(mquant_args.visual_static):
                    act_skip_names = list(mquant_args.skip_names) + list(
                        getattr(mquant_args, "act_skip_names", [])
                    )
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[source_model.vision_model, source_model.mlp1],
                        bits=int(mquant_args.visual_a_bits),
                        groupsize=int(mquant_args.a_groupsize),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        static=bool(mquant_args.visual_static),
                        act_per_tensor=bool(mquant_args.act_per_tensor),
                        skip_names=act_skip_names,
                    )
                if int(mquant_args.llm_a_bits) < 16 or bool(mquant_args.llm_static):
                    act_skip_names = list(mquant_args.skip_names) + list(
                        getattr(mquant_args, "act_skip_names", [])
                    )
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[source_model.language_model],
                        bits=int(mquant_args.llm_a_bits),
                        groupsize=int(mquant_args.a_groupsize),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        static=bool(mquant_args.llm_static),
                        act_per_tensor=bool(mquant_args.act_per_tensor),
                        skip_names=act_skip_names,
                    )
                if (bool(mquant_args.visual_static) or bool(mquant_args.llm_static)) and mquant_dataset is not None:
                    mquant_quant_utils.calib_vqa_plus(
                        proxy,
                        mquant_args,
                        mquant_dataset,
                        int(mquant_args.calib_num),
                    )

            elif family == "minicpmv":
                if use_gptq:
                    tokenizer = getattr(tokenizer_bundle, "tokenizer", None)
                    if tokenizer is None:
                        raise ValueError(
                            "MQuant MiniCPM-V GPTQ path requires a tokenizer in TokenizerBundle."
                        )
                    if mquant_args.dataset_name:
                        mquant_dataset = mquant_build_dataset(mquant_args.dataset_name)
                        if mquant_dataset is None:
                            raise ValueError(f"Failed to build VLM dataset: {mquant_args.dataset_name}")
                        if mquant_calib_num is not None and hasattr(mquant_dataset, "data"):
                            mquant_dataset.data = mquant_dataset.data.head(int(mquant_calib_num))
                    else:
                        raise ValueError("MQuant MiniCPM-V GPTQ requires `--mquant_dataset_name`.")
                    proxy = _MindPipeMiniCPMVGPTQWrapper(
                        source_model=source_model,
                        tokenizer=tokenizer,
                        target_device=args.device,
                        max_new_tokens=int(getattr(args, "mquant_max_new_tokens", 20)),
                        use_cache=False,
                    )
                if not mquant_args.not_fuse_layer_norms:
                    fuse_minicpmv_layer_norms(source_model, mquant_args)
                if mquant_args.rotate:
                    rotate_minicpmv_model(source_model, mquant_args)
                if (
                    int(mquant_args.visual_a_bits) < 16
                    or int(mquant_args.llm_a_bits) < 16
                    or bool(mquant_args.visual_static)
                    or bool(mquant_args.llm_static)
                ):
                    mquant_quant_utils.minicpmv_add_act_qaunt(source_model, mquant_args)
                    self._configure_minicpmv_online_hadamard_wrappers(
                        mquant_quant_utils,
                        mquant_get_hadK,
                        mquant_utils,
                        source_model=source_model,
                        mquant_args=mquant_args,
                    )
                elif bool(mquant_args.quant_visual_clip) and not hasattr(
                    source_model.vpm.embeddings.patch_embedding,
                    "module",
                ):
                    source_model.vpm.embeddings.patch_embedding = _MiniCPMVLeafModuleProxy(
                        source_model.vpm.embeddings.patch_embedding
                    )
                quantizer_map = _minicpmv_rtn_gptq_fwrd_plus(
                    mquant_minicpmv_gptq_plus,
                    proxy,
                    dataset=mquant_dataset,
                    dev=args.device,
                    dataset_name=mquant_args.dataset_name,
                    args=mquant_args,
                )
                if int(mquant_args.visual_a_bits) < 16 or bool(mquant_args.visual_static):
                    act_skip_names = list(mquant_args.skip_names) + list(
                        getattr(mquant_args, "act_skip_names", [])
                    )
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[source_model.vpm, source_model.resampler],
                        bits=int(mquant_args.visual_a_bits),
                        groupsize=int(mquant_args.a_groupsize),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        static=bool(mquant_args.visual_static),
                        act_per_tensor=bool(mquant_args.act_per_tensor),
                        skip_names=act_skip_names,
                    )
                if int(mquant_args.llm_a_bits) < 16 or bool(mquant_args.llm_static):
                    act_skip_names = list(mquant_args.skip_names) + list(
                        getattr(mquant_args, "act_skip_names", [])
                    )
                    configured_act_quantizers += self._configure_activation_quantizers(
                        mquant_quant_utils,
                        modules=[source_model.llm.model],
                        bits=int(mquant_args.llm_a_bits),
                        groupsize=int(mquant_args.a_groupsize),
                        symmetric=bool(args.activation_symmetric),
                        clip_ratio=1.0,
                        static=bool(mquant_args.llm_static),
                        act_per_tensor=bool(mquant_args.act_per_tensor),
                        skip_names=act_skip_names,
                    )

        quantized_names = sorted(quantizer_map.keys())
        return {
            "mquant_root": str(mquant_root),
            "mquant_family": family,
            "weight_method": str(args.weight_method),
            "mquant_dataset_name": mquant_args.dataset_name or None,
            "mquant_calib_num": int(mquant_args.calib_num),
            "rotation_mode": str(args.rotation_mode),
            "quantized_linear_count": len(quantized_names),
            "quantized_linear_layers": quantized_names,
            "configured_act_quantizer_count": int(configured_act_quantizers),
            "mquant_config": {
                "llm_w_bits": int(mquant_args.llm_w_bits),
                "visual_w_bits": int(mquant_args.visual_w_bits),
                "llm_a_bits": int(mquant_args.llm_a_bits),
                "visual_a_bits": int(mquant_args.visual_a_bits),
                "visual_w_clip": bool(mquant_args.visual_w_clip),
                "llm_w_clip": bool(mquant_args.llm_w_clip),
                "visual_static": bool(mquant_args.visual_static),
                "llm_static": bool(mquant_args.llm_static),
                "w_groupsize": int(mquant_args.w_groupsize),
                "a_groupsize": int(mquant_args.a_groupsize),
                "quant_llm": bool(mquant_args.quant_llm),
                "quant_visual_clip": bool(mquant_args.quant_visual_clip),
                "quant_cross_attention": bool(mquant_args.quant_cross_attention),
                "not_fuse_layer_norms": bool(mquant_args.not_fuse_layer_norms),
                "no_fuse_visual_clip": bool(mquant_args.no_fuse_visual_clip),
                "no_fuse_visual_cross_attn": bool(mquant_args.no_fuse_visual_cross_attn),
                "no_fuse_llm": bool(mquant_args.no_fuse_llm),
                "rotate": bool(mquant_args.rotate),
                "rotate_visual_clip": bool(mquant_args.rotate_visual_clip),
                "rotate_visual_cross_attn": bool(mquant_args.rotate_visual_cross_attn),
                "rotate_llm": bool(mquant_args.rotate_llm),
                "act_per_tensor": bool(mquant_args.act_per_tensor),
                "online_visual_hadamard": bool(mquant_args.online_visual_hadamard),
                "online_llm_hadamard": bool(mquant_args.online_llm_hadamard),
                "fp32_had": bool(mquant_args.fp32_had),
                "visual_split": bool(mquant_args.visual_split),
                "llm_split": bool(mquant_args.llm_split),
                "skip_names": list(mquant_args.skip_names),
                "act_skip_names": list(getattr(mquant_args, "act_skip_names", [])),
            },
        }
