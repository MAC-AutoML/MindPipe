"""Batch collators for compression-aware LoRA training stages."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import importlib.util
from pathlib import Path
from typing import Any
from typing import Sequence

from PIL import Image
import torch

from algorithm.common.modeling import MiniCPMTokenizerAdapter


IGNORE_INDEX = -100


@dataclass
class RawTextCPTCollator:
    tokenizer: Any

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for CPT training.")
        input_ids_list = [
            torch.tensor(example["input_ids"], dtype=torch.long)
            for example in instances
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = input_ids.ne(pad_token_id)
        labels = input_ids.clone().masked_fill(~attention_mask, IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class TextSFTCollator:
    tokenizer: Any
    max_length: int
    min_response_tokens: int = 8

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for SFT training.")

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        eos = self.tokenizer.eos_token or ""
        for example in instances:
            prompt = str(example["prompt"])
            response = str(example["response"]) + eos
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            response_ids = self.tokenizer(
                response,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            response_limit = max(0, int(self.max_length) - len(prompt_ids))
            if response_limit <= 0:
                continue
            response_ids = response_ids[:response_limit]
            if len(response_ids) < int(self.min_response_tokens):
                continue
            input_ids = prompt_ids + response_ids
            labels = ([IGNORE_INDEX] * len(prompt_ids)) + response_ids
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        if not input_ids_list:
            raise ValueError("Text SFT batch has no valid examples with assistant labels.")

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(pad_token_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _load_minicpm_sft_module():
    source_path = Path("/mnt/42_store/wxx/flatQuant/FlatQuant/lora_utils/prepare_data_sft_minicpm.py")
    if not source_path.exists():
        raise FileNotFoundError(f"MiniCPM-V SFT preprocessing module not found: {source_path}")
    spec = importlib.util.spec_from_file_location("mindpipe_minicpm_sft_data", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MiniCPM-V SFT preprocessing module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class MiniCPMVImageTextSFTCollator:
    tokenizer: Any
    model: Any
    max_length: int

    def __post_init__(self) -> None:
        self._minicpm_sft = _load_minicpm_sft_module()
        source_model = getattr(self.model, "_source_model", self.model)
        if not hasattr(source_model, "transform"):
            raise AttributeError("MiniCPM-V source model must expose `transform` for image preprocessing.")
        if not hasattr(self.tokenizer, "im_start"):
            self.tokenizer = MiniCPMTokenizerAdapter(self.tokenizer)
        self.transform = source_model.transform
        source_config = getattr(source_model, "config", getattr(self.model, "config", None))
        self.query_nums = int(getattr(source_config, "query_num", 64))
        self.batch_vision = bool(getattr(source_config, "batch_vision_input", False))
        self._collate = partial(self._minicpm_sft.data_collator, max_length=int(self.max_length))

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        examples: list[dict[str, Any]] = []
        for instance in instances:
            image_path = instance.get("image")
            if not image_path:
                images = instance.get("images") or []
                image_path = images[0] if images else None
            if not image_path:
                continue
            image = Image.open(str(image_path)).convert("RGB")
            conversations = instance.get("conversations") or instance.get("messages")
            if not conversations:
                continue
            raw_data = {"image": str(image_path), **dict(instance)}
            ret = self._minicpm_sft.preprocess(
                image,
                conversations,
                self.tokenizer,
                self.transform,
                query_nums=self.query_nums,
                llm_type="minicpm",
                batch_vision=self.batch_vision,
                raw_data=raw_data,
            )
            labels = ret["target"]
            if not torch.is_tensor(labels) or not labels.ne(IGNORE_INDEX).any().item():
                continue
            examples.append(
                {
                    "input_ids": ret["input_ids"],
                    "position_ids": ret["position_ids"],
                    "labels": labels,
                    "attention_mask": torch.ones_like(ret["input_ids"], dtype=torch.bool),
                    "pixel_values": ret["pixel_values"],
                    "tgt_sizes": ret["tgt_sizes"],
                    "image_bound": ret["image_bound"],
                }
            )
        if not examples:
            raise ValueError("MiniCPM-V SFT batch has no valid image-text examples.")
        return self._collate(examples)


def _qwen_message_content(
    text: str,
    image_path: str | None,
    image_max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    text = text.replace("<image>", "").strip()
    content: list[dict[str, Any]] = []
    if image_path:
        image_item: dict[str, Any] = {"type": "image", "image": str(image_path)}
        if image_max_pixels is not None and image_max_pixels > 0:
            image_item["max_pixels"] = int(image_max_pixels)
        content.append(image_item)
    if text:
        content.append({"type": "text", "text": text})
    return content


def _as_qwen_messages(
    instance: dict[str, Any],
    image_max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    image_path = instance.get("image")
    if not image_path:
        images = instance.get("images") or []
        image_path = images[0] if images else None
    conversations = instance.get("conversations") or instance.get("messages") or []
    messages: list[dict[str, Any]] = []
    image_consumed = False
    for message in conversations:
        role = message.get("role", message.get("from")) if isinstance(message, dict) else None
        content = message.get("content", message.get("value")) if isinstance(message, dict) else None
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role not in {"user", "assistant"} or content is None:
            continue
        text = str(content)
        if role == "user":
            use_image = str(image_path) if image_path and not image_consumed else None
            messages.append({"role": role, "content": _qwen_message_content(text, use_image, image_max_pixels)})
            image_consumed = image_consumed or bool(use_image)
        else:
            messages.append({"role": role, "content": text.strip()})
    return messages


def _pad_1d_tensors(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [tensor.to(dtype=torch.long).view(-1) for tensor in tensors],
        batch_first=True,
        padding_value=padding_value,
    )


@dataclass
class QwenVLImageTextSFTCollator:
    processor: Any
    tokenizer: Any
    max_length: int
    model_type: str
    image_max_pixels: int | None = 262144

    def __post_init__(self) -> None:
        if self.processor is None:
            raise AttributeError("Qwen-VL SFT requires a tokenizer bundle with `processor`.")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Qwen-VL tokenizer must define pad_token_id.")
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError(
                "qwen_vl_utils is required for Qwen-VL LLaVA SFT. "
                "Please install `qwen-vl-utils`."
            ) from err
        self._process_vision_info = process_vision_info

    def _vision_info(self, messages: list[dict[str, Any]]):
        if self.model_type in {"qwen3_vl", "qwen3_5"}:
            result = self._process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            images, videos, video_kwargs = result
            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            return images, videos, video_metadatas, dict(video_kwargs or {})
        images, videos = self._process_vision_info([messages])
        return images, videos, None, {}

    def _encode(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool):
        template_input: Any = messages if self.model_type == "qwen3_vl" else [messages]
        prompt = self.processor.apply_chat_template(
            template_input,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        images, videos, video_metadatas, video_kwargs = self._vision_info(messages)
        processor_kwargs = {
            "text": prompt,
            "images": images,
            "videos": videos,
            "return_tensors": "pt",
            "padding": False,
        }
        if self.model_type in {"qwen3_vl", "qwen3_5"}:
            processor_kwargs["do_resize"] = False
            processor_kwargs["video_metadata"] = video_metadatas
            processor_kwargs.update(video_kwargs)
        return self.processor(**processor_kwargs)

    def _image_pixel_budgets(self) -> list[int | None]:
        if self.image_max_pixels is None or self.image_max_pixels <= 0:
            return [None]
        budgets: list[int | None] = []
        value = int(self.image_max_pixels)
        while value >= 32768:
            budgets.append(value)
            value //= 2
        if 32768 not in budgets:
            budgets.append(32768)
        return budgets

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded_examples: list[dict[str, Any]] = []
        for instance in instances:
            for image_max_pixels in self._image_pixel_budgets():
                messages = _as_qwen_messages(dict(instance), image_max_pixels=image_max_pixels)
                if len(messages) < 2 or messages[-1].get("role") != "assistant":
                    break
                prompt_messages = messages[:-1]
                full_inputs = self._encode(messages, add_generation_prompt=False)
                prompt_inputs = self._encode(prompt_messages, add_generation_prompt=True)
                input_ids = full_inputs["input_ids"][0][: self.max_length]
                prompt_len = min(int(prompt_inputs["input_ids"].shape[-1]), int(input_ids.numel()))
                labels = input_ids.clone()
                labels[:prompt_len] = IGNORE_INDEX
                if not labels.ne(IGNORE_INDEX).any().item():
                    continue
                item = {key: value for key, value in dict(full_inputs).items() if value is not None}
                item["input_ids"] = input_ids
                for key in ("mm_token_type_ids", "token_type_ids"):
                    value = item.get(key)
                    if torch.is_tensor(value) and value.dim() >= 2 and value.shape[-1] >= input_ids.numel():
                        item[key] = value[0][: input_ids.numel()]
                    elif torch.is_tensor(value) and value.dim() == 1 and value.numel() >= input_ids.numel():
                        item[key] = value[: input_ids.numel()]
                item["labels"] = labels
                encoded_examples.append(item)
                break

        if not encoded_examples:
            raise ValueError("Qwen-VL SFT batch has no valid image-text examples.")

        pad_token_id = int(self.tokenizer.pad_token_id)
        batch: dict[str, Any] = {
            "input_ids": _pad_1d_tensors([item["input_ids"] for item in encoded_examples], pad_token_id),
            "labels": _pad_1d_tensors([item["labels"] for item in encoded_examples], IGNORE_INDEX),
        }
        batch["attention_mask"] = batch["input_ids"].ne(pad_token_id).long()
        mm_token_type_tensors = [
            item.get("mm_token_type_ids")
            for item in encoded_examples
            if torch.is_tensor(item.get("mm_token_type_ids"))
        ]
        if mm_token_type_tensors:
            batch["mm_token_type_ids"] = _pad_1d_tensors(mm_token_type_tensors, 0)
        token_type_tensors = [
            item.get("token_type_ids")
            for item in encoded_examples
            if torch.is_tensor(item.get("token_type_ids"))
        ]
        if token_type_tensors:
            batch["token_type_ids"] = _pad_1d_tensors(token_type_tensors, 0)

        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            tensors = [item.get(key) for item in encoded_examples if torch.is_tensor(item.get(key))]
            if not tensors:
                continue
            if key.startswith("pixel_values"):
                batch[key] = torch.cat(tensors, dim=0)
            elif all(tensor.dim() > 0 for tensor in tensors):
                batch[key] = torch.cat(tensors, dim=0)
            else:
                batch[key] = torch.stack(tensors, dim=0)
        return batch
