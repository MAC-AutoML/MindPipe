"""VLMEvalKit integration for in-memory multimodal model evaluation."""

from __future__ import annotations

import contextlib
import gc
import importlib
import inspect
import json
import logging
import math
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from algorithm.common.device import empty_cache
from algorithm.common.device import resolve_device
from algorithm.common.io import ensure_dir
from algorithm.common.io import model_slug
from algorithm.common.io import write_json
from algorithm.common.modeling import MiniCPMTokenizerAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLMEVALKIT_ROOT = os.environ.get(
    "VLMEVALKIT_ROOT",
    str(REPO_ROOT / "third_party" / "VLMEvalKit"),
)

QWEN3_5_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
    }
)
QWEN3_VL_MODEL_TYPES = frozenset({"qwen3_vl", *QWEN3_5_MODEL_TYPES})


def _qwen3_disable_thinking(model_type: str) -> bool:
    return model_type in QWEN3_5_MODEL_TYPES


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return {
            "columns": [str(column) for column in value.columns],
            "records": _json_safe(value.to_dict(orient="records")),
        }
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


@contextlib.contextmanager
def _temporary_env(**updates: str | None):
    previous_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_vlmeval_modules(vlm_eval_kit_root: str) -> dict[str, Any]:
    _ensure_vlmeval_transformers_compat()
    _ensure_vlmeval_moviepy_compat()
    root = Path(vlm_eval_kit_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"VLMEvalKit root does not exist: {root}. "
            "Initialize the `third_party/VLMEvalKit` submodule or set `VLMEVALKIT_ROOT`."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    dataset_module = importlib.import_module("vlmeval.dataset")
    inference_module = importlib.import_module("vlmeval.inference")
    smp_module = importlib.import_module("vlmeval.smp")
    base_module = importlib.import_module("vlmeval.vlm.base")
    dataset_modality_resolver = getattr(dataset_module, "DATASET_MODALITY", None)
    if dataset_modality_resolver is None:
        dataset_modality_resolver = lambda _dataset_name: "IMAGE"

    return {
        "BaseModel": base_module.BaseModel,
        "build_dataset": dataset_module.build_dataset,
        "DATASET_TYPE": dataset_module.DATASET_TYPE,
        "DATASET_MODALITY": dataset_modality_resolver,
        "infer_data_job": inference_module.infer_data_job,
        "get_pred_file_path": getattr(smp_module, "get_pred_file_path", None),
        "listinstr": smp_module.listinstr,
        "MMBenchOfficialServer": smp_module.MMBenchOfficialServer,
    }


def _ensure_vlmeval_transformers_compat() -> None:
    """Patch removed/renamed HF symbols expected by older VLMEvalKit revisions."""
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        fallback = getattr(transformers, "AutoModelForImageTextToText", None)
        if fallback is not None:
            transformers.AutoModelForVision2Seq = fallback


def _ensure_vlmeval_moviepy_compat() -> None:
    """Patch moviepy symbol locations expected by older VLMEvalKit revisions."""
    try:
        import moviepy
    except Exception:
        return

    if hasattr(moviepy, "VideoFileClip") and hasattr(moviepy, "ImageSequenceClip"):
        return

    try:
        from moviepy.editor import ImageSequenceClip
        from moviepy.editor import VideoFileClip
    except Exception:
        return

    if not hasattr(moviepy, "VideoFileClip"):
        moviepy.VideoFileClip = VideoFileClip
    if not hasattr(moviepy, "ImageSequenceClip"):
        moviepy.ImageSequenceClip = ImageSequenceClip


def _resolve_work_dir(common_args: dict[str, Any]) -> Path:
    explicit = common_args.get("vlm_work_dir")
    if explicit:
        return ensure_dir(Path(explicit))
    output_root = common_args.get("evaluation_output_dir")
    if output_root:
        return ensure_dir(Path(output_root) / "vlm_eval")
    return ensure_dir(Path("results") / "vlm_eval")


def _resolve_model_name(common_args: dict[str, Any]) -> str:
    model_path = common_args.get("model_path")
    if model_path:
        return model_slug(model_path)
    return "mindpipe_model"


def _resolve_vlm_progress_path(common_args: dict[str, Any]) -> Path:
    output_root = common_args.get("evaluation_output_dir")
    if output_root:
        return ensure_dir(Path(output_root)) / "vlm_eval_progress.json"
    explicit_work_dir = common_args.get("vlm_work_dir")
    if explicit_work_dir:
        return ensure_dir(Path(explicit_work_dir)).parent / "vlm_eval_progress.json"
    return ensure_dir(Path("results")) / "vlm_eval_progress.json"


def _load_json_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_existing_vlm_records(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if "vlm_eval" in payload and isinstance(payload.get("vlm_eval"), dict):
        datasets = payload["vlm_eval"].get("datasets")
        return datasets if isinstance(datasets, dict) else {}
    datasets = payload.get("datasets")
    return datasets if isinstance(datasets, dict) else {}


def _vlm_record_is_complete(record: dict[str, Any], mode: str) -> bool:
    if not isinstance(record, dict):
        return False
    if mode == "infer":
        return bool(record.get("inference_completed"))
    if mode == "eval":
        return "evaluation" in record or "evaluation_skipped" in record
    return bool(record.get("inference_completed")) and (
        "evaluation" in record or "evaluation_skipped" in record
    )


def _persist_vlm_progress(
    progress_path: Path,
    *,
    mode: str,
    model_name: str,
    work_dir: Path,
    results: dict[str, Any],
) -> None:
    write_json(
        progress_path,
        {
            "vlm_eval": {
                "mode": mode,
                "model_name": model_name,
                "work_dir": str(work_dir),
                "datasets": results,
            }
        },
    )


def _default_max_new_tokens(dataset_name: str | None, dataset_type: str) -> int:
    if dataset_name == "OCRBench":
        return 32
    if dataset_name == "ChartQA_TEST":
        return 16
    if dataset_name in {"TextVQA_VAL", "InfoVQA_VAL"}:
        return 32
    if dataset_type == "MCQ":
        return 128
    if dataset_type == "Y/N":
        return 32
    return 512


def _maybe_to_device(batch: Any, device) -> Any:
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except TypeError:
            pass
    if isinstance(batch, dict):
        return {key: _maybe_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_maybe_to_device(value, device) for value in batch)
    return batch


def _model_input_device(model, fallback_device):
    try:
        parameter = next(model.parameters())
        return parameter.device
    except Exception:
        pass
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return fallback_device


def _open_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _resolve_vlmeval_img_root(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return dataset_name
    try:
        image_base_module = importlib.import_module("vlmeval.dataset.image_base")
        resolver = getattr(image_base_module, "img_root_map", None)
        if callable(resolver):
            return str(resolver(dataset_name))
    except Exception:
        pass
    return dataset_name


def _looks_like_url(value: str) -> bool:
    return "://" in value


def _guess_media_kind(value: str) -> str | None:
    mime, _ = mimetypes.guess_type(value)
    if not mime:
        return None
    return mime.split("/")[0]


def _resolve_media_value(value: Any, dataset_name: str | None) -> str:
    candidate = str(value).strip()
    if not candidate or _looks_like_url(candidate):
        return candidate

    path = Path(candidate).expanduser()
    if path.exists():
        return str(path.resolve())

    lmu_root = os.environ.get("LMUData")
    if not lmu_root:
        return candidate

    dataset_root_name = _resolve_vlmeval_img_root(dataset_name)
    dataset_root = Path(lmu_root).expanduser() / "images"
    if dataset_root_name:
        dataset_root = dataset_root / dataset_root_name

    fallback_candidates: list[Path] = []
    if path.is_absolute():
        parts = list(path.parts)
        if "images" in parts:
            image_index = parts.index("images")
            relative_parts = parts[image_index + 1 :]
            if relative_parts:
                fallback_candidates.append(Path(lmu_root).expanduser() / "images" / Path(*relative_parts))
    else:
        fallback_candidates.append(dataset_root / path)
    fallback_candidates.append(dataset_root / path.name)

    for fallback in fallback_candidates:
        if fallback.exists():
            return str(fallback.resolve())
    return candidate


def _normalize_vlm_message(message: Any, dataset_name: str | None) -> list[dict[str, Any]]:
    def normalize_item(item_type: str, value: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized: dict[str, Any] = {"type": item_type}
        if extra:
            normalized.update(extra)
        if item_type == "text":
            normalized["value"] = str(value)
            return normalized
        if item_type not in {"image", "video"}:
            raise ValueError(f"Unsupported VLM message item type: {item_type}")
        resolved = _resolve_media_value(value, dataset_name)
        if not _looks_like_url(resolved):
            resolved_path = Path(resolved).expanduser()
            if not resolved_path.exists():
                raise AssertionError(
                    f"Invalid {item_type} value for dataset `{dataset_name}`: {value!r}"
                )
        normalized["value"] = resolved
        return normalized

    if isinstance(message, str):
        return [normalize_item("text", message)]
    if isinstance(message, dict):
        if "type" not in message or "value" not in message:
            raise AssertionError(f"Invalid VLM message dict: {message!r}")
        extra = {key: value for key, value in message.items() if key not in {"type", "value"}}
        return [normalize_item(str(message["type"]), message["value"], extra)]
    if isinstance(message, list):
        normalized_items: list[dict[str, Any]] = []
        for item in message:
            if isinstance(item, str):
                media_kind = _guess_media_kind(item)
                item_type = media_kind if media_kind in {"image", "video"} else "text"
                normalized_items.append(normalize_item(item_type, item))
                continue
            if not isinstance(item, dict) or "type" not in item or "value" not in item:
                raise AssertionError(f"Invalid VLM message item: {item!r}")
            extra = {key: value for key, value in item.items() if key not in {"type", "value"}}
            normalized_items.append(normalize_item(str(item["type"]), item["value"], extra))
        return normalized_items
    raise AssertionError(f"Unsupported VLM message type: {type(message)!r}")


@contextlib.contextmanager
def _temporary_generation_cache(model):
    states = []
    visited = set()
    stack = [model]

    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        config = getattr(current, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            states.append((config, "use_cache", config.use_cache))
            config.use_cache = True

        generation_config = getattr(current, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            states.append((generation_config, "use_cache", generation_config.use_cache))
            generation_config.use_cache = True

        for attr_name in ("llm", "language_model", "model"):
            nested = getattr(current, attr_name, None)
            if nested is not None and nested is not current:
                stack.append(nested)

    try:
        yield
    finally:
        for target, attr_name, value in reversed(states):
            setattr(target, attr_name, value)


def _build_internvl_transform(input_size: int):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except Exception as err:
        raise RuntimeError(
            "InternVL2 VLMEval wrapper requires torchvision for image preprocessing."
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
    image = _open_image(image_path)
    transform = transform or _build_internvl_transform(input_size)
    processed_images = _dynamic_preprocess_internvl_image(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    pixel_values = torch.stack([transform(item) for item in processed_images])
    return pixel_values, pixel_values.size(0)


def _internvl_prompt_from_message(message, dataset: str | None) -> tuple[str, int]:
    image_num = len([item for item in message if item["type"] == "image"])
    if image_num == 1:
        prompt = "<image>\n" + "\n".join(
            item["value"] for item in message if item["type"] == "text"
        )
    else:
        prompt = ""
        image_idx = 1
        for item in message:
            if item["type"] == "text":
                prompt += item["value"]
            elif item["type"] == "image":
                prompt += f"<image-{image_idx}>"
                image_idx += 1
            elif item["type"] == "video":
                raise NotImplementedError("MindPipe VLM evaluation does not support video datasets yet.")
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
    if dataset == "OCRBench":
        prompt = (
            prompt.rstrip()
            + "\nAnswer with the exact text only. Reply with a short phrase only. Do not explain."
        )
    return prompt.strip(), image_num


def _internvl_default_max_num(dataset: str | None) -> int:
    if dataset in {"ChartQA_TEST", "MMMU_DEV_VAL"}:
        return 12
    if dataset in {"DocVQA_VAL", "DocVQA_TEST"}:
        return 18
    if dataset in {"InfoVQA_VAL", "InfoVQA_TEST", "OCRBench", "HRBench4K", "HRBench8K"}:
        return 24
    return 6


def _attach_generation_cleanup(wrapper, device, *, enabled: bool = True):
    """给任意兼容 VLMEvalKit 的 wrapper 统一挂上生成期 cache 与样本级清理逻辑。"""
    if not enabled:
        return wrapper
    original_generate_inner = wrapper.generate_inner

    def wrapped_generate_inner(message, dataset=None):
        try:
            with _temporary_generation_cache(getattr(wrapper, "model", None)):
                return original_generate_inner(message, dataset)
        finally:
            gc.collect()
            empty_cache(device)

    wrapper.generate_inner = wrapped_generate_inner
    return wrapper


def _build_qwen2_messages(message, dataset: str | None):
    conversation = []
    for item in message:
        if item["type"] == "text":
            conversation.append({"type": "text", "text": item["value"]})
            continue
        if item["type"] == "image":
            image_path = Path(item["value"]).expanduser().resolve()
            content_item = {"type": "image", "image": image_path.as_uri()}
            if dataset == "OCRBench":
                content_item["min_pixels"] = 10 * 10 * 28 * 28
            for key in ("min_pixels", "max_pixels", "total_pixels", "resized_height", "resized_width"):
                if key in item and item[key] is not None:
                    content_item[key] = item[key]
            conversation.append(content_item)
            continue
        if item["type"] == "video":
            raise NotImplementedError("MindPipe VLM evaluation does not support video datasets yet.")
        raise ValueError(f"Unsupported message item: {item}")
    if dataset == "OCRBench":
        # OCRBench uses exact matching; verbose chain-of-thought style answers hurt both
        # accuracy and throughput for Qwen3-VL style instruction-tuned checkpoints.
        conversation.append(
            {
                "type": "text",
                "text": "Answer with the exact text only. Reply with a short phrase only. Do not explain.",
            }
        )
    return [{"role": "user", "content": conversation}]


def _build_qwen2_wrapper(
    model,
    tokenizer_bundle,
    common_args: dict[str, Any],
    base_model_cls,
    dataset_type_resolver,
):
    source_model = getattr(model, "_source_model", model)
    processor = tokenizer_bundle.processor
    tokenizer = tokenizer_bundle.tokenizer
    if processor is None:
        raise ValueError("Qwen2/Qwen2.5-VL evaluation requires TokenizerBundle.processor.")
    target_device = resolve_device(common_args.get("device", "auto"))
    use_cache = bool(common_args.get("vlm_use_cache", False))
    max_new_tokens_override = common_args.get("vlm_max_new_tokens")
    max_new_tokens_override = (
        int(max_new_tokens_override) if max_new_tokens_override is not None else None
    )

    class MindPipeQwen2VLWrapper(base_model_cls):
        INTERLEAVE = True

        def __init__(self):
            super().__init__()
            self.model = source_model
            self.processor = processor
            self.tokenizer = tokenizer
            self.target_device = target_device
            self._model_prepared = False

        def use_custom_prompt(self, dataset):
            return False

        def build_prompt(self, line, dataset):
            raise NotImplementedError("MindPipe Qwen2/Qwen2.5-VL wrapper relies on dataset prompts.")

        def generate(self, message, dataset=None):
            normalized = _normalize_vlm_message(message, dataset)
            return self.generate_inner(normalized, dataset)

        def _ensure_model_ready(self):
            if self._model_prepared:
                return
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.target_device)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = use_cache
            if hasattr(self.model, "llm"):
                if hasattr(self.model.llm, "config"):
                    self.model.llm.config.use_cache = use_cache
                if getattr(self.model.llm, "generation_config", None) is not None:
                    self.model.llm.generation_config.use_cache = use_cache
            self.model.eval()
            self._model_prepared = True

        def generate_inner(self, message, dataset=None):
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
                raise err
            self._ensure_model_ready()
            messages = _build_qwen2_messages(message, dataset)
            prompt = self.processor.apply_chat_template(
                [messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = process_vision_info([messages])
            inputs = self.processor(
                text=prompt,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
            inputs = _maybe_to_device(inputs, _model_input_device(self.model, self.target_device))
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            max_new_tokens = (
                max_new_tokens_override
                if max_new_tokens_override is not None
                else _default_max_new_tokens(dataset, dataset_type_resolver(dataset))
            )
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )
            trimmed_ids = [
                output_ids[len(input_row):]
                for input_row, output_ids in zip(input_ids, generated_ids)
            ]
            responses = self.tokenizer.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return responses[0].strip()

    return MindPipeQwen2VLWrapper()


def _build_qwen3_wrapper(
    model,
    tokenizer_bundle,
    common_args: dict[str, Any],
    base_model_cls,
    dataset_type_resolver,
):
    source_model = getattr(model, "_source_model", model)
    processor = tokenizer_bundle.processor
    tokenizer = tokenizer_bundle.tokenizer
    if processor is None:
        raise ValueError("Qwen3/Qwen3.5-VL evaluation requires TokenizerBundle.processor.")
    target_device = resolve_device(common_args.get("device", "auto"))
    model_type = getattr(getattr(source_model, "config", None), "model_type", "")
    disable_thinking = _qwen3_disable_thinking(model_type)
    use_cache = bool(common_args.get("vlm_use_cache", False))
    max_new_tokens_override = common_args.get("vlm_max_new_tokens")
    max_new_tokens_override = (
        int(max_new_tokens_override) if max_new_tokens_override is not None else None
    )

    class MindPipeQwen3VLWrapper(base_model_cls):
        INTERLEAVE = True

        def __init__(self):
            super().__init__()
            self.model = source_model
            self.processor = processor
            self.tokenizer = tokenizer
            self.target_device = target_device
            self._model_prepared = False

        def use_custom_prompt(self, dataset):
            return False

        def build_prompt(self, line, dataset):
            raise NotImplementedError("MindPipe Qwen3/Qwen3.5-VL wrapper relies on dataset prompts.")

        def generate(self, message, dataset=None):
            normalized = _normalize_vlm_message(message, dataset)
            return self.generate_inner(normalized, dataset)

        def _ensure_model_ready(self):
            if self._model_prepared:
                return
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.target_device)
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = use_cache
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.use_cache = use_cache
            if hasattr(self.model, "llm"):
                if hasattr(self.model.llm, "config") and hasattr(self.model.llm.config, "use_cache"):
                    self.model.llm.config.use_cache = use_cache
                if getattr(self.model.llm, "generation_config", None) is not None:
                    self.model.llm.generation_config.use_cache = use_cache
            self.model.eval()
            self._model_prepared = True

        def generate_inner(self, message, dataset=None):
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
                raise err
            self._ensure_model_ready()
            messages = _build_qwen2_messages(message, dataset)
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            prompt = self.processor.apply_chat_template(
                messages,
                **chat_template_kwargs,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=prompt,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                padding=True,
                **(video_kwargs or {}),
            )
            model_device = _model_input_device(self.model, self.target_device)
            inputs = _maybe_to_device(inputs, model_device)
            if hasattr(inputs, "to") and hasattr(self.model, "dtype"):
                inputs = inputs.to(dtype=self.model.dtype)
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            max_new_tokens = (
                max_new_tokens_override
                if max_new_tokens_override is not None
                else _default_max_new_tokens(dataset, dataset_type_resolver(dataset))
            )
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )
            trimmed_ids = [
                output_ids[len(input_row):]
                for input_row, output_ids in zip(input_ids, generated_ids)
            ]
            responses = self.tokenizer.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return responses[0].strip()

    return MindPipeQwen3VLWrapper()


def _build_minicpm_wrapper(
    model,
    tokenizer_bundle,
    common_args: dict[str, Any],
    base_model_cls,
    dataset_type_resolver,
):
    source_model = getattr(model, "_source_model", model)
    tokenizer = tokenizer_bundle.tokenizer
    if not isinstance(tokenizer, MiniCPMTokenizerAdapter):
        tokenizer = MiniCPMTokenizerAdapter(tokenizer)
    target_device = resolve_device(common_args.get("device", "auto"))
    use_cache = bool(common_args.get("vlm_use_cache", False))
    max_new_tokens_override = common_args.get("vlm_max_new_tokens")
    max_new_tokens_override = (
        int(max_new_tokens_override) if max_new_tokens_override is not None else None
    )
    try:
        chat_signature = inspect.signature(source_model.chat)
    except (TypeError, ValueError):
        chat_signature = None

    class MindPipeMiniCPMVWrapper(base_model_cls):
        INTERLEAVE = False

        def __init__(self):
            super().__init__()
            self.model = source_model
            self.tokenizer = tokenizer
            self.target_device = target_device
            self._model_prepared = False

        def use_custom_prompt(self, dataset):
            return False

        def build_prompt(self, line, dataset):
            raise NotImplementedError("MindPipe MiniCPM-V wrapper relies on dataset prompts.")

        def generate(self, message, dataset=None):
            normalized = _normalize_vlm_message(message, dataset)
            return self.generate_inner(normalized, dataset)

        def _ensure_model_ready(self):
            if self._model_prepared:
                return
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.target_device)
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = use_cache
            if hasattr(self.model, "llm"):
                if hasattr(self.model.llm, "config") and hasattr(self.model.llm.config, "use_cache"):
                    self.model.llm.config.use_cache = use_cache
                if getattr(self.model.llm, "generation_config", None) is not None:
                    self.model.llm.generation_config.use_cache = use_cache
            self.model.eval()
            self._model_prepared = True

        def _unwrap_response(self, response):
            if isinstance(response, tuple) and len(response) > 0:
                return response[0]
            return response

        def _generation_context(self):
            model_device = resolve_device(self.target_device)
            try:
                model_dtype = next(self.model.parameters()).dtype
            except Exception:
                model_dtype = torch.bfloat16
            if model_device.type in ("cuda", "npu"):
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
                    use_cache=use_cache,
                    )
                )

        def _generate_interleaved(self, message, max_new_tokens: int):
            content = []
            for item in message:
                if item["type"] == "text":
                    content.append(item["value"])
                elif item["type"] == "image":
                    content.append(_open_image(item["value"]))
                elif item["type"] == "video":
                    raise NotImplementedError("MindPipe VLM evaluation does not support video datasets yet.")
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
                        use_cache=use_cache,
                    )
                )

        def generate_inner(self, message, dataset=None):
            self._ensure_model_ready()
            dataset_type = dataset_type_resolver(dataset)
            max_new_tokens = (
                max_new_tokens_override
                if max_new_tokens_override is not None
                else _default_max_new_tokens(dataset, dataset_type)
            )
            prompt = "\n".join(item["value"] for item in message if item["type"] == "text")
            images = [_open_image(item["value"]) for item in message if item["type"] == "image"]
            if chat_signature is not None and "image" in chat_signature.parameters:
                return self._generate_legacy(prompt=prompt, images=images, max_new_tokens=max_new_tokens)
            try:
                return self._generate_legacy(prompt=prompt, images=images, max_new_tokens=max_new_tokens)
            except TypeError:
                return self._generate_interleaved(message=message, max_new_tokens=max_new_tokens)

    return MindPipeMiniCPMVWrapper()

def _build_internvl_wrapper(
    model,
    tokenizer_bundle,
    common_args: dict[str, Any],
    base_model_cls,
    dataset_type_resolver,
):
    source_model = getattr(model, "_source_model", model)
    tokenizer = tokenizer_bundle.tokenizer
    target_device = resolve_device(common_args.get("device", "auto"))
    use_cache = bool(common_args.get("vlm_use_cache", False))
    max_new_tokens_override = common_args.get("vlm_max_new_tokens")
    max_new_tokens_override = (
        int(max_new_tokens_override) if max_new_tokens_override is not None else None
    )
    default_input_size = getattr(
        getattr(getattr(source_model, "config", None), "vision_config", None),
        "image_size",
        448,
    )
    input_size = int(common_args.get("internvl_image_size", default_input_size))
    use_thumbnail = bool(common_args.get("internvl_use_thumbnail", True))
    explicit_max_num = common_args.get("internvl_max_num")
    chat_signature = None
    try:
        chat_signature = inspect.signature(source_model.chat)
    except (TypeError, ValueError):
        pass

    class MindPipeInternVLWrapper(base_model_cls):
        INTERLEAVE = True

        def __init__(self):
            super().__init__()
            self.model = source_model
            self.tokenizer = tokenizer
            self.target_device = target_device
            self.input_size = input_size
            self.use_thumbnail = use_thumbnail
            self._transform = _build_internvl_transform(self.input_size)
            self._model_prepared = False

        def use_custom_prompt(self, dataset):
            return False

        def build_prompt(self, line, dataset):
            raise NotImplementedError("MindPipe InternVL2 wrapper relies on dataset prompts.")

        def generate(self, message, dataset=None):
            normalized = _normalize_vlm_message(message, dataset)
            return self.generate_inner(normalized, dataset)

        def _ensure_model_ready(self):
            if self._model_prepared:
                return
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.target_device)
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = use_cache
            language_model = getattr(self.model, "language_model", None)
            if language_model is not None:
                if hasattr(language_model, "config") and hasattr(language_model.config, "use_cache"):
                    language_model.config.use_cache = use_cache
                if getattr(language_model, "generation_config", None) is not None:
                    language_model.generation_config.use_cache = use_cache
            self.model.eval()
            self._model_prepared = True

        def _model_dtype(self):
            model_dtype = getattr(self.model, "dtype", None)
            if isinstance(model_dtype, torch.dtype) and model_dtype.is_floating_point:
                return model_dtype
            try:
                parameter_dtype = next(self.model.parameters()).dtype
                if parameter_dtype.is_floating_point:
                    return parameter_dtype
            except Exception:
                pass
            return torch.bfloat16

        def generate_inner(self, message, dataset=None):
            self._ensure_model_ready()
            prompt, image_num = _internvl_prompt_from_message(message, dataset)
            image_paths = [item["value"] for item in message if item["type"] == "image"]
            max_num = (
                int(explicit_max_num)
                if explicit_max_num is not None
                else _internvl_default_max_num(dataset)
            )
            model_device = _model_input_device(self.model, self.target_device)
            model_dtype = self._model_dtype()

            pixel_values = None
            num_patches_list: list[int] = []
            if image_paths:
                pixel_values_list = []
                for image_path in image_paths:
                    current_pixels, num_patches = _load_internvl_image(
                        image_path,
                        input_size=self.input_size,
                        max_num=max_num,
                        use_thumbnail=self.use_thumbnail,
                        transform=self._transform,
                    )
                    pixel_values_list.append(current_pixels)
                    num_patches_list.append(num_patches)
                pixel_values = torch.cat(pixel_values_list, dim=0).to(
                    device=model_device,
                    dtype=model_dtype,
                )

            generation_config = {
                "max_new_tokens": (
                    max_new_tokens_override
                    if max_new_tokens_override is not None
                    else _default_max_new_tokens(dataset, dataset_type_resolver(dataset))
                ),
                "do_sample": False,
                "num_beams": 1,
            }
            chat_kwargs: dict[str, Any] = {
                "tokenizer": self.tokenizer,
                "pixel_values": pixel_values,
                "question": prompt,
                "generation_config": generation_config,
            }
            if image_num > 0 and chat_signature is not None and "num_patches_list" in chat_signature.parameters:
                chat_kwargs["num_patches_list"] = num_patches_list
            if chat_signature is not None and "verbose" in chat_signature.parameters:
                chat_kwargs["verbose"] = False

            with torch.no_grad():
                response = self.model.chat(**chat_kwargs)
            if isinstance(response, tuple) and response:
                response = response[0]
            return str(response).strip()

    return MindPipeInternVLWrapper()


def _build_wrapper(model, tokenizer_bundle, common_args: dict[str, Any], modules: dict[str, Any]):
    config = getattr(getattr(model, "_source_model", model), "config", getattr(model, "config", None))
    model_type = getattr(config, "model_type", "") if config is not None else ""
    base_model_cls = modules["BaseModel"]
    dataset_type_resolver = modules["DATASET_TYPE"]

    if model_type in {"qwen2_vl", "qwen2_5_vl"}:
        return _build_qwen2_wrapper(
            model,
            tokenizer_bundle,
            common_args,
            base_model_cls,
            dataset_type_resolver,
        )
    if model_type in QWEN3_VL_MODEL_TYPES:
        return _build_qwen3_wrapper(
            model,
            tokenizer_bundle,
            common_args,
            base_model_cls,
            dataset_type_resolver,
        )
    if model_type == "internvl_chat":
        return _build_internvl_wrapper(
            model,
            tokenizer_bundle,
            common_args,
            base_model_cls,
            dataset_type_resolver,
        )
    if hasattr(getattr(model, "_source_model", model), "chat") or "minicpm" in str(model_type):
        return _build_minicpm_wrapper(
            model,
            tokenizer_bundle,
            common_args,
            base_model_cls,
            dataset_type_resolver,
        )
    raise NotImplementedError(
        "VLM benchmark evaluation currently supports Qwen2/Qwen2.5/Qwen3-VL/Qwen3.5, InternVL2, and MiniCPM-V style models only."
    )


def _build_judge_kwargs(dataset, dataset_name: str, common_args: dict[str, Any], listinstr) -> dict[str, Any]:
    judge_kwargs = {
        "nproc": int(common_args.get("vlm_api_nproc", 4)),
        "verbose": bool(common_args.get("vlm_verbose", False)),
        "retry": 3,
    }
    explicit_judge = common_args.get("vlm_judge")
    if explicit_judge:
        judge_kwargs["model"] = explicit_judge
        return judge_kwargs

    dataset_type = getattr(dataset, "TYPE", None)
    if dataset_type in {"MCQ", "Y/N", "MCQ_MMMU_Pro"} or listinstr(
        ["moviechat1k", "mme-reasoning"], dataset_name.lower()
    ):
        if listinstr(["WeMath", "MME-Reasoning", "VisualPuzzles", "PuzzleVQA", "VisuLogic"], dataset_name):
            judge_kwargs["model"] = "exact_matching"
        else:
            judge_kwargs["model"] = "chatgpt-0125"
    elif listinstr(["MMVet", "LLaVABench", "MMBench_Video"], dataset_name):
        judge_kwargs["model"] = "gpt-4-turbo"
    elif listinstr(["VGRPBench"], dataset_name):
        judge_kwargs["model"] = "gpt-4o"
    elif listinstr(
        [
            "MathVista",
            "MathVerse",
            "MathVision",
            "LENS",
            "DynaMath",
            "VL-RewardBench",
            "LogicVista",
            "MOAT",
            "OCR_Reasoning",
            "VTCBench",
            "Asclepius",
            "MMSafetyBench",
            "MSSBench",
            "SIUO",
            "SIUO_GEN",
            "XSTest",
            "Flames",
        ],
        dataset_name,
    ):
        judge_kwargs["model"] = "gpt-4o-mini"
    elif listinstr(["MMLongBench", "MMDU", "DUDE", "SLIDEVQA", "MIA-Bench", "WildVision", "MMAlignBench"], dataset_name):
        judge_kwargs["model"] = "gpt-4o"
    elif listinstr(["ChartMimic", "MMVMBench", "M4Bench"], dataset_name):
        judge_kwargs["model"] = "gpt-4o"
    elif listinstr(["CVQA_EN", "CVQA_LOC", "AyaVisionBench", "CoreCognition"], dataset_name):
        judge_kwargs["model"] = "gpt-4.1"
    elif listinstr(["MathCanvas"], dataset_name):
        judge_kwargs["model"] = "gpt-4.1-2025-04-14"
    elif listinstr(["WorldVQA"], dataset_name):
        judge_kwargs["model"] = "gpt-4o-1120"
    else:
        judge_kwargs["model"] = "exact_matching"
    return judge_kwargs


def _eval_skip_reason(dataset_name: str, mmbench_official_server) -> str | None:
    if "MLLMGuard_DS" in dataset_name:
        return "dataset evaluation is not supported"
    if dataset_name == "AesBench_TEST":
        return "test-only split requires external submission"
    if dataset_name in {"DocVQA_TEST", "InfoVQA_TEST", "Q-Bench1_TEST", "A-Bench_TEST"}:
        return "test-only split has no public ground truth"
    if dataset_name in {
        "MMBench_TEST_CN",
        "MMBench_TEST_EN",
        "MMBench",
        "MMBench_CN",
        "MMBench_TEST_CN_V11",
        "MMBench_TEST_EN_V11",
        "MMBench_V11",
        "MMBench_CN_V11",
    } and not mmbench_official_server(dataset_name):
        return "MMBench evaluation is only available on the official server"
    return None


def _resolve_result_file_path(
    modules: dict[str, Any],
    work_dir: Path,
    model_name: str,
    dataset_name: str,
    pred_format: str,
) -> str:
    getter = modules.get("get_pred_file_path")
    if callable(getter):
        try:
            return getter(str(work_dir), model_name, dataset_name, use_env_format=True)
        except TypeError:
            try:
                return getter(str(work_dir), model_name, dataset_name)
            except TypeError:
                pass
    # Backward-compat for older VLMEvalKit revisions.
    return str(Path(work_dir) / f"{model_name}_{dataset_name}.{pred_format}")


def _resolve_existing_result_file(result_file: str) -> str:
    candidate = Path(result_file)
    if candidate.exists():
        return str(candidate)
    for suffix in ("xlsx", "json", "jsonl", "tsv", "csv"):
        alternate = candidate.with_suffix(f".{suffix}")
        if alternate.exists():
            return str(alternate)
    return str(candidate)


def evaluate_vlm(model, tokenizer_bundle, common_args: dict[str, Any]) -> dict[str, Any]:
    dataset_names = list(common_args.get("vlm_datasets") or [])
    if not dataset_names:
        raise ValueError("`--eval_vlm` requires at least one dataset in `--vlm_datasets`.")
    if tokenizer_bundle is None or not hasattr(tokenizer_bundle, "tokenizer"):
        raise ValueError("VLM evaluation requires the full TokenizerBundle, not just a tokenizer.")

    modules = _load_vlmeval_modules(common_args.get("vlm_eval_kit_root", DEFAULT_VLMEVALKIT_ROOT))
    work_dir = _resolve_work_dir(common_args)
    model_name = _resolve_model_name(common_args)
    mode = str(common_args.get("vlm_mode", "all"))
    resume_enabled = bool(common_args.get("vlm_resume", False))
    progress_path = _resolve_vlm_progress_path(common_args)
    wrapper = _build_wrapper(model, tokenizer_bundle, common_args, modules)
    # 把清理逻辑放在模型 wrapper 外层，后续新接入的 VLM 模型也能直接复用，
    # 不需要在每个模型分支里重复写一套显存回收代码。
    wrapper = _attach_generation_cleanup(
        wrapper,
        common_args.get("device", "auto"),
        enabled=bool(common_args.get("vlm_sample_cleanup", True)),
    )

    results: dict[str, Any] = {}
    if resume_enabled:
        metrics_root = common_args.get("evaluation_output_dir")
        metrics_path = (
            ensure_dir(Path(metrics_root)) / "metrics.json"
            if metrics_root
            else work_dir.parent / "metrics.json"
        )
        for payload in (_load_json_payload(metrics_path), _load_json_payload(progress_path)):
            for dataset_name, record in _extract_existing_vlm_records(payload).items():
                if isinstance(record, dict):
                    results[dataset_name] = record

    with _temporary_env(PRED_FORMAT=str(common_args.get("vlm_pred_format", "xlsx"))):
        for dataset_name in dataset_names:
            existing_record = results.get(dataset_name)
            if resume_enabled and _vlm_record_is_complete(existing_record, mode):
                print(f"[vlm_eval] Skip completed dataset: {dataset_name}")
                continue

            dataset = modules["build_dataset"](dataset_name)
            if dataset is None:
                raise ValueError(f"Failed to build VLMEvalKit dataset: {dataset_name}")

            dataset_modality = getattr(dataset, "MODALITY", modules["DATASET_MODALITY"](dataset_name))
            dataset_type = getattr(dataset, "TYPE", modules["DATASET_TYPE"](dataset_name))
            if dataset_modality != "IMAGE":
                raise NotImplementedError(
                    f"Dataset `{dataset_name}` uses modality `{dataset_modality}`; only IMAGE datasets are supported."
                )
            if dataset_type == "MT":
                raise NotImplementedError(f"Dataset `{dataset_name}` is multi-turn and is not supported yet.")

            num_samples = common_args.get("num_samples")
            if num_samples is not None:
                original_len = len(dataset)
                if hasattr(dataset, "data"):
                    dataset.data = dataset.data.head(int(num_samples))
                elif hasattr(dataset, "dataset_map"):
                    for dname in dataset.dataset_map:
                        dataset.dataset_map[dname].data = dataset.dataset_map[dname].data.head(int(num_samples))
                    dataset.data = dataset.data.head(int(num_samples))
                print(f"[vlm_eval] Dataset {dataset_name}: {original_len} -> {len(dataset)} samples (num_samples={num_samples})")

            pred_format = str(common_args.get("vlm_pred_format", "xlsx"))
            result_file = _resolve_result_file_path(
                modules=modules,
                work_dir=work_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                pred_format=pred_format,
            )
            record: dict[str, Any] = {
                "dataset_type": dataset_type,
                "dataset_modality": dataset_modality,
                "result_file": result_file,
            }
            if isinstance(existing_record, dict):
                record.update(existing_record)
                record["dataset_type"] = dataset_type
                record["dataset_modality"] = dataset_modality
                record["result_file"] = result_file

            if mode != "eval":
                resolved_existing_result = _resolve_existing_result_file(result_file)
                if resume_enabled and Path(resolved_existing_result).exists():
                    result_file = resolved_existing_result
                    record["result_file"] = resolved_existing_result
                    record["inference_completed"] = True
                    print(f"[vlm_eval] Reuse existing predictions for dataset: {dataset_name}")
                else:
                    infer_fn = modules["infer_data_job"]
                    infer_kwargs = {
                        "work_dir": str(work_dir),
                        "model_name": model_name,
                        "dataset": dataset,
                        "verbose": bool(common_args.get("vlm_verbose", False)),
                        "api_nproc": int(common_args.get("vlm_api_nproc", 4)),
                        "ignore_failed": bool(common_args.get("vlm_ignore_failed", False)),
                    }
                    if "use_vllm" in inspect.signature(infer_fn).parameters:
                        infer_kwargs["use_vllm"] = False
                    wrapper = infer_fn(wrapper, **infer_kwargs)
                    result_file = _resolve_existing_result_file(result_file)
                    record["result_file"] = result_file
                    record["inference_completed"] = True

            if mode != "infer":
                skip_reason = _eval_skip_reason(dataset_name, modules["MMBenchOfficialServer"])
                if skip_reason is not None:
                    record["evaluation_skipped"] = skip_reason
                else:
                    judge_kwargs = _build_judge_kwargs(
                        dataset=dataset,
                        dataset_name=dataset_name,
                        common_args=common_args,
                        listinstr=modules["listinstr"],
                    )
                    record["judge"] = judge_kwargs.get("model")
                    record["evaluation"] = _json_safe(dataset.evaluate(result_file, **judge_kwargs))

            results[dataset_name] = record
            if resume_enabled:
                _persist_vlm_progress(
                    progress_path,
                    mode=mode,
                    model_name=model_name,
                    work_dir=work_dir,
                    results=results,
                )
            # 在数据集切换处再清一次，避免前一个 benchmark split 的高水位缓存
            # 继续带到下一个 split。
            gc.collect()
            empty_cache(common_args.get("device", "auto"))

    if resume_enabled:
        _persist_vlm_progress(
            progress_path,
            mode=mode,
            model_name=model_name,
            work_dir=work_dir,
            results=results,
        )

    return {
        "mode": mode,
        "model_name": model_name,
        "work_dir": str(work_dir),
        "datasets": results,
    }
