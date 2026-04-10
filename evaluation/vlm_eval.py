"""VLMEvalKit integration for in-memory multimodal model evaluation."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from algorithm.common.device import resolve_device
from algorithm.common.io import ensure_dir
from algorithm.common.io import model_slug


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLMEVALKIT_ROOT = os.environ.get(
    "VLMEVALKIT_ROOT",
    str(REPO_ROOT / "third_party" / "VLMEvalKit"),
)


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

    return {
        "BaseModel": base_module.BaseModel,
        "build_dataset": dataset_module.build_dataset,
        "DATASET_TYPE": dataset_module.DATASET_TYPE,
        "DATASET_MODALITY": dataset_module.DATASET_MODALITY,
        "infer_data_job": inference_module.infer_data_job,
        "get_pred_file_path": smp_module.get_pred_file_path,
        "listinstr": smp_module.listinstr,
        "MMBenchOfficialServer": smp_module.MMBenchOfficialServer,
    }


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


def _default_max_new_tokens(dataset_type: str) -> int:
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

        def _ensure_model_ready(self):
            if self._model_prepared:
                return
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.target_device)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if hasattr(self.model, "llm"):
                if hasattr(self.model.llm, "config"):
                    self.model.llm.config.use_cache = False
                if getattr(self.model.llm, "generation_config", None) is not None:
                    self.model.llm.generation_config.use_cache = False
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
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=_default_max_new_tokens(dataset_type_resolver(dataset)),
                do_sample=False,
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


def _build_minicpm_wrapper(
    model,
    tokenizer_bundle,
    common_args: dict[str, Any],
    base_model_cls,
    dataset_type_resolver,
):
    source_model = getattr(model, "_source_model", model)
    tokenizer = tokenizer_bundle.tokenizer
    target_device = resolve_device(common_args.get("device", "auto"))
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

        def _ensure_model_ready(self):
            if self._model_prepared:
                return
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.target_device)
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
                    use_cache=False,
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
                        use_cache=False,
                    )
                )

        def generate_inner(self, message, dataset=None):
            self._ensure_model_ready()
            dataset_type = dataset_type_resolver(dataset)
            max_new_tokens = _default_max_new_tokens(dataset_type)
            prompt = "\n".join(item["value"] for item in message if item["type"] == "text")
            images = [_open_image(item["value"]) for item in message if item["type"] == "image"]
            if chat_signature is not None and "image" in chat_signature.parameters:
                return self._generate_legacy(prompt=prompt, images=images, max_new_tokens=max_new_tokens)
            try:
                return self._generate_legacy(prompt=prompt, images=images, max_new_tokens=max_new_tokens)
            except TypeError:
                return self._generate_interleaved(message=message, max_new_tokens=max_new_tokens)

    return MindPipeMiniCPMVWrapper()

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
    if hasattr(getattr(model, "_source_model", model), "chat") or "minicpm" in str(model_type):
        return _build_minicpm_wrapper(
            model,
            tokenizer_bundle,
            common_args,
            base_model_cls,
            dataset_type_resolver,
        )
    raise NotImplementedError(
        "VLM benchmark evaluation currently supports Qwen2/Qwen2.5-VL and MiniCPM-V style models only."
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
    wrapper = _build_wrapper(model, tokenizer_bundle, common_args, modules)

    results: dict[str, Any] = {}
    with _temporary_env(PRED_FORMAT=str(common_args.get("vlm_pred_format", "xlsx"))):
        for dataset_name in dataset_names:
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

            result_file = modules["get_pred_file_path"](
                str(work_dir),
                model_name,
                dataset_name,
                use_env_format=True,
            )
            record: dict[str, Any] = {
                "dataset_type": dataset_type,
                "dataset_modality": dataset_modality,
                "result_file": result_file,
            }

            if mode != "eval":
                wrapper = modules["infer_data_job"](
                    wrapper,
                    work_dir=str(work_dir),
                    model_name=model_name,
                    dataset=dataset,
                    verbose=bool(common_args.get("vlm_verbose", False)),
                    api_nproc=int(common_args.get("vlm_api_nproc", 4)),
                    ignore_failed=bool(common_args.get("vlm_ignore_failed", False)),
                    use_vllm=False,
                )
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

    return {
        "mode": mode,
        "model_name": model_name,
        "work_dir": str(work_dir),
        "datasets": results,
    }
