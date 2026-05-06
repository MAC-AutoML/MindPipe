"""Unified GPTQ runner."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import build_decoder_layer_groups
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import find_linear_layers
from ....common.modeling import get_layer_device
from ....common.modeling import get_text_backbone
from ....common.modeling import move_tensors_to_device
from ....common.modeling import unwrap_layer_output
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


logger = logging.getLogger(__name__)


class GPTQMethod(BaseQuantizationMethod):
    name = "gptq"
    default_calibration_dataset = "pileval"
    quantization_block_size = 32
    _SUPPORTED_VLM_MODEL_TYPES = {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "internvl_chat",
        "minicpmv",
    }

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        resolved = resolve_device(args.device)
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        source_root = Path(__file__).resolve().parent / "source"
        with prepend_python_path(source_root):
            from gptq import GPTQ
            from quant import Quantizer
            source_model = self._resolve_source_model(model)
            model_type = self._resolve_model_type(source_model)
            vlm_dataset_name = self._resolve_vlm_dataset_name(args)
            if vlm_dataset_name:
                if model_type not in self._SUPPORTED_VLM_MODEL_TYPES:
                    raise NotImplementedError(
                        "Multimodal GPTQ currently supports "
                        f"{sorted(self._SUPPORTED_VLM_MODEL_TYPES)}, got {model_type!r}."
                    )
                return self._apply_vlm_gptq(
                    model=model,
                    tokenizer_bundle=tokenizer_bundle,
                    args=args,
                    source_model=source_model,
                    model_type=model_type,
                    source_root=source_root,
                    GPTQ=GPTQ,
                    Quantizer=Quantizer,
                    dataset_name=vlm_dataset_name,
                )
            return self._apply_text_backbone_gptq(
                model=model,
                tokenizer_bundle=tokenizer_bundle,
                args=args,
                source_root=source_root,
                GPTQ=GPTQ,
                Quantizer=Quantizer,
            )

    @staticmethod
    def _resolve_source_model(model):
        return getattr(model, "_source_model", model)

    @staticmethod
    def _resolve_model_type(model) -> str:
        return str(getattr(getattr(model, "config", None), "model_type", "") or "")

    @staticmethod
    def _resolve_vlm_dataset_name(args) -> str | None:
        explicit = getattr(args, "gptq_vlm_dataset_name", None)
        if explicit:
            return str(explicit)
        legacy = getattr(args, "mquant_dataset_name", None)
        if legacy:
            return str(legacy)
        return None

    @staticmethod
    def _resolve_vlm_calib_num(args) -> int:
        explicit = getattr(args, "gptq_vlm_calib_num", None)
        if explicit is not None:
            return int(explicit)
        legacy = getattr(args, "mquant_calib_num", None)
        if legacy is not None:
            return int(legacy)
        return int(args.calibration_samples)

    @staticmethod
    def _resolve_bool(value, *, default: bool) -> bool:
        if value is None:
            return bool(default)
        return bool(value)

    @staticmethod
    def _move_inputs_to_device(model_inputs, device):
        if isinstance(model_inputs, dict) or hasattr(model_inputs, "items"):
            return {
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in model_inputs.items()
            }
        if hasattr(model_inputs, "to"):
            return model_inputs.to(device)
        return model_inputs

    @staticmethod
    def _prepare_gptq_state(linear, args, GPTQ, Quantizer):
        gptq_state = GPTQ(linear)
        gptq_state.quantizer = Quantizer()
        gptq_state.quantizer.configure(
            args.weight_bits,
            perchannel=True,
            sym=args.weight_symmetric,
            mse=False,
        )
        return gptq_state

    def _finalize_group_quantization(
        self,
        *,
        group_prefix: str,
        gptq_states: dict[str, Any],
        quantizer_artifacts: dict[str, dict[str, object]],
        args,
    ) -> None:
        for name, gptq_state in gptq_states.items():
            # Stabilize Hessian before factorization. Multimodal branches are even more likely
            # to produce NaN/Inf or slightly asymmetric statistics.
            gptq_state.H = torch.nan_to_num(gptq_state.H, nan=0.0, posinf=0.0, neginf=0.0)
            gptq_state.H = 0.5 * (gptq_state.H + gptq_state.H.T)

            damp_schedule = []
            for damp in (args.damp_percent, 0.05, 0.1, 0.25, 0.5, 1.0):
                if damp not in damp_schedule:
                    damp_schedule.append(damp)
            actorder_schedule = [args.use_activation_order]
            if args.use_activation_order:
                actorder_schedule.append(False)

            last_error = None
            quantized = False
            for actorder in actorder_schedule:
                for damp in damp_schedule:
                    try:
                        gptq_state.fasterquant(
                            blocksize=self.quantization_block_size,
                            percdamp=damp,
                            groupsize=args.weight_group_size,
                            actorder=actorder,
                            static_groups=args.static_groups,
                        )
                        quantized = True
                        last_error = None
                        break
                    except RuntimeError as error:
                        last_error = error
                        if "not positive-definite" not in str(error):
                            raise
                if quantized:
                    break

            if not quantized and last_error is not None:
                raise last_error

            quantizer_artifacts[f"{group_prefix}.{name}"] = {
                "bits": args.weight_bits,
                "group_size": args.weight_group_size,
                "symmetric": args.weight_symmetric,
            }
            gptq_state.free()

    def _apply_text_backbone_gptq(
        self,
        *,
        model,
        tokenizer_bundle,
        args,
        source_root: Path,
        GPTQ,
        Quantizer,
    ) -> dict[str, object]:
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )
        backbone = get_text_backbone(model)
        input_states, layer_kwargs = capture_first_block_inputs(
            model=model,
            backbone=backbone,
            calibration_batches=calibration_batches,
            device=args.device,
        )
        output_states = torch.zeros_like(input_states)
        quantizer_artifacts = {}
        max_layers = getattr(args, "gptq_max_layers", None)
        layers = backbone.layers
        if max_layers is not None:
            layers = layers[: int(max_layers)]

        for layer_index, block in enumerate(layers):
            target_device = get_layer_device(backbone, layer_index)
            input_states = input_states.to(target_device)
            output_states = output_states.to(target_device)
            layer_kwargs = move_tensors_to_device(layer_kwargs, target_device)
            linear_layers = find_linear_layers(block)
            layer_groups = build_decoder_layer_groups(block, set(linear_layers))

            for group in layer_groups:
                subset = {name: linear_layers[name] for name in group}
                gptq_states = {
                    name: self._prepare_gptq_state(linear, args, GPTQ, Quantizer)
                    for name, linear in subset.items()
                }

                def add_batch(name: str):
                    def hook(_module, inputs, outputs):
                        gptq_states[name].add_batch(
                            torch.nan_to_num(inputs[0].data, nan=0.0, posinf=0.0, neginf=0.0),
                            torch.nan_to_num(outputs.data, nan=0.0, posinf=0.0, neginf=0.0),
                        )

                    return hook

                handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        )
                for handle in handles:
                    handle.remove()

                self._finalize_group_quantization(
                    group_prefix=f"{backbone.prefix}.layers.{layer_index}",
                    gptq_states=gptq_states,
                    quantizer_artifacts=quantizer_artifacts,
                    args=args,
                )
                del gptq_states
                empty_cache(args.device)

            for sample_index in range(args.calibration_samples):
                with torch.no_grad():
                    output_states[sample_index] = unwrap_layer_output(
                        block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                    )

            empty_cache(args.device)
            input_states, output_states = output_states, input_states

        return {
            "source_root": str(source_root),
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }

    @staticmethod
    def _model_accepts_kwarg(model, kwarg_name: str) -> bool:
        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            return False
        return kwarg_name in signature.parameters

    def _run_multimodal_forward(self, *, source_model, prepared_inputs) -> None:
        forward_kwargs = dict(prepared_inputs)
        forward_kwargs["use_cache"] = False
        if self._model_accepts_kwarg(source_model, "logits_to_keep"):
            forward_kwargs["logits_to_keep"] = 1
        model_dtype = getattr(source_model, "dtype", None)
        if not isinstance(model_dtype, torch.dtype):
            try:
                model_dtype = next(source_model.parameters()).dtype
            except Exception:
                model_dtype = None
        if isinstance(model_dtype, torch.dtype):
            for key, value in list(forward_kwargs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    forward_kwargs[key] = value.to(dtype=model_dtype)
        with torch.inference_mode():
            source_model(**forward_kwargs)

    def _load_vlm_dataset_modules(self, *, dataset_name: str, args) -> tuple[dict[str, Any], Any, int]:
        from evaluation.vlm_eval import DEFAULT_VLMEVALKIT_ROOT
        from evaluation.vlm_eval import _load_vlmeval_modules

        requested_root = getattr(args, "vlm_eval_kit_root", None) or DEFAULT_VLMEVALKIT_ROOT
        candidate_roots: list[str] = []
        for root in (
            requested_root,
            "/mnt/42_store/zy/HUAWEI/work1/MQuant/third/VLMEvalKit",
        ):
            if root and root not in candidate_roots:
                candidate_roots.append(str(root))

        modules = None
        last_error: Exception | None = None
        for root in candidate_roots:
            try:
                modules = _load_vlmeval_modules(root)
                if root != requested_root:
                    logger.warning(
                        "Requested VLMEvalKit root %s unavailable; falling back to %s for GPTQ multimodal calibration.",
                        requested_root,
                        root,
                    )
                break
            except (FileNotFoundError, ModuleNotFoundError) as error:
                last_error = error
        if modules is None:
            if last_error is not None:
                raise last_error
            raise FileNotFoundError("Failed to resolve a usable VLMEvalKit root for multimodal GPTQ calibration.")

        dataset = modules["build_dataset"](dataset_name)
        if dataset is None:
            raise ValueError(f"Failed to build VLM calibration dataset: {dataset_name}")

        sample_count = min(self._resolve_vlm_calib_num(args), len(dataset))
        return modules, dataset, sample_count

    def _build_vlm_calibration_messages(self, *, dataset_name: str, args) -> tuple[dict[str, Any], list[Any]]:
        modules, dataset, sample_count = self._load_vlm_dataset_modules(
            dataset_name=dataset_name,
            args=args,
        )
        messages = [dataset.build_prompt(dataset.data.iloc[index]) for index in range(sample_count)]
        logger.info(
            "Prepared %s multimodal GPTQ calibration prompts from %s.",
            len(messages),
            dataset_name,
        )
        return modules, messages

    def _apply_vlm_gptq(
        self,
        *,
        model,
        tokenizer_bundle,
        args,
        source_model,
        model_type: str,
        source_root: Path,
        GPTQ,
        Quantizer,
        dataset_name: str,
    ) -> dict[str, object]:
        processor = getattr(tokenizer_bundle, "processor", None)
        if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"} and processor is None:
            raise ValueError(
                f"Multimodal GPTQ for {model_type} requires TokenizerBundle.processor."
            )

        quant_visual = self._resolve_bool(
            getattr(args, "gptq_vlm_quant_visual", None),
            default=True,
        )
        quant_connector = self._resolve_bool(
            getattr(args, "gptq_vlm_quant_connector", None),
            default=True,
        )
        quant_llm = self._resolve_bool(
            getattr(args, "gptq_vlm_quant_llm", None),
            default=True,
        )
        if not any((quant_visual, quant_connector, quant_llm)):
            raise ValueError("At least one GPTQ multimodal branch must be enabled.")

        if getattr(source_model, "hf_device_map", None):
            raise NotImplementedError(
                "Multimodal GPTQ currently expects a single-device model (no hf_device_map sharding)."
            )

        target_device = resolve_device(args.device)
        source_model.to(target_device)
        source_model.eval()
        if hasattr(source_model, "config") and hasattr(source_model.config, "use_cache"):
            source_model.config.use_cache = False

        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            calibration_inputs = self._build_qwen2_vlm_calibration_inputs(
                processor=processor,
                dataset_name=dataset_name,
                args=args,
            )
            branch_groups = self._build_qwen2_vlm_groups(
                source_model=source_model,
                quant_visual=quant_visual,
                quant_connector=quant_connector,
                quant_llm=quant_llm,
            )
            run_calibration_sample = lambda model_input: self._run_multimodal_forward(
                source_model=source_model,
                prepared_inputs=self._move_inputs_to_device(model_input, target_device),
            )
            sample_count = len(calibration_inputs)
        elif model_type == "qwen3_vl":
            calibration_inputs = self._build_qwen3_vlm_calibration_inputs(
                processor=processor,
                source_model=source_model,
                dataset_name=dataset_name,
                args=args,
            )
            branch_groups = self._build_qwen3_vlm_groups(
                source_model=source_model,
                quant_visual=quant_visual,
                quant_connector=quant_connector,
                quant_llm=quant_llm,
            )
            run_calibration_sample = lambda model_input: self._run_multimodal_forward(
                source_model=source_model,
                prepared_inputs=self._move_inputs_to_device(model_input, target_device),
            )
            sample_count = len(calibration_inputs)
        elif model_type == "internvl_chat":
            calibration_inputs = self._build_internvl_vlm_calibration_inputs(
                tokenizer=tokenizer_bundle.tokenizer,
                source_model=source_model,
                dataset_name=dataset_name,
                args=args,
            )
            branch_groups = self._build_internvl_vlm_groups(
                source_model=source_model,
                quant_visual=quant_visual,
                quant_connector=quant_connector,
                quant_llm=quant_llm,
            )
            run_calibration_sample = lambda model_input: self._run_multimodal_forward(
                source_model=source_model,
                prepared_inputs=self._move_inputs_to_device(model_input, target_device),
            )
            sample_count = len(calibration_inputs)
        else:
            _, calibration_messages = self._build_vlm_calibration_messages(
                dataset_name=dataset_name,
                args=args,
            )
            calibration_inputs = calibration_messages
            branch_groups = self._build_generic_vlm_groups(
                source_model=source_model,
                model_type=model_type,
                quant_visual=quant_visual,
                quant_connector=quant_connector,
                quant_llm=quant_llm,
            )
            calibration_runner = self._build_wrapper_vlm_runner(
                model=model,
                tokenizer_bundle=tokenizer_bundle,
                args=args,
                model_type=model_type,
            )
            run_calibration_sample = lambda prompt_message: calibration_runner(
                prompt_message,
                dataset_name,
            )
            sample_count = len(calibration_messages)

        quantizer_artifacts: dict[str, dict[str, object]] = {}
        for group_prefix, subset in branch_groups:
            gptq_states = {
                name: self._prepare_gptq_state(linear, args, GPTQ, Quantizer)
                for name, linear in subset.items()
            }

            def add_batch(name: str):
                def hook(_module, inputs, outputs):
                    output_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    gptq_states[name].add_batch(
                        torch.nan_to_num(inputs[0].data, nan=0.0, posinf=0.0, neginf=0.0),
                        torch.nan_to_num(output_tensor.data, nan=0.0, posinf=0.0, neginf=0.0),
                    )

                return hook

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            try:
                for model_inputs in calibration_inputs:
                    run_calibration_sample(model_inputs)
            finally:
                for handle in handles:
                    handle.remove()

            self._finalize_group_quantization(
                group_prefix=group_prefix,
                gptq_states=gptq_states,
                quantizer_artifacts=quantizer_artifacts,
                args=args,
            )
            del gptq_states
            empty_cache(args.device)

        return {
            "source_root": str(source_root),
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
            "multimodal_calibration": {
                "model_type": model_type,
                "dataset_name": dataset_name,
                "sample_count": sample_count,
                "quant_visual": quant_visual,
                "quant_connector": quant_connector,
                "quant_llm": quant_llm,
            },
        }

    def _build_qwen2_vlm_calibration_inputs(self, *, processor, dataset_name: str, args) -> list[Any]:
        from evaluation.vlm_eval import _build_qwen2_messages

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError(
                "qwen_vl_utils is required for multimodal GPTQ calibration. "
                "Please install `qwen-vl-utils`."
            ) from err

        modules, dataset, sample_count = self._load_vlm_dataset_modules(
            dataset_name=dataset_name,
            args=args,
        )
        calibration_inputs: list[Any] = []
        for index in range(sample_count):
            line = dataset.data.iloc[index]
            prompt_message = dataset.build_prompt(line)
            messages = _build_qwen2_messages(prompt_message, dataset_name)
            prompt = processor.apply_chat_template(
                [messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = process_vision_info([messages])
            calibration_inputs.append(
                processor(
                    text=prompt,
                    images=images,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                )
            )
        logger.info(
            "Prepared %s multimodal GPTQ calibration samples from %s.",
            len(calibration_inputs),
            dataset_name,
        )
        return calibration_inputs

    def _build_qwen3_vlm_calibration_inputs(
        self,
        *,
        processor,
        source_model,
        dataset_name: str,
        args,
    ) -> list[Any]:
        from evaluation.vlm_eval import _build_qwen2_messages

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError(
                "qwen_vl_utils is required for multimodal GPTQ calibration. "
                "Please install `qwen-vl-utils`."
            ) from err

        _, dataset, sample_count = self._load_vlm_dataset_modules(
            dataset_name=dataset_name,
            args=args,
        )
        calibration_inputs: list[Any] = []
        for index in range(sample_count):
            line = dataset.data.iloc[index]
            prompt_message = dataset.build_prompt(line)
            messages = _build_qwen2_messages(prompt_message, dataset_name)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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
            model_inputs = processor(
                text=prompt,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                padding=True,
                **(video_kwargs or {}),
            )
            if hasattr(model_inputs, "to") and hasattr(source_model, "dtype"):
                model_inputs = model_inputs.to(dtype=source_model.dtype)
            calibration_inputs.append(model_inputs)
        logger.info(
            "Prepared %s multimodal GPTQ calibration samples from %s for Qwen3-VL.",
            len(calibration_inputs),
            dataset_name,
        )
        return calibration_inputs

    def _build_internvl_vlm_calibration_inputs(
        self,
        *,
        tokenizer,
        source_model,
        dataset_name: str,
        args,
    ) -> list[Any]:
        from evaluation.vlm_eval import _internvl_default_max_num
        from evaluation.vlm_eval import _internvl_prompt_from_message
        from evaluation.vlm_eval import _load_internvl_image

        _, dataset, sample_count = self._load_vlm_dataset_modules(
            dataset_name=dataset_name,
            args=args,
        )
        img_context_token = "<IMG_CONTEXT>"
        img_start_token = "<img>"
        img_end_token = "</img>"
        img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
        source_model.img_context_token_id = img_context_token_id

        input_size = int(
            getattr(
                getattr(getattr(source_model, "config", None), "vision_config", None),
                "image_size",
                448,
            )
        )

        calibration_inputs: list[Any] = []
        for index in range(sample_count):
            line = dataset.data.iloc[index]
            prompt_message = dataset.build_prompt(line)
            question, _image_num = _internvl_prompt_from_message(prompt_message, dataset_name)
            image_paths = [item["value"] for item in prompt_message if item.get("type") == "image"]
            max_num = _internvl_default_max_num(dataset_name)

            pixel_values = None
            num_patches_list: list[int] = []
            if image_paths:
                pixel_values_list = []
                for image_path in image_paths:
                    current_pixels, num_patches = _load_internvl_image(
                        image_path,
                        input_size=input_size,
                        max_num=max_num,
                        use_thumbnail=True,
                    )
                    pixel_values_list.append(current_pixels)
                    num_patches_list.append(num_patches)
                pixel_values = torch.cat(pixel_values_list, dim=0)

            query = question
            if pixel_values is not None and "<image>" not in query:
                query = "<image>\n" + query
            for num_patches in num_patches_list:
                image_tokens = (
                    img_start_token
                    + img_context_token * int(source_model.num_image_token) * int(num_patches)
                    + img_end_token
                )
                query = query.replace("<image>", image_tokens, 1)

            model_inputs = tokenizer(query, return_tensors="pt")
            model_inputs = dict(model_inputs)
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_flags"] = (
                torch.ones((pixel_values.shape[0], 1), dtype=torch.long)
                if pixel_values is not None
                else torch.zeros((0, 1), dtype=torch.long)
            )
            calibration_inputs.append(model_inputs)

        logger.info(
            "Prepared %s multimodal GPTQ calibration samples from %s for InternVL2.",
            len(calibration_inputs),
            dataset_name,
        )
        return calibration_inputs

    @staticmethod
    def _build_visual_layer_groups(linear_layers: dict[str, Any]) -> list[list[str]]:
        if "attn.qkv" in linear_layers:
            groups = [["attn.qkv"]]
            if "attn.proj" in linear_layers:
                groups.append(["attn.proj"])
            if "mlp.fc1" in linear_layers and "mlp.fc2" in linear_layers:
                groups.extend([["mlp.fc1"], ["mlp.fc2"]])
            elif "mlp.linear_fc1" in linear_layers and "mlp.linear_fc2" in linear_layers:
                groups.extend([["mlp.linear_fc1"], ["mlp.linear_fc2"]])
            elif {
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            }.issubset(linear_layers):
                groups.extend([["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]])
            else:
                remaining = [name for name in sorted(linear_layers) if name not in {item for group in groups for item in group}]
                groups.extend([[name] for name in remaining])
            return groups
        return [[name] for name in sorted(linear_layers)]

    @staticmethod
    def _find_gptq_layers(module: nn.Module, prefix: str = "") -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        for child_name, child in module.named_children():
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                result[qualified_name] = child
                continue
            result.update(GPTQMethod._find_gptq_layers(child, qualified_name))
        return result

    @staticmethod
    def _resolve_group_variants(
        available_names: set[str],
        candidate_groups: list[list[list[str]]],
    ) -> list[list[str]]:
        groups: list[list[str]] = []
        used_names: set[str] = set()
        for variants in candidate_groups:
            resolved = next((variant for variant in variants if all(name in available_names for name in variant)), None)
            if resolved is None:
                continue
            groups.append(list(resolved))
            used_names.update(resolved)
        for name in sorted(available_names):
            if name not in used_names:
                groups.append([name])
        return groups

    def _build_qwen2_vlm_groups(
        self,
        *,
        source_model,
        quant_visual: bool,
        quant_connector: bool,
        quant_llm: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        groups: list[tuple[str, dict[str, Any]]] = []
        multimodal_root = getattr(source_model, "model", source_model)
        visual_root = getattr(multimodal_root, "visual", None)
        llm_root = getattr(multimodal_root, "language_model", None)
        if visual_root is None or llm_root is None:
            raise AttributeError(
                "Expected Qwen2/Qwen2.5-VL multimodal root to expose `model.visual` and `model.language_model`."
            )

        if quant_visual:
            for layer_index, block in enumerate(getattr(visual_root, "blocks", [])):
                linear_layers = find_linear_layers(block)
                for group in self._build_visual_layer_groups(linear_layers):
                    groups.append(
                        (
                            f"model.visual.blocks.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )

        if quant_connector:
            merger = getattr(visual_root, "merger", None)
            if merger is None:
                raise AttributeError("Expected Qwen2/Qwen2.5-VL visual root to expose `merger`.")
            linear_layers = find_linear_layers(merger)
            for group in [[name] for name in sorted(linear_layers)]:
                groups.append(
                    (
                        "model.visual.merger",
                        {name: linear_layers[name] for name in group},
                    )
                )

        if quant_llm:
            for layer_index, block in enumerate(llm_root.layers):
                linear_layers = find_linear_layers(block)
                layer_groups = build_decoder_layer_groups(block, set(linear_layers))
                for group in layer_groups:
                    groups.append(
                        (
                            f"model.language_model.layers.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )
        return groups

    def _build_qwen3_vlm_groups(
        self,
        *,
        source_model,
        quant_visual: bool,
        quant_connector: bool,
        quant_llm: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        groups: list[tuple[str, dict[str, Any]]] = []
        multimodal_root = getattr(source_model, "model", source_model)
        visual_root = getattr(multimodal_root, "visual", None)
        llm_root = getattr(multimodal_root, "language_model", None)
        if visual_root is None or llm_root is None:
            raise AttributeError(
                "Expected Qwen3-VL multimodal root to expose `model.visual` and `model.language_model`."
            )

        if quant_visual:
            patch_proj = getattr(getattr(visual_root, "patch_embed", None), "proj", None)
            if patch_proj is not None and not isinstance(patch_proj, nn.Conv2d):
                logger.warning(
                    "Qwen3-VL visual patch embedding %s is not currently quantized by GPTQMethod; "
                    "only visual blocks and mergers are quantized.",
                    type(patch_proj).__name__,
                )
            for layer_index, block in enumerate(getattr(visual_root, "blocks", [])):
                linear_layers = find_linear_layers(block)
                for group in self._build_visual_layer_groups(linear_layers):
                    groups.append(
                        (
                            f"model.visual.blocks.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )

        if quant_connector:
            merger_modules = [("model.visual.merger", getattr(visual_root, "merger", None))]
            merger_modules.extend(
                (
                    f"model.visual.deepstack_merger_list.{index}",
                    merger,
                )
                for index, merger in enumerate(getattr(visual_root, "deepstack_merger_list", []))
            )
            for group_prefix, merger in merger_modules:
                if merger is None:
                    continue
                linear_layers = find_linear_layers(merger)
                for group in [[name] for name in sorted(linear_layers)]:
                    groups.append(
                        (
                            group_prefix,
                            {name: linear_layers[name] for name in group},
                        )
                    )

        if quant_llm:
            for layer_index, block in enumerate(llm_root.layers):
                linear_layers = find_linear_layers(block)
                layer_groups = build_decoder_layer_groups(block, set(linear_layers))
                for group in layer_groups:
                    groups.append(
                        (
                            f"model.language_model.layers.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )
        return groups

    def _build_internvl_vlm_groups(
        self,
        *,
        source_model,
        quant_visual: bool,
        quant_connector: bool,
        quant_llm: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        groups: list[tuple[str, dict[str, Any]]] = []

        if quant_visual:
            patch_embedding = getattr(getattr(source_model.vision_model, "embeddings", None), "patch_embedding", None)
            if isinstance(patch_embedding, nn.Conv2d):
                groups.append(("vision_model.embeddings", {"patch_embedding": patch_embedding}))
            for layer_index, block in enumerate(getattr(source_model.vision_model.encoder, "layers", [])):
                linear_layers = self._find_gptq_layers(block)
                layer_groups = self._resolve_group_variants(
                    set(linear_layers),
                    [
                        [["attn.qkv"], ["attn.qkv.module"]],
                        [["attn.proj"], ["attn.proj.module"]],
                        [["mlp.fc1"], ["mlp.fc1.module"]],
                        [["mlp.fc2"], ["mlp.fc2.module"]],
                    ],
                )
                for group in layer_groups:
                    groups.append(
                        (
                            f"vision_model.encoder.layers.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )

        if quant_connector:
            linear_layers = self._find_gptq_layers(source_model.mlp1)
            layer_groups = self._resolve_group_variants(
                set(linear_layers),
                [
                    [["1"], ["1.module"]],
                    [["3"], ["3.module"]],
                ],
            )
            for group in layer_groups:
                groups.append(
                    (
                        "mlp1",
                        {name: linear_layers[name] for name in group},
                    )
                )

        if quant_llm:
            for layer_index, block in enumerate(source_model.language_model.model.layers):
                linear_layers = self._find_gptq_layers(block)
                layer_groups = self._resolve_group_variants(
                    set(linear_layers),
                    [
                        [["attention.wqkv"], ["attention.wqkv.module"]],
                        [["attention.wo"], ["attention.wo.module"]],
                        [
                            ["feed_forward.w1", "feed_forward.w3"],
                            ["feed_forward.w1.module", "feed_forward.w3.module"],
                        ],
                        [["feed_forward.w2"], ["feed_forward.w2.module"]],
                    ],
                )
                for group in layer_groups:
                    groups.append(
                        (
                            f"language_model.model.layers.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )
        return groups

    def _build_minicpmv_vlm_groups(
        self,
        *,
        source_model,
        quant_visual: bool,
        quant_connector: bool,
        quant_llm: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        from ..mquant.method import _ensure_minicpmv_mquant_compat

        _ensure_minicpmv_mquant_compat(source_model)
        groups: list[tuple[str, dict[str, Any]]] = []

        if quant_visual:
            patch_embedding = getattr(getattr(source_model.vpm, "patch_embed", None), "proj", None)
            if isinstance(patch_embedding, nn.Conv2d):
                groups.append(("vpm.patch_embed", {"proj": patch_embedding}))
            for layer_index, block in enumerate(getattr(source_model.vpm, "blocks", [])):
                linear_layers = self._find_gptq_layers(block)
                layer_groups = self._resolve_group_variants(
                    set(linear_layers),
                    [
                        [
                            ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
                            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                        ],
                        [["attn.proj"], ["attn.out_proj"], ["self_attn.out_proj"]],
                        [["mlp.fc1"]],
                        [["mlp.fc2"]],
                    ],
                )
                for group in layer_groups:
                    groups.append(
                        (
                            f"vpm.blocks.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )

        if quant_connector:
            linear_layers = self._find_gptq_layers(source_model.resampler)
            layer_groups = self._resolve_group_variants(
                set(linear_layers),
                [
                    [["kv_proj"]],
                    [["attn.q_proj", "attn.k_proj", "attn.v_proj"]],
                    [["attn.out_proj"]],
                    [["proj_fc"]],
                ],
            )
            for group in layer_groups:
                groups.append(
                    (
                        "resampler",
                        {name: linear_layers[name] for name in group},
                    )
                )

        if quant_llm:
            for layer_index, block in enumerate(source_model.llm.model.layers):
                linear_layers = find_linear_layers(block)
                layer_groups = build_decoder_layer_groups(block, set(linear_layers))
                for group in layer_groups:
                    groups.append(
                        (
                            f"llm.model.layers.{layer_index}",
                            {name: linear_layers[name] for name in group},
                        )
                    )
        return groups

    def _build_generic_vlm_groups(
        self,
        *,
        source_model,
        model_type: str,
        quant_visual: bool,
        quant_connector: bool,
        quant_llm: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        if model_type == "internvl_chat":
            return self._build_internvl_vlm_groups(
                source_model=source_model,
                quant_visual=quant_visual,
                quant_connector=quant_connector,
                quant_llm=quant_llm,
            )
        if model_type == "minicpmv":
            return self._build_minicpmv_vlm_groups(
                source_model=source_model,
                quant_visual=quant_visual,
                quant_connector=quant_connector,
                quant_llm=quant_llm,
            )
        raise NotImplementedError(f"Unsupported generic multimodal GPTQ model_type={model_type!r}.")

    def _build_wrapper_vlm_runner(
        self,
        *,
        model,
        tokenizer_bundle,
        args,
        model_type: str,
    ):
        from evaluation.vlm_eval import _build_internvl_wrapper
        from evaluation.vlm_eval import _build_minicpm_wrapper

        base_model_cls = type("_GPTQCalibrationBase", (), {"__init__": lambda self: None})
        common_args = {
            "device": args.device,
            "vlm_use_cache": False,
            "vlm_max_new_tokens": 2 if model_type == "minicpmv" else 1,
        }
        dataset_type_resolver = lambda _dataset_name: "IMAGE"

        if model_type == "internvl_chat":
            wrapper = _build_internvl_wrapper(
                model,
                tokenizer_bundle,
                common_args,
                base_model_cls,
                dataset_type_resolver,
            )
            return lambda prompt_message, dataset_name: wrapper.generate_inner(prompt_message, dataset=dataset_name)
        if model_type == "minicpmv":
            wrapper = _build_minicpm_wrapper(
                model,
                tokenizer_bundle,
                common_args,
                base_model_cls,
                dataset_type_resolver,
            )

            def _run_minicpmv(prompt_message, dataset_name):
                try:
                    return wrapper.generate_inner(prompt_message, dataset=dataset_name)
                except IndexError as error:
                    if "index 0 is out of bounds" not in str(error):
                        raise
                    logger.warning(
                        "MiniCPM-V GPTQ calibration hit an empty decode after forward replay; "
                        "ignoring it because GPTQ only needs the captured module activations."
                    )
                    return ""

            return _run_minicpmv
        raise NotImplementedError(f"Unsupported wrapper GPTQ runner for model_type={model_type!r}.")
