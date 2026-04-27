"""Unified AWQ runner."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import torch

from ....common.device import backend_module
from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod

logger = logging.getLogger(__name__)


class AWQMethod(BaseQuantizationMethod):
    name = "awq"
    default_calibration_dataset = "pileval"
    _SUPPORTED_VLM_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "minicpmv"}

    @staticmethod
    def _resolve_source_model(model):
        return getattr(model, "_source_model", model)

    @staticmethod
    def _resolve_model_type(model) -> str:
        return str(getattr(getattr(model, "config", None), "model_type", "") or "")

    @staticmethod
    def _resolve_bool(value, *, default: bool) -> bool:
        if value is None:
            return bool(default)
        return bool(value)

    @staticmethod
    def _resolve_vlm_dataset_name(args) -> str | None:
        explicit = getattr(args, "awq_vlm_dataset_name", None)
        if explicit:
            return str(explicit)
        legacy = getattr(args, "mquant_dataset_name", None)
        if legacy:
            return str(legacy)
        return None

    @staticmethod
    def _resolve_vlm_calib_num(args) -> int:
        explicit = getattr(args, "awq_vlm_calib_num", None)
        if explicit is not None:
            return int(explicit)
        legacy = getattr(args, "mquant_calib_num", None)
        if legacy is not None:
            return int(legacy)
        return int(args.calibration_samples)

    @staticmethod
    def _resolve_branch_weight_bits(args, branch: str) -> int:
        branch_to_attr = {
            "visual": "awq_visual_w_bits",
            "connector": "awq_connector_w_bits",
            "llm": "awq_llm_w_bits",
        }
        if branch not in branch_to_attr:
            raise ValueError(f"Unsupported AWQ branch name: {branch!r}")
        if branch == "connector":
            explicit = getattr(args, branch_to_attr[branch], None)
            if explicit is not None:
                return int(explicit)
            visual_override = getattr(args, branch_to_attr["visual"], None)
            if visual_override is not None:
                return int(visual_override)
            return int(args.weight_bits)
        explicit = getattr(args, branch_to_attr[branch], None)
        if explicit is not None:
            return int(explicit)
        return int(args.weight_bits)

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
    def _move_nested_to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, tuple):
            return tuple(AWQMethod._move_nested_to_cpu(item) for item in value)
        if isinstance(value, list):
            return [AWQMethod._move_nested_to_cpu(item) for item in value]
        if isinstance(value, dict):
            return {key: AWQMethod._move_nested_to_cpu(item) for key, item in value.items()}
        return value

    @staticmethod
    def _move_nested_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(AWQMethod._move_nested_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [AWQMethod._move_nested_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {key: AWQMethod._move_nested_to_device(item, device) for key, item in value.items()}
        return value

    @staticmethod
    def _model_accepts_kwarg(model, kwarg_name: str) -> bool:
        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            return False
        return kwarg_name in signature.parameters

    def _run_multimodal_forward(self, *, source_model, prepared_inputs) -> None:
        if (
            isinstance(prepared_inputs, dict)
            and callable(prepared_inputs.get("_mindpipe_forward_callable"))
        ):
            prepared_inputs["_mindpipe_forward_callable"]()
            return
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
                        "Requested VLMEvalKit root %s unavailable; falling back to %s for AWQ multimodal calibration.",
                        requested_root,
                        root,
                    )
                break
            except (FileNotFoundError, ModuleNotFoundError) as error:
                last_error = error
        if modules is None:
            if last_error is not None:
                raise last_error
            raise FileNotFoundError("Failed to resolve a usable VLMEvalKit root for multimodal AWQ calibration.")

        dataset = modules["build_dataset"](dataset_name)
        if dataset is None:
            raise ValueError(f"Failed to build VLM calibration dataset: {dataset_name}")

        sample_count = min(self._resolve_vlm_calib_num(args), len(dataset))
        return modules, dataset, sample_count

    def _build_qwen2_vlm_calibration_inputs(self, *, processor, dataset_name: str, args) -> list[Any]:
        from evaluation.vlm_eval import _build_qwen2_messages

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError(
                "qwen_vl_utils is required for multimodal AWQ calibration. Please install `qwen-vl-utils`."
            ) from err

        _modules, dataset, sample_count = self._load_vlm_dataset_modules(
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
            "Prepared %s multimodal AWQ calibration samples from %s.",
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
                "qwen_vl_utils is required for multimodal AWQ calibration. Please install `qwen-vl-utils`."
            ) from err

        _modules, dataset, sample_count = self._load_vlm_dataset_modules(
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
            "Prepared %s multimodal AWQ calibration samples from %s for Qwen3-VL.",
            len(calibration_inputs),
            dataset_name,
        )
        return calibration_inputs

    def _build_minicpmv_vlm_calibration_inputs(
        self,
        *,
        source_model,
        tokenizer,
        dataset_name: str,
        args,
    ) -> list[Any]:
        from ..mquant.method import _MindPipeMiniCPMVGPTQWrapper

        _modules, dataset, sample_count = self._load_vlm_dataset_modules(
            dataset_name=dataset_name,
            args=args,
        )
        wrapper = _MindPipeMiniCPMVGPTQWrapper(
            source_model=source_model,
            tokenizer=tokenizer,
            target_device=args.device,
            max_new_tokens=2,
            use_cache=False,
        )
        calibration_inputs: list[Any] = []
        for index in range(sample_count):
            line = dataset.data.iloc[index]
            prompt_message = dataset.build_prompt(line)

            def _forward_callable(message=prompt_message, current_dataset=dataset_name):
                wrapper.generate_inner(message, dataset=current_dataset)

            calibration_inputs.append(
                {
                    "_mindpipe_forward_callable": _forward_callable,
                    "_mindpipe_dataset_name": dataset_name,
                }
            )
        logger.info(
            "Prepared %s multimodal AWQ calibration samples from %s for MiniCPM-V.",
            len(calibration_inputs),
            dataset_name,
        )
        return calibration_inputs

    def _capture_block_samples(self, *, source_model, calibration_inputs, block, device) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        captured_samples: list[tuple[torch.Tensor, dict[str, Any]]] = []

        def pre_hook(_module, args, kwargs):
            if not args:
                raise RuntimeError(f"Expected positional block input for {type(block)}.")
            hidden_states = self._move_nested_to_cpu(args[0])
            captured_kwargs = self._move_nested_to_cpu(kwargs)
            captured_samples.append((hidden_states, captured_kwargs))

        handle = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            for model_inputs in calibration_inputs:
                self._run_multimodal_forward(
                    source_model=source_model,
                    prepared_inputs=self._move_inputs_to_device(model_inputs, device),
                )
        finally:
            handle.remove()

        if not captured_samples:
            raise RuntimeError(f"Failed to capture calibration samples for block {type(block)}.")
        return captured_samples

    def _apply_visual_block_awq(
        self,
        *,
        block,
        block_prefix: str,
        source_model,
        calibration_inputs,
        args,
        quantization_config,
        auto_scale_block,
        apply_scale,
        auto_clip_block,
        apply_clip,
        get_named_linears,
        pseudo_quantize_tensor,
        weight_bits: int,
        awq_auto_scale: bool,
        awq_mse_range: bool,
        awq_clip_targets: str,
    ) -> dict[str, object]:
        runtime_device = resolve_device(args.device)
        block_samples = self._capture_block_samples(
            source_model=source_model,
            calibration_inputs=calibration_inputs,
            block=block,
            device=runtime_device,
        )

        block = block.to(runtime_device)
        named_linears = get_named_linears(block, model=source_model)
        if not named_linears:
            logger.warning("Skip visual block %s because no linear layers were found.", block_prefix)
            return {"quantized_linear_count": 0, "quantized_linear_names": []}

        input_feat: dict[str, list[torch.Tensor]] = {name: [] for name in named_linears}
        replay_samples_by_name: dict[str, list[tuple[torch.Tensor, dict[str, Any]]]] = {
            "attn": [],
            "mlp": [],
        }

        def cache_input_hook(_module, inputs, _outputs, name):
            input_feat[name].append(inputs[0].detach().cpu())

        def cache_module_pre_hook(module_name: str):
            def _hook(_module, args, kwargs):
                if not args:
                    raise RuntimeError(
                        f"Expected positional input when capturing {module_name} replay samples for {block_prefix}."
                    )
                replay_samples_by_name[module_name].append(
                    (
                        self._move_nested_to_cpu(args[0]),
                        self._move_nested_to_cpu(kwargs),
                    )
                )

            return _hook

        handles = [
            named_linears[name].register_forward_hook(lambda module, inputs, outputs, layer_name=name: cache_input_hook(module, inputs, outputs, layer_name))
            for name in named_linears
        ]
        if hasattr(block, "attn"):
            handles.append(
                block.attn.register_forward_pre_hook(
                    cache_module_pre_hook("attn"),
                    with_kwargs=True,
                )
            )
        if hasattr(block, "mlp"):
            handles.append(
                block.mlp.register_forward_pre_hook(
                    cache_module_pre_hook("mlp"),
                    with_kwargs=True,
                )
            )
        try:
            replay_samples = [
                (
                    sample_input.to(runtime_device),
                    self._move_nested_to_device(sample_kwargs, runtime_device),
                )
                for sample_input, sample_kwargs in block_samples
            ]
            for sample_input, sample_kwargs in replay_samples:
                with torch.no_grad():
                    block(sample_input, **sample_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        input_feat = {
            name: torch.cat(feats, dim=0)
            for name, feats in input_feat.items()
            if feats
        }

        if awq_auto_scale:
            try:
                scales_list = auto_scale_block(
                    source_model,
                    block,
                    {},
                    w_bit=weight_bits,
                    q_config=quantization_config,
                    input_feat=input_feat,
                    sample_inputs=block_samples,
                    replay_samples_by_name=replay_samples_by_name,
                )
                apply_scale(block, scales_list, input_feat_dict=input_feat, device=runtime_device)
            except Exception as error:
                logger.warning(
                    "Visual AWQ auto_scale failed on %s; skip scale search for this block. Error: %s",
                    block_prefix,
                    error,
                )
                block.to(runtime_device)
                scales_list = []
        else:
            scales_list = []

        if awq_mse_range:
            clip_list = auto_clip_block(
                block,
                w_bit=weight_bits,
                q_config=quantization_config,
                input_feat=input_feat,
                device=runtime_device,
                model=source_model,
                clip_targets=awq_clip_targets,
            )
            apply_clip(block, clip_list, device=runtime_device)
        else:
            clip_list = []

        quantized_names: list[str] = []
        for name, linear in named_linears.items():
            linear.to(runtime_device)
            linear.weight.data = pseudo_quantize_tensor(
                linear.weight.data,
                n_bit=weight_bits,
                **quantization_config,
            )
            quantized_names.append(f"{block_prefix}.{name}")

        # Upstream AWQ helpers move touched submodules back to CPU after scale/clip.
        # Restore the whole visual block onto the runtime device before the next
        # multimodal forward, otherwise subsequent block-sample capture will hit
        # CPU/CUDA mismatches inside the already-quantized prefix blocks.
        block.to(runtime_device)
        del input_feat
        empty_cache(runtime_device)
        return {
            "quantized_linear_count": len(quantized_names),
            "quantized_linear_names": quantized_names,
            "scale_entry_count": len(scales_list),
            "clip_entry_count": len(clip_list),
        }

    def _apply_connector_awq(
        self,
        *,
        connector,
        connector_prefix: str,
        source_model,
        calibration_inputs,
        args,
        quantization_config,
        auto_scale_block,
        apply_scale,
        auto_clip_block,
        apply_clip,
        get_named_linears,
        pseudo_quantize_tensor,
        weight_bits: int,
        awq_auto_scale: bool,
        awq_mse_range: bool,
        awq_clip_targets: str,
    ) -> dict[str, object]:
        runtime_device = resolve_device(args.device)
        connector_samples = self._capture_block_samples(
            source_model=source_model,
            calibration_inputs=calibration_inputs,
            block=connector,
            device=runtime_device,
        )

        connector = connector.to(runtime_device)
        named_linears = get_named_linears(connector, model=source_model)
        if not named_linears:
            logger.warning("Skip connector %s because no linear layers were found.", connector_prefix)
            return {"quantized_linear_count": 0, "quantized_linear_names": []}

        input_feat: dict[str, list[torch.Tensor]] = {name: [] for name in named_linears}
        replay_samples_by_name: dict[str, list[tuple[torch.Tensor, dict[str, Any]]]] = {}

        def cache_input_hook(_module, inputs, _outputs, name):
            input_feat[name].append(inputs[0].detach().cpu())

        def cache_module_pre_hook(module_name: str):
            def _hook(_module, args, kwargs):
                if not args:
                    raise RuntimeError(
                        f"Expected positional input when capturing {module_name} replay samples for {connector_prefix}."
                    )
                replay_samples_by_name.setdefault(module_name, []).append(
                    (
                        self._move_nested_to_cpu(args[0]),
                        self._move_nested_to_cpu(kwargs),
                    )
                )

            return _hook

        handles = [
            named_linears[name].register_forward_hook(
                lambda module, inputs, outputs, layer_name=name: cache_input_hook(module, inputs, outputs, layer_name)
            )
            for name in named_linears
        ]
        if hasattr(connector, "norm"):
            handles.append(
                connector.norm.register_forward_pre_hook(
                    cache_module_pre_hook("norm"),
                    with_kwargs=True,
                )
            )
        try:
            replay_samples = [
                (
                    sample_input.to(runtime_device),
                    self._move_nested_to_device(sample_kwargs, runtime_device),
                )
                for sample_input, sample_kwargs in connector_samples
            ]
            for sample_input, sample_kwargs in replay_samples:
                with torch.no_grad():
                    connector(sample_input, **sample_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        input_feat = {
            name: torch.cat(feats, dim=0)
            for name, feats in input_feat.items()
            if feats
        }

        if awq_auto_scale:
            try:
                scales_list = auto_scale_block(
                    source_model,
                    connector,
                    {},
                    w_bit=weight_bits,
                    q_config=quantization_config,
                    input_feat=input_feat,
                    sample_inputs=connector_samples,
                    replay_samples_by_name=replay_samples_by_name,
                )
                apply_scale(connector, scales_list, input_feat_dict=input_feat, device=runtime_device)
            except Exception as error:
                logger.warning(
                    "Connector AWQ auto_scale failed on %s; skip scale search for this module. Error: %s",
                    connector_prefix,
                    error,
                )
                connector.to(runtime_device)
                scales_list = []
        else:
            scales_list = []

        if awq_mse_range:
            clip_list = auto_clip_block(
                connector,
                w_bit=weight_bits,
                q_config=quantization_config,
                input_feat=input_feat,
                device=runtime_device,
                model=source_model,
                clip_targets=awq_clip_targets,
            )
            apply_clip(connector, clip_list, device=runtime_device)
        else:
            clip_list = []

        quantized_names: list[str] = []
        for name, linear in named_linears.items():
            linear.to(runtime_device)
            linear.weight.data = pseudo_quantize_tensor(
                linear.weight.data,
                n_bit=weight_bits,
                **quantization_config,
            )
            quantized_names.append(f"{connector_prefix}.{name}")

        connector.to(runtime_device)
        del input_feat
        empty_cache(runtime_device)
        return {
            "quantized_linear_count": len(quantized_names),
            "quantized_linear_names": quantized_names,
            "scale_entry_count": len(scales_list),
            "clip_entry_count": len(clip_list),
        }

    def _apply_qwen2_vl_multimodal_awq(
        self,
        *,
        source_model,
        processor,
        args,
        quant_visual: bool,
        quant_connector: bool,
        quantization_config,
        auto_scale_block,
        apply_scale,
        auto_clip_block,
        apply_clip,
        get_named_linears,
        pseudo_quantize_tensor,
        visual_weight_bits: int,
        connector_weight_bits: int,
        awq_auto_scale: bool,
        awq_mse_range: bool,
        awq_clip_targets: str,
        dataset_name: str,
    ) -> dict[str, object]:
        runtime_device = resolve_device(args.device)
        source_model.to(runtime_device)
        source_model.eval()
        if hasattr(source_model, "config") and hasattr(source_model.config, "use_cache"):
            source_model.config.use_cache = False

        calibration_inputs = self._build_qwen2_vlm_calibration_inputs(
            processor=processor,
            dataset_name=dataset_name,
            args=args,
        )
        multimodal_root = getattr(source_model, "model", source_model)
        visual_root = getattr(multimodal_root, "visual", None)
        if visual_root is None or not hasattr(visual_root, "blocks"):
            raise AttributeError("Expected Qwen2/Qwen2.5-VL source model to expose `model.visual.blocks`.")

        quantized_blocks = []
        connector_artifacts = None
        total_quantized = 0
        if quant_visual:
            for layer_index, block in enumerate(visual_root.blocks):
                block_prefix = f"model.visual.blocks.{layer_index}"
                logger.info("Running visual AWQ on %s", block_prefix)
                block_artifacts = self._apply_visual_block_awq(
                    block=block,
                    block_prefix=block_prefix,
                    source_model=source_model,
                    calibration_inputs=calibration_inputs,
                    args=args,
                    quantization_config=quantization_config,
                    auto_scale_block=auto_scale_block,
                    apply_scale=apply_scale,
                    auto_clip_block=auto_clip_block,
                    apply_clip=apply_clip,
                    get_named_linears=get_named_linears,
                    pseudo_quantize_tensor=pseudo_quantize_tensor,
                    weight_bits=visual_weight_bits,
                    awq_auto_scale=awq_auto_scale,
                    awq_mse_range=awq_mse_range,
                    awq_clip_targets=awq_clip_targets,
                )
                total_quantized += int(block_artifacts["quantized_linear_count"])
                quantized_blocks.append({"prefix": block_prefix, **block_artifacts})

        if quant_connector:
            connector = getattr(visual_root, "merger", None)
            if connector is None:
                raise AttributeError("Expected Qwen2/Qwen2.5-VL visual root to expose `merger`.")
            connector_prefix = "model.visual.merger"
            logger.info("Running connector AWQ on %s", connector_prefix)
            connector_artifacts = self._apply_connector_awq(
                connector=connector,
                connector_prefix=connector_prefix,
                source_model=source_model,
                calibration_inputs=calibration_inputs,
                args=args,
                quantization_config=quantization_config,
                auto_scale_block=auto_scale_block,
                apply_scale=apply_scale,
                auto_clip_block=auto_clip_block,
                apply_clip=apply_clip,
                get_named_linears=get_named_linears,
                pseudo_quantize_tensor=pseudo_quantize_tensor,
                weight_bits=connector_weight_bits,
                awq_auto_scale=awq_auto_scale,
                awq_mse_range=awq_mse_range,
                awq_clip_targets=awq_clip_targets,
            )
            total_quantized += int(connector_artifacts["quantized_linear_count"])

        artifacts = {
            "dataset_name": dataset_name,
            "sample_count": len(calibration_inputs),
            "quantized_linear_count": total_quantized,
            "quantized_blocks": quantized_blocks,
        }
        if connector_artifacts is not None:
            artifacts["quantized_connector"] = {"prefix": "model.visual.merger", **connector_artifacts}
        return artifacts

    def _apply_qwen3_vl_multimodal_awq(
        self,
        *,
        source_model,
        processor,
        args,
        quant_visual: bool,
        quant_connector: bool,
        quantization_config,
        auto_scale_block,
        apply_scale,
        auto_clip_block,
        apply_clip,
        get_named_linears,
        pseudo_quantize_tensor,
        visual_weight_bits: int,
        connector_weight_bits: int,
        awq_auto_scale: bool,
        awq_mse_range: bool,
        awq_clip_targets: str,
        dataset_name: str,
    ) -> dict[str, object]:
        runtime_device = resolve_device(args.device)
        source_model.to(runtime_device)
        source_model.eval()
        if hasattr(source_model, "config") and hasattr(source_model.config, "use_cache"):
            source_model.config.use_cache = False

        calibration_inputs = self._build_qwen3_vlm_calibration_inputs(
            processor=processor,
            source_model=source_model,
            dataset_name=dataset_name,
            args=args,
        )
        multimodal_root = getattr(source_model, "model", source_model)
        visual_root = getattr(multimodal_root, "visual", None)
        if visual_root is None or not hasattr(visual_root, "blocks"):
            raise AttributeError("Expected Qwen3-VL source model to expose `model.visual.blocks`.")

        quantized_blocks = []
        quantized_connectors = []
        total_quantized = 0
        if quant_visual:
            for layer_index, block in enumerate(visual_root.blocks):
                block_prefix = f"model.visual.blocks.{layer_index}"
                logger.info("Running visual AWQ on %s", block_prefix)
                block_artifacts = self._apply_visual_block_awq(
                    block=block,
                    block_prefix=block_prefix,
                    source_model=source_model,
                    calibration_inputs=calibration_inputs,
                    args=args,
                    quantization_config=quantization_config,
                    auto_scale_block=auto_scale_block,
                    apply_scale=apply_scale,
                    auto_clip_block=auto_clip_block,
                    apply_clip=apply_clip,
                    get_named_linears=get_named_linears,
                    pseudo_quantize_tensor=pseudo_quantize_tensor,
                    weight_bits=visual_weight_bits,
                    awq_auto_scale=awq_auto_scale,
                    awq_mse_range=awq_mse_range,
                    awq_clip_targets=awq_clip_targets,
                )
                total_quantized += int(block_artifacts["quantized_linear_count"])
                quantized_blocks.append({"prefix": block_prefix, **block_artifacts})

        if quant_connector:
            connector_modules = [("model.visual.merger", getattr(visual_root, "merger", None))]
            connector_modules.extend(
                (
                    f"model.visual.deepstack_merger_list.{index}",
                    merger,
                )
                for index, merger in enumerate(getattr(visual_root, "deepstack_merger_list", []))
            )
            for connector_prefix, connector in connector_modules:
                if connector is None:
                    continue
                logger.info("Running connector AWQ on %s", connector_prefix)
                connector_artifacts = self._apply_connector_awq(
                    connector=connector,
                    connector_prefix=connector_prefix,
                    source_model=source_model,
                    calibration_inputs=calibration_inputs,
                    args=args,
                    quantization_config=quantization_config,
                    auto_scale_block=auto_scale_block,
                    apply_scale=apply_scale,
                    auto_clip_block=auto_clip_block,
                    apply_clip=apply_clip,
                    get_named_linears=get_named_linears,
                    pseudo_quantize_tensor=pseudo_quantize_tensor,
                    weight_bits=connector_weight_bits,
                    awq_auto_scale=awq_auto_scale,
                    awq_mse_range=awq_mse_range,
                    awq_clip_targets=awq_clip_targets,
                )
                total_quantized += int(connector_artifacts["quantized_linear_count"])
                quantized_connectors.append({"prefix": connector_prefix, **connector_artifacts})

        artifacts = {
            "dataset_name": dataset_name,
            "sample_count": len(calibration_inputs),
            "quantized_linear_count": total_quantized,
            "quantized_blocks": quantized_blocks,
        }
        if quantized_connectors:
            artifacts["quantized_connector"] = quantized_connectors
        return artifacts

    def _apply_minicpmv_multimodal_awq(
        self,
        *,
        source_model,
        tokenizer,
        args,
        quant_visual: bool,
        quant_connector: bool,
        quantization_config,
        auto_scale_block,
        apply_scale,
        auto_clip_block,
        apply_clip,
        get_named_linears,
        pseudo_quantize_tensor,
        visual_weight_bits: int,
        connector_weight_bits: int,
        awq_auto_scale: bool,
        awq_mse_range: bool,
        awq_clip_targets: str,
        dataset_name: str,
    ) -> dict[str, object]:
        from ..mquant.method import _ensure_minicpmv_mquant_compat

        runtime_device = resolve_device(args.device)
        source_model.to(runtime_device)
        source_model.eval()
        if hasattr(source_model, "config") and hasattr(source_model.config, "use_cache"):
            source_model.config.use_cache = False
        if hasattr(source_model, "llm") and hasattr(source_model.llm, "config") and hasattr(source_model.llm.config, "use_cache"):
            source_model.llm.config.use_cache = False

        _ensure_minicpmv_mquant_compat(source_model)
        calibration_inputs = self._build_minicpmv_vlm_calibration_inputs(
            source_model=source_model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            args=args,
        )

        quantized_blocks = []
        connector_artifacts = None
        total_quantized = 0

        if quant_visual:
            for layer_index, block in enumerate(getattr(source_model.vpm, "blocks", [])):
                block_prefix = f"vpm.blocks.{layer_index}"
                logger.info("Running visual AWQ on %s", block_prefix)
                block_artifacts = self._apply_visual_block_awq(
                    block=block,
                    block_prefix=block_prefix,
                    source_model=source_model,
                    calibration_inputs=calibration_inputs,
                    args=args,
                    quantization_config=quantization_config,
                    auto_scale_block=auto_scale_block,
                    apply_scale=apply_scale,
                    auto_clip_block=auto_clip_block,
                    apply_clip=apply_clip,
                    get_named_linears=get_named_linears,
                    pseudo_quantize_tensor=pseudo_quantize_tensor,
                    weight_bits=visual_weight_bits,
                    awq_auto_scale=awq_auto_scale,
                    awq_mse_range=awq_mse_range,
                    awq_clip_targets=awq_clip_targets,
                )
                total_quantized += int(block_artifacts["quantized_linear_count"])
                quantized_blocks.append({"prefix": block_prefix, **block_artifacts})

        if quant_connector:
            connector = getattr(source_model, "resampler", None)
            if connector is None:
                raise AttributeError("Expected MiniCPM-V source model to expose `resampler`.")
            connector_prefix = "model.resampler"
            logger.info("Running connector AWQ on %s", connector_prefix)
            connector_artifacts = self._apply_connector_awq(
                connector=connector,
                connector_prefix=connector_prefix,
                source_model=source_model,
                calibration_inputs=calibration_inputs,
                args=args,
                quantization_config=quantization_config,
                auto_scale_block=auto_scale_block,
                apply_scale=apply_scale,
                auto_clip_block=auto_clip_block,
                apply_clip=apply_clip,
                get_named_linears=get_named_linears,
                pseudo_quantize_tensor=pseudo_quantize_tensor,
                weight_bits=connector_weight_bits,
                awq_auto_scale=awq_auto_scale,
                awq_mse_range=awq_mse_range,
                awq_clip_targets=awq_clip_targets,
            )
            total_quantized += int(connector_artifacts["quantized_linear_count"])

        artifacts = {
            "dataset_name": dataset_name,
            "sample_count": len(calibration_inputs),
            "quantized_linear_count": total_quantized,
            "quantized_blocks": quantized_blocks,
        }
        if connector_artifacts is not None:
            artifacts["quantized_connector"] = {"prefix": "model.resampler", **connector_artifacts}
        return artifacts

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        output_dir = self.resolve_output_dir(args)
        awq_state_path = output_dir / "awq_search.pt"
        awq_search_sequence_length = int(getattr(args, "awq_search_sequence_length", 512) or 512)
        awq_reuse_search_result = bool(getattr(args, "awq_reuse_search_result", False))
        awq_auto_scale = bool(getattr(args, "awq_auto_scale", True))
        awq_mse_range = bool(getattr(args, "awq_mse_range", True))
        awq_clip_targets = str(getattr(args, "awq_clip_targets", "auto") or "auto")
        awq_qwen3_5_quantize_linear_attn = bool(
            getattr(args, "awq_qwen3_5_quantize_linear_attn", True)
        )
        source_model = self._resolve_source_model(model)
        model_type = self._resolve_model_type(source_model)
        awq_vlm_dataset_name = self._resolve_vlm_dataset_name(args)
        awq_vlm_quant_visual = self._resolve_bool(
            getattr(args, "awq_vlm_quant_visual", None),
            default=False,
        )
        awq_vlm_quant_connector = self._resolve_bool(
            getattr(args, "awq_vlm_quant_connector", None),
            default=False,
        )
        awq_vlm_quant_llm = self._resolve_bool(
            getattr(args, "awq_vlm_quant_llm", None),
            default=True,
        )
        visual_weight_bits = self._resolve_branch_weight_bits(args, "visual")
        connector_weight_bits = self._resolve_branch_weight_bits(args, "connector")
        llm_weight_bits = self._resolve_branch_weight_bits(args, "llm")
        if awq_vlm_dataset_name and model_type not in self._SUPPORTED_VLM_MODEL_TYPES:
            raise NotImplementedError(
                "Multimodal AWQ currently supports "
                f"{sorted(self._SUPPORTED_VLM_MODEL_TYPES)}, got {model_type!r}."
            )
        if awq_search_sequence_length <= 0:
            raise ValueError(
                f"awq_search_sequence_length must be positive, got {awq_search_sequence_length}."
            )
        if int(args.activation_bits) < 16:
            logger.warning(
                "Pure AWQ currently only fake-quantizes weights in this MindPipe path; "
                "requested activation_bits=%s will not add AWQ activation quantization.",
                args.activation_bits,
            )
        quantization_config = {
            "zero_point": not args.weight_symmetric,
            "q_group_size": args.weight_group_size,
        }
        runtime_device = resolve_device(args.device)
        if runtime_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        runtime_backend = backend_module(runtime_device)
        if runtime_backend is not None and hasattr(runtime_backend, "set_device"):
            runtime_backend.set_device(runtime_device)

        with prepend_python_path(source_root):
            from awq.quantize.auto_clip import apply_clip
            from awq.quantize.auto_clip import auto_clip_block
            from awq.quantize.auto_scale import apply_scale
            from awq.quantize.auto_scale import auto_scale_block
            from awq.quantize.pre_quant import apply_awq
            from awq.quantize.pre_quant import get_named_linears
            from awq.quantize.pre_quant import run_awq
            from awq.quantize.quantizer import pseudo_quantize_model_weight
            from awq.quantize.quantizer import pseudo_quantize_tensor

            awq_search_enabled = bool(args.awq_search)
            awq_search_reused = False
            awq_state = None
            enable_vlm_awq = bool(awq_vlm_dataset_name) and model_type in self._SUPPORTED_VLM_MODEL_TYPES
            run_text_awq = not enable_vlm_awq or awq_vlm_quant_llm

            if awq_search_enabled and run_text_awq:
                if awq_reuse_search_result and awq_state_path.exists():
                    logger.info("Reusing existing AWQ search state from %s.", awq_state_path)
                    awq_state = torch.load(awq_state_path, map_location="cpu")
                    apply_awq(model, awq_state, device=args.device)
                    awq_search_reused = True
                else:
                    if awq_search_sequence_length != int(args.sequence_length):
                        logger.info(
                            "AWQ search uses sequence length %s while the global sequence_length remains %s.",
                            awq_search_sequence_length,
                            args.sequence_length,
                        )
                    awq_state = run_awq(
                        model,
                        tokenizer_bundle.tokenizer,
                        w_bit=llm_weight_bits,
                        q_config=quantization_config,
                        n_samples=args.calibration_samples,
                        seqlen=awq_search_sequence_length,
                        auto_scale=awq_auto_scale,
                        mse_range=awq_mse_range,
                        clip_targets=awq_clip_targets,
                        qwen3_5_quantize_linear_attn=awq_qwen3_5_quantize_linear_attn,
                        calib_data=args.calibration_dataset,
                        device=args.device,
                        data_path=args.data_path,
                    )
                    torch.save(awq_state, awq_state_path)

            if run_text_awq:
                pseudo_quantize_model_weight(
                    model,
                    w_bit=llm_weight_bits,
                    q_config=quantization_config,
                    qwen3_5_quantize_linear_attn=awq_qwen3_5_quantize_linear_attn,
                    device=args.device,
                )

            vlm_multimodal_artifacts = None
            if enable_vlm_awq and (awq_vlm_quant_visual or awq_vlm_quant_connector):
                multimodal_awq_kwargs = dict(
                    source_model=source_model,
                    args=args,
                    quant_visual=awq_vlm_quant_visual,
                    quant_connector=awq_vlm_quant_connector,
                    quantization_config=quantization_config,
                    auto_scale_block=auto_scale_block,
                    apply_scale=apply_scale,
                    auto_clip_block=auto_clip_block,
                    apply_clip=apply_clip,
                    get_named_linears=get_named_linears,
                    pseudo_quantize_tensor=pseudo_quantize_tensor,
                    visual_weight_bits=visual_weight_bits,
                    connector_weight_bits=connector_weight_bits,
                    awq_auto_scale=awq_auto_scale if awq_search_enabled else False,
                    awq_mse_range=awq_mse_range if awq_search_enabled else False,
                    awq_clip_targets=awq_clip_targets,
                    dataset_name=awq_vlm_dataset_name,
                )
                if model_type in {"qwen2_vl", "qwen2_5_vl"}:
                    processor = getattr(tokenizer_bundle, "processor", None)
                    if processor is None:
                        raise ValueError(
                            f"Multimodal AWQ for {model_type} requires TokenizerBundle.processor."
                        )
                    multimodal_awq_kwargs["processor"] = processor
                    vlm_multimodal_artifacts = self._apply_qwen2_vl_multimodal_awq(
                        **multimodal_awq_kwargs,
                    )
                elif model_type == "qwen3_vl":
                    processor = getattr(tokenizer_bundle, "processor", None)
                    if processor is None:
                        raise ValueError(
                            f"Multimodal AWQ for {model_type} requires TokenizerBundle.processor."
                        )
                    multimodal_awq_kwargs["processor"] = processor
                    vlm_multimodal_artifacts = self._apply_qwen3_vl_multimodal_awq(
                        **multimodal_awq_kwargs,
                    )
                elif model_type == "minicpmv":
                    tokenizer = getattr(tokenizer_bundle, "tokenizer", None)
                    if tokenizer is None:
                        raise ValueError(
                            "Multimodal AWQ for MiniCPM-V requires TokenizerBundle.tokenizer."
                        )
                    multimodal_awq_kwargs["tokenizer"] = tokenizer
                    vlm_multimodal_artifacts = self._apply_minicpmv_multimodal_awq(
                        **multimodal_awq_kwargs,
                    )
                else:
                    raise NotImplementedError(
                        f"Multimodal AWQ dispatch not implemented for model_type={model_type!r}."
                    )

        artifacts = {
            "source_root": str(source_root),
            "quantization_config": quantization_config,
            "awq_search_enabled": awq_search_enabled,
            "awq_search_reused": awq_search_reused,
            "awq_search_sequence_length": awq_search_sequence_length,
            "awq_auto_scale": awq_auto_scale,
            "awq_mse_range": awq_mse_range,
            "awq_clip_targets": awq_clip_targets,
            "awq_qwen3_5_quantize_linear_attn": awq_qwen3_5_quantize_linear_attn,
            "awq_vlm_dataset_name": awq_vlm_dataset_name,
            "awq_vlm_quant_visual": awq_vlm_quant_visual,
            "awq_vlm_quant_connector": awq_vlm_quant_connector,
            "awq_vlm_quant_llm": awq_vlm_quant_llm,
            "resolved_weight_bits": {
                "visual": visual_weight_bits,
                "connector": connector_weight_bits,
                "llm": llm_weight_bits,
            },
        }
        if awq_state_path.exists():
            artifacts["awq_search_path"] = str(awq_state_path)
        if vlm_multimodal_artifacts is not None:
            artifacts["awq_vlm_multimodal"] = vlm_multimodal_artifacts
        return artifacts
