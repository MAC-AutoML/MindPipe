"""Unified QuaRot runner."""

from __future__ import annotations

import functools
import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import build_decoder_layer_groups
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import find_linear_layers
from ....common.modeling import get_text_backbone
from ....common.modeling import load_model_and_tokenizer
from ....common.modeling import unwrap_layer_output
from ....common.runtime import prepend_python_path

from ...base import BaseQuantizationMethod


class QuaRotMethod(BaseQuantizationMethod):
    name = "quarot"
    npu_ready = False  # Hadamard fallback 需在 NPU 上验证

    @staticmethod
    def _is_minicpm_like(model) -> bool:
        return getattr(getattr(model, "config", None), "model_type", None) == "minicpmv"

    @staticmethod
    def _try_get_hadK(hadamard_utils, size: int):
        try:
            return hadamard_utils.get_hadK(size)
        except (AssertionError, ValueError):
            return None, None

    def load_resources(self, args):
        force_eager = not self._should_use_runtime_path_for_args(args)
        return load_model_and_tokenizer(args.model_path, dtype=args.dtype, force_eager=force_eager)

    @staticmethod
    def _should_use_runtime_path_for_args(args) -> bool:
        return args.activation_bits >= 16 and args.key_bits >= 16 and args.value_bits >= 16

    @staticmethod
    def _quantization_root(model):
        if model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
            return model.language_model
        return model

    def _build_source_args(self, args) -> SimpleNamespace:
        return SimpleNamespace(
            model=args.model_path,
            seed=args.seed,
            hf_token=args.hf_token,
            rotate=True,
            rotate_mode=args.rotation_mode,
            optimized_rotation_path=args.rotation_checkpoint,
            fp32_had=False,
            w_bits=args.weight_bits,
            w_groupsize=args.weight_group_size,
            w_asym=not args.weight_symmetric,
            w_rtn=args.weight_method == "rtn",
            w_clip=False,
            nsamples=args.calibration_samples,
            percdamp=args.damp_percent,
            act_order=args.use_activation_order,
            int8_down_proj=False,
            a_bits=args.activation_bits,
            a_groupsize=args.activation_group_size,
            a_asym=not args.activation_symmetric,
            a_clip_ratio=1.0,
            k_bits=args.key_bits,
            k_groupsize=args.kv_group_size,
            k_asym=not args.key_symmetric,
            k_clip_ratio=1.0,
            k_pre_rope=False,
            v_bits=args.value_bits,
            v_groupsize=args.kv_group_size,
            v_asym=not args.value_symmetric,
            v_clip_ratio=1.0,
        )

    @staticmethod
    def _bind_rotation_device(rotation_utils, device: str) -> None:
        original_get_orthogonal_matrix = rotation_utils.get_orthogonal_matrix

        def get_orthogonal_matrix(size, mode, device_override=None):
            resolved_device = torch.device(device if device_override is None else device_override)
            return original_get_orthogonal_matrix(size, mode, device=resolved_device)

        rotation_utils.get_orthogonal_matrix = get_orthogonal_matrix

    @staticmethod
    def _ensure_forward_global(module, function_name: str) -> None:
        function_object = getattr(module, "forward").__func__
        if function_name in function_object.__globals__:
            return
        source_module = importlib.import_module(function_object.__module__)
        if not hasattr(source_module, function_name):
            raise KeyError(f"Missing {function_name} in {function_object.__module__}.")
        function_object.__globals__[function_name] = getattr(source_module, function_name)

    def _patch_model_utils(self, model_utils, model):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        qwen_model_types = (Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration)

        def _is_qwen_like(model) -> bool:
            return isinstance(model, qwen_model_types)

        def _is_llama_like(model) -> bool:
            return _is_qwen_like(model) or self._is_minicpm_like(model) or isinstance(model, model_utils.LLAMA_MODEL)

        def model_type_extractor(model):
            if _is_llama_like(model):
                return model_utils.LLAMA_MODEL
            if isinstance(model, model_utils.OPT_MODEL):
                return model_utils.OPT_MODEL
            raise ValueError(f"Unknown model type {model}")

        def get_model_type(model):
            if _is_llama_like(model):
                return model_utils.LLAMA_MODEL
            if isinstance(model, model_utils.OPT_MODEL):
                return model_utils.OPT_MODEL
            raise ValueError(f"Unknown model type {model}")

        def get_rope_function_name(model):
            if _is_llama_like(model):
                return "apply_rotary_pos_emb"
            raise NotImplementedError

        def get_layers(model):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                return backbone.layers
            return model_utils.get_layers.__wrapped__(model)  # type: ignore[attr-defined]

        def get_embeddings(model, model_type):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                return [backbone.root.embed_tokens]
            return model_utils.get_embeddings.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        def get_transformer_layers(model, model_type):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                return list(backbone.layers)
            return model_utils.get_transformer_layers.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        def get_lm_head(model, model_type):
            if _is_llama_like(model):
                return model.lm_head
            return model_utils.get_lm_head.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        def get_pre_head_layernorm(model, model_type):
            if _is_llama_like(model):
                backbone = get_text_backbone(model)
                return backbone.root.norm
            return model_utils.get_pre_head_layernorm.__wrapped__(model, model_type)  # type: ignore[attr-defined]

        model_utils.model_type_extractor = functools.wraps(model_utils.model_type_extractor)(model_type_extractor)
        model_utils.get_model_type = functools.wraps(model_utils.get_model_type)(get_model_type)
        model_utils.get_rope_function_name = functools.wraps(model_utils.get_rope_function_name)(get_rope_function_name)
        model_utils.get_layers = functools.wraps(model_utils.get_layers)(get_layers)
        model_utils.get_embeddings = functools.wraps(model_utils.get_embeddings)(get_embeddings)
        model_utils.get_transformer_layers = functools.wraps(model_utils.get_transformer_layers)(get_transformer_layers)
        model_utils.get_lm_head = functools.wraps(model_utils.get_lm_head)(get_lm_head)
        model_utils.get_pre_head_layernorm = functools.wraps(model_utils.get_pre_head_layernorm)(get_pre_head_layernorm)
        extra_norm_types = [Qwen2RMSNorm]
        try:
            extra_norm_types.append(get_text_backbone(model).layers[0].input_layernorm.__class__)
        except Exception:
            pass
        return tuple(dict.fromkeys(extra_norm_types))

    def _fuse_layer_norms(self, model, model_utils, rotation_utils, extra_norm_types):
        import transformers

        model_type = model_utils.get_model_type(model)
        kwargs = {"model": model, "model_type": model_type}

        for embedding in model_utils.get_embeddings(**kwargs):
            weight = embedding.weight.data.double()
            embedding.weight.data = (weight - weight.mean(dim=-1, keepdim=True)).to(embedding.weight.data.dtype)

        for layer in model_utils.get_transformer_layers(**kwargs):
            rotation_utils.fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            rotation_utils.fuse_ln_linear(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            )

        rotation_utils.fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
        norm_types = (transformers.models.llama.modeling_llama.LlamaRMSNorm, *extra_norm_types)
        model_utils.replace_modules(
            model,
            norm_types,
            lambda _: model_utils.RMSN(model.config.hidden_size),
            replace_layers=False,
        )

    def _configure_activation_quantizers(
        self,
        model,
        source_args,
        quant_utils,
        hadamard_utils,
        rotation_utils,
        model_utils,
        ref_utils,
    ) -> None:
        quantization_root = self._quantization_root(model)
        qlayers = quant_utils.find_qlayers(quantization_root)
        if source_args.rotate:
            for layer_name, qlayer in qlayers.items():
                if "down_proj" in layer_name:
                    had_k, k_value = self._try_get_hadK(hadamard_utils, model.config.intermediate_size)
                    if k_value is not None:
                        qlayer.online_full_had = True
                        qlayer.had_K = had_k
                        qlayer.K = k_value
                        qlayer.fp32_had = source_args.fp32_had
                if "o_proj" in layer_name:
                    had_k, k_value = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayer.online_partial_had = True
                    qlayer.had_K = had_k
                    qlayer.K = k_value
                    qlayer.had_dim = model.config.hidden_size // model.config.num_attention_heads
                    qlayer.fp32_had = source_args.fp32_had

        if source_args.a_bits < 16 or source_args.v_bits < 16:
            down_proj_groupsize = -1
            if source_args.a_groupsize > 0:
                down_proj_groupsize = ref_utils.llama_down_proj_groupsize(quantization_root, source_args.a_groupsize)

            act_qlayers = quant_utils.find_qlayers(quantization_root, layers=[quant_utils.ActQuantWrapper])
            for layer_name, qlayer in act_qlayers.items():
                layer_input_bits = source_args.a_bits
                layer_groupsize = source_args.a_groupsize
                if "v_proj" in layer_name and source_args.v_bits < 16:
                    qlayer.out_quantizer.configure(
                        bits=source_args.v_bits,
                        groupsize=source_args.v_groupsize,
                        sym=not source_args.v_asym,
                        clip_ratio=source_args.v_clip_ratio,
                    )
                if "lm_head" in layer_name:
                    layer_input_bits = 16
                if "down_proj" in layer_name:
                    layer_groupsize = down_proj_groupsize
                qlayer.quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=not source_args.a_asym,
                    clip_ratio=source_args.a_clip_ratio,
                )

        if source_args.k_bits < 16:
            if source_args.k_pre_rope:
                raise NotImplementedError("QuaRot pre-RoPE key quantization is not supported.")
            rope_function_name = model_utils.get_rope_function_name(model)
            backbone = get_text_backbone(model)
            k_quant_config = {
                "k_bits": source_args.k_bits,
                "k_groupsize": source_args.k_groupsize,
                "k_sym": not source_args.k_asym,
                "k_clip_ratio": source_args.k_clip_ratio,
            }
            for layer in backbone.layers:
                self._ensure_forward_global(layer.self_attn, rope_function_name)
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    @staticmethod
    def _should_use_runtime_path(model, args) -> bool:
        model_type = getattr(model.config, "model_type", None)
        return model_type in {"qwen2", "qwen2_5_vl"} and QuaRotMethod._should_use_runtime_path_for_args(args)

    @staticmethod
    def _get_extra_rotation_modules(model):
        merger = getattr(getattr(model, "visual", None), "merger", None)
        if (
            getattr(model.config, "model_type", None) == "qwen2_5_vl"
            and merger is not None
            and hasattr(merger, "mlp")
            and isinstance(merger.mlp[-1], torch.nn.Linear)
        ):
            return [merger.mlp[-1]]
        return []

    @staticmethod
    def _neutralize_norm(norm) -> None:
        if hasattr(norm, "weight") and norm.weight is not None:
            norm.weight.data.fill_(1)
        if hasattr(norm, "bias") and norm.bias is not None:
            norm.bias.data.zero_()

    def _fuse_norm_into_linears(self, norm, linears) -> None:
        if not hasattr(norm, "weight") or norm.weight is None:
            return
        scale = norm.weight.data.double()
        bias = getattr(norm, "bias", None)
        for linear in linears:
            weight = linear.weight.data.double()
            linear.weight.data = (weight * scale.view(1, -1)).to(linear.weight.dtype)
            if bias is not None:
                if linear.bias is None:
                    linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
                linear.bias.data = (
                    linear.bias.data.double() + torch.matmul(weight, bias.data.double())
                ).to(linear.weight.dtype)
        self._neutralize_norm(norm)

    def _apply_runtime_rotation(self, model, backbone, args, quarot_runtime) -> None:
        rotation_device = torch.device(args.device)
        q_matrix = quarot_runtime.functional.random_hadamard_matrix(backbone.hidden_size, device=rotation_device)

        embed = backbone.embed_tokens
        if embed is None:
            raise AttributeError("QuaRot runtime path requires embedding layer.")
        embed_weight = embed.weight.data.double()
        embed.weight.data = (embed_weight - embed_weight.mean(dim=-1, keepdim=True)).to(embed.weight.dtype)

        for block in backbone.layers:
            self._fuse_norm_into_linears(
                block.input_layernorm,
                [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj],
            )
            self._fuse_norm_into_linears(
                block.post_attention_layernorm,
                [block.mlp.gate_proj, block.mlp.up_proj],
            )

        head = None
        if hasattr(model, "lm_head"):
            head = model.lm_head
            if head.weight.data_ptr() == embed.weight.data_ptr():
                head.weight = torch.nn.Parameter(embed.weight.detach().clone())
            if backbone.final_norm is not None:
                self._fuse_norm_into_linears(backbone.final_norm, [head])

        embed = embed.to(rotation_device)
        embed.weight.data = torch.matmul(embed.weight.data.double(), q_matrix).to(embed.weight.dtype)
        backbone.root.embed_tokens = embed.cpu()

        if head is not None:
            head = head.to(rotation_device)
            head.weight.data = torch.matmul(head.weight.data.double(), q_matrix).to(head.weight.dtype)
            if hasattr(model, "lm_head"):
                model.lm_head = head.cpu()

        for module in self._get_extra_rotation_modules(model):
            module = module.to(rotation_device)
            module.weight.data = torch.matmul(q_matrix.T, module.weight.data.double()).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data = torch.matmul(q_matrix.T, module.bias.data.double()).to(module.bias.dtype)
            module.cpu()

        for layer_index, block in enumerate(backbone.layers):
            block = block.to(rotation_device)
            for linear in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
                linear.weight.data = torch.matmul(linear.weight.data.double(), q_matrix).to(linear.weight.dtype)

            block.self_attn.o_proj.weight.data = torch.matmul(
                q_matrix.T,
                block.self_attn.o_proj.weight.data.double(),
            ).to(block.self_attn.o_proj.weight.dtype)
            if block.self_attn.o_proj.bias is not None:
                block.self_attn.o_proj.bias.data = torch.matmul(
                    q_matrix.T,
                    block.self_attn.o_proj.bias.data.double(),
                ).to(block.self_attn.o_proj.bias.dtype)
            quarot_runtime.functional.apply_exact_had_to_linear(block.self_attn.o_proj, had_dim=-1, output=False, device=args.device)

            for linear in (block.mlp.gate_proj, block.mlp.up_proj):
                linear.weight.data = torch.matmul(linear.weight.data.double(), q_matrix).to(linear.weight.dtype)

            block.mlp.down_proj.weight.data = torch.matmul(
                q_matrix.T,
                block.mlp.down_proj.weight.data.double(),
            ).to(block.mlp.down_proj.weight.dtype)
            if block.mlp.down_proj.bias is not None:
                block.mlp.down_proj.bias.data = torch.matmul(
                    q_matrix.T,
                    block.mlp.down_proj.bias.data.double(),
                ).to(block.mlp.down_proj.bias.dtype)
            quarot_runtime.functional.apply_exact_had_to_linear(block.mlp.down_proj, had_dim=-1, output=False, device=args.device)

            backbone.layers[layer_index] = block.cpu()
            del block
            empty_cache(args.device)

    def _apply_runtime_gptq_quantization(
        self,
        model,
        backbone,
        calibration_batches,
        gptq_utils,
        quant_utils,
        args,
    ) -> dict[str, object]:
        input_states, layer_kwargs = capture_first_block_inputs(
            model=model,
            backbone=backbone,
            calibration_batches=calibration_batches,
            device=args.device,
        )
        output_states = torch.zeros_like(input_states)
        quantizer_artifacts: dict[str, object] = {}

        for layer_index, block in enumerate(backbone.layers):
            block = block.to(args.device)
            linear_layers = find_linear_layers(block)
            layer_groups = build_decoder_layer_groups(block, set(linear_layers))

            for group in layer_groups:
                subset = {name: linear_layers[name] for name in group}
                gptq_states = {}
                for layer_name, linear in subset.items():
                    gptq_state = gptq_utils.GPTQ(linear)
                    gptq_state.quantizer = quant_utils.WeightQuantizer()
                    gptq_state.quantizer.configure(
                        args.weight_bits,
                        perchannel=True,
                        sym=args.weight_symmetric,
                        mse=False,
                    )
                    gptq_states[layer_name] = gptq_state

                def add_batch(layer_name: str):
                    def hook(_module, inputs, outputs):
                        gptq_states[layer_name].add_batch(
                            torch.nan_to_num(inputs[0].data, nan=0.0, posinf=0.0, neginf=0.0),
                            torch.nan_to_num(outputs.data, nan=0.0, posinf=0.0, neginf=0.0),
                        )

                    return hook

                handles = [subset[layer_name].register_forward_hook(add_batch(layer_name)) for layer_name in subset]
                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = torch.nan_to_num(
                            unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            ),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                for handle in handles:
                    handle.remove()

                for layer_name, gptq_state in gptq_states.items():
                    gptq_state.H = torch.nan_to_num(gptq_state.H, nan=0.0, posinf=0.0, neginf=0.0)
                    gptq_state.H = 0.5 * (gptq_state.H + gptq_state.H.T)
                    original_weight = gptq_state.layer.weight.data.clone()
                    hessian_snapshot = gptq_state.H.clone()
                    damp_schedule = []
                    for damp_value in (args.damp_percent, 0.05, 0.1, 0.25, 1.0):
                        if damp_value not in damp_schedule:
                            damp_schedule.append(damp_value)
                    last_error = None
                    for damp_value in damp_schedule:
                        try:
                            gptq_state.layer.weight.data = original_weight.clone()
                            gptq_state.H = hessian_snapshot.clone()
                            gptq_state.fasterquant(
                                percdamp=damp_value,
                                groupsize=args.weight_group_size,
                                actorder=args.use_activation_order,
                                static_groups=args.static_groups,
                            )
                            last_error = None
                            break
                        except RuntimeError as error:
                            if "not positive-definite" not in str(error):
                                raise
                            last_error = error
                    if last_error is not None:
                        gptq_state.layer.weight.data = original_weight
                        raise last_error
                    quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{layer_name}"] = {
                        "bits": args.weight_bits,
                        "group_size": args.weight_group_size,
                        "symmetric": args.weight_symmetric,
                    }
                    gptq_state.free()
                del gptq_states
                empty_cache(args.device)

            for sample_index in range(args.calibration_samples):
                with torch.no_grad():
                    output_states[sample_index] = torch.nan_to_num(
                        unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

            backbone.layers[layer_index] = block.cpu()
            del block
            empty_cache(args.device)
            input_states, output_states = output_states, input_states

        return quantizer_artifacts

    def _apply_rtn_quantization(self, backbone, quant_utils, args) -> dict[str, object]:
        if args.weight_group_size != -1:
            raise NotImplementedError("QuaRot RTN only supports per-channel weights; set --weight_group_size -1.")

        quantizer_artifacts: dict[str, object] = {}
        for layer_index, block in enumerate(backbone.layers):
            block = block.to(args.device)
            qlayers = quant_utils.find_qlayers(block, layers=[quant_utils.ActQuantWrapper])
            for layer_name, qlayer in qlayers.items():
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(
                    args.weight_bits,
                    perchannel=True,
                    sym=args.weight_symmetric,
                    mse=False,
                )
                weights = qlayer.weight.data
                quantizer.find_params(weights)
                qlayer.weight.data = quantizer.quantize(weights).to(weights.dtype)
                quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{layer_name}"] = {
                    "bits": args.weight_bits,
                    "group_size": -1,
                    "symmetric": args.weight_symmetric,
                }
            backbone.layers[layer_index] = block.cpu()
            del block
            empty_cache(args.device)
        return quantizer_artifacts

    def _apply_gptq_quantization(
        self,
        model,
        backbone,
        calibration_batches,
        gptq_utils,
        quant_utils,
        args,
    ) -> dict[str, object]:
        input_states, layer_kwargs = capture_first_block_inputs(
            model=model,
            backbone=backbone,
            calibration_batches=calibration_batches,
            device=args.device,
        )
        output_states = torch.zeros_like(input_states)
        quantizer_artifacts: dict[str, object] = {}

        for layer_index, block in enumerate(backbone.layers):
            block = block.to(args.device)
            qlayers = quant_utils.find_qlayers(block, layers=[quant_utils.ActQuantWrapper])
            layer_groups = build_decoder_layer_groups(block, set(qlayers))

            for group in layer_groups:
                subset = {name: qlayers[name] for name in group if name in qlayers}
                gptq_states = {}
                for layer_name, qlayer in subset.items():
                    gptq_state = gptq_utils.GPTQ(qlayer)
                    gptq_state.quantizer = quant_utils.WeightQuantizer()
                    gptq_state.quantizer.configure(
                        args.weight_bits,
                        perchannel=True,
                        sym=args.weight_symmetric,
                        mse=False,
                    )
                    gptq_states[layer_name] = gptq_state

                def add_batch(layer_name: str):
                    def hook(_module, inputs, outputs):
                        gptq_states[layer_name].add_batch(
                            torch.nan_to_num(inputs[0].data, nan=0.0, posinf=0.0, neginf=0.0),
                            torch.nan_to_num(outputs.data, nan=0.0, posinf=0.0, neginf=0.0),
                        )

                    return hook

                handles = [subset[layer_name].register_forward_hook(add_batch(layer_name)) for layer_name in subset]
                for sample_index in range(args.calibration_samples):
                    with torch.no_grad():
                        output_states[sample_index] = torch.nan_to_num(
                            unwrap_layer_output(
                                block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                            ),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                for handle in handles:
                    handle.remove()

                for layer_name, gptq_state in gptq_states.items():
                    gptq_state.H = torch.nan_to_num(gptq_state.H, nan=0.0, posinf=0.0, neginf=0.0)
                    gptq_state.H = 0.5 * (gptq_state.H + gptq_state.H.T)
                    original_weight = gptq_state.layer.weight.data.clone()
                    hessian_snapshot = gptq_state.H.clone()
                    damp_schedule = []
                    for damp_value in (args.damp_percent, 0.05, 0.1, 0.25, 1.0):
                        if damp_value not in damp_schedule:
                            damp_schedule.append(damp_value)
                    last_error = None
                    for damp_value in damp_schedule:
                        try:
                            gptq_state.layer.weight.data = original_weight.clone()
                            gptq_state.H = hessian_snapshot.clone()
                            gptq_state.fasterquant(
                                percdamp=damp_value,
                                groupsize=args.weight_group_size,
                                actorder=args.use_activation_order,
                                static_groups=args.static_groups,
                            )
                            last_error = None
                            break
                        except RuntimeError as error:
                            if "not positive-definite" not in str(error):
                                raise
                            last_error = error
                    if last_error is not None:
                        gptq_state.layer.weight.data = original_weight
                        raise last_error
                    quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{layer_name}"] = {
                        "bits": args.weight_bits,
                        "group_size": args.weight_group_size,
                        "symmetric": args.weight_symmetric,
                    }
                    gptq_state.free()
                del gptq_states
                empty_cache(args.device)

            for sample_index in range(args.calibration_samples):
                with torch.no_grad():
                    output_states[sample_index] = torch.nan_to_num(
                        unwrap_layer_output(
                            block(input_states[sample_index].unsqueeze(0), **layer_kwargs)
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

            backbone.layers[layer_index] = block.cpu()
            del block
            empty_cache(args.device)
            input_states, output_states = output_states, input_states

        return quantizer_artifacts

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        source_root = Path(__file__).resolve().parent / "source"
        source_args = self._build_source_args(args)
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
        )
        backbone = get_text_backbone(model)

        with prepend_python_path(source_root):
            if self._should_use_runtime_path(model, args):
                import quarot
                import gptq_utils
                import quant_utils
                import runtime_models

                self._apply_runtime_rotation(model, backbone, args, quarot)
                runtime_models.install_runtime_quarot_layers(model)

                quantizer_artifacts: dict[str, object] = {}
                weight_quantizer_name = "none"
                if args.weight_bits < 16:
                    if args.weight_method == "rtn":
                        raise NotImplementedError("QuaRot runtime path currently supports GPTQ weights only.")
                    quantizer_artifacts = self._apply_runtime_gptq_quantization(
                        model,
                        backbone,
                        calibration_batches,
                        gptq_utils,
                        quant_utils,
                        args,
                    )
                    weight_quantizer_name = "gptq"

                return {
                    "source_root": str(source_root),
                    "quarot_config": {
                        "rotate": True,
                        "rotation_mode": source_args.rotate_mode,
                        "rotation_checkpoint": source_args.optimized_rotation_path,
                        "weight_bits": source_args.w_bits,
                        "activation_bits": source_args.a_bits,
                        "key_bits": source_args.k_bits,
                        "value_bits": source_args.v_bits,
                        "weight_group_size": source_args.w_groupsize,
                        "activation_group_size": source_args.a_groupsize,
                        "kv_group_size": source_args.k_groupsize,
                        "weight_quantizer": weight_quantizer_name,
                        "calibration_samples": source_args.nsamples,
                        "source_variant": "runtime_weight_only",
                    },
                    "quantized_linear_count": len(quantizer_artifacts),
                    "quantized_linear_layers": quantizer_artifacts,
                }

            import gptq_utils
            import hadamard_utils
            import model_utils
            import quant_utils
            import rotation_utils
            import utils as ref_utils

            ref_utils.DEV = torch.device(args.device)
            if ref_utils.DEV.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            extra_norm_types = self._patch_model_utils(model_utils, model)
            self._bind_rotation_device(rotation_utils, args.device)

            if source_args.rotate:
                self._fuse_layer_norms(model, model_utils, rotation_utils, extra_norm_types)
                rotation_utils.rotate_model(model, source_args)
                ref_utils.cleanup_memory(verbos=False)

            quant_utils.add_actquant(self._quantization_root(model))
            self._configure_activation_quantizers(
                model,
                source_args,
                quant_utils,
                hadamard_utils,
                rotation_utils,
                model_utils,
                ref_utils,
            )

            quantizer_artifacts: dict[str, object] = {}
            weight_quantizer_name = "none"
            if args.weight_bits < 16:
                if args.weight_method == "rtn":
                    quantizer_artifacts = self._apply_rtn_quantization(backbone, quant_utils, args)
                    weight_quantizer_name = "rtn"
                else:
                    quantizer_artifacts = self._apply_gptq_quantization(
                        model,
                        backbone,
                        calibration_batches,
                        gptq_utils,
                        quant_utils,
                        args,
                    )
                    weight_quantizer_name = "gptq"

        return {
            "source_root": str(source_root),
            "quarot_config": {
                "rotate": source_args.rotate,
                "rotation_mode": source_args.rotate_mode,
                "rotation_checkpoint": source_args.optimized_rotation_path,
                "weight_bits": source_args.w_bits,
                "activation_bits": source_args.a_bits,
                "key_bits": source_args.k_bits,
                "value_bits": source_args.v_bits,
                "weight_group_size": source_args.w_groupsize,
                "activation_group_size": source_args.a_groupsize,
                "kv_group_size": source_args.k_groupsize,
                "weight_quantizer": weight_quantizer_name,
                "calibration_samples": source_args.nsamples,
            },
            "quantized_linear_count": len(quantizer_artifacts),
            "quantized_linear_layers": quantizer_artifacts,
        }
