"""Unified SpinQuant runner."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import build_decoder_layer_groups
from ....common.modeling import capture_first_block_inputs
from ....common.modeling import get_text_backbone
from ....common.modeling import load_model_and_tokenizer
from ....common.modeling import unwrap_layer_output
from ....common.runtime import prepend_python_path

from ...base import BaseQuantizationMethod


class SpinQuantMethod(BaseQuantizationMethod):
    name = "spinquant"

    def load_resources(self, args):
        return load_model_and_tokenizer(args.model_path, dtype=args.dtype, force_eager=False)

    def _build_source_args(self, args) -> SimpleNamespace:
        return SimpleNamespace(
            input_model=args.model_path,
            seed=args.seed,
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
            export_to_et=False,
            load_qmodel_path=None,
            save_qmodel_path=None,
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
    def _is_qwen_like(model) -> bool:
        return getattr(model.config, "model_type", None) in {"qwen2", "qwen2_5_vl"}

    @staticmethod
    def _random_orthogonal_matrix(size: int) -> torch.Tensor:
        q_matrix, r_matrix = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64))
        q_matrix *= torch.sign(torch.diag(r_matrix)).unsqueeze(0)
        return q_matrix

    def _materialize_default_rotation_checkpoint(
        self,
        model,
        backbone,
        args,
        hadamard_utils,
    ) -> str:
        output_dir = self.resolve_output_dir(args)
        rotation_dir = output_dir / "rotation"
        rotation_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = rotation_dir / f"{self.name}_{args.rotation_mode}_identity_r2.bin"

        hidden_size = backbone.hidden_size
        num_layers = len(backbone.layers)
        num_heads = model.config.num_attention_heads
        head_dim = hidden_size // num_heads
        layer_key_prefix = f"{backbone.prefix}.layers"
        expected_key = f"{layer_key_prefix}.0.self_attn.R2"

        if checkpoint_path.exists():
            try:
                existing_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "R1" in existing_checkpoint and expected_key in existing_checkpoint:
                    return str(checkpoint_path)
            except Exception:
                pass

        if args.rotation_mode == "hadamard":
            r1 = hadamard_utils.random_hadamard_matrix(hidden_size, "cpu").to(torch.float32)
        elif args.rotation_mode == "random":
            r1 = self._random_orthogonal_matrix(hidden_size).to(torch.float32)
        else:
            raise ValueError(f"Unsupported SpinQuant rotation mode: {args.rotation_mode}")

        payload = {"R1": r1}
        identity_r2 = torch.eye(head_dim, dtype=torch.float32)
        for layer_index in range(num_layers):
            payload[f"{layer_key_prefix}.{layer_index}.self_attn.R2"] = identity_r2.clone()
        torch.save(payload, checkpoint_path)
        return str(checkpoint_path)

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

    def _apply_rtn_quantization(self, backbone, quant_utils, args) -> dict[str, object]:
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
                    weight_groupsize=args.weight_group_size,
                )
                weights = qlayer.weight.data
                quantizer.find_params(weights)
                q_weight, _int_weight, _scale = quantizer.fake_quantize(weights)
                qlayer.weight.data = q_weight.to(weights.dtype)
                quantizer_artifacts[f"{backbone.prefix}.layers.{layer_index}.{layer_name}"] = {
                    "bits": args.weight_bits,
                    "group_size": args.weight_group_size,
                    "symmetric": args.weight_symmetric,
                }
            backbone.layers[layer_index] = block.cpu()
            del block
            torch.cuda.empty_cache()
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
                        weight_groupsize=-1,
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
                torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()
            input_states, output_states = output_states, input_states

        return quantizer_artifacts

    def _configure_activation_quantizers(
        self,
        model,
        backbone,
        source_args,
        quant_utils,
        hadamard_utils,
        rotation_utils,
        ref_utils,
    ) -> None:
        backbone_root = backbone.root
        qlayers = quant_utils.find_qlayers(backbone_root)
        if source_args.rotate:
            for layer_name, qlayer in qlayers.items():
                if "down_proj" in layer_name:
                    had_k, k_value = hadamard_utils.get_hadK(backbone_root.config.intermediate_size)
                    qlayer.online_full_had = True
                    qlayer.had_K = had_k
                    qlayer.K = k_value
                    qlayer.fp32_had = source_args.fp32_had

        if source_args.a_bits < 16 or source_args.v_bits < 16:
            act_qlayers = quant_utils.find_qlayers(backbone_root, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if source_args.a_groupsize > 0:
                down_proj_groupsize = ref_utils.llama_down_proj_groupsize(backbone_root, source_args.a_groupsize)

            num_heads = backbone_root.config.num_attention_heads
            head_dim = backbone_root.config.hidden_size // num_heads
            for layer_name, qlayer in act_qlayers.items():
                layer_input_bits = source_args.a_bits
                layer_groupsize = source_args.a_groupsize
                if "v_proj" in layer_name and source_args.v_bits < 16:
                    qlayer.out_quantizer.configure(
                        bits=source_args.v_bits,
                        groupsize=head_dim,
                        sym=not source_args.v_asym,
                        clip_ratio=source_args.v_clip_ratio,
                    )
                if "o_proj" in layer_name:
                    layer_groupsize = head_dim
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
                raise NotImplementedError("SpinQuant pre-RoPE key quantization is not supported.")
            rope_function_name = "apply_multimodal_rotary_pos_emb"
            if getattr(backbone_root.config, "model_type", None) != "qwen2_5_vl":
                rope_function_name = "apply_rotary_pos_emb"
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
                    config=backbone_root.config,
                    **k_quant_config,
                )

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
            from eval_utils import gptq_utils
            from eval_utils import rotation_utils
            from utils import fuse_norm_utils
            from utils import hadamard_utils
            from utils import quant_utils
            from utils import utils as ref_utils

            ref_utils.DEV = torch.device(args.device)
            self._bind_rotation_device(rotation_utils, args.device)

            effective_rotation_checkpoint = source_args.optimized_rotation_path
            rotation_fallback = None
            if effective_rotation_checkpoint is None and self._is_qwen_like(model):
                effective_rotation_checkpoint = self._materialize_default_rotation_checkpoint(
                    model=model,
                    backbone=backbone,
                    args=args,
                    hadamard_utils=hadamard_utils,
                )
                source_args.optimized_rotation_path = effective_rotation_checkpoint
                rotation_fallback = "identity_r2"

            if source_args.rotate:
                fuse_norm_utils.fuse_layer_norms(model)
                rotation_utils.rotate_model(model, source_args)
                ref_utils.cleanup_memory(verbos=False)

            quant_utils.add_actquant(backbone.root)
            self._configure_activation_quantizers(
                model,
                backbone,
                source_args,
                quant_utils,
                hadamard_utils,
                rotation_utils,
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
            "spinquant_config": {
                "rotate": source_args.rotate,
                "rotation_mode": source_args.rotate_mode,
                "rotation_checkpoint": source_args.optimized_rotation_path,
                "rotation_fallback": rotation_fallback,
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
