"""Unified SliderQuant runner.

This adapter vendors the upstream SliderQuant implementation under ``source/``
and drives its core ``sliderquant`` function with the model/tokenizer already
loaded by MindPipe.  The first pass intentionally targets text-only
Llama/Qwen-family causal LMs; multimodal branches need a separate mapping layer.
"""

from __future__ import annotations

import importlib
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.runtime import prepend_python_path
from ...base import BaseQuantizationMethod


LOGGER = logging.getLogger(__name__)


def _purge_conflicting_modules(module_names: tuple[str, ...], allowed_root: Path) -> None:
    import sys

    allowed_root = allowed_root.resolve()
    for module_name in module_names:
        prefix = f"{module_name}."
        for name, module in list(sys.modules.items()):
            if name != module_name and not name.startswith(prefix):
                continue

            module_file = getattr(module, "__file__", None)
            module_path = getattr(module, "__path__", None)
            candidates: list[Path] = []
            if module_file:
                candidates.append(Path(module_file))
            if module_path:
                candidates.extend(Path(path) for path in module_path)

            if any(path.resolve().is_relative_to(allowed_root) for path in candidates):
                continue
            sys.modules.pop(name, None)


class _MindPipeSliderLM:
    def __init__(self, model, tokenizer, device: torch.device, seqlen: int, batch_size: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.seqlen = seqlen
        self.batch_size_per_gpu = batch_size
        self.vocab_size = getattr(tokenizer, "vocab_size", None)

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu


class SliderQuantMethod(BaseQuantizationMethod):
    name = "sliderquant"
    npu_ready = False  # Upstream kernels and training flow have not been NPU-smoked yet.
    default_calibration_dataset = "wikitext2"

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        rotate_suffix = "_rot" if getattr(args, "sliderquant_rotate", False) else ""
        run_spec = (
            f"{self.name}_w{args.weight_bits}a{args.activation_bits}"
            f"_win{args.sliderquant_num_layer}"
            f"_step{args.sliderquant_quant_step}"
            f"_seq{args.sequence_length}"
            f"{rotate_suffix}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    @staticmethod
    def _resolve_text_model(model):
        if type(model).__name__ == "TextModelAdapter" and hasattr(model, "text_model"):
            return model.text_model
        return model

    @staticmethod
    def _infer_net_name(model_path: str, model) -> str:
        model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
        path_name = Path(str(model_path).rstrip("/")).name
        if "qwen" in model_type or "qwen" in path_name.lower():
            return path_name or "Qwen"
        if "llama" in model_type or "llama" in path_name.lower():
            return path_name or "Llama"
        if "vicuna" in path_name.lower():
            return path_name
        return path_name or model_type or "model"

    @staticmethod
    def _validate_model(model) -> None:
        if type(model).__name__ == "TextModelAdapter" and hasattr(model, "text_model"):
            raise ValueError(
                "SliderQuant adapter currently supports text-only causal LMs. "
                "Use a text model path, not a multimodal wrapper, for this first integration."
            )
        model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
        if not any(key in model_type for key in ("llama", "qwen")):
            raise NotImplementedError(
                f"SliderQuant text adapter currently supports Llama/Qwen-like model_type, got {model_type!r}."
            )
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise NotImplementedError(
                "SliderQuant upstream code expects model.model.layers; this model layout is not supported yet."
            )

    @staticmethod
    def _validate_args(args) -> None:
        if args.sliderquant_teach_model is not None:
            raise NotImplementedError(
                "SliderQuant teacher-model distillation is not wired into MindPipe yet; "
                "leave --sliderquant_teach_model unset for now."
            )
        if args.sliderquant_quant_step <= 0:
            raise ValueError("--sliderquant_quant_step must be positive.")
        if args.sliderquant_num_layer <= 0:
            raise ValueError("--sliderquant_num_layer must be positive.")
        if args.sliderquant_max_layers is not None and args.sliderquant_max_layers <= 0:
            raise ValueError("--sliderquant_max_layers must be positive when set.")
        if args.sliderquant_epochs > 0 and not (args.sliderquant_lwc or args.sliderquant_let):
            raise ValueError("SliderQuant training requires at least one of --sliderquant_lwc or --sliderquant_let.")

    def _build_source_args(self, args, output_dir: Path, model) -> SimpleNamespace:
        weight_bits = int(args.weight_bits)
        activation_bits = int(args.activation_bits)
        quantize = bool(weight_bits < 16 or activation_bits < 16 or args.sliderquant_quant_warp)
        dtype_name = str(args.dtype).lower()
        use_bfloat16 = bool(dtype_name == "bfloat16")
        net_name = args.sliderquant_net or self._infer_net_name(args.model_path, model)
        group_size = args.weight_group_size if args.weight_group_size and args.weight_group_size > 0 else None
        quant_rate_list = None
        if args.sliderquant_quant_rate_list:
            quant_rate_list = [float(value) for value in str(args.sliderquant_quant_rate_list).split(",") if value.strip()]

        return SimpleNamespace(
            abits=activation_bits,
            act_symmetric=bool(args.activation_symmetric),
            a_dynamic_method="per_token",
            attn_implementation=args.attn_implementation,
            auto_lr_scale=bool(args.sliderquant_auto_lr_scale),
            batch_size=int(args.sliderquant_batch_size),
            cache_dir=str(output_dir / "cache"),
            calib_dataset=args.calibration_dataset,
            circular_aug=bool(args.sliderquant_circular_aug),
            config=None,
            deactive_amp=bool(args.sliderquant_deactive_amp),
            debug=bool(args.sliderquant_debug),
            disable_zero_point=not bool(args.weight_symmetric),
            epochs=int(args.sliderquant_epochs),
            eval_ppl=False,
            export_model_mode="quant",
            export_model_path=None,
            fill_end_window_size=None,
            fill_start_window_size=None,
            fill_window_size=args.sliderquant_fill_window_size,
            fp16_act=bool(args.sliderquant_fp16_act),
            gqa_scales=str(args.sliderquant_gqa_scales),
            grad_clip=args.sliderquant_grad_clip,
            group_size=group_size,
            huber_loss_max=float(args.sliderquant_huber_loss_max),
            inference_batch_size=int(args.sliderquant_inference_batch_size),
            items=None,
            last_round_inp_num=int(args.sliderquant_last_round_inp_num),
            layers_assigned_gpu=args.sliderquant_layers_assigned_gpu,
            layer_windows_scheduler=args.sliderquant_layer_windows_scheduler,
            let=bool(args.sliderquant_let),
            limit=-1,
            littlt_bs_round=args.sliderquant_little_bs_round,
            lm_eval_batch_size="64",
            load_rotate_model_path=None,
            lora_iter_num_list=None,
            lora_layer_list=None,
            lora_lr=float(args.sliderquant_lora_lr),
            lora_quant=bool(args.sliderquant_lora_quant),
            lora_rank=int(args.sliderquant_lora_rank),
            lora_r_list=None,
            loss_function=str(args.sliderquant_loss_function),
            loss_type=str(args.sliderquant_loss_type),
            low_cpu_memory=bool(args.sliderquant_low_cpu_memory),
            low_memory=bool(args.sliderquant_low_memory),
            lwc=bool(args.sliderquant_lwc),
            lwc_lr=float(args.sliderquant_lwc_lr),
            max_layers=args.sliderquant_max_layers,
            model=args.model_path,
            model_family=net_name.split("-")[0],
            multigpu=False,
            net=net_name,
            nsamples=int(args.calibration_samples),
            num_layer=int(args.sliderquant_num_layer),
            output_dir=str(output_dir),
            parallelize=False,
            quant_gate=bool(args.sliderquant_quant_gate),
            quant_layer_list=None,
            quant_mode=str(args.sliderquant_quant_mode),
            quant_mode_layer_list=None,
            quant_rate=float(args.sliderquant_quant_rate),
            quant_rate_list=quant_rate_list,
            quant_step=int(args.sliderquant_quant_step),
            quant_warp=bool(args.sliderquant_quant_warp),
            resume=args.sliderquant_resume_from,
            resume_layers_num=args.sliderquant_resume_layers_num,
            rotate=bool(args.sliderquant_rotate),
            rotate_mode=args.rotation_mode,
            save_qmodel_path=None,
            scale_lr=float(args.sliderquant_scale_lr),
            seed=int(args.seed),
            seqlen=int(args.sequence_length),
            sliding_layer=args.sliderquant_sliding_layer,
            start_round=int(args.sliderquant_start_round),
            symmetric=bool(args.weight_symmetric),
            tasks="",
            teach_model=args.sliderquant_teach_model,
            test_datasets=args.evaluation_dataset,
            test_mode=bool(args.sliderquant_test_mode),
            train_resume=args.sliderquant_train_resume,
            update_gate=bool(args.sliderquant_update_gate),
            use_base_loss=str(args.sliderquant_use_base_loss),
            use_bfloat16=use_bfloat16,
            use_ddp=False,
            use_down_scale=bool(args.sliderquant_use_down_scale),
            use_fp_inp_loss=bool(args.sliderquant_use_fp_inp_loss),
            use_lora=bool(args.sliderquant_use_lora),
            use_lr_scheduler=bool(args.sliderquant_use_lr_scheduler),
            use_quant_tar_loss=bool(args.sliderquant_use_quant_tar_loss),
            warmup_ratio=float(args.sliderquant_warmup_ratio),
            wbits=weight_bits,
            w_dynamic_method="per_channel",
            weight_merge=bool(args.sliderquant_weight_merge),
            wo_lwc=not bool(args.sliderquant_lwc),
            yaml=None,
        )

    @staticmethod
    def _install_quant_params(source_args: SimpleNamespace) -> None:
        source_args.weight_quant_params = {
            "n_bits": source_args.wbits,
            "per_channel_axes": [0],
            "symmetric": source_args.symmetric,
            "dynamic_method": source_args.w_dynamic_method,
            "group_size": source_args.group_size,
            "lwc": source_args.lwc,
            "disable_zero_point": source_args.disable_zero_point,
        }
        source_args.act_quant_params = {
            "n_bits": source_args.abits,
            "per_channel_axes": [],
            "symmetric": source_args.act_symmetric,
            "dynamic_method": source_args.a_dynamic_method,
        }
        source_args.q_quant_params = {
            "n_bits": source_args.abits,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": source_args.a_dynamic_method,
        }
        source_args.k_quant_params = dict(source_args.q_quant_params)
        source_args.v_quant_params = dict(source_args.q_quant_params)
        source_args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(args)
        self._validate_model(model)
        output_dir = self.resolve_output_dir(args)
        source_root = Path(__file__).resolve().parent / "source"
        source_args = self._build_source_args(args, output_dir, model)
        self._install_quant_params(source_args)

        random.seed(source_args.seed)
        np.random.seed(source_args.seed)
        torch.manual_seed(source_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(source_args.seed)

        if not bool(source_args.wbits < 16 or source_args.abits < 16 or source_args.quant_warp):
            return {
                "source_root": str(source_root),
                "sliderquant_config": {
                    "weight_bits": source_args.wbits,
                    "activation_bits": source_args.abits,
                    "rotate": source_args.rotate,
                    "rotate_mode": source_args.rotate_mode,
                    "epochs": 0,
                    "quant_step": source_args.quant_step,
                    "num_layer": source_args.num_layer,
                    "sliding_layer": source_args.sliding_layer,
                },
                "skipped_reason": "all quantization bit-widths are 16; keep baseline model unchanged",
            }

        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer_bundle.tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )

        resolved_device = resolve_device(args.device)
        text_model = self._resolve_text_model(model)
        text_model.seqlen = int(args.sequence_length)
        text_model.eval()
        for param in text_model.parameters():
            param.requires_grad = False

        lm = _MindPipeSliderLM(
            model=text_model,
            tokenizer=tokenizer_bundle.tokenizer,
            device=resolved_device,
            seqlen=int(args.sequence_length),
            batch_size=int(args.sliderquant_batch_size),
        )

        with prepend_python_path(source_root):
            importlib.invalidate_caches()
            _purge_conflicting_modules(
                ("models", "quantize", "train_utils", "parallel_utils", "datautils", "utils"),
                source_root,
            )
            if source_args.rotate:
                rotation_utils = importlib.import_module("models.rotation_utils")
                LOGGER.info("Applying SliderQuant rotation before quantization: mode=%s", source_args.rotate_mode)
                rotation_utils.fuse_layer_norms(text_model)
                rotation_utils.rotate_model(text_model, source_args, add_online_rotate=False)
                empty_cache(args.device)
            sliderquant_module = importlib.import_module("quantize.sliderquant")
            try:
                quantized_model, _inputs, _attention_mask, _position_ids = sliderquant_module.sliderquant(
                    lm,
                    source_args,
                    calibration_batches,
                    LOGGER,
                    teach_lm=None,
                )
            finally:
                empty_cache(args.device)

        return {
            "_updated_model": model,
            "source_root": str(source_root),
            "sliderquant_config": {
                "weight_bits": source_args.wbits,
                "activation_bits": source_args.abits,
                "weight_group_size": source_args.group_size,
                "calibration_dataset": source_args.calib_dataset,
                "calibration_samples": source_args.nsamples,
                "epochs": source_args.epochs,
                "quant_step": source_args.quant_step,
                "quant_rate_list": source_args.quant_rate_list,
                "rotate": source_args.rotate,
                "rotate_mode": source_args.rotate_mode,
                "max_layers": source_args.max_layers,
                "num_layer": source_args.num_layer,
                "sliding_layer": source_args.sliding_layer,
                "quant_mode": source_args.quant_mode,
                "use_lora": source_args.use_lora,
                "lora_rank": source_args.lora_rank,
                "lwc": source_args.lwc,
                "let": source_args.let,
                "low_memory": source_args.low_memory,
            },
            "slider_parameters_path": str(output_dir / "slider_parameters.pth"),
            "quantized_model_class": type(quantized_model).__name__,
        }
