"""MindPipe-native basic QA-LoRA runner.

This first-stage implementation keeps the base model in a fake-quantized training
view and trains QA-LoRA adapters with group-pooled inputs. It does not export an
AutoGPTQ packed checkpoint or merge adapters into qzeros.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.io import write_json
from ...base import BaseQuantizationMethod
from ..qlora.method import Qwen2_5_VLTextOnlyForCausalLM
from ..qlora.method import Qwen3VLTextOnlyForCausalLM
from ..qlora.method import Qwen3_5TextOnlyForCausalLM
from ..qlora.method import SUPPORTED_TEXT_MODEL_TYPES
from ..qlora.method import SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES
from ..qlora.method import _SupervisedCollator
from ..qlora.method import _build_plain_text_train_dataset
from ..qlora.method import _find_lora_target_modules
from ..qlora.method import _load_supervised_dataset
from ..qlora.method import _prepare_model_for_fake_quant_training
from ..qlora.method import _replace_with_fake_quant_linears
from ..qlora.method import _resolve_compute_dtype
from ..qlora.method import _validate_supervised_columns


LOGGER = logging.getLogger(__name__)


class QALoRALinear(torch.nn.Module):
    """Frozen base linear plus QA-LoRA's group-pooled low-rank adapter."""

    def __init__(
        self,
        base_layer: torch.nn.Linear,
        *,
        r: int,
        alpha: float,
        dropout: float,
        group_size: int,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("QA-LoRA requires a positive LoRA rank.")
        if group_size <= 0:
            raise ValueError("QA-LoRA requires a positive group size.")
        if base_layer.in_features % group_size != 0:
            raise ValueError(
                f"QA-LoRA group_size={group_size} must divide in_features={base_layer.in_features}."
            )

        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.qa_group_size = int(group_size)
        self.qa_pooled_in_features = self.in_features // self.qa_group_size
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = torch.nn.Dropout(float(dropout))
        self.lora_A = torch.nn.Linear(self.qa_pooled_in_features, self.r, bias=False)
        self.lora_B = torch.nn.Linear(self.r, self.out_features, bias=False)
        self.reset_parameters()

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lora_A.weight)
        torch.nn.init.zeros_(self.lora_B.weight)

    def _pool_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, 1, self.in_features)
        pooled = F.avg_pool1d(flat, kernel_size=self.qa_group_size, stride=self.qa_group_size)
        return pooled.reshape(*original_shape, self.qa_pooled_in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        adapter_input = self._pool_last_dim(x.to(self.lora_A.weight.dtype))
        adapter_output = self.lora_B(self.lora_A(self.lora_dropout(adapter_input))) * self.scaling
        return result + adapter_output.to(result.dtype)


def _resolve_qalora_group_size(args) -> int:
    requested = getattr(args, "qalora_group_size", None)
    if requested is not None:
        return int(requested)
    if args.weight_group_size is not None:
        return int(args.weight_group_size)
    return int(args.group_size)


def _replace_with_qalora_adapters(
    model: torch.nn.Module,
    *,
    target_modules: list[str],
    args,
) -> dict[str, object]:
    target_set = set(target_modules)
    group_size = _resolve_qalora_group_size(args)
    replacements: list[tuple[torch.nn.Module, str, torch.nn.Linear, str]] = []

    def collect(parent: torch.nn.Module, prefix: str = "") -> None:
        for child_name, child in parent.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if child_name in target_set and isinstance(child, torch.nn.Linear):
                replacements.append((parent, child_name, child, full_name))
                continue
            collect(child, full_name)

    collect(model)
    adapter_layers: dict[str, object] = {}
    for parent, child_name, linear, full_name in replacements:
        adapter = QALoRALinear(
            linear,
            r=int(args.qlora_lora_r),
            alpha=float(args.qlora_lora_alpha),
            dropout=float(args.qlora_lora_dropout),
            group_size=group_size,
        )
        adapter.to(device=linear.weight.device)
        setattr(parent, child_name, adapter)
        adapter_layers[full_name] = {
            "in_features": int(linear.in_features),
            "pooled_in_features": int(linear.in_features) // group_size,
            "out_features": int(linear.out_features),
            "group_size": group_size,
            "r": int(args.qlora_lora_r),
            "alpha": float(args.qlora_lora_alpha),
            "dropout": float(args.qlora_lora_dropout),
        }
    return adapter_layers


def _qalora_adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QALoRALinear):
            continue
        state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
        state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


class QALoRAMethod(BaseQuantizationMethod):
    name = "qalora"
    npu_ready = False
    default_calibration_dataset = "pileval"

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        group_size = _resolve_qalora_group_size(args)
        run_spec = (
            f"{self.name}_w{args.weight_bits}"
            f"_g{group_size}"
            f"_r{args.qlora_lora_r}"
            f"_lr{args.qlora_learning_rate:g}"
            f"_seq{args.sequence_length}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, model, tokenizer_bundle, args) -> None:
        model_type = getattr(model.config, "model_type", None)
        if model_type not in SUPPORTED_TEXT_MODEL_TYPES | SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES:
            raise NotImplementedError(
                "QA-LoRA v1 currently supports decoder-only LLaMA-family/Qwen2/Qwen3 text models, "
                "plus Qwen2.5-VL/Qwen3-VL/Qwen3.5 in language-only mode; "
                f"got model_type={model_type!r}."
            )
        if (
            getattr(tokenizer_bundle, "processor", None) is not None
            and model_type not in SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES
        ):
            raise NotImplementedError("QA-LoRA v1 is text-only; multimodal processors are not supported.")
        if args.eval_vlm:
            raise ValueError("QA-LoRA v1 is text-only; set --eval_vlm false.")
        if int(args.weight_bits) not in {2, 3, 4}:
            raise ValueError("QA-LoRA currently supports --weight_bits in {2, 3, 4}.")
        resolved_device = resolve_device(args.device)
        if resolved_device.type != "cuda":
            raise NotImplementedError(f"QA-LoRA v1 currently supports CUDA only; got device={resolved_device}.")

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(model, tokenizer_bundle, args)

        from transformers import AutoModelForCausalLM
        from transformers import Trainer
        from transformers import TrainingArguments

        output_dir = self.resolve_output_dir(args)
        trainer_output_dir = ensure_dir(output_dir / "trainer")
        adapter_dir = ensure_dir(output_dir / "adapter_model")
        tokenizer = tokenizer_bundle.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must define eos_token or pad_token for QA-LoRA.")
            tokenizer.pad_token = tokenizer.eos_token

        source_max_len = (
            int(args.qlora_source_max_len)
            if args.qlora_source_max_len is not None
            else max(1, int(args.sequence_length) - int(args.qlora_target_max_len))
        )
        target_max_len = int(args.qlora_target_max_len)
        model_type = getattr(model.config, "model_type", None)

        dataset_mode = "supervised" if args.qlora_train_file else "plain_text"
        if dataset_mode == "supervised":
            train_dataset, eval_dataset = _load_supervised_dataset(
                train_file=Path(args.qlora_train_file),
                eval_file=Path(args.qlora_eval_file) if args.qlora_eval_file else None,
                eval_split_ratio=float(args.qlora_eval_split_ratio),
                seed=int(args.seed),
                max_train_samples=args.qlora_max_train_samples,
                max_eval_samples=args.qlora_max_eval_samples,
            )
            _validate_supervised_columns(train_dataset, args.qlora_input_field, args.qlora_output_field, "train")
            if eval_dataset is not None:
                _validate_supervised_columns(eval_dataset, args.qlora_input_field, args.qlora_output_field, "eval")

            rename_mapping = {}
            if args.qlora_input_field != "input":
                rename_mapping[args.qlora_input_field] = "input"
            if args.qlora_output_field != "output":
                rename_mapping[args.qlora_output_field] = "output"
            if rename_mapping:
                train_dataset = train_dataset.rename_columns(rename_mapping)
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.rename_columns(rename_mapping)
        else:
            plain_text_samples = (
                int(args.qlora_max_train_samples)
                if args.qlora_max_train_samples is not None
                else int(args.qlora_plain_text_default_samples)
            )
            train_dataset = _build_plain_text_train_dataset(
                tokenizer=tokenizer,
                dataset_name=args.calibration_dataset,
                sequence_length=int(args.sequence_length),
                sample_count=plain_text_samples,
                seed=int(args.seed),
                data_path=args.data_path,
            )
            eval_dataset = None

        compute_dtype = _resolve_compute_dtype(args.dtype)
        resolved_device = resolve_device(args.device)
        device_index = 0 if resolved_device.index is None else int(resolved_device.index)

        base_config = model.config
        del model
        empty_cache(args.device)

        wrapper_cls = None
        if model_type in SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES:
            if model_type == "qwen2_5_vl":
                wrapper_cls = Qwen2_5_VLTextOnlyForCausalLM
            elif model_type == "qwen3_vl":
                wrapper_cls = Qwen3VLTextOnlyForCausalLM
            elif model_type == "qwen3_5":
                wrapper_cls = Qwen3_5TextOnlyForCausalLM
            else:  # pragma: no cover
                raise NotImplementedError(f"Unsupported language-only VLM QA-LoRA wrapper for model_type={model_type!r}.")

        if wrapper_cls is not None:
            qalora_model = wrapper_cls.from_pretrained(
                args.model_path,
                config=base_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=compute_dtype,
                attn_implementation=args.attn_implementation,
                device_map={"": device_index},
            )
        else:
            qalora_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=compute_dtype,
                attn_implementation=args.attn_implementation,
                device_map={"": device_index},
            )

        quantized_linear_layers = _replace_with_fake_quant_linears(qalora_model, args)
        qalora_model.config.use_cache = False
        qalora_model = _prepare_model_for_fake_quant_training(
            qalora_model,
            use_gradient_checkpointing=bool(args.qlora_gradient_checkpointing),
        )
        target_modules = _find_lora_target_modules(qalora_model)
        if not target_modules:
            raise RuntimeError("Could not find any eligible linear modules for QA-LoRA target injection.")
        adapter_layers = _replace_with_qalora_adapters(qalora_model, target_modules=target_modules, args=args)
        if not adapter_layers:
            raise RuntimeError("Could not inject any QA-LoRA adapter layers.")

        train_args = TrainingArguments(
            output_dir=str(trainer_output_dir),
            per_device_train_batch_size=int(args.qlora_per_device_train_batch_size),
            per_device_eval_batch_size=int(args.qlora_per_device_eval_batch_size),
            gradient_accumulation_steps=int(args.qlora_gradient_accumulation_steps),
            num_train_epochs=float(args.qlora_num_train_epochs),
            max_steps=int(args.qlora_max_steps),
            learning_rate=float(args.qlora_learning_rate),
            lr_scheduler_type=args.qlora_lr_scheduler_type,
            warmup_ratio=float(args.qlora_warmup_ratio),
            weight_decay=float(args.qlora_weight_decay),
            logging_strategy="steps",
            logging_steps=int(args.qlora_logging_steps),
            save_strategy="no",
            report_to="none",
            bf16=compute_dtype == torch.bfloat16,
            fp16=compute_dtype == torch.float16,
            gradient_checkpointing=bool(args.qlora_gradient_checkpointing),
            remove_unused_columns=False,
            do_train=True,
            do_eval=eval_dataset is not None,
            eval_strategy="no",
            dataloader_num_workers=int(args.qlora_dataloader_num_workers),
            seed=int(args.seed),
            use_cpu=False,
            optim="adamw_torch",
        )
        data_collator = _SupervisedCollator(
            tokenizer=tokenizer,
            source_max_len=source_max_len,
            target_max_len=target_max_len,
            train_on_source=bool(args.qlora_train_on_source),
        )
        trainer = Trainer(
            model=qalora_model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        tokenizer.save_pretrained(str(adapter_dir))
        adapter_state_path = adapter_dir / "qalora_adapter.pt"
        torch.save(_qalora_adapter_state_dict(qalora_model), adapter_state_path)
        adapter_config_path = write_json(
            adapter_dir / "qalora_config.json",
            {
                "adapter_type": "qalora",
                "weight_bits": int(args.weight_bits),
                "backend": "fake_quant",
                "group_size": _resolve_qalora_group_size(args),
                "lora_r": int(args.qlora_lora_r),
                "lora_alpha": float(args.qlora_lora_alpha),
                "lora_dropout": float(args.qlora_lora_dropout),
                "target_modules": target_modules,
                "merge_to_autogptq_qzeros": False,
            },
        )

        train_metrics = dict(train_result.metrics)
        train_metrics.update(
            {
                "train_examples": len(train_dataset),
                "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
                "target_modules": target_modules,
                "qalora_adapter_layers": len(adapter_layers),
            }
        )
        train_metrics_path = write_json(output_dir / "qalora_train_metrics.json", train_metrics)

        eval_metrics_path = None
        eval_metrics = None
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            eval_metrics_path = write_json(output_dir / "qalora_eval_metrics.json", dict(eval_metrics))

        qalora_model.config.use_cache = True
        qalora_model.eval()
        qalora_model.seqlen = int(args.sequence_length)

        artifacts: dict[str, object] = {
            "qalora_config": {
                "weight_bits": int(args.weight_bits),
                "backend": "fake_quant",
                "adapter": "group_pooled_lora",
                "merge_to_autogptq_qzeros": False,
                "lora_r": int(args.qlora_lora_r),
                "lora_alpha": float(args.qlora_lora_alpha),
                "lora_dropout": float(args.qlora_lora_dropout),
                "learning_rate": float(args.qlora_learning_rate),
                "num_train_epochs": float(args.qlora_num_train_epochs),
                "max_steps": int(args.qlora_max_steps),
                "train_on_source": bool(args.qlora_train_on_source),
                "gradient_checkpointing": bool(args.qlora_gradient_checkpointing),
                "source_max_len": source_max_len,
                "target_max_len": target_max_len,
                "dataset_mode": dataset_mode,
                "qalora_group_size": _resolve_qalora_group_size(args),
                "weight_group_size": args.weight_group_size,
                "group_size": args.group_size,
                "weight_symmetric": bool(args.weight_symmetric),
                "weight_clip": bool(args.weight_clip),
            },
            "train_file": None if not args.qlora_train_file else str(Path(args.qlora_train_file)),
            "eval_file": None if not args.qlora_eval_file else str(Path(args.qlora_eval_file)),
            "calibration_dataset": args.calibration_dataset,
            "trainer_output_dir": str(trainer_output_dir),
            "adapter_model_dir": str(adapter_dir),
            "adapter_state_path": str(adapter_state_path),
            "adapter_config_path": str(adapter_config_path),
            "train_examples": len(train_dataset),
            "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
            "target_modules": target_modules,
            "quantized_linear_count": len(quantized_linear_layers),
            "quantized_linear_layers": quantized_linear_layers,
            "qalora_adapter_count": len(adapter_layers),
            "qalora_adapter_layers": adapter_layers,
            "train_metrics_path": str(train_metrics_path),
        }
        if eval_metrics_path is not None:
            artifacts["eval_metrics_path"] = str(eval_metrics_path)
        if eval_metrics is not None:
            artifacts["eval_metrics"] = dict(eval_metrics)
        artifacts["_updated_model"] = qalora_model
        artifacts["_updated_tokenizer_bundle"] = tokenizer_bundle
        return artifacts
