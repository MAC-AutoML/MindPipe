"""MindPipe-native text-only QLoRA runner."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from typing import Any
from typing import Sequence

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers.generation import GenerationMixin
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from ....common.datasets import get_calibration_and_evaluation_data
from ....common.device import empty_cache
from ....common.device import resolve_device
from ....common.io import ensure_dir
from ....common.io import model_slug
from ....common.io import write_json
from ...base import BaseQuantizationMethod
from ..splitquant.source.splitquant.quant_utils import WeightQuantizer


LOGGER = logging.getLogger(__name__)
IGNORE_INDEX = -100
SUPPORTED_TEXT_MODEL_TYPES = {"llama", "qwen2", "qwen3"}
SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES = {"qwen2_5_vl", "qwen3_vl", "qwen3_5"}


class FakeQuantLinear(torch.nn.Linear):
    """Linear layer with frozen-weight fake quantization for low-bit QLoRA experiments."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        bits: int,
        symmetric: bool,
        group_size: int,
        mse: bool,
    ) -> None:
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.weight = linear.weight
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = linear.bias
        self.fake_quant_bits = int(bits)
        self.fake_quant_symmetric = bool(symmetric)
        self.fake_quant_group_size = int(group_size)
        self.fake_quant_mse = bool(mse)
        self.weight_quantizers = torch.nn.ModuleList()
        self.refresh_quantizers()

    def refresh_quantizers(self) -> None:
        self.weight_quantizers = torch.nn.ModuleList()
        if self.fake_quant_bits >= 16:
            return
        if self.fake_quant_group_size <= 0:
            quantizer = WeightQuantizer()
            quantizer.configure(
                bits=self.fake_quant_bits,
                perchannel=True,
                sym=self.fake_quant_symmetric,
                mse=self.fake_quant_mse,
            )
            quantizer.find_params(self.weight.detach())
            self.weight_quantizers.append(quantizer)
            return

        for start in range(0, self.in_features, self.fake_quant_group_size):
            end = start + self.fake_quant_group_size
            quantizer = WeightQuantizer()
            quantizer.configure(
                bits=self.fake_quant_bits,
                perchannel=True,
                sym=self.fake_quant_symmetric,
                mse=self.fake_quant_mse,
            )
            quantizer.find_params(self.weight[:, start:end].detach())
            self.weight_quantizers.append(quantizer)

    def _quantized_weight(self) -> torch.Tensor:
        if self.fake_quant_bits >= 16 or not self.weight_quantizers:
            return self.weight
        if self.fake_quant_group_size <= 0:
            return self.weight_quantizers[0](self.weight)

        chunks = []
        for idx, start in enumerate(range(0, self.in_features, self.fake_quant_group_size)):
            end = start + self.fake_quant_group_size
            chunks.append(self.weight_quantizers[idx](self.weight[:, start:end]))
        return torch.cat(chunks, dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self._quantized_weight(), self.bias)


class Qwen2_5_VLTextOnlyForCausalLM(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    """Text-only CausalLM view over Qwen2.5-VL, used for language-branch-only QLoRA."""

    _supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

        self.config = config
        self.model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.name_or_path = getattr(config, "_name_or_path", None)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = {"past_key_values": past_key_values, "use_cache": kwargs.get("use_cache")}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            model_inputs["position_ids"] = kwargs["position_ids"]
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3VLTextOnlyForCausalLM(Qwen3VLPreTrainedModel, GenerationMixin):
    """Text-only CausalLM view over Qwen3-VL, used for language-branch-only QLoRA."""

    _supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        self.config = config
        self.model = torch.nn.Module()
        self.model.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.name_or_path = getattr(config, "_name_or_path", None)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model.language_model, "gradient_checkpointing_enable"):
            self.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.model.language_model, "gradient_checkpointing_disable"):
            self.model.language_model.gradient_checkpointing_disable()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = {"past_key_values": past_key_values, "use_cache": kwargs.get("use_cache")}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            model_inputs["position_ids"] = kwargs["position_ids"]
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3_5TextOnlyForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    """Text-only CausalLM view over Qwen3.5, used for language-branch-only QLoRA."""

    _supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        self.config = config
        self.model = torch.nn.Module()
        self.model.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.name_or_path = getattr(config, "_name_or_path", None)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model.language_model, "gradient_checkpointing_enable"):
            self.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.model.language_model, "gradient_checkpointing_disable"):
            self.model.language_model.gradient_checkpointing_disable()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = {"past_key_values": past_key_values, "use_cache": kwargs.get("use_cache")}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            model_inputs["position_ids"] = kwargs["position_ids"]
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _dataset_loader_name(path: Path) -> str:
    suffixes = tuple(s.lower() for s in path.suffixes)
    if suffixes[-2:] == (".json", ".gz") or suffixes[-1:] in {(".json",), (".jsonl",)}:
        return "json"
    if suffixes[-2:] == (".csv", ".gz") or suffixes[-1:] == (".csv",):
        return "csv"
    if suffixes[-2:] == (".parquet", ".gz") or suffixes[-1:] == (".parquet",):
        return "parquet"
    raise ValueError(
        f"Unsupported QLoRA dataset file extension for {path}. "
        "Use a local .json, .jsonl, .csv, or .parquet file."
    )


def _load_supervised_dataset(
    train_file: Path,
    eval_file: Path | None,
    eval_split_ratio: float,
    seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    from datasets import load_dataset

    if not train_file.exists():
        raise FileNotFoundError(f"QLoRA train file does not exist: {train_file}")
    if eval_file is not None and not eval_file.exists():
        raise FileNotFoundError(f"QLoRA eval file does not exist: {eval_file}")

    loader_name = _dataset_loader_name(train_file)
    train_dataset = load_dataset(loader_name, data_files=str(train_file), split="train")
    eval_dataset = None

    if eval_file is not None:
        eval_loader_name = _dataset_loader_name(eval_file)
        eval_dataset = load_dataset(eval_loader_name, data_files=str(eval_file), split="train")
    elif eval_split_ratio > 0.0:
        if not 0.0 < eval_split_ratio < 1.0:
            raise ValueError("--qlora_eval_split_ratio must be in (0, 1) when provided.")
        if len(train_dataset) >= 2:
            split = train_dataset.train_test_split(test_size=eval_split_ratio, seed=seed, shuffle=True)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            LOGGER.warning(
                "QLoRA eval split requested, but train dataset has fewer than 2 samples; skipping eval split."
            )

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(int(max_train_samples), len(train_dataset))))
    if eval_dataset is not None and max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(int(max_eval_samples), len(eval_dataset))))
    return train_dataset, eval_dataset


def _validate_supervised_columns(dataset, input_field: str, output_field: str, split_name: str) -> None:
    available = set(getattr(dataset, "column_names", []) or [])
    missing = [name for name in (input_field, output_field) if name not in available]
    if missing:
        raise ValueError(
            f"QLoRA {split_name} dataset is missing required columns {missing}. "
            f"Available columns: {sorted(available)}"
        )


def _find_lora_target_modules(model, bnb_module=None) -> list[str]:
    linear_classes: tuple[type, ...] = (torch.nn.Linear,)
    if bnb_module is not None and hasattr(bnb_module.nn, "Linear4bit"):
        linear_classes = linear_classes + (bnb_module.nn.Linear4bit,)
    if bnb_module is not None and hasattr(bnb_module.nn, "Linear8bitLt"):
        linear_classes = linear_classes + (bnb_module.nn.Linear8bitLt,)

    module_names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, linear_classes):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name == "lm_head":
            continue
        module_names.add(leaf_name)
    return sorted(module_names)


def _resolve_fake_quant_group_size(linear: torch.nn.Linear, requested_group_size: int | None) -> int:
    if requested_group_size is None:
        return -1
    group_size = int(requested_group_size)
    if group_size <= 0 or group_size >= linear.in_features:
        return -1
    if linear.in_features % group_size != 0:
        return -1
    return group_size


def _replace_with_fake_quant_linears(model, args) -> dict[str, object]:
    requested_group_size = args.weight_group_size if args.weight_group_size is not None else args.group_size
    replacements: list[tuple[torch.nn.Module, str, torch.nn.Linear, str]] = []

    def _collect(parent: torch.nn.Module, prefix: str = "") -> None:
        for child_name, child in parent.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, torch.nn.Linear) and not isinstance(child, FakeQuantLinear):
                replacements.append((parent, child_name, child, full_name))
            _collect(child, full_name)

    _collect(model)
    quantized_layers: dict[str, object] = {}
    for parent, child_name, linear, full_name in replacements:
        resolved_group_size = _resolve_fake_quant_group_size(linear, requested_group_size)
        if requested_group_size not in (None, 0, -1) and resolved_group_size < 0:
            LOGGER.warning(
                "QLoRA fake-quant group size %s does not divide %s.in_features=%s; falling back to per-channel.",
                requested_group_size,
                full_name,
                linear.in_features,
            )
        fake_quant_linear = FakeQuantLinear(
            linear,
            bits=int(args.weight_bits),
            symmetric=bool(args.weight_symmetric),
            group_size=resolved_group_size,
            mse=bool(args.weight_clip),
        )
        setattr(parent, child_name, fake_quant_linear)
        quantized_layers[full_name] = {
            "bits": int(args.weight_bits),
            "group_size": resolved_group_size,
            "symmetric": bool(args.weight_symmetric),
            "mse_clip": bool(args.weight_clip),
            "in_features": int(linear.in_features),
            "out_features": int(linear.out_features),
        }
    return quantized_layers


def _prepare_model_for_fake_quant_training(model, use_gradient_checkpointing: bool) -> torch.nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if isinstance(module, torch.nn.LayerNorm) or "rmsnorm" in class_name:
            module.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )
        if supports_gc_kwargs:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=None)
        else:
            model.gradient_checkpointing_enable()
    return model


def _build_plain_text_train_dataset(
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    sample_count: int,
    seed: int,
    data_path: str | Path,
):
    calibration_batches, _ = get_calibration_and_evaluation_data(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sequence_length=sequence_length,
        sample_count=sample_count,
        seed=seed,
        data_path=data_path,
    )
    rows = []
    for input_ids, _labels in calibration_batches:
        rows.append({"input_ids": input_ids.squeeze(0).tolist()})
    return Dataset.from_list(rows)


@dataclass
class _SupervisedCollator:
    tokenizer: Any
    source_max_len: int
    target_max_len: int
    train_on_source: bool

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if instances and "input_ids" in instances[0]:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Tokenizer must define pad_token_id before QLoRA collation.")
            input_ids_list = [
                torch.tensor(example["input_ids"], dtype=torch.long)
                for example in instances
            ]
            padded_inputs = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=pad_token_id,
            )
            attention_mask = padded_inputs.ne(pad_token_id)
            labels = padded_inputs.clone()
            labels = labels.masked_fill(~attention_mask, IGNORE_INDEX)
            return {
                "input_ids": padded_inputs,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        bos = self.tokenizer.bos_token or ""
        eos = self.tokenizer.eos_token or ""
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before QLoRA collation.")

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for example in instances:
            source_text = f"{bos}{str(example['input'])}"
            target_text = f"{str(example['output'])}{eos}"

            source_ids = self.tokenizer(
                source_text,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            target_ids = self.tokenizer(
                target_text,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = source_ids + target_ids
            if self.train_on_source:
                labels = list(input_ids)
            else:
                labels = [IGNORE_INDEX] * len(source_ids) + list(target_ids)

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = padded_inputs.ne(pad_token_id)
        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }


class QLoRAMethod(BaseQuantizationMethod):
    name = "qlora"
    npu_ready = False
    default_calibration_dataset = "pileval"

    def resolve_output_dir(self, args) -> Path:
        model_name = model_slug(args.model_path)
        run_spec = (
            f"{self.name}_w{args.weight_bits}"
            f"_r{args.qlora_lora_r}"
            f"_lr{args.qlora_learning_rate:g}"
            f"_seq{args.sequence_length}"
        )
        return ensure_dir(Path(args.output_root) / model_name / self.name / run_spec)

    def _validate_args(self, model, tokenizer_bundle, args) -> None:
        model_type = getattr(model.config, "model_type", None)
        if model_type not in SUPPORTED_TEXT_MODEL_TYPES | SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES:
            raise NotImplementedError(
                "QLoRA v1 currently supports decoder-only LLaMA-family/Qwen2/Qwen3 text models, "
                "plus Qwen2.5-VL/Qwen3-VL/Qwen3.5 in language-only mode; "
                f"got model_type={model_type!r}."
            )
        if (
            getattr(tokenizer_bundle, "processor", None) is not None
            and model_type not in SUPPORTED_TEXT_ONLY_VLM_MODEL_TYPES
        ):
            raise NotImplementedError("QLoRA v1 is text-only; multimodal processors are not supported.")
        if args.eval_vlm:
            raise ValueError("QLoRA v1 is text-only; set --eval_vlm false.")
        if int(args.weight_bits) not in {2, 3, 4}:
            raise ValueError("QLoRA currently supports --weight_bits in {2, 3, 4}.")
        resolved_device = resolve_device(args.device)
        if resolved_device.type != "cuda":
            raise NotImplementedError(
                f"QLoRA v1 currently supports CUDA only; got device={resolved_device}."
            )

    def apply_fake_quantization(self, model, tokenizer_bundle, args) -> dict[str, object]:
        self._validate_args(model, tokenizer_bundle, args)

        from peft import LoraConfig
        from peft import get_peft_model
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
                raise ValueError("Tokenizer must define eos_token or pad_token for QLoRA.")
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
        quant_backend = "bnb_4bit" if int(args.weight_bits) == 4 else "fake_quant"

        # The base model was already loaded by the executor for general workflows.
        # QLoRA needs a dedicated training model instance so that LoRA injection and quantized/fake-quant
        # wrappers do not mutate the executor's original model object.
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
                raise NotImplementedError(f"Unsupported language-only VLM QLoRA wrapper for model_type={model_type!r}.")

        quantized_linear_layers: dict[str, object] = {}
        if quant_backend == "bnb_4bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise RuntimeError(
                    "QLoRA 4bit backend requires bitsandbytes, but it is not installed in the current environment. "
                    "Install bitsandbytes in the `mindpipe` environment before running `--quantization qlora`."
                ) from exc

            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(args.qlora_double_quant),
                bnb_4bit_quant_type=args.qlora_quant_type,
            )
            if wrapper_cls is not None:
                qlora_model = wrapper_cls.from_pretrained(
                    args.model_path,
                    config=base_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=compute_dtype,
                    attn_implementation=args.attn_implementation,
                    device_map={"": device_index},
                    quantization_config=quantization_config,
                )
            else:
                qlora_model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=compute_dtype,
                    attn_implementation=args.attn_implementation,
                    device_map={"": device_index},
                    quantization_config=quantization_config,
                )
            qlora_model.config.use_cache = False
            qlora_model = prepare_model_for_kbit_training(
                qlora_model,
                use_gradient_checkpointing=bool(args.qlora_gradient_checkpointing),
            )
            target_modules = _find_lora_target_modules(qlora_model, bnb)
        else:
            if wrapper_cls is not None:
                qlora_model = wrapper_cls.from_pretrained(
                    args.model_path,
                    config=base_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=compute_dtype,
                    attn_implementation=args.attn_implementation,
                    device_map={"": device_index},
                )
            else:
                qlora_model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=compute_dtype,
                    attn_implementation=args.attn_implementation,
                    device_map={"": device_index},
                )
            quantized_linear_layers = _replace_with_fake_quant_linears(qlora_model, args)
            qlora_model.config.use_cache = False
            qlora_model = _prepare_model_for_fake_quant_training(
                qlora_model,
                use_gradient_checkpointing=bool(args.qlora_gradient_checkpointing),
            )
            target_modules = _find_lora_target_modules(qlora_model)
        if not target_modules:
            raise RuntimeError("Could not find any eligible linear modules for QLoRA target injection.")
        lora_config = LoraConfig(
            r=int(args.qlora_lora_r),
            lora_alpha=int(args.qlora_lora_alpha),
            target_modules=target_modules,
            lora_dropout=float(args.qlora_lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )
        qlora_model = get_peft_model(qlora_model, lora_config)

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
            save_strategy="steps",
            save_steps=int(args.qlora_save_steps),
            save_total_limit=int(args.qlora_save_total_limit),
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
            model=qlora_model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        train_metrics = dict(train_result.metrics)
        train_metrics.update(
            {
                "train_examples": len(train_dataset),
                "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
                "target_modules": target_modules,
            }
        )
        train_metrics_path = write_json(output_dir / "qlora_train_metrics.json", train_metrics)

        eval_metrics_path = None
        eval_metrics = None
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            eval_metrics_path = write_json(output_dir / "qlora_eval_metrics.json", dict(eval_metrics))

        qlora_model.config.use_cache = True
        qlora_model.eval()
        qlora_model.seqlen = int(args.sequence_length)

        if bool(args.qlora_merge_adapter):
            if not hasattr(qlora_model, "merge_and_unload"):
                raise RuntimeError(
                    "The current PEFT/QLoRA runtime does not expose merge_and_unload(); "
                    "leave --qlora_merge_adapter false in this environment."
                )
            qlora_model = qlora_model.merge_and_unload()
            qlora_model.eval()
            qlora_model.seqlen = int(args.sequence_length)

        artifacts: dict[str, object] = {
            "qlora_config": {
                "weight_bits": int(args.weight_bits),
                "backend": quant_backend,
                "quant_type": args.qlora_quant_type if quant_backend == "bnb_4bit" else "uniform_fake_quant",
                "double_quant": bool(args.qlora_double_quant) if quant_backend == "bnb_4bit" else False,
                "lora_r": int(args.qlora_lora_r),
                "lora_alpha": int(args.qlora_lora_alpha),
                "lora_dropout": float(args.qlora_lora_dropout),
                "learning_rate": float(args.qlora_learning_rate),
                "num_train_epochs": float(args.qlora_num_train_epochs),
                "max_steps": int(args.qlora_max_steps),
                "train_on_source": bool(args.qlora_train_on_source),
                "gradient_checkpointing": bool(args.qlora_gradient_checkpointing),
                "source_max_len": source_max_len,
                "target_max_len": target_max_len,
                "merge_adapter": bool(args.qlora_merge_adapter),
                "dataset_mode": dataset_mode,
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
            "train_examples": len(train_dataset),
            "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
            "target_modules": target_modules,
            "quantized_linear_count": len(quantized_linear_layers),
            "quantized_linear_layers": quantized_linear_layers,
            "train_metrics_path": str(train_metrics_path),
        }
        if eval_metrics_path is not None:
            artifacts["eval_metrics_path"] = str(eval_metrics_path)
        if eval_metrics is not None:
            artifacts["eval_metrics"] = dict(eval_metrics)
        artifacts["_updated_model"] = qlora_model
        artifacts["_updated_tokenizer_bundle"] = tokenizer_bundle
        return artifacts
