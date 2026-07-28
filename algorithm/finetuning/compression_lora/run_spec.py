"""Run-spec helpers for compression LoRA outputs."""

from __future__ import annotations

from typing import Any


def _sft_dataset_slug(train_file: str | None) -> str | None:
    if not train_file:
        return None
    normalized = str(train_file).lower().replace("_", "-")
    if "flan-mini" in normalized or "flanmini" in normalized:
        return "flan-mini"
    if "openorca" in normalized or "open-orca" in normalized:
        return "openorca"
    if "alpaca-gpt4" in normalized or "alpaca-gpt-4" in normalized:
        return "alpaca-gpt4"
    if "alpaca" in normalized:
        return "alpaca"
    if "boolq" in normalized or "bool-q" in normalized:
        return "boolq"
    if "multirc" in normalized or "multi-rc" in normalized:
        return "multirc"
    if "arc-challenge" in normalized or "arcchallenge" in normalized:
        return "arc-challenge"
    if "arc-easy" in normalized or "arceasy" in normalized:
        return "arc-easy"
    if "openbookqa" in normalized or "open-book-qa" in normalized:
        return "openbookqa"
    if "qasper-bool-train" in normalized or "qasperbooltrain" in normalized:
        return "qasper-bool"
    if "anli-r1-train" in normalized or "anlir1train" in normalized:
        return "anli-r1"
    if "eval-near-sft-4096" in normalized or "eval_near_sft_4096" in normalized:
        return "self-near"
    if "cb-train" in normalized or "commitmentbank" in normalized:
        return "cb"
    if "qnli" in normalized:
        return "qnli"
    if "mnli" in normalized:
        return "mnli"
    if "rte" in normalized:
        return "rte"
    return None


def parse_compression_lora_train_plan(plan: str | None) -> list[str]:
    if not plan:
        return ["cpt", "sft"]
    stages = [stage.strip().lower() for stage in plan.split(",") if stage.strip()]
    if not stages:
        return ["cpt", "sft"]
    allowed = {"cpt", "sft"}
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise ValueError(f"Unsupported compression_lora train stage(s): {unknown}. Allowed: {sorted(allowed)}")
    return stages


def compression_lora_run_spec(args: Any, *, include_sequence_length: bool = True) -> str:
    parts = [f"compression_lora_r{args.compression_lora_rank}", f"init{args.compression_lora_init}"]
    adapter_type = getattr(args, "compression_lora_adapter_type", "lora")
    if adapter_type != "lora":
        parts.append(f"adapter{adapter_type}")
    if getattr(args, "compression_lora_resume_from", None):
        parts.append(str(getattr(args, "compression_lora_resume_mode", "strict")))
    train_plan = parse_compression_lora_train_plan(getattr(args, "compression_lora_train_plan", None))
    if "cpt" in train_plan:
        parts.append(f"cptn{args.compression_lora_cpt_samples}")
        parts.append(f"cpte{args.compression_lora_cpt_num_train_epochs:g}")
        parts.append(f"cptlr{args.compression_lora_cpt_learning_rate:g}")
    if "sft" in train_plan:
        sft_dataset = _sft_dataset_slug(getattr(args, "compression_lora_sft_train_file", None))
        if sft_dataset:
            parts.append(f"sftdata{sft_dataset}")
        parts.append(f"sftn{args.compression_lora_sft_samples}")
        sft_sample_start = int(getattr(args, "compression_lora_sft_sample_start", 0))
        if sft_sample_start:
            parts.append(f"sftstart{sft_sample_start}")
        parts.append(f"sfte{args.compression_lora_sft_num_train_epochs:g}")
        parts.append(f"sftlr{args.compression_lora_sft_learning_rate:g}")
    if include_sequence_length:
        parts.append(f"seq{args.sequence_length}")
    return "_".join(parts)
