"""Run-spec helpers for compression LoRA outputs."""

from __future__ import annotations

from typing import Any


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
    train_plan = parse_compression_lora_train_plan(getattr(args, "compression_lora_train_plan", None))
    if "cpt" in train_plan:
        parts.append(f"cptn{args.compression_lora_cpt_samples}")
        parts.append(f"cpte{args.compression_lora_cpt_num_train_epochs:g}")
        parts.append(f"cptlr{args.compression_lora_cpt_learning_rate:g}")
    if "sft" in train_plan:
        parts.append(f"sftn{args.compression_lora_sft_samples}")
        parts.append(f"sfte{args.compression_lora_sft_num_train_epochs:g}")
        parts.append(f"sftlr{args.compression_lora_sft_learning_rate:g}")
    if include_sequence_length:
        parts.append(f"seq{args.sequence_length}")
    return "_".join(parts)
