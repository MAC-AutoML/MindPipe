#!/usr/bin/env python3
"""Verify the allocation-free FIA call against the legacy call."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu


NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 128
MASK_SIZE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_case(
    *,
    name: str,
    query_lengths: list[int],
    kv_lengths: list[int],
    device: torch.device,
) -> dict[str, object]:
    if len(query_lengths) != len(kv_lengths):
        raise ValueError("query and KV batch sizes must match")
    query_cumulative: list[int] = []
    total = 0
    for length in query_lengths:
        total += length
        query_cumulative.append(total)

    blocks_per_sequence = [math.ceil(length / BLOCK_SIZE) for length in kv_lengths]
    max_blocks = max(blocks_per_sequence)
    block_table_cpu = torch.full(
        (len(kv_lengths), max_blocks), -1, dtype=torch.int32
    )
    next_block = 0
    for row, block_count in enumerate(blocks_per_sequence):
        block_table_cpu[row, :block_count] = torch.arange(
            next_block, next_block + block_count, dtype=torch.int32
        )
        next_block += block_count

    query = torch.randn(
        total, NUM_HEADS, HEAD_SIZE, dtype=torch.float16, device=device
    )
    key_cache = torch.randn(
        next_block,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        dtype=torch.float16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = block_table_cpu.to(device)
    attention_mask = torch.triu(
        torch.ones(MASK_SIZE, MASK_SIZE, dtype=torch.int8, device=device),
        diagonal=1,
    )
    query_cumulative_device = torch.tensor(
        query_cumulative, dtype=torch.int32, device=device
    )
    kv_lengths_device = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    key = key_cache.view(next_block, BLOCK_SIZE, -1)
    value = value_cache.view(next_block, BLOCK_SIZE, -1)
    common = {
        "atten_mask": attention_mask,
        "block_table": block_table,
        "input_layout": "TND",
        "block_size": BLOCK_SIZE,
        "actual_seq_lengths_kv": kv_lengths_device,
        "num_key_value_heads": NUM_KV_HEADS,
        "num_heads": NUM_HEADS,
        "scale": 1.0 / math.sqrt(HEAD_SIZE),
        "sparse_mode": 3,
    }

    legacy, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        actual_seq_lengths=query_cumulative_device,
        **common,
    )
    preallocated = torch.empty_like(query)
    softmax_lse = torch.empty(1, dtype=query.dtype, device=device)
    candidate, _ = torch_npu.npu_fused_infer_attention_score.out(
        query,
        key,
        value,
        actual_seq_lengths=query_cumulative,
        out=[preallocated, softmax_lse],
        **common,
    )
    torch.npu.synchronize()

    difference = (legacy.float() - candidate.float()).abs()
    exact_elements = int((legacy == candidate).sum().cpu().item())
    total_elements = legacy.numel()
    return {
        "name": name,
        "query_lengths": query_lengths,
        "query_cumulative": query_cumulative,
        "kv_lengths": kv_lengths,
        "query_shape": list(query.shape),
        "kv_cache_shape": list(key_cache.shape),
        "candidate_reuses_output_storage": (
            candidate.data_ptr() == preallocated.data_ptr()
        ),
        "both_finite": bool(
            torch.isfinite(legacy).all().cpu().item()
            and torch.isfinite(candidate).all().cpu().item()
        ),
        "max_abs_error": float(difference.max().cpu().item()),
        "mean_abs_error": float(difference.mean().cpu().item()),
        "exact_elements": exact_elements,
        "total_elements": total_elements,
        "exact_fraction": exact_elements / total_elements,
    }


def main() -> int:
    args = parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    torch_npu.npu.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    cases = [
        run_case(
            name="mixed_short_prefill",
            query_lengths=[5, 7, 3, 9],
            kv_lengths=[32, 48, 17, 64],
            device=device,
        ),
        run_case(
            name="image_heavy_prefill",
            query_lengths=[128, 257, 511, 791],
            kv_lengths=[128, 257, 511, 1687],
            device=device,
        ),
    ]
    passed = all(
        case["candidate_reuses_output_storage"]
        and case["both_finite"]
        and case["max_abs_error"] == 0.0
        for case in cases
    )
    report = {
        "kind": "qwen25_vl_fia_fastpath_correctness",
        "valid": passed,
        "device": "Ascend 910B3",
        "seed": args.seed,
        "reference": "device cumulative lengths + allocating FIA return",
        "candidate": "host cumulative lengths + preallocated FIA out",
        "cases": cases,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
