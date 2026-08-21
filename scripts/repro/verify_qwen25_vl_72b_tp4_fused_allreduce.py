#!/usr/bin/env python3
"""Verify TP4 W8A8 fused matmul/all-reduce at Qwen2.5-VL-72B shapes."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu


WORLD_SIZE = 4
OUTPUT_SIZE = 8192
ACL_FORMAT_FRACTAL_NZ = 29
PROJECTIONS = {"o_proj": 2048, "down_proj": 7392}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def hcom_name(rank: int) -> str:
    group = dist.distributed_c10d._get_default_group()
    backend = group._get_backend(torch.device("npu"))
    global_rank = dist.get_global_rank(group, rank)
    return backend.get_hccl_comm_name(global_rank)


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu().item())


def worker(rank: int, port: int, args: dict[str, object], output: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        "hccl",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=300),
    )
    device = torch.device(f"npu:{rank}")
    torch.manual_seed(int(args["seed"]) + rank)
    torch_npu.npu.manual_seed(int(args["seed"]) + rank)
    hcom = hcom_name(rank)
    rows: list[dict[str, object]] = []

    try:
        for projection in args["projections"]:
            input_size = PROJECTIONS[str(projection)]
            for token_count in args["tokens"]:
                tokens = int(token_count)
                qinput = torch.randint(
                    -127,
                    128,
                    (tokens, input_size),
                    dtype=torch.int8,
                    device=device,
                )
                input_scale = (
                    torch.rand(tokens, dtype=torch.float32, device=device)
                    * 0.02
                    + 1e-5
                )
                weight = torch.randint(
                    -127,
                    128,
                    (input_size, OUTPUT_SIZE),
                    dtype=torch.int8,
                    device=device,
                )
                weight = torch_npu.npu_format_cast(
                    weight.contiguous(), ACL_FORMAT_FRACTAL_NZ
                )
                weight_scale = (
                    torch.rand(OUTPUT_SIZE, dtype=torch.float32, device=device)
                    * 5e-4
                    + 1e-6
                )

                separate = torch_npu.npu_quant_matmul(
                    qinput,
                    weight,
                    weight_scale,
                    pertoken_scale=input_scale,
                    output_dtype=torch.float16,
                )
                dist.all_reduce(separate, op=dist.ReduceOp.SUM)
                fused = torch_npu.npu_mm_all_reduce_base(
                    qinput,
                    weight,
                    hcom,
                    reduce_op="sum",
                    dequant_scale=weight_scale,
                    pertoken_scale=input_scale,
                )
                torch_npu.npu.synchronize()

                separate_fp32 = separate.float()
                fused_fp32 = fused.float()
                diff = (fused_fp32 - separate_fp32).abs()
                reference_abs = separate_fp32.abs()
                local = {
                    "both_finite": bool(
                        torch.isfinite(separate).all().item()
                        and torch.isfinite(fused).all().item()
                    ),
                    "max_abs_diff": float(diff.max().cpu().item()),
                    "mean_abs_diff": float(diff.mean().cpu().item()),
                    "reference_mean_abs": float(
                        reference_abs.mean().cpu().item()
                    ),
                    "reference_max_abs": float(reference_abs.max().cpu().item()),
                    "exact_fraction": float((diff == 0).float().mean().cpu().item()),
                }
                finite_tensor = torch.tensor(
                    [int(local["both_finite"])],
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
                local["both_finite"] = bool(finite_tensor.cpu().item())
                for key in (
                    "max_abs_diff",
                    "mean_abs_diff",
                    "reference_mean_abs",
                    "reference_max_abs",
                ):
                    local[key] = max_across_ranks(float(local[key]), device)
                exact_tensor = torch.tensor(
                    [float(local["exact_fraction"])],
                    dtype=torch.float32,
                    device=device,
                )
                dist.all_reduce(exact_tensor, op=dist.ReduceOp.MIN)
                local["exact_fraction_min_rank"] = float(
                    exact_tensor.cpu().item()
                )
                local["mean_abs_over_reference_mean_abs"] = (
                    float(local["mean_abs_diff"])
                    / max(float(local["reference_mean_abs"]), 1e-12)
                )
                rows.append(
                    {
                        "projection": str(projection),
                        "shape_mkn": [tokens, input_size, OUTPUT_SIZE],
                        "correctness": local,
                    }
                )
                del qinput, input_scale, weight, weight_scale, separate, fused
                torch_npu.npu.empty_cache()

        if rank == 0:
            Path(output).write_text(
                json.dumps(
                    {
                        "kind": "qwen25_vl_72b_tp4_fused_allreduce_verify",
                        "world_size": WORLD_SIZE,
                        "device": "Ascend 910B3",
                        "seed": int(args["seed"]),
                        "reference": "npu_quant_matmul + HCCL all_reduce",
                        "candidate": "npu_mm_all_reduce_base",
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 1687])
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=sorted(PROJECTIONS),
        default=list(PROJECTIONS),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(tokens <= 0 for tokens in args.tokens):
        raise ValueError("--tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    port = args.master_port or free_port()
    mp.spawn(
        worker,
        args=(port, vars(args), str(args.output.resolve())),
        nprocs=WORLD_SIZE,
        join=True,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
