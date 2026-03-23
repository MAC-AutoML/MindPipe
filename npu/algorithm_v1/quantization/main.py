"""Quantization CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common.logging import setup_logging
from .registry import METHOD_REGISTRY
from .registry import get_method


def build_parser() -> argparse.ArgumentParser:
    default_output_root = Path("/mnt/42_store/lcw/data2/Huawei/algorithm-v1/results/quantization")
    parser = argparse.ArgumentParser(description="Unified fake-quant launcher.")
    parser.add_argument("--algorithm", required=True, choices=sorted(METHOD_REGISTRY))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=str(default_output_root))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--calibration_dataset", default="pileval", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=16)
    parser.add_argument("--query_bits", type=int, default=16)
    parser.add_argument("--key_bits", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--weight_group_size", type=int, default=None)
    parser.add_argument("--activation_group_size", type=int, default=None)
    parser.add_argument("--kv_group_size", type=int, default=None)
    parser.add_argument("--weight_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--key_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight_method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--use_activation_order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_epochs", type=int, default=15)
    parser.add_argument("--flatquant_calibration_batch_size", type=int, default=4)
    parser.add_argument("--flatquant_lr", type=float, default=1e-5)
    parser.add_argument("--flatquant_cali_trans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_add_diag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lwc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lac", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_diag_init", default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--flatquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--flatquant_warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_deactive_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_direct_inv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_separate_vtrans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static_groups", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--awq_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation_mode", default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_checkpoint", default=None)
    parser.add_argument("--save_fake_model", action=argparse.BooleanOptionalAction, default=False)
    return parser


def normalize_args(args):
    if args.weight_group_size is None:
        args.weight_group_size = args.group_size
    if args.activation_group_size is None:
        args.activation_group_size = args.group_size
    if args.kv_group_size is None:
        args.kv_group_size = args.group_size
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    normalize_args(args)
    setup_logging(args.log_level)
    result = get_method(args.algorithm).run(args)
    print(
        json.dumps(
            {
                "algorithm_name": result.algorithm_name,
                "model_path": result.model_path,
                "output_dir": result.output_dir,
                "metrics_path": result.metrics_path,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
