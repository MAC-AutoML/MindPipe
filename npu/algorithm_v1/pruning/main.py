"""Pruning CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common.logging import setup_logging
from .registry import METHOD_REGISTRY
from .registry import get_method


def build_parser() -> argparse.ArgumentParser:
    default_output_root = Path("/mnt/42_store/lcw/data2/Huawei/algorithm-v1/results/pruning")
    parser = argparse.ArgumentParser(description="Unified pruning launcher.")
    parser.add_argument("--algorithm", required=True, choices=sorted(METHOD_REGISTRY))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=str(default_output_root))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--calibration_dataset", default="c4", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--structure_pattern", default="unstructured")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--use_variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flap_metrics", default="WIFV", choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument("--flap_remove_heads", type=int, default=8)
    parser.add_argument("--pseudo_pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_pruned_model", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
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
