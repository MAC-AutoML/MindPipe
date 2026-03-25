#!/usr/bin/env python3
"""Summarize workflow experiment metrics into a compact JSON table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "workflow"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize workflow metrics.")
    parser.add_argument("--results_root", action="append", default=None)
    parser.add_argument("--output_path", default=None)
    return parser


def build_run_key(payload: dict) -> str:
    quantization_algorithm = payload.get("quantization_algorithm", "unknown")
    pruning_algorithm = payload.get("pruning_algorithm", "unknown")
    parts = [
        f"{quantization_algorithm}__{pruning_algorithm}",
        f"w{payload.get('weight_bits')}",
        f"a{payload.get('activation_bits')}",
    ]
    query_bits = payload.get("query_bits")
    key_bits = payload.get("key_bits")
    value_bits = payload.get("value_bits")
    if quantization_algorithm == "flatquant" or any(bit is not None for bit in (query_bits, key_bits, value_bits)):
        parts.append(f"q{query_bits}")
        parts.append(f"k{key_bits}")
        parts.append(f"v{value_bits}")
    parts.append(f"s{payload.get('sparsity_ratio')}")
    if pruning_algorithm == "flap":
        parts.append(str(payload.get("flap_metrics")))
        parts.append(f"h{payload.get('flap_remove_heads')}")
    return "__".join(parts)


def collect_metrics(results_roots: list[Path]) -> dict:
    summary: dict[str, dict] = {}
    for results_root in results_roots:
        for metrics_path in sorted(results_root.rglob("metrics.json")):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_name = Path(payload["model_path"].rstrip("/")).name
            execution_order = payload.get("execution_order", "unknown")
            quantization_algorithm = payload.get("quantization_algorithm", "unknown")
            pruning_algorithm = payload.get("pruning_algorithm", "unknown")
            model_summary = summary.setdefault(model_name, {})
            order_summary = model_summary.setdefault(execution_order, {})
            run_key = build_run_key(payload)
            order_summary[run_key] = {
                "quantization_algorithm": quantization_algorithm,
                "pruning_algorithm": pruning_algorithm,
                "perplexity": payload["perplexity"],
                "metrics_path": str(metrics_path),
                "results_root": str(results_root),
                "weight_bits": payload.get("weight_bits"),
                "activation_bits": payload.get("activation_bits"),
                "query_bits": payload.get("query_bits"),
                "key_bits": payload.get("key_bits"),
                "value_bits": payload.get("value_bits"),
                "sparsity_ratio": payload.get("sparsity_ratio"),
                "structure_pattern": payload.get("structure_pattern"),
                "quantization_calibration_dataset": payload.get("quantization_calibration_dataset"),
                "pruning_calibration_dataset": payload.get("pruning_calibration_dataset"),
                "quantization_calibration_samples": payload.get("quantization_calibration_samples"),
                "pruning_calibration_samples": payload.get("pruning_calibration_samples"),
                "weight_method": payload.get("weight_method"),
                "flap_metrics": payload.get("flap_metrics"),
                "flap_remove_heads": payload.get("flap_remove_heads"),
                "pseudo_pruning": payload.get("pseudo_pruning"),
            }
    return summary


def main() -> int:
    args = build_parser().parse_args()
    raw_roots = args.results_root or [str(DEFAULT_RESULTS_ROOT)]
    results_roots = [Path(path) for path in raw_roots]
    summary = collect_metrics(results_roots)
    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_path is None:
        print(output_text)
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
