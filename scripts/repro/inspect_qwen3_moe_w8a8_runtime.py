#!/usr/bin/env python3
"""Load Qwen3 MoE W8A8 EP2 and prove single-storage expert residency."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.repro.qwen3_moe_runtime_inspection import (  # noqa: E402
    AUDIT_ENV,
    DEFAULT_SPEC,
    FORBIDDEN_REPLICATION_ENV,
    inspect_model,
    validate_worker_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="float16"
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--compat-shims", action="store_true")
    args = parser.parse_args()
    if args.tensor_parallel_size != DEFAULT_SPEC.world_size:
        parser.error(
            "single-storage acceptance is fixed to TP2+EP2; "
            f"got tensor_parallel_size={args.tensor_parallel_size}"
        )
    return args


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "qwen3_moe_single_storage_runtime_audit",
        "model": str(model_path),
        "quantization": "ascend",
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "expected_model": {
            "layers": DEFAULT_SPEC.expected_layers,
            "global_num_experts": DEFAULT_SPEC.global_num_experts,
            "local_num_experts_per_rank": DEFAULT_SPEC.local_num_experts,
            "top_k": DEFAULT_SPEC.top_k,
            "hidden_size": DEFAULT_SPEC.hidden_size,
            "intermediate_size": DEFAULT_SPEC.intermediate_size,
            "quant_method": DEFAULT_SPEC.expected_quant_method,
            "main_weight_storage_count_per_rank": (
                DEFAULT_SPEC.expected_layers * 2
            ),
            "main_weight_unique_bytes_per_rank": (
                DEFAULT_SPEC.expected_layers
                * DEFAULT_SPEC.local_num_experts
                * (
                    2
                    * DEFAULT_SPEC.intermediate_size
                    * DEFAULT_SPEC.hidden_size
                    + DEFAULT_SPEC.hidden_size * DEFAULT_SPEC.intermediate_size
                )
            ),
        },
        "required_environment": {AUDIT_ENV: "1"},
        "forbidden_replication_environment": list(FORBIDDEN_REPLICATION_ENV),
    }

    try:
        if not model_path.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {model_path}")
        forbidden = {
            name: os.getenv(name)
            for name in FORBIDDEN_REPLICATION_ENV
            if os.getenv(name) not in (None, "", "0")
        }
        if forbidden:
            raise ValueError(
                f"replicated-local environment must be disabled: {forbidden}"
            )
        os.environ[AUDIT_ENV] = "1"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if args.compat_shims:
            from scripts.repro.inspect_vllm_ascend_loaded_quant_params import (
                _install_compat_shims,
            )

            _install_compat_shims()

        from vllm import LLM

        llm = LLM(
            model=str(model_path),
            trust_remote_code=True,
            dtype=args.dtype,
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_expert_parallel=True,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization="ascend",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
        )
        worker_inspector = partial(
            inspect_model,
            require_load_audit=True,
        )
        workers = llm.apply_model(worker_inspector)
        worker_reports = workers if isinstance(workers, list) else [workers]
        failures = validate_worker_reports(worker_reports)
        report.update({
            "workers": worker_reports,
            "worker_count": len(worker_reports),
            "passed": not failures,
            "failures": failures,
        })
    except Exception as exc:
        report.update({
            "workers": [],
            "worker_count": 0,
            "passed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc(),
        })

    _write_report(output_path, report)
    print(json.dumps({
        "passed": report["passed"],
        "worker_count": report["worker_count"],
        "failure_count": len(report["failures"]),
        "main_weight_unique_bytes_per_worker": [
            worker.get("storage_summary", {}).get("main_weight_unique_bytes")
            if isinstance(worker, dict)
            else None
            for worker in report["workers"]
        ],
        "output_json": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
