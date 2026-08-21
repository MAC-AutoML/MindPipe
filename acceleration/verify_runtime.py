#!/usr/bin/env python3
"""Verify that all four MindPipe acceleration paths are installed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from install_runtime_patch import patch_state, runtime_patches, validate_layout


REQUIRED_MARKERS = {
    "vllm": {
        "vllm/model_executor/models/qwen3_moe.py": (
            "MINDPIPE_QWEN3_MOE_SINGLE_STORAGE_AUDIT",
        ),
        "vllm/model_executor/models/qwen2.py": (
            "MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT",
            "MINDPIPE_QWEN2_MLP_FUSED_ALLREDUCE_W8A8",
        ),
        "vllm/v1/engine/core.py": ("MINDPIPE_ENGINE_IDLE_COALESCE_MS",),
        "vllm/model_executor/models/mindpipe_qwen2_loop_out.py": (
            "mindpipe_qwen2_grouped_swiglu_loop_out",
        ),
    },
    "vllm-ascend": {
        "vllm_ascend/attention/attention_v1.py": (
            "MINDPIPE_W8A8_FIA_FASTPATH",
        ),
        "vllm_ascend/ops/linear_op.py": (
            "MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER",
            "MINDPIPE_MIXTRAL_ATTN_COMM_QUANT",
        ),
        "vllm_ascend/ops/rotary_embedding.py": ("MINDPIPE_QWEN3_SP_FAST_ROPE",),
        "vllm_ascend/ops/moe/token_dispatcher.py": (
            "MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE",
        ),
        "vllm_ascend/ops/moe/mixtral_tp4_routing_exact_chain.py": (
            "MINDPIPE_MIXTRAL_TP4_MOE_PIPELINE",
        ),
        "vllm_ascend/ops/layernorm.py": (
            "MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT",
        ),
    },
}

FORBIDDEN_MARKERS = (
    "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS",
    "MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES",
    "MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS",
    "MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--vllm-ascend-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    roots = {
        "vllm": args.vllm_root.expanduser().resolve(),
        "vllm-ascend": args.vllm_ascend_root.expanduser().resolve(),
    }
    report: dict[str, object] = {"valid": True, "runtimes": {}}
    try:
        for item in runtime_patches(roots["vllm"], roots["vllm-ascend"]):
            validate_layout(item)
            state = patch_state(item)
            if state != "applied":
                raise RuntimeError(f"{item.name} patch is not installed")
        for name, files in REQUIRED_MARKERS.items():
            runtime_report: dict[str, object] = {"root": str(roots[name]), "files": {}}
            for relative, markers in files.items():
                path = roots[name] / relative
                if not path.is_file():
                    raise RuntimeError(f"Required installed file is absent: {path}")
                text = path.read_text(encoding="utf-8")
                missing = [marker for marker in markers if marker not in text]
                forbidden = [marker for marker in FORBIDDEN_MARKERS if marker in text]
                if missing or forbidden:
                    raise RuntimeError(
                        f"Runtime marker verification failed for {path}: "
                        f"missing={missing}, forbidden={forbidden}"
                    )
                runtime_report["files"][relative] = {"markers": list(markers)}
            report["runtimes"][name] = runtime_report
    except RuntimeError as exc:
        report["valid"] = False
        report["error"] = str(exc)

    rendered = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
