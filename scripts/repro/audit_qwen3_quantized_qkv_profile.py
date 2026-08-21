#!/usr/bin/env python3
"""Prove Qwen3 quantized QKV gather from two-rank profiler databases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def profiler_dbs(root: Path) -> list[Path]:
    paths = sorted(
        root.expanduser().resolve().rglob("ascend_pytorch_profiler_*.db")
    )
    if len(paths) != 2:
        raise ValueError(f"expected two profiler DBs under {root}, found {len(paths)}")
    return paths


def database_evidence(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_unix_ns": stat.st_mtime_ns,
    }


def read_rank(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        communication = connection.execute(
            """
            SELECT dtype.name, communication.count, COUNT(*)
            FROM COMMUNICATION_OP AS communication
            JOIN STRING_IDS AS op_type ON op_type.id = communication.opType
            JOIN ENUM_HCCL_DATA_TYPE AS dtype ON dtype.id = communication.dataType
            WHERE op_type.value = 'hcom_allGather_'
            GROUP BY dtype.name, communication.count
            ORDER BY dtype.name, communication.count
            """
        ).fetchall()
        dynamic_quant_calls = connection.execute(
            """
            SELECT COUNT(*)
            FROM CANN_API AS api
            JOIN STRING_IDS AS name ON name.id = api.name
            WHERE name.value = 'aclnnDynamicQuantV2'
            """
        ).fetchone()[0]
        npus = [list(row) for row in connection.execute("SELECT * FROM NPU_INFO")]
        rank_devices = [
            list(row) for row in connection.execute("SELECT * FROM RANK_DEVICE_MAP")
        ]
    finally:
        connection.close()
    gathers: dict[str, dict[int, int]] = defaultdict(dict)
    for dtype, count, calls in communication:
        gathers[str(dtype)][int(count)] = int(calls)
    return {
        "database": database_evidence(path),
        "dynamic_quant_calls": int(dynamic_quant_calls),
        "all_gathers": dict(gathers),
        "npus": npus,
        "rank_devices": rank_devices,
    }


def paired_shapes(record: dict[str, object]) -> list[dict[str, int]]:
    gathers = record["all_gathers"]
    result = []
    for activation_elements, activation_calls in sorted(
        gathers.get("INT8", {}).items()
    ):
        if activation_elements % 2048:
            continue
        local_tokens = activation_elements // 2048
        scale_calls = gathers.get("FP32", {}).get(local_tokens)
        if scale_calls != activation_calls:
            continue
        residual_calls = gathers.get("FP16", {}).get(activation_elements, 0)
        result.append({
            "local_token_count": local_tokens,
            "activation_elements": activation_elements,
            "int8_activation_gather_calls": activation_calls,
            "fp32_token_scale_gather_calls": scale_calls,
            "residual_fp16_activation_gather_calls": residual_calls,
        })
    return result


def main() -> int:
    args = parse_args()
    controls = [read_rank(path) for path in profiler_dbs(args.control_root)]
    candidates = [read_rank(path) for path in profiler_dbs(args.candidate_root)]
    failures: list[str] = []
    ranks = []
    for rank, (control, candidate) in enumerate(zip(controls, candidates)):
        control_pairs = paired_shapes(control)
        candidate_pairs = paired_shapes(candidate)
        dynamic_delta = (
            candidate["dynamic_quant_calls"] - control["dynamic_quant_calls"]
        )
        gather_calls = sum(
            pair["int8_activation_gather_calls"] for pair in candidate_pairs
        )
        checks = {
            "control_has_no_quantized_qkv_pairs": not control_pairs,
            "candidate_has_quantized_qkv_pairs": bool(candidate_pairs),
            "candidate_has_47_layer_pattern": all(
                pair["int8_activation_gather_calls"]
                == 47 * max(1, pair["residual_fp16_activation_gather_calls"])
                for pair in candidate_pairs
            ),
            "int8_and_scale_calls_match": all(
                pair["int8_activation_gather_calls"]
                == pair["fp32_token_scale_gather_calls"]
                for pair in candidate_pairs
            ),
            "profiles_ascend_910b": (
                control["npus"] == candidate["npus"] == [[rank, "Ascend910B"]]
            ),
            "rank_device_mapping": (
                control["rank_devices"]
                == candidate["rank_devices"]
                == [[rank, rank]]
            ),
        }
        bad = [name for name, passed in checks.items() if not passed]
        if bad:
            failures.append(f"rank {rank}: failed checks: {', '.join(bad)}")
        ranks.append({
            "rank": rank,
            "control_database": control["database"],
            "candidate_database": candidate["database"],
            "dynamic_quant_call_delta": dynamic_delta,
            "quantized_qkv_gather_calls": gather_calls,
            "paired_shapes": candidate_pairs,
            "checks": checks,
        })
    result = {
        "schema_version": 3,
        "kind": "qwen3_sp_quantized_qkv_profile_audit",
        "passed": not failures,
        "failures": failures,
        "ranks": ranks,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
