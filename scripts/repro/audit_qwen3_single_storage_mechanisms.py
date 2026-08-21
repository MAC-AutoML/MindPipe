#!/usr/bin/env python3
"""Prove Qwen3 Fast RoPE and quantized EP2 finalize from profiler DBs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


HIDDEN_SIZE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def profiler_dbs(root: Path) -> list[Path]:
    paths = sorted(
        root.expanduser().resolve().rglob("ascend_pytorch_profiler_*.db")
    )
    if len(paths) != 2:
        raise ValueError(f"expected two profiler DBs under {root}, found {len(paths)}")
    return paths


def named_counts(
    connection: sqlite3.Connection, table: str, needle: str
) -> dict[str, int]:
    rows = connection.execute(
        f"""
        SELECT names.value, COUNT(*)
        FROM {table} AS events
        JOIN STRING_IDS AS names ON names.id = events.name
        WHERE names.value LIKE ?
        GROUP BY names.value
        ORDER BY names.value
        """,
        (needle,),
    ).fetchall()
    return {str(name): int(count) for name, count in rows}


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
        rows = connection.execute(
            """
            SELECT dtype.name, communication.count, COUNT(*)
            FROM COMMUNICATION_OP AS communication
            JOIN STRING_IDS AS op_type ON op_type.id = communication.opType
            JOIN ENUM_HCCL_DATA_TYPE AS dtype ON dtype.id = communication.dataType
            WHERE op_type.value = 'hcom_allReduce_'
            GROUP BY dtype.name, communication.count
            ORDER BY dtype.name, communication.count
            """
        ).fetchall()
        result = {
            "database": database_evidence(path),
            "pytorch_rope": named_counts(connection, "PYTORCH_API", "%rotary%"),
            "cann_rope": named_counts(connection, "CANN_API", "%ApplyRotaryPosEmb%"),
            "kernel_rope": named_counts(
                connection, "COMPUTE_TASK_INFO", "%ApplyRotaryPosEmb%"
            ),
            "npus": [list(row) for row in connection.execute("SELECT * FROM NPU_INFO")],
            "rank_devices": [
                list(row)
                for row in connection.execute("SELECT * FROM RANK_DEVICE_MAP")
            ],
        }
    finally:
        connection.close()
    all_reduces: dict[str, dict[int, int]] = {}
    for dtype, count, calls in rows:
        all_reduces.setdefault(str(dtype), {})[int(count)] = int(calls)
    result["all_reduces"] = all_reduces
    return result


def ep2_pairs(record: dict[str, object]) -> list[dict[str, int]]:
    reductions = record["all_reduces"]
    pairs = []
    for token_count, scale_calls in sorted(reductions.get("FP32", {}).items()):
        payload_calls = reductions.get("INT8", {}).get(token_count * HIDDEN_SIZE)
        if payload_calls is None:
            continue
        pairs.append({
            "token_count": token_count,
            "fp32_scale_all_reduce_calls": scale_calls,
            "int8_payload_all_reduce_calls": payload_calls,
        })
    return pairs


def main() -> int:
    args = parse_args()
    controls = [read_rank(path) for path in profiler_dbs(args.control_root)]
    candidates = [read_rank(path) for path in profiler_dbs(args.candidate_root)]
    failures: list[str] = []
    ep2_ranks = []
    rope_ranks = []
    for rank, (control, candidate) in enumerate(zip(controls, candidates)):
        pairs = ep2_pairs(candidate)
        ep2_checks = {
            "has_quantized_finalize_pairs": bool(pairs),
            "scale_and_payload_calls_match": all(
                pair["fp32_scale_all_reduce_calls"]
                == pair["int8_payload_all_reduce_calls"]
                for pair in pairs
            ),
            "all_pairs_cover_complete_48_layer_passes": all(
                pair["fp32_scale_all_reduce_calls"] > 0
                and pair["fp32_scale_all_reduce_calls"] % 48 == 0
                for pair in pairs
            ),
            "profiles_ascend_910b": candidate["npus"] == [[rank, "Ascend910B"]],
            "rank_device_mapping": candidate["rank_devices"] == [[rank, rank]],
        }
        control_fast = control["pytorch_rope"].get(
            "npu::npu_apply_rotary_pos_emb", 0
        )
        control_old = sum(
            count
            for name, count in control["pytorch_rope"].items()
            if "rotary_embedding" in name
        )
        candidate_fast = candidate["pytorch_rope"].get(
            "npu::npu_apply_rotary_pos_emb", 0
        )
        candidate_api = candidate["pytorch_rope"].get(
            "aclnnApplyRotaryPosEmbV2", 0
        )
        candidate_kernel = candidate["kernel_rope"].get("ApplyRotaryPosEmb", 0)
        rope_checks = {
            "control_uses_old_rope_path": control_old > 0 and control_fast == 0,
            "candidate_hits_npu_fast_rope": candidate_fast > 0,
            "candidate_api_and_kernel_match": (
                candidate_fast == candidate_api == candidate_kernel
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
        bad_ep2 = [name for name, passed in ep2_checks.items() if not passed]
        bad_rope = [name for name, passed in rope_checks.items() if not passed]
        if bad_ep2:
            failures.append(f"ep2 rank {rank}: failed checks: {', '.join(bad_ep2)}")
        if bad_rope:
            failures.append(f"rope rank {rank}: failed checks: {', '.join(bad_rope)}")
        ep2_ranks.append({
            "rank": rank,
            "database": candidate["database"],
            "quantized_finalize_pairs": pairs,
            "checks": ep2_checks,
        })
        rope_ranks.append({
            "rank": rank,
            "control_database": control["database"],
            "candidate_database": candidate["database"],
            "fast_rope_calls": candidate_fast,
            "checks": rope_checks,
        })
    result = {
        "schema_version": 3,
        "kind": "qwen3_single_storage_mechanism_audit",
        "passed": not failures,
        "failures": failures,
        "mechanisms": {
            "quantized_ep2_finalize": {"ranks": ep2_ranks},
            "fast_rope": {"ranks": rope_ranks},
        },
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
