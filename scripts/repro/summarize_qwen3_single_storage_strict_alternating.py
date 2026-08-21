#!/usr/bin/env python3
"""Validate strict alternating Qwen3 FP16/single-storage W8A8 runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


SEQUENCE = (
    (1, "fp16"),
    (1, "w8a8"),
    (2, "w8a8"),
    (2, "fp16"),
    (3, "fp16"),
    (3, "w8a8"),
)
REPLICATION_SWITCHES = (
    "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS",
    "MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES",
    "MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS",
    "MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE",
)
DISABLED_EXPERIMENTS = (
    "MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING",
    "MINDPIPE_QWEN3_MOE_PREQUANT_MULTICAST",
    "MINDPIPE_QWEN3_MOE_MULTICAST_REDUCE_SCATTER",
    "MINDPIPE_QWEN3_MOE_MULTICAST_BATCHED_P2P",
)
W8A8_SWITCHES = {
    "MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE": "1",
    "MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE_MIN_TOKENS": "8192",
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "1",
    "MINDPIPE_QWEN3_SP_FAST_ROPE": "1",
    "MINDPIPE_QWEN3_SP_SPARSE_LOGITS": "0",
    "MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER": "1",
}
FP16_SWITCHES = {
    "MINDPIPE_QWEN3_MOE_QUANTIZED_EP2_FINALIZE": "0",
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "0",
    "MINDPIPE_QWEN3_SP_FAST_ROPE": "0",
    "MINDPIPE_QWEN3_SP_SPARSE_LOGITS": "0",
    "MINDPIPE_QWEN3_SP_QUANTIZED_QKV_GATHER": "0",
}
MODE_ENV_KEYS = set(W8A8_SWITCHES) | {"VLLM_DISABLE_COMPILE_CACHE"}
RUN_SPECIFIC_ARGUMENT_KEYS = {"mode", "model", "port", "tag", "output_dir", "env"}
EXPECTED_ARGUMENTS = {
    "dtype": "float16",
    "served_model_name": "qwen3-30b-a3b",
    "device": "0,1",
    "host": "127.0.0.1",
    "input_len": 2048,
    "output_len": 16,
    "num_prompts": 64,
    "warmup_num_prompts": 64,
    "warmup_max_concurrency": 32,
    "request_rate": "inf",
    "request_timeout": 1800.0,
    "fixed_synchronized_start": False,
    "seed": 20260712,
    "max_concurrency": 32,
    "max_model_len": 2304,
    "max_num_batched_tokens": 65536,
    "max_num_seqs": 32,
    "num_gpu_blocks_override": None,
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "enable_expert_parallel": True,
    "additional_config": (
        '{"torchair_graph_config":{"enabled":false},'
        '"ascend_scheduler_config":{"enabled":true},"refresh":true}'
    ),
    "compilation_config": (
        '{"cudagraph_mode":"FULL_DECODE_ONLY",'
        '"cudagraph_capture_sizes":[32]}'
    ),
    "disable_prefix_caching": True,
    "disable_chunked_prefill": True,
    "enforce_eager": False,
    "async_scheduling": False,
    "aiv": True,
    "profile": False,
    "startup_timeout": 1800,
    "quality_prompt": ["The capital of France is", "Complete exactly: 2 + 2 ="],
    "quality_max_tokens": 8,
    "quality_timeout": 120,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--fp16-model", type=Path, required=True)
    parser.add_argument("--w8a8-model", type=Path, required=True)
    parser.add_argument("--storage-audit", type=Path, required=True)
    parser.add_argument("--qkv-profile-audit", type=Path, required=True)
    parser.add_argument("--mechanism-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def env_map(values: object) -> dict[str, str]:
    if not isinstance(values, list):
        return {}
    return dict(
        item.split("=", 1)
        for item in values
        if isinstance(item, str) and "=" in item
    )


def request_file_vector(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def artifact_request_vector(path: Path) -> list[dict[str, object]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        body = row.get("request_body")
        if isinstance(body, str):
            body = json.loads(body)
        if not isinstance(body, dict):
            raise ValueError(f"{path}: request_body is not an object")
        result.append(body)
    return result


def successful_response_request_ids(path: Path) -> list[int]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        request_id = row.get("request_id")
        if not isinstance(request_id, int):
            raise ValueError(f"{path}: request_id is not an integer")
        if row.get("line_number") != request_id + 1:
            raise ValueError(f"{path}: line_number does not match request_id")
        if row.get("status") != 200:
            raise ValueError(f"{path}: response status is not 200")
        result.append(request_id)
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate request_id")
    return result


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def server_log_hits(path: Path) -> dict[str, int]:
    text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
    return {
        "quantization": text.count("Using the vLLM Ascend Quantization now!"),
        "sequence_parallel_fast_path": text.count(
            "Sequence-parallel fast path hit"
        ),
    }


def normalized_common_arguments(arguments: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in sorted(arguments.items())
        if key not in RUN_SPECIFIC_ARGUMENT_KEYS
    }


def normalized_common_environment(environment: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in environment.items()
        if key not in MODE_ENV_KEYS
    }


def load_summary(
    root: Path, pair: int, mode: str
) -> tuple[Path, dict[str, object]]:
    directory = root / f"pair{pair}_{mode}"
    paths = [
        path
        for path in sorted(directory.glob("*_summary.json"))
        if not path.name.endswith("_fixed_summary.json")
    ]
    if len(paths) != 1:
        raise ValueError(
            f"expected one service summary in {directory}, found {len(paths)}"
        )
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def summary_artifact_path(
    summary: dict[str, object], key: str, summary_path: Path
) -> Path:
    raw = summary.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{summary_path}: {key} is absent")
    return Path(raw).expanduser().resolve()


def quality_signature(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    for check in result.get("checks", []):
        if isinstance(check, dict):
            check.pop("elapsed_seconds", None)
    return result


def phase_shape_checks(summary: dict[str, object]) -> dict[str, bool]:
    expected = {
        "num_prompts": 64,
        "http_completed": 64,
        "completed": 64,
        "failed": 0,
        "usage_failed": 0,
        "input_tokens": 131072,
        "output_tokens": 1024,
        "prompt_tokens": 131072,
        "completion_tokens": 1024,
        "total_tokens": 132096,
        "prompt_token_vector": [2048] * 64,
        "completion_token_vector": [16] * 64,
    }
    warmup = (summary.get("warmup") or {}).get("summary") or {}
    return {
        "formal_shape": all(summary.get(key) == value for key, value in expected.items()),
        "warmup_shape": all(warmup.get(key) == value for key, value in expected.items()),
    }


def storage_checks(storage: dict[str, object]) -> dict[str, bool]:
    workers = storage.get("workers") or []
    return {
        "passed": storage.get("passed") is True,
        "worker_count": storage.get("worker_count") == 2,
        "local_experts": [
            worker.get("local_global_expert_ids") for worker in workers
        ] == [list(range(64)), list(range(64, 128))],
        "main_weight_bytes": len(workers) == 2 and all(
            worker.get("storage_summary", {}).get("main_weight_unique_bytes")
            == 14_495_514_624
            for worker in workers
        ),
        "single_load": len(workers) == 2 and all(
            worker.get("load_audit", {}).get("duplicate_source_count") == 0
            and worker.get("load_audit", {}).get("loaded_count") == 27_648
            and worker.get("load_audit", {}).get("skipped_nonlocal_count")
            == 27_648
            for worker in workers
        ),
        "no_weight_aliases": len(workers) == 2 and all(
            worker.get("storage_summary", {}).get("aliased_main_weights") == {}
            and worker.get("storage_summary", {}).get(
                "unique_main_weight_storage_count"
            ) == 96
            for worker in workers
        ),
    }


def qkv_profile_checks(value: dict[str, object]) -> dict[str, bool]:
    ranks = value.get("ranks") or []
    return {
        "passed": value.get("passed") is True,
        "kind": value.get("kind") == "qwen3_sp_quantized_qkv_profile_audit",
        "two_ranks": len(ranks) == 2,
        "actual_hits": len(ranks) == 2 and all(
            rank.get("rank") == index
            and rank.get("quantized_qkv_gather_calls", 0) > 0
            and all((rank.get("checks") or {}).values())
            for index, rank in enumerate(ranks)
        ),
    }


def mechanism_checks(value: dict[str, object]) -> dict[str, bool]:
    mechanisms = value.get("mechanisms") or {}
    ep2 = (mechanisms.get("quantized_ep2_finalize") or {}).get("ranks") or []
    rope = (mechanisms.get("fast_rope") or {}).get("ranks") or []
    return {
        "passed": value.get("passed") is True,
        "kind": value.get("kind") == "qwen3_single_storage_mechanism_audit",
        "two_ranks": len(ep2) == len(rope) == 2,
        "quantized_ep2_actual_hits": len(ep2) == 2 and all(
            rank.get("rank") == index
            and bool(rank.get("quantized_finalize_pairs"))
            and all((rank.get("checks") or {}).values())
            for index, rank in enumerate(ep2)
        ),
        "fast_rope_actual_hits": len(rope) == 2 and all(
            rank.get("rank") == index
            and rank.get("fast_rope_calls", 0) > 0
            and all((rank.get("checks") or {}).values())
            for index, rank in enumerate(rope)
        ),
    }


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    request_file = args.request_file.expanduser().resolve()
    expected_requests = request_file_vector(request_file)
    expected_models = {
        "fp16": str(args.fp16_model.expanduser().resolve()),
        "w8a8": str(args.w8a8_model.expanduser().resolve()),
    }
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    common_arguments: list[dict[str, object]] = []
    common_environments: list[dict[str, str]] = []
    quality_signatures: dict[str, list[dict[str, object]]] = {
        "fp16": [],
        "w8a8": [],
    }

    if len(expected_requests) != 64:
        failures.append(
            f"request file contains {len(expected_requests)} rows, expected 64"
        )

    for sequence_index, (pair, mode) in enumerate(SEQUENCE):
        path, summary = load_summary(root, pair, mode)
        label = f"pair{pair}_{mode}"
        environment = env_map(summary.get("env_overrides"))
        arguments = summary.get("arguments") or {}
        checks = {
            "returncode": summary.get("returncode") == 0,
            "status": summary.get("status") == "completed",
            "diagnostic_only": summary.get("diagnostic_only") is False,
            "issues_empty": summary.get("issues") == [],
            "teardown": summary.get("teardown_complete") is True,
            "post_benchmark_health": summary.get("post_benchmark_health") is True,
            "process_group_gone": (
                summary.get("teardown_evidence") or {}
            ).get("process_group_gone") is True,
            "port_released": (
                summary.get("teardown_evidence") or {}
            ).get("port_released") is True,
            "single_global_wall_clock": (
                summary.get("aggregation") or {}
            ).get("basis") == "single_global_wall_clock",
            "endpoint_throughputs_not_summed": (
                summary.get("aggregation") or {}
            ).get("endpoint_local_throughputs_summed") is False,
            "sequence_index": summary.get("acceptance_sequence_index")
            == sequence_index,
            "common_arguments_exact": all(
                arguments.get(key) == value
                for key, value in EXPECTED_ARGUMENTS.items()
            ),
            "request_path": Path(
                str(arguments.get("request_file", ""))
            ).expanduser().resolve() == request_file,
            "replication_disabled": all(
                environment.get(name) == "0" for name in REPLICATION_SWITCHES
            ),
            "experiments_disabled": all(
                environment.get(name) == "0" for name in DISABLED_EXPERIMENTS
            ),
            "storage_audit_disabled_during_timing": environment.get(
                "MINDPIPE_QWEN3_MOE_SINGLE_STORAGE_AUDIT"
            ) == "0",
            "quality": summary.get("quality_completed") == 2
            and summary.get("quality_failed") == 0,
            "engine_dtype": summary.get("engine_dtype") == "torch.float16",
            "model_path": summary.get("model") == expected_models[mode]
            and arguments.get("model") == expected_models[mode],
            "weight_rank_vector": summary.get("weights_memory_gb_vector")
            == (
                [14.6194, 14.6194]
                if mode == "w8a8"
                else [28.4573, 28.4573]
            ),
            **phase_shape_checks(summary),
        }

        log_path = summary_artifact_path(summary, "server_log", path)
        checks["server_log_exists"] = log_path.is_file()
        hits = server_log_hits(log_path) if log_path.is_file() else {}
        if mode == "w8a8":
            checks.update({
                "candidate_switches": all(
                    environment.get(key) == value
                    for key, value in W8A8_SWITCHES.items()
                ),
                "weight_watermark": math.isclose(
                    float(summary.get("weights_memory_gb", -1)),
                    14.6194,
                    rel_tol=0.0,
                    abs_tol=1e-4,
                ),
                "quantization": summary.get("engine_quantization") == "ascend",
                "quantization_summary_hit": summary.get(
                    "ascend_quantization_log"
                ) is True,
                "quantization_two_rank_log_hit": hits.get("quantization") == 2,
                "sequence_parallel_two_rank_log_hit": hits.get(
                    "sequence_parallel_fast_path"
                ) == 2,
            })
        else:
            checks.update({
                "candidate_switches_disabled": all(
                    environment.get(key) == value
                    for key, value in FP16_SWITCHES.items()
                ),
                "quantization_disabled": summary.get("engine_quantization") is None,
                "no_quantization_log_hit": hits.get("quantization") == 0,
                "no_sequence_parallel_log_hit": hits.get(
                    "sequence_parallel_fast_path"
                ) == 0,
            })

        artifacts = {
            "formal_requests": summary_artifact_path(summary, "requests_jsonl", path),
            "formal_responses": summary_artifact_path(summary, "responses_jsonl", path),
            "warmup_responses": Path(
                str((summary.get("warmup") or {}).get("responses_jsonl", ""))
            ).expanduser().resolve(),
            "quality": summary_artifact_path(summary, "quality_result_json", path),
        }
        for artifact_name, artifact_path in artifacts.items():
            checks[f"{artifact_name}_exists"] = artifact_path.is_file()
        checks["formal_requests_matches_request_file"] = (
            artifacts["formal_requests"].is_file()
            and artifact_request_vector(artifacts["formal_requests"])
            == expected_requests
        )
        expected_request_ids = list(range(len(expected_requests)))
        for artifact_name in ("formal_responses", "warmup_responses"):
            artifact_path = artifacts[artifact_name]
            checks[f"{artifact_name}_covers_request_file"] = (
                artifact_path.is_file()
                and successful_response_request_ids(artifact_path)
                == expected_request_ids
            )
        if artifacts["quality"].is_file():
            quality_signatures[mode].append(quality_signature(artifacts["quality"]))

        duration = float(summary["duration"])
        throughput = float(summary["total_token_throughput"])
        checks["throughput_formula"] = math.isclose(
            throughput,
            132_096 / duration,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        bad = [name for name, passed in checks.items() if not passed]
        if bad:
            failures.append(f"{label}: failed checks: {', '.join(bad)}")
        rows.append({
            "sequence_index": sequence_index,
            "pair": pair,
            "mode": mode,
            "duration": duration,
            "total_token_throughput": throughput,
            "summary": str(path),
            "checks": checks,
            "server_log_hits": hits,
        })
        common_arguments.append(normalized_common_arguments(arguments))
        common_environments.append(normalized_common_environment(environment))

    common_workload_identical = len(common_arguments) == 6 and all(
        item == common_arguments[0] for item in common_arguments[1:]
    )
    common_environment_identical = len(common_environments) == 6 and all(
        item == common_environments[0] for item in common_environments[1:]
    )
    if not common_workload_identical:
        failures.append("six runs: normalized workload arguments differ")
    if not common_environment_identical:
        failures.append("six runs: normalized common environments differ")

    quality_determinism = {}
    for mode, signatures in quality_signatures.items():
        identical = len(signatures) == 3 and all(
            item == signatures[0] for item in signatures[1:]
        )
        quality_determinism[f"{mode}_three_runs_identical"] = identical
        if not identical:
            failures.append(
                f"{mode}: independent quality results differ across fresh servers"
            )

    pair_results = []
    for pair in (1, 2, 3):
        fp16 = next(
            row for row in rows if row["pair"] == pair and row["mode"] == "fp16"
        )
        w8a8 = next(
            row for row in rows if row["pair"] == pair and row["mode"] == "w8a8"
        )
        ratio = (
            float(w8a8["total_token_throughput"])
            / float(fp16["total_token_throughput"])
        )
        passed = ratio >= 1.5
        if not passed:
            failures.append(f"pair{pair}: speedup {ratio:.15f}x is below 1.5x")
        pair_results.append({"pair": pair, "speedup": ratio, "passed": passed})

    fp16_values = [
        float(row["total_token_throughput"])
        for row in rows
        if row["mode"] == "fp16"
    ]
    w8a8_values = [
        float(row["total_token_throughput"])
        for row in rows
        if row["mode"] == "w8a8"
    ]
    mean_fp16 = statistics.mean(fp16_values)
    mean_w8a8 = statistics.mean(w8a8_values)
    aggregate_ratio = mean_w8a8 / mean_fp16
    if aggregate_ratio < 1.5:
        failures.append(
            f"aggregate speedup {aggregate_ratio:.15f}x is below 1.5x"
        )

    storage_path = args.storage_audit.expanduser().resolve()
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    single_storage_checks = storage_checks(storage)
    bad_storage = [
        name for name, passed in single_storage_checks.items() if not passed
    ]
    if bad_storage:
        failures.append(
            f"storage audit: failed checks: {', '.join(bad_storage)}"
        )

    qkv_path = args.qkv_profile_audit.expanduser().resolve()
    qkv_audit = json.loads(qkv_path.read_text(encoding="utf-8"))
    qkv_checks = qkv_profile_checks(qkv_audit)
    bad_qkv = [name for name, passed in qkv_checks.items() if not passed]
    if bad_qkv:
        failures.append(
            f"QKV profiler audit: failed checks: {', '.join(bad_qkv)}"
        )

    mechanism_path = args.mechanism_audit.expanduser().resolve()
    mechanism_audit = json.loads(mechanism_path.read_text(encoding="utf-8"))
    natural_mechanism_checks = mechanism_checks(mechanism_audit)
    bad_mechanisms = [
        name for name, passed in natural_mechanism_checks.items() if not passed
    ]
    if bad_mechanisms:
        failures.append(
            "mechanism profiler audit: failed checks: "
            + ", ".join(bad_mechanisms)
        )

    result = {
        "schema_version": 4,
        "kind": "qwen3_single_storage_strict_alternating_acceptance",
        "passed": not failures,
        "failures": failures,
        "strict_speedup_threshold": 1.5,
        "comparison_policy": "unrounded",
        "request_file": str(request_file),
        "request_count": len(expected_requests),
        "common_workload_profile": {
            "identical_across_six_runs": common_workload_identical,
            "common_environment_identical_across_six_runs": (
                common_environment_identical
            ),
            "normalized_arguments": common_arguments,
            "normalized_environments": common_environments,
        },
        "rows": rows,
        "pairs": pair_results,
        "aggregate": {
            "formula": (
                "mean(W8A8 total_token_throughput) / "
                "mean(FP16 total_token_throughput)"
            ),
            "mean_fp16_total_token_throughput": mean_fp16,
            "mean_w8a8_total_token_throughput": mean_w8a8,
            "mean_throughput_ratio": aggregate_ratio,
            "passed": aggregate_ratio >= 1.5,
        },
        "single_storage_audit": {
            "path": str(storage_path),
            "checks": single_storage_checks,
        },
        "mechanism_evidence": {
            "qkv_profile_audit": {
                "path": str(qkv_path),
                "checks": qkv_checks,
            },
            "natural_mechanism_audit": {
                "path": str(mechanism_path),
                "checks": natural_mechanism_checks,
            },
        },
        "quality_determinism": quality_determinism,
    }
    output = args.output.expanduser().resolve()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
