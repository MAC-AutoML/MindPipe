#!/usr/bin/env python3
"""Validate and summarize the strict alternating Qwen3 C32 acceptance runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_SEQUENCE = (
    (1, "fp16"),
    (1, "w8a8"),
    (2, "w8a8"),
    (2, "fp16"),
    (3, "fp16"),
    (3, "w8a8"),
)

STRICT_SPEEDUP_THRESHOLD = 1.5
THROUGHPUT_REL_TOLERANCE = 1e-6
THROUGHPUT_ABS_TOLERANCE = 1e-6

PINNED_RUNTIME_HEADS = {
    "vllm": "8ce5d3198d00631a76e1aa02a57947b46bc7218c",
    "vllm_ascend": "00ba07102212c7c7a40de427f09848f2e203c498",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
VLLM_VERSION_GIT_PATTERN = re.compile(
    r"vLLM API server version [^\r\n]*\+g([0-9a-f]{7,40})"
)
LEGACY_COMMON_LOG_MARKERS = (
    "Engine idle request coalescing enabled",
)
LEGACY_W8A8_LOG_MARKERS = (
    "Using experimental Qwen3 W8A8 replicated-local MoE",
    "Using runtime Qwen3 replicated-local single-pass 128-expert MLP",
)

EXPECTED_PROFILE: dict[str, Any] = {
    "device": "0,1",
    "host": "127.0.0.1",
    "input_len": 2048,
    "output_len": 16,
    "num_prompts": 64,
    "warmup_num_prompts": 64,
    "warmup_max_concurrency": 32,
    "request_rate": "inf",
    "seed": 20260712,
    "max_concurrency": 32,
    "max_model_len": 2304,
    "max_num_batched_tokens": 65536,
    "max_num_seqs": 32,
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "enable_expert_parallel": True,
    "disable_prefix_caching": True,
    "disable_chunked_prefill": True,
    "enforce_eager": False,
    "async_scheduling": False,
    "aiv": True,
    "compilation_config": {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [32],
    },
    "additional_config": {
        "torchair_graph_config": {"enabled": False},
        "ascend_scheduler_config": {"enabled": True},
        "refresh": True,
    },
}

COMMON_ENV = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MINDPIPE_ENGINE_IDLE_COALESCE_US": "30000",
    "MINDPIPE_ENGINE_IDLE_COALESCE_TARGET_ADDS": "31",
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "0",
    "VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP": "0",
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "0",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE_W8A8": "0",
}

MODE_ENV = {
    "fp16": {
        "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL": "0",
        "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH": "0",
        "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS": "0",
    },
    "w8a8": {
        "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL": "1",
        "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH": "1",
        "MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT": "1",
        "MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_TARGETS": "qkv",
        "MINDPIPE_W8A8_DYNAMIC_RMSNORM_QUANT_MAX_TOKENS": "0",
        "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS": "1",
        "MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES": "0",
        "MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS": "1",
        "MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE": "0",
        "MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING": "0",
        "MINDPIPE_QWEN3_MOE_GLOBAL_ROUTING_QUANT": "0",
        "MINDPIPE_QWEN3_MOE_QUANTIZED_PEER_REDUCE_SCATTER": "0",
        "MINDPIPE_QWEN3_MOE_GMM2_TUNING": "0",
    },
}


class EvidenceError(RuntimeError):
    """Raised when the acceptance evidence cannot be loaded."""


def _normalized_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(Path(value).expanduser().resolve())


def _runtime_identity(
    sources: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "head_commit": record["head_commit"],
            "source_fingerprint_sha256": record[
                "source_fingerprint_sha256"
            ],
        }
        for name, record in sorted(sources.items())
    }


def _normalize_runtime_sources(
    value: Any,
    label: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list) or not value:
        return {}, [f"{label}: runtime_sources must be a non-empty list"]

    pinned_by_head = {head: name for name, head in PINNED_RUNTIME_HEADS.items()}
    for index, record in enumerate(value):
        record_label = f"{label}.runtime_sources[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{record_label}: must be an object")
            continue
        head = record.get("head_commit")
        fingerprint = record.get("source_fingerprint_sha256")
        if not isinstance(head, str) or GIT_COMMIT_PATTERN.fullmatch(head) is None:
            errors.append(f"{record_label}: invalid head_commit={head!r}")
            continue
        runtime_name = pinned_by_head.get(head)
        if runtime_name is None:
            errors.append(
                f"{record_label}: head_commit={head!r} is not a pinned runtime HEAD"
            )
            continue
        if runtime_name in normalized:
            errors.append(f"{record_label}: duplicate pinned runtime {runtime_name}")
            continue
        if (
            not isinstance(fingerprint, str)
            or SHA256_PATTERN.fullmatch(fingerprint) is None
        ):
            errors.append(
                f"{record_label}: invalid source_fingerprint_sha256="
                f"{fingerprint!r}"
            )
            continue
        normalized[runtime_name] = {
            "head_commit": head,
            "source_fingerprint_sha256": fingerprint,
            "pythonpath": _normalized_path(record.get("pythonpath")),
            "git_root": _normalized_path(record.get("git_root")),
        }

    missing = sorted(set(PINNED_RUNTIME_HEADS) - set(normalized))
    if missing:
        errors.append(f"{label}: missing pinned runtime sources: {missing!r}")
    if len(value) != len(PINNED_RUNTIME_HEADS):
        errors.append(
            f"{label}: expected exactly {len(PINNED_RUNTIME_HEADS)} runtime "
            f"sources, got {len(value)}"
        )
    return normalized, errors


def _summary_runtime_sources(summary: dict[str, Any]) -> tuple[Any, str]:
    if "runtime_sources" in summary:
        return summary["runtime_sources"], "summary.runtime_sources"
    metadata = summary.get("runtime_metadata")
    if isinstance(metadata, dict) and "runtime_sources" in metadata:
        return metadata["runtime_sources"], "summary.runtime_metadata.runtime_sources"
    arguments = summary.get("arguments")
    if isinstance(arguments, dict) and "runtime_sources" in arguments:
        return arguments["runtime_sources"], "arguments.runtime_sources"
    return None, "missing"


def _evidence_path(value: Any, summary_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _env_map(assignments: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(assignments, list):
        return result
    for assignment in assignments:
        if not isinstance(assignment, str) or "=" not in assignment:
            continue
        key, value = assignment.split("=", 1)
        result[key] = value
    return result


def _equal_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be numeric, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise EvidenceError(f"{name} must be finite and positive, got {value!r}")
    return numeric


def _load_result_json(summary: dict[str, Any], summary_path: Path) -> tuple[dict[str, Any], Path]:
    raw_path = summary.get("result_json")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError(f"{summary_path}: missing result_json")
    result_path = Path(raw_path)
    if not result_path.is_absolute():
        result_path = summary_path.parent / result_path
    if not result_path.is_file():
        raise EvidenceError(f"{summary_path}: result_json does not exist: {result_path}")
    try:
        return json.loads(result_path.read_text(encoding="utf-8")), result_path.resolve()
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"failed to load {result_path}: {exc}") from exc


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in (1, 2, 3):
        for mode in ("fp16", "w8a8"):
            directory = root / f"pair{pair}_{mode}"
            summaries = sorted(directory.glob("*_summary.json"))
            if len(summaries) != 1:
                raise EvidenceError(
                    f"expected one summary in {directory}, got {len(summaries)}"
                )
            summary_path = summaries[0]
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise EvidenceError(f"failed to load {summary_path}: {exc}") from exc
            result, result_path = _load_result_json(summary, summary_path)
            rows.append(
                {
                    "pair": pair,
                    "mode": mode,
                    "summary": summary,
                    "summary_path": summary_path.resolve(),
                    "result": result,
                    "result_path": result_path,
                }
            )
    return rows


def _validate_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    pair = row["pair"]
    mode = row["mode"]
    summary = row["summary"]
    result = row["result"]
    arguments = summary.get("arguments")
    errors: list[str] = []
    label = f"pair{pair}_{mode}"

    if not isinstance(arguments, dict):
        arguments = {}
        errors.append(f"{label}: missing arguments object")

    model = _normalized_path(summary.get("model"))
    argument_model = _normalized_path(arguments.get("model"))
    python = _normalized_path(summary.get("python"))
    argument_python = _normalized_path(arguments.get("python"))
    device = summary.get("device")
    if model is None:
        errors.append(f"{label}: missing summary model")
    if argument_model is None:
        errors.append(f"{label}: missing arguments.model")
    elif model is not None and argument_model != model:
        errors.append(
            f"{label}: arguments.model={argument_model!r}, expected {model!r}"
        )
    if python is None:
        errors.append(f"{label}: missing summary python")
    if argument_python is None:
        errors.append(f"{label}: missing arguments.python")
    elif python is not None and argument_python != python:
        errors.append(
            f"{label}: arguments.python={argument_python!r}, expected {python!r}"
        )

    runtime_value, runtime_location = _summary_runtime_sources(summary)
    runtime_sources: dict[str, dict[str, Any]] | None = None
    runtime_origin = "missing"
    if runtime_value is not None:
        runtime_sources, runtime_errors = _normalize_runtime_sources(
            runtime_value,
            label,
        )
        errors.extend(runtime_errors)
        runtime_origin = runtime_location

    if summary.get("mode") != mode:
        errors.append(f"{label}: summary mode={summary.get('mode')!r}")
    if arguments.get("mode") != mode:
        errors.append(f"{label}: arguments.mode={arguments.get('mode')!r}")
    if result.get("mode") != mode:
        errors.append(f"{label}: result mode={result.get('mode')!r}")

    for field, expected in EXPECTED_PROFILE.items():
        actual = _json_value(arguments.get(field))
        if actual != expected:
            errors.append(f"{label}: {field}={actual!r}, expected {expected!r}")
    if summary.get("device") != EXPECTED_PROFILE["device"]:
        errors.append(
            f"{label}: summary device={summary.get('device')!r}, "
            f"expected {EXPECTED_PROFILE['device']!r}"
        )
    if summary.get("seed") != EXPECTED_PROFILE["seed"]:
        errors.append(
            f"{label}: summary seed={summary.get('seed')!r}, "
            f"expected {EXPECTED_PROFILE['seed']!r}"
        )

    env = _env_map(arguments.get("env"))
    for key, expected in {**COMMON_ENV, **MODE_ENV[mode]}.items():
        if env.get(key) != expected:
            errors.append(f"{label}: env {key}={env.get(key)!r}, expected {expected!r}")

    for field, expected in (("completed", 64), ("failed", 0), ("returncode", 0)):
        if summary.get(field) != expected:
            errors.append(f"{label}: {field}={summary.get(field)!r}, expected {expected}")

    duration = _require_number(summary.get("duration"), f"{label}.duration")
    throughput = _require_number(
        summary.get("total_token_throughput"),
        f"{label}.total_token_throughput",
    )
    input_tokens = summary.get("input_tokens")
    output_tokens = summary.get("output_tokens")
    input_tokens_valid = (
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens > 0
    )
    output_tokens_valid = (
        isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens > 0
    )
    if not input_tokens_valid:
        errors.append(f"{label}: invalid input_tokens={input_tokens!r}")
    if output_tokens != EXPECTED_PROFILE["num_prompts"] * EXPECTED_PROFILE["output_len"]:
        errors.append(
            f"{label}: output_tokens={output_tokens!r}, expected "
            f"{EXPECTED_PROFILE['num_prompts'] * EXPECTED_PROFILE['output_len']}"
        )

    total_tokens = None
    derived_throughput = None
    throughput_consistent = False
    if input_tokens_valid and output_tokens_valid:
        total_tokens = input_tokens + output_tokens
        derived_throughput = total_tokens / duration
        throughput_consistent = math.isclose(
            throughput,
            derived_throughput,
            rel_tol=THROUGHPUT_REL_TOLERANCE,
            abs_tol=THROUGHPUT_ABS_TOLERANCE,
        )
        if not throughput_consistent:
            errors.append(
                f"{label}: total_token_throughput={throughput!r} does not match "
                f"(input_tokens + output_tokens) / duration="
                f"{derived_throughput!r}"
            )

    result_fields = {
        "duration": duration,
        "completed": summary.get("completed"),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_token_throughput": throughput,
    }
    for field, expected in result_fields.items():
        if not _equal_number(result.get(field), expected):
            errors.append(
                f"{label}: result {field}={result.get(field)!r}, expected {expected!r}"
            )

    run_date = result.get("date")
    try:
        parsed_date = datetime.strptime(run_date, "%Y%m%d-%H%M%S")
    except (TypeError, ValueError):
        parsed_date = None
        errors.append(f"{label}: invalid result date={run_date!r}")

    return (
        {
            "pair": pair,
            "mode": mode,
            "model": model,
            "python": python,
            "device": device,
            "runtime_sources": runtime_sources,
            "runtime_provenance_origin": runtime_origin,
            "summary": str(row["summary_path"]),
            "result_json": str(row["result_path"]),
            "run_date": run_date,
            "_parsed_date": parsed_date,
            "duration": duration,
            "completed": summary.get("completed"),
            "failed": summary.get("failed"),
            "returncode": summary.get("returncode"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_token_throughput": throughput,
            "derived_total_token_throughput": derived_throughput,
            "total_token_throughput_consistent": throughput_consistent,
            "quality_prompt_count": len(arguments.get("quality_prompt") or []),
            "_server_command": summary.get("server_command"),
            "_server_log_path": _evidence_path(
                summary.get("server_log"), row["summary_path"]
            ),
            "_raw_model": summary.get("model"),
            "_raw_python": summary.get("python"),
            "valid": not errors,
            "validation_errors": errors,
        },
        errors,
    )


def _validate_semantic_gate(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {
            "provided": False,
            "passed": False,
            "note": "A passing semantic gate is required for strict acceptance.",
        }, ["semantic gate summary is required"]
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"failed to load semantic summary {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceError(f"semantic summary must be a JSON object: {path}")

    comparisons = data.get("comparisons")
    if not isinstance(comparisons, dict):
        comparisons = {}
    acceptance = comparisons.get("acceptance")
    if not isinstance(acceptance, dict):
        acceptance = {}
    reference_mode = acceptance.get("reference_mode")
    algorithm_reference = acceptance.get("algorithm_reference")
    if not isinstance(algorithm_reference, dict):
        algorithm_reference = {}
    eager_passed = algorithm_reference.get("request_parallel_eager", {}).get(
        "passed"
    )
    graph_passed = algorithm_reference.get("request_parallel_graph", {}).get(
        "passed"
    )
    graph_equivalence_passed = acceptance.get("graph_equivalence", {}).get(
        "passed"
    )
    repeat_stability = acceptance.get("repeat_stability")
    if not isinstance(repeat_stability, dict):
        repeat_stability = {}
    repeat_stability_passed = all(
        repeat_stability.get(mode, {}).get("passed") is True
        for mode in (
            "standard_single_seq",
            "request_parallel_eager",
            "request_parallel_graph",
        )
    )
    model = _normalized_path(data.get("model"))
    python = _normalized_path(data.get("python"))
    device = data.get("device")
    runtime_sources, runtime_errors = _normalize_runtime_sources(
        data.get("runtime_sources"),
        "semantic gate",
    )
    errors = []
    checks = {
        "summary_kind": (
            data.get("kind")
            == "qwen3_attention_request_parallel_semantics_gate_summary"
        ),
        "schema_version": data.get("schema_version") == 2,
        "summary_passed": data.get("passed") is True,
        "verdict_pass": data.get("verdict") == "PASS",
        "execution_passed": data.get("execution_passed") is True,
        "acceptance_passed": acceptance.get("passed") is True,
        "reference_is_standard_single_seq": reference_mode == "standard_single_seq",
        "request_parallel_eager_exact": eager_passed is True,
        "request_parallel_graph_exact": graph_passed is True,
        "graph_equivalence_exact": graph_equivalence_passed is True,
        "repeat_stability_exact": repeat_stability_passed,
        "model_nonempty": model is not None,
        "python_nonempty": python is not None,
        "device_matches_profile": device == EXPECTED_PROFILE["device"],
        "runtime_sources_complete": not runtime_errors,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"semantic gate check failed: {name}")
    errors.extend(runtime_errors)
    return {
        "provided": True,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "passed": not errors,
        "verdict": data.get("verdict"),
        "reference_mode": reference_mode,
        "checks": checks,
        "provenance": {
            "model": model,
            "python": python,
            "device": device,
            "runtime_sources": runtime_sources,
            "runtime_identity": (
                _runtime_identity(runtime_sources) if not runtime_errors else {}
            ),
        },
    }, errors


def _profile_fingerprint() -> str:
    payload = {
        "profile": EXPECTED_PROFILE,
        "common_env": COMMON_ENV,
        "mode_env": MODE_ENV,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_throughput(
    rows: list[dict[str, Any]],
) -> tuple[int | None, float, float | None]:
    duration = sum(row["duration"] for row in rows)
    token_counts = [row["total_tokens"] for row in rows]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in token_counts
    ):
        return None, duration, None
    total_tokens = sum(token_counts)
    return total_tokens, duration, total_tokens / duration


def _revalidate_semantic_runtime_sources(
    semantic_sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    paths: list[Path] = []
    for runtime_name in PINNED_RUNTIME_HEADS:
        pythonpath = semantic_sources.get(runtime_name, {}).get("pythonpath")
        if not isinstance(pythonpath, str) or not pythonpath:
            errors.append(
                f"legacy runtime reconstruction: semantic {runtime_name} "
                "has no pythonpath"
            )
        else:
            paths.append(Path(pythonpath))
    if errors:
        return {}, errors

    try:
        try:
            from scripts.repro.gate_qwen3_attention_request_parallel import (
                _runtime_source_records,
            )
        except ModuleNotFoundError:
            from gate_qwen3_attention_request_parallel import (  # type: ignore
                _runtime_source_records,
            )
        current_records = _runtime_source_records(paths)
    except Exception as exc:
        return {}, [f"legacy runtime reconstruction failed: {exc}"]

    current_sources, current_errors = _normalize_runtime_sources(
        current_records,
        "legacy runtime reconstruction",
    )
    errors.extend(current_errors)
    if not current_errors and (
        _runtime_identity(current_sources) != _runtime_identity(semantic_sources)
    ):
        errors.append(
            "legacy runtime reconstruction: current source fingerprints do not "
            "match semantic gate runtime_sources"
        )
    return current_sources, errors


def _legacy_row_runtime_evidence(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    label = f"pair{row['pair']}_{row['mode']}"
    errors: list[str] = []
    command = row.get("_server_command")
    if not isinstance(command, str) or not command:
        errors.append(f"{label}: legacy runtime reconstruction needs server_command")
        command_tokens: list[str] = []
    else:
        try:
            command_tokens = shlex.split(command)
        except ValueError as exc:
            errors.append(f"{label}: invalid server_command: {exc}")
            command_tokens = []

    for field, raw_value in (
        ("model", row.get("_raw_model")),
        ("python", row.get("_raw_python")),
    ):
        if not isinstance(raw_value, str) or raw_value not in command_tokens:
            errors.append(
                f"{label}: server_command does not bind summary {field}="
                f"{raw_value!r}"
            )
    device_assignment = f"ASCEND_RT_VISIBLE_DEVICES={row.get('device')}"
    if device_assignment not in command_tokens:
        errors.append(
            f"{label}: server_command does not bind {device_assignment}"
        )

    server_log_path = row.get("_server_log_path")
    if not isinstance(server_log_path, Path) or not server_log_path.is_file():
        errors.append(
            f"{label}: legacy runtime reconstruction needs an existing server_log"
        )
        return {"server_log": None, "server_log_sha256": None}, errors
    try:
        server_log_raw = server_log_path.read_bytes()
        server_log = server_log_raw.decode("utf-8", errors="replace")
    except OSError as exc:
        errors.append(f"{label}: failed to read server_log: {exc}")
        return {"server_log": str(server_log_path), "server_log_sha256": None}, errors

    observed_versions = VLLM_VERSION_GIT_PATTERN.findall(server_log)
    pinned_vllm_head = PINNED_RUNTIME_HEADS["vllm"]
    if not any(pinned_vllm_head.startswith(version) for version in observed_versions):
        errors.append(
            f"{label}: server_log does not identify pinned vLLM HEAD "
            f"{pinned_vllm_head}"
        )
    required_markers = list(LEGACY_COMMON_LOG_MARKERS)
    if row["mode"] == "w8a8":
        required_markers.extend(LEGACY_W8A8_LOG_MARKERS)
    missing_markers = [marker for marker in required_markers if marker not in server_log]
    if missing_markers:
        errors.append(
            f"{label}: server_log is missing runtime markers {missing_markers!r}"
        )
    return {
        "server_log": str(server_log_path),
        "server_log_sha256": hashlib.sha256(server_log_raw).hexdigest(),
        "vllm_git_versions": observed_versions,
        "required_markers": required_markers,
    }, errors


def _bind_provenance(
    rows: list[dict[str, Any]],
    semantic_gate: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    semantic_provenance = semantic_gate.get("provenance")
    if not isinstance(semantic_provenance, dict):
        semantic_provenance = {}
    semantic_sources = semantic_provenance.get("runtime_sources")
    if not isinstance(semantic_sources, dict):
        semantic_sources = {}

    missing_runtime_rows = [row for row in rows if row["runtime_sources"] is None]
    reconstruction: dict[str, Any] = {
        "used": bool(missing_runtime_rows),
        "method": (
            "server_command + server_log pinned vLLM version/feature markers + "
            "current source fingerprint revalidation against semantic gate"
            if missing_runtime_rows
            else "per-run runtime_sources"
        ),
        "rows": [],
    }
    if missing_runtime_rows:
        current_sources, current_errors = _revalidate_semantic_runtime_sources(
            semantic_sources
        )
        errors.extend(current_errors)
        reconstruction["current_runtime_identity"] = (
            _runtime_identity(current_sources) if not current_errors else {}
        )
        for row in missing_runtime_rows:
            evidence, row_errors = _legacy_row_runtime_evidence(row)
            reconstruction["rows"].append(
                {
                    "pair": row["pair"],
                    "mode": row["mode"],
                    "evidence": evidence,
                    "passed": not row_errors and not current_errors,
                }
            )
            errors.extend(row_errors)
            if not row_errors and not current_errors:
                row["runtime_sources"] = current_sources
                row["runtime_provenance_origin"] = "legacy_reconstructed"

    python_values = {row["python"] for row in rows if row["python"] is not None}
    device_values = {row["device"] for row in rows if row["device"] is not None}
    if len(python_values) != 1 or any(row["python"] is None for row in rows):
        errors.append(
            f"performance provenance: six runs do not share one Python: "
            f"{sorted(python_values)!r}"
        )
    if len(device_values) != 1 or any(row["device"] is None for row in rows):
        errors.append(
            f"performance provenance: six runs do not share one device: "
            f"{sorted(device_values, key=repr)!r}"
        )

    models_by_mode: dict[str, set[str]] = {"fp16": set(), "w8a8": set()}
    for row in rows:
        if row["model"] is not None:
            models_by_mode[row["mode"]].add(row["model"])
    for mode, models in models_by_mode.items():
        if len(models) != 1 or any(
            row["model"] is None for row in rows if row["mode"] == mode
        ):
            errors.append(
                f"performance provenance: {mode} runs do not share one model: "
                f"{sorted(models)!r}"
            )

    shared_python = next(iter(python_values)) if len(python_values) == 1 else None
    shared_device = next(iter(device_values)) if len(device_values) == 1 else None
    w8a8_model = (
        next(iter(models_by_mode["w8a8"]))
        if len(models_by_mode["w8a8"]) == 1
        else None
    )
    for field, actual, expected in (
        ("model", semantic_provenance.get("model"), w8a8_model),
        ("python", semantic_provenance.get("python"), shared_python),
        ("device", semantic_provenance.get("device"), shared_device),
    ):
        if actual != expected or actual is None:
            errors.append(
                f"semantic/performance provenance mismatch: {field}="
                f"{actual!r}, expected {expected!r}"
            )

    semantic_identity = (
        _runtime_identity(semantic_sources)
        if len(semantic_sources) == len(PINNED_RUNTIME_HEADS)
        else {}
    )
    row_identities: list[dict[str, dict[str, str]]] = []
    for row in rows:
        row_sources = row["runtime_sources"]
        if not isinstance(row_sources, dict) or len(row_sources) != len(
            PINNED_RUNTIME_HEADS
        ):
            errors.append(
                f"pair{row['pair']}_{row['mode']}: no complete runtime provenance"
            )
            continue
        identity = _runtime_identity(row_sources)
        row_identities.append(identity)
        if identity != semantic_identity:
            errors.append(
                f"pair{row['pair']}_{row['mode']}: runtime fingerprint does not "
                "match semantic gate"
            )
    if row_identities and any(
        identity != row_identities[0] for identity in row_identities[1:]
    ):
        errors.append("performance provenance: six runtime fingerprints differ")

    reconstruction["passed"] = not errors if missing_runtime_rows else True
    return {
        "passed": not errors,
        "shared_python": shared_python,
        "shared_device": shared_device,
        "fp16_model": (
            next(iter(models_by_mode["fp16"]))
            if len(models_by_mode["fp16"]) == 1
            else None
        ),
        "w8a8_model": w8a8_model,
        "runtime_identity": semantic_identity,
        "pinned_runtime_heads": PINNED_RUNTIME_HEADS,
        "legacy_reconstruction": reconstruction,
    }, errors


def summarize(root: Path, semantic_summary: Path | None = None) -> dict[str, Any]:
    loaded_rows = _load_rows(root)
    raw_rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for loaded in loaded_rows:
        row, errors = _validate_row(loaded)
        raw_rows.append(row)
        validation_errors.extend(errors)

    observed_sequence = [
        (row["pair"], row["mode"])
        for row in sorted(
            raw_rows,
            key=lambda item: item["_parsed_date"] or datetime.max,
        )
    ]
    execution_order_valid = tuple(observed_sequence) == EXPECTED_SEQUENCE
    if not execution_order_valid:
        validation_errors.append(
            f"execution order={observed_sequence!r}, expected={EXPECTED_SEQUENCE!r}"
        )

    input_token_values = {row["input_tokens"] for row in raw_rows}
    output_token_values = {row["output_tokens"] for row in raw_rows}
    token_counts_equal = len(input_token_values) == 1 and len(output_token_values) == 1
    if not token_counts_equal:
        validation_errors.append(
            f"token counts differ: "
            f"input={sorted(input_token_values, key=repr)!r}, "
            f"output={sorted(output_token_values, key=repr)!r}"
        )

    semantic_gate, semantic_errors = _validate_semantic_gate(semantic_summary)
    validation_errors.extend(semantic_errors)
    provenance_binding, provenance_errors = _bind_provenance(
        raw_rows,
        semantic_gate,
    )
    validation_errors.extend(provenance_errors)

    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in raw_rows:
        by_pair.setdefault(row["pair"], {})[row["mode"]] = row
    pairs = []
    for pair in (1, 2, 3):
        fp16 = by_pair[pair]["fp16"]
        w8a8 = by_pair[pair]["w8a8"]
        fp16_derived_throughput = fp16["derived_total_token_throughput"]
        w8a8_derived_throughput = w8a8["derived_total_token_throughput"]
        speedup = None
        if (
            isinstance(fp16_derived_throughput, (int, float))
            and isinstance(w8a8_derived_throughput, (int, float))
        ):
            speedup = w8a8_derived_throughput / fp16_derived_throughput
        pairs.append(
            {
                "pair": pair,
                "fp16_duration": fp16["duration"],
                "w8a8_duration": w8a8["duration"],
                "fp16_total_token_throughput": fp16["total_token_throughput"],
                "w8a8_total_token_throughput": w8a8["total_token_throughput"],
                "speedup": speedup,
                "speedup_at_least_1_5x": (
                    speedup is not None
                    and speedup >= STRICT_SPEEDUP_THRESHOLD
                ),
                "valid": fp16["valid"] and w8a8["valid"],
            }
        )

    fp16_rows = [row for row in raw_rows if row["mode"] == "fp16"]
    w8a8_rows = [row for row in raw_rows if row["mode"] == "w8a8"]
    fp16_mean_throughput = statistics.mean(
        row["total_token_throughput"] for row in fp16_rows
    )
    w8a8_mean_throughput = statistics.mean(
        row["total_token_throughput"] for row in w8a8_rows
    )
    arithmetic_mean_throughput_ratio = (
        w8a8_mean_throughput / fp16_mean_throughput
    )
    fp16_mean_seconds = statistics.mean(row["duration"] for row in fp16_rows)
    w8a8_mean_seconds = statistics.mean(row["duration"] for row in w8a8_rows)

    (
        fp16_aggregated_tokens,
        fp16_aggregated_seconds,
        fp16_aggregated_throughput,
    ) = _aggregate_throughput(fp16_rows)
    (
        w8a8_aggregated_tokens,
        w8a8_aggregated_seconds,
        w8a8_aggregated_throughput,
    ) = _aggregate_throughput(w8a8_rows)
    throughput_ratio = None
    if (
        fp16_aggregated_throughput is not None
        and w8a8_aggregated_throughput is not None
    ):
        throughput_ratio = (
            w8a8_aggregated_throughput / fp16_aggregated_throughput
        )

    prompt_tokens = next(iter(input_token_values)) if len(input_token_values) == 1 else None
    fixed_prompt_ratio = None
    if isinstance(prompt_tokens, int) and prompt_tokens > 0:
        fp16_prompt_mean = statistics.mean(
            prompt_tokens / row["duration"] for row in fp16_rows
        )
        w8a8_prompt_mean = statistics.mean(
            prompt_tokens / row["duration"] for row in w8a8_rows
        )
        fixed_prompt_ratio = w8a8_prompt_mean / fp16_prompt_mean

    all_six_measurements_valid = (
        all(row["valid"] for row in raw_rows)
        and execution_order_valid
        and token_counts_equal
    )
    all_six_valid = (
        all_six_measurements_valid and provenance_binding["passed"] is True
    )
    semantic_required_and_passed = semantic_gate["passed"] is True
    aggregated_throughput_pass = (
        throughput_ratio is not None
        and throughput_ratio >= STRICT_SPEEDUP_THRESHOLD
    )
    all_pair_speedups_pass = all(
        pair["speedup_at_least_1_5x"] for pair in pairs
    )
    strict_pass = (
        all_six_valid
        and semantic_required_and_passed
        and provenance_binding["passed"] is True
        and not validation_errors
        and aggregated_throughput_pass
        and all_pair_speedups_pass
    )

    for row in raw_rows:
        row.pop("_parsed_date", None)
        row.pop("_server_command", None)
        row.pop("_server_log_path", None)
        row.pop("_raw_model", None)
        row.pop("_raw_python", None)

    return {
        "schema_version": 2,
        "kind": "qwen3_strict_alternating_c32_summary",
        "profile": EXPECTED_PROFILE,
        "profile_fingerprint_sha256": _profile_fingerprint(),
        "mode_specific_execution": {
            "fp16": "standard TP2 attention and sharded expert path",
            "w8a8": "request-parallel attention and replicated-local 128-expert path",
        },
        "execution_order": {
            "expected": [list(item) for item in EXPECTED_SEQUENCE],
            "observed": [list(item) for item in observed_sequence],
            "valid": execution_order_valid,
        },
        "token_counts_equal": token_counts_equal,
        "pairs": pairs,
        "fp16_mean_total_token_throughput": fp16_mean_throughput,
        "w8a8_mean_total_token_throughput": w8a8_mean_throughput,
        "arithmetic_mean_throughput_ratio_context": (
            arithmetic_mean_throughput_ratio
        ),
        "fp16_aggregated_tokens": fp16_aggregated_tokens,
        "w8a8_aggregated_tokens": w8a8_aggregated_tokens,
        "fp16_aggregated_seconds": fp16_aggregated_seconds,
        "w8a8_aggregated_seconds": w8a8_aggregated_seconds,
        "fp16_aggregated_total_token_throughput": (
            fp16_aggregated_throughput
        ),
        "w8a8_aggregated_total_token_throughput": (
            w8a8_aggregated_throughput
        ),
        "throughput_ratio_primary_method": (
            "ratio of sum(input_tokens + output_tokens) / sum(duration)"
        ),
        "throughput_ratio_primary": throughput_ratio,
        "aggregated_throughput_at_least_1_5x": aggregated_throughput_pass,
        "all_pair_speedups_at_least_1_5x": all_pair_speedups_pass,
        "fp16_mean_seconds": fp16_mean_seconds,
        "w8a8_mean_seconds": w8a8_mean_seconds,
        "wall_time_ratio_context": fp16_mean_seconds / w8a8_mean_seconds,
        "fp16_wall_seconds_population_stddev": statistics.pstdev(
            row["duration"] for row in fp16_rows
        ),
        "w8a8_wall_seconds_population_stddev": statistics.pstdev(
            row["duration"] for row in w8a8_rows
        ),
        "fixed_prompt_throughput_ratio": fixed_prompt_ratio,
        "quality_evidence": {
            "embedded_in_performance_runs": all(
                row["quality_prompt_count"] > 0 for row in raw_rows
            ),
            "note": (
                "The six performance runs used no embedded quality prompts; "
                "execution correctness is bound through semantic_gate."
            ),
        },
        "semantic_gate": semantic_gate,
        "provenance_binding": provenance_binding,
        "all_six_measurements_valid": all_six_measurements_valid,
        "all_six_valid": all_six_valid,
        "validation_errors": validation_errors,
        "strict_1_5x_pass": strict_pass,
        "raw_rows": raw_rows,
    }


def _failure_result(message: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "qwen3_strict_alternating_c32_summary",
        "all_six_valid": False,
        "validation_errors": [message],
        "strict_1_5x_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-summary",
        type=Path,
        required=True,
        help="Layered request-parallel semantic gate summary to bind to the result.",
    )
    args = parser.parse_args()

    try:
        result = summarize(args.root, args.semantic_summary)
    except EvidenceError as exc:
        result = _failure_result(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    keys = (
        "fp16_aggregated_total_token_throughput",
        "w8a8_aggregated_total_token_throughput",
        "throughput_ratio_primary",
        "arithmetic_mean_throughput_ratio_context",
        "aggregated_throughput_at_least_1_5x",
        "all_pair_speedups_at_least_1_5x",
        "wall_time_ratio_context",
        "all_six_valid",
        "strict_1_5x_pass",
        "validation_errors",
    )
    print(json.dumps({key: result.get(key) for key in keys}, indent=2))
    return 0 if result["strict_1_5x_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
