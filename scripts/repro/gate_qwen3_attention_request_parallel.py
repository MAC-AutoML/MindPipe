#!/usr/bin/env python3
"""Gate Qwen3 request-parallel attention and decode ACL Graph semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from scripts.repro.gate_qwen3_long_prefill_semantics import (
        DEFAULT_ASCEND_ENV, GENERATED_TOKENS, THRESHOLDS,
        _atomic_write_json, _fixed_prompts, _is_port_open,
        _preflight_runtime, _runtime_environment, _sourced_command,
        _terminate_process_tree, _token_ids_sha256, _utc_now,
        _validate_comparator, _validate_response, _wait_for_health,
        compare_responses)
except ModuleNotFoundError:
    from gate_qwen3_long_prefill_semantics import (  # type: ignore
        DEFAULT_ASCEND_ENV, GENERATED_TOKENS, THRESHOLDS,
        _atomic_write_json, _fixed_prompts, _is_port_open,
        _preflight_runtime, _runtime_environment, _sourced_command,
        _terminate_process_tree, _token_ids_sha256, _utc_now,
        _validate_comparator, _validate_response, _wait_for_health,
        compare_responses)


REFERENCE_MODE = "standard_single_seq"
CHARACTERIZATION_MODE = "standard_batched_eager"
REQUEST_PARALLEL_MODES = ("request_parallel_eager",
                          "request_parallel_graph")
ACCEPTANCE_MODES = (REFERENCE_MODE, *REQUEST_PARALLEL_MODES)
MODES = (REFERENCE_MODE, CHARACTERIZATION_MODE, *REQUEST_PARALLEL_MODES)
REPEATS = 2
GRAPH_CAPTURE_SIZE = 2
SUMMARY_SCHEMA_VERSION = 2
RUNTIME_SOURCE_FINGERPRINT_VERSION = 1
RUNTIME_SOURCE_SUFFIXES = frozenset({".py", ".cpp", ".h"})
RUNTIME_SOURCE_EXCLUDED_DIRS = frozenset({
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    "__pycache__",
    "build",
    "dist",
    "generated",
    "kernel_meta",
    "logs",
    "my_results",
    "profiler",
    "profiles",
    "results",
})
RUNTIME_SOURCE_EXCLUDED_DIR_GLOBS = (
    "build-*",
    "build_*",
    "cmake-build-*",
    "generated-*",
    "generated_*",
    "my_results-*",
    "my_results_*",
    "results-*",
    "results_*",
)
DEFAULT_ADDITIONAL_CONFIG = json.dumps(
    {
        "torchair_graph_config": {
            "enabled": False
        },
        "ascend_scheduler_config": {
            "enabled": True
        },
        "refresh": True,
    },
    separators=(",", ":"),
)

MANAGED_ENV = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MINDPIPE_ENGINE_IDLE_COALESCE_US": "1000000",
    "MINDPIPE_ENGINE_IDLE_COALESCE_TARGET_ADDS": "1",
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
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "0",
    "VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP": "0",
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "0",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE_W8A8": "0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19058)
    parser.add_argument("--served-model-name", default="qwen3-30b-a3b")
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--termination-timeout", type=float, default=30.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--additional-config", default=DEFAULT_ADDITIONAL_CONFIG)
    parser.add_argument("--ascend-env", type=Path, default=DEFAULT_ASCEND_ENV)
    parser.add_argument(
        "--pythonpath",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="Prepend a runtime source directory to PYTHONPATH. May be repeated.",
    )
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _parse_env(assignments: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid --env assignment: {assignment!r}")
        if key in result:
            raise ValueError(f"Duplicate --env key: {key}")
        result[key] = value
    return result


def validate_args(args: argparse.Namespace) -> None:
    devices = [part.strip() for part in args.device.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("--device must identify two distinct devices")
    if not (1 <= args.port <= 65535):
        raise ValueError("--port must be in [1, 65535]")
    if not (0 < args.gpu_memory_utilization <= 1):
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    parsed_config = json.loads(args.additional_config)
    if not isinstance(parsed_config, dict):
        raise ValueError("--additional-config must decode to a JSON object")
    _parse_env(args.env)
    if args.validate_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    if not args.model.expanduser().is_dir():
        raise FileNotFoundError(args.model)
    if not args.ascend_env.expanduser().is_file():
        raise FileNotFoundError(args.ascend_env)
    if not (Path(args.python).expanduser().is_file() or shutil.which(args.python)):
        raise FileNotFoundError(args.python)
    for path in args.pythonpath:
        if not path.expanduser().is_dir():
            raise FileNotFoundError(f"PYTHONPATH directory does not exist: {path}")


def _resolved_python(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    executable = shutil.which(value)
    if executable is None:
        raise FileNotFoundError(value)
    return str(Path(executable).resolve())


def _git_output(repo: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fingerprint runtime sources") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Could not inspect runtime source Git repository at {repo}: {detail}"
        ) from exc
    return completed.stdout


def _is_relevant_untracked_source(relative_path: bytes) -> bool:
    display_path = os.fsdecode(relative_path)
    path = Path(display_path)
    excluded_prefixes = tuple(
        pattern[:-1] for pattern in RUNTIME_SOURCE_EXCLUDED_DIR_GLOBS
    )
    return (
        path.suffix in RUNTIME_SOURCE_SUFFIXES
        and not any(part in RUNTIME_SOURCE_EXCLUDED_DIRS for part in path.parts)
        and not any(
            part.startswith(excluded_prefixes) for part in path.parts
        )
    )


def _fingerprint_part(digest: Any, label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _runtime_source_record(pythonpath: Path) -> dict[str, Any]:
    resolved_path = pythonpath.expanduser().resolve()
    root_raw = _git_output(resolved_path, "rev-parse", "--show-toplevel")
    git_root = Path(os.fsdecode(root_raw.rstrip(b"\n"))).resolve()
    head = _git_output(git_root, "rev-parse", "HEAD").strip().decode("ascii")
    tracked_diff = _git_output(
        git_root,
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "HEAD",
        "--",
    )

    exclusions = [
        f":(exclude,glob)**/{directory}/**"
        for directory in sorted(RUNTIME_SOURCE_EXCLUDED_DIRS)
    ]
    exclusions.extend(
        f":(exclude,glob)**/{directory_glob}/**"
        for directory_glob in RUNTIME_SOURCE_EXCLUDED_DIR_GLOBS
    )
    untracked_raw = _git_output(
        git_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "*.py",
        "*.cpp",
        "*.h",
        *exclusions,
    )
    relative_paths = sorted({
        path
        for path in untracked_raw.split(b"\0")
        if path and _is_relevant_untracked_source(path)
    })

    digest = hashlib.sha256()
    _fingerprint_part(
        digest,
        b"fingerprint-version",
        str(RUNTIME_SOURCE_FINGERPRINT_VERSION).encode("ascii"),
    )
    _fingerprint_part(digest, b"head", head.encode("ascii"))
    _fingerprint_part(digest, b"tracked-diff", tracked_diff)
    untracked_files: list[str] = []
    untracked_bytes = 0
    for relative_path in relative_paths:
        source_path = git_root / os.fsdecode(relative_path)
        metadata = source_path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            kind = b"regular"
            contents = source_path.read_bytes()
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"symlink"
            contents = os.fsencode(os.readlink(source_path))
        else:
            raise RuntimeError(
                f"Unsupported untracked runtime source type: {source_path}"
            )
        _fingerprint_part(digest, b"untracked-path", relative_path)
        _fingerprint_part(digest, b"untracked-kind", kind)
        _fingerprint_part(digest, b"untracked-contents", contents)
        untracked_bytes += len(contents)
        untracked_files.append(
            relative_path.decode("utf-8", errors="backslashreplace")
        )

    return {
        "pythonpath": str(resolved_path),
        "git_root": str(git_root),
        "head_commit": head,
        "dirty": bool(tracked_diff or relative_paths),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked_source_count": len(untracked_files),
        "untracked_source_bytes": untracked_bytes,
        "untracked_source_files": untracked_files,
        "source_fingerprint_sha256": digest.hexdigest(),
    }


def _runtime_source_records(pythonpaths: list[Path]) -> list[dict[str, Any]]:
    return [_runtime_source_record(path) for path in pythonpaths]


def _mode_environment(args: argparse.Namespace, mode: str) -> dict[str, str]:
    environment = _parse_env(args.env)
    environment.update(MANAGED_ENV)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    request_parallel = mode in REQUEST_PARALLEL_MODES
    graph = mode == "request_parallel_graph"
    environment["MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL"] = (
        "1" if request_parallel else "0")
    environment["MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH"] = (
        "1" if graph else "0")
    return environment


def _mode_max_num_seqs(mode: str) -> int:
    return 1 if mode == REFERENCE_MODE else 2


def _server_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        args.python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model.expanduser().resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--dtype",
        "float16",
        "--served-model-name",
        args.served_model_name,
        "--max-model-len",
        "2304",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        "2",
        "--quantization",
        "ascend",
        "--enable-expert-parallel",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        str(_mode_max_num_seqs(mode)),
        "--disable-log-requests",
        "--disable-log-stats",
        "--additional-config",
        args.additional_config,
    ]
    if mode == "request_parallel_graph":
        command.extend([
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [GRAPH_CAPTURE_SIZE],
                },
                separators=(",", ":"),
            ),
        ])
    else:
        command.append("--enforce-eager")
    return command


def _pairs(prompts: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "balanced_2048_2048",
            "left": prompts[0],
            "right": prompts[1],
        },
        {
            "name": "uneven_2048_1000",
            "left": prompts[2],
            "right": prompts[3],
        },
    )


def _request_payload(args: argparse.Namespace, mode: str,
                     pair: dict[str, Any], repeat: int) -> tuple[dict[str, Any],
                                                                 list[dict[str,
                                                                           Any]]]:
    ordered = ([pair["left"], pair["right"]]
               if repeat == 0 else [pair["right"], pair["left"]])
    payload = {
        "model": args.served_model_name,
        "prompt": [prompt["token_ids"] for prompt in ordered],
        "max_tokens": GENERATED_TOKENS,
        "min_tokens": GENERATED_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "ignore_eos": True,
        "stream": False,
        "echo": False,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "prompt_logprobs": 5,
        "logprobs": 5,
        "request_id": f"attention-gate-{mode}-{pair['name']}-r{repeat}",
    }
    return payload, ordered


def _split_and_validate_response(
    body: Any,
    ordered_prompts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    split: dict[str, dict[str, Any]] = {}
    if not isinstance(body, dict):
        return {"valid": False, "errors": ["response is not an object"]}, split
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 2:
        return {"valid": False, "errors": ["response must contain two choices"]}, split
    if [choice.get("index") for choice in choices
            if isinstance(choice, dict)] != [0, 1]:
        errors.append("choice indices must be [0, 1]")
    usage = body.get("usage")
    expected_prompt_tokens = sum(prompt["length"] for prompt in ordered_prompts)
    if not isinstance(usage, dict):
        errors.append("response usage is missing")
    else:
        if usage.get("prompt_tokens") != expected_prompt_tokens:
            errors.append("usage.prompt_tokens does not match the prompt pair")
        if usage.get("completion_tokens") != 2 * GENERATED_TOKENS:
            errors.append("usage.completion_tokens must equal 32")

    for index, prompt in enumerate(ordered_prompts):
        if index >= len(choices) or not isinstance(choices[index], dict):
            errors.append(f"choice {index} is malformed")
            continue
        choice = choices[index]
        single_body = {
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt["length"],
                "completion_tokens": GENERATED_TOKENS,
            },
        }
        validation = _validate_response(single_body, prompt)
        if not validation["valid"]:
            errors.extend(
                f"{prompt['name']}: {message}"
                for message in validation["errors"])
        if choice.get("finish_reason") != "length":
            errors.append(f"{prompt['name']}: finish_reason is not length")
        if not isinstance(choice.get("text"), str):
            errors.append(f"{prompt['name']}: generated text is missing")
        split[prompt["name"]] = single_body
    return {"valid": not errors, "errors": errors}, split


def _send_pair(args: argparse.Namespace, mode: str, pair: dict[str, Any],
               repeat: int, path: Path) -> tuple[dict[str, Any],
                                                 dict[str, dict[str, Any]]]:
    payload, ordered_prompts = _request_payload(args, mode, pair, repeat)
    request = Request(
        f"http://{args.host}:{args.port}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status: int | None = None
    body: Any = None
    error: str | None = None
    try:
        with urlopen(request, timeout=args.request_timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        status = exc.code
        error = repr(exc)
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw_error_body": raw}
    except Exception as exc:
        error = repr(exc)
    validation, split = (
        _split_and_validate_response(body, ordered_prompts)
        if status == 200 else ({
            "valid": False,
            "errors": [error or f"HTTP status {status}"],
        }, {}))
    record = {
        "schema_version": 1,
        "mode": mode,
        "pair": pair["name"],
        "repeat": repeat,
        "ordered_prompts": [{
            "name": prompt["name"],
            "length": prompt["length"],
            "sha256_le_u32": prompt["sha256_le_u32"],
        } for prompt in ordered_prompts],
        "elapsed_seconds": time.perf_counter() - started,
        "http_status": status,
        "request": payload,
        "response": body,
        "validation": validation,
        "error": error,
    }
    _atomic_write_json(path, record)
    return record, split


def _activation_evidence(mode: str, log_path: Path) -> dict[str, Any]:
    text = (log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists() else "")
    marker = "Using experimental Qwen3 attention request parallelism"
    marker_count = text.count(marker)
    expected_aclgraph = mode == "request_parallel_graph"
    configured_marker = f"{marker} (decode_aclgraph={expected_aclgraph})"
    configured_marker_count = text.count(configured_marker)
    capture_count = text.count("Graph capturing finished")
    replay_count = text.count("Replaying aclgraph")
    coalesced = [
        int(value) for value in re.findall(
            r"Engine idle request coalescing collected (\d+) additional ADD",
            text)
    ]
    expected_marker = 2 if mode in REQUEST_PARALLEL_MODES else 0
    passed = (marker_count == expected_marker
              and configured_marker_count == expected_marker
              and len(coalesced) == 4 and coalesced == [1, 1, 1, 1])
    if mode == "request_parallel_graph":
        passed = passed and capture_count == 2 and replay_count == 2
    else:
        passed = passed and capture_count == 0 and replay_count == 0
    return {
        "passed": passed,
        "request_parallel_marker_count": marker_count,
        "request_parallel_configured_marker_count": configured_marker_count,
        "expected_decode_aclgraph": (expected_aclgraph
                                     if mode in REQUEST_PARALLEL_MODES else None),
        "graph_capture_count": capture_count,
        "graph_replay_count": replay_count,
        "coalesced_additional_requests": coalesced,
    }


def _run_mode(
    args: argparse.Namespace,
    mode: str,
    prompt_pairs: tuple[dict[str, Any], ...],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    log_path = mode_dir / "server.log"
    command = _server_command(args, mode)
    shell_command = _sourced_command(args, command)
    overrides = _mode_environment(args, mode)
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": mode,
        "max_num_seqs": _mode_max_num_seqs(mode),
        "server_command": command,
        "server_shell_command": shell_command,
        "environment_overrides": overrides,
        "server_log": str(log_path.resolve()),
    }
    _atomic_write_json(mode_dir / "mode_manifest.json", manifest)
    responses: dict[str, list[dict[str, Any]]] = {
        prompt[side]["name"]: []
        for prompt in prompt_pairs for side in ("left", "right")
    }
    requests: list[dict[str, Any]] = []
    proc: subprocess.Popen[Any] | None = None
    termination: dict[str, Any] | None = None
    error: str | None = None
    started = time.perf_counter()
    environment = _runtime_environment(args, overrides)
    if _is_port_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use")
    try:
        with log_path.open("w", encoding="utf-8") as server_log:
            proc = subprocess.Popen(
                ["bash", "-lc", shell_command],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                _wait_for_health(args, proc, log_path)
                for pair in prompt_pairs:
                    for repeat in range(REPEATS):
                        path = mode_dir / f"request_{pair['name']}_r{repeat}.json"
                        record, split = _send_pair(args, mode, pair, repeat,
                                                   path)
                        requests.append({
                            "pair": pair["name"],
                            "repeat": repeat,
                            "path": str(path.resolve()),
                            "http_status": record["http_status"],
                            "validation": record["validation"],
                            "error": record["error"],
                        })
                        for prompt_name, body in split.items():
                            responses[prompt_name].append(body)
            finally:
                termination = _terminate_process_tree(
                    proc, args.termination_timeout)
    except Exception:
        error = traceback.format_exc()
        if proc is not None and termination is None:
            termination = _terminate_process_tree(proc,
                                                   args.termination_timeout)
    evidence = _activation_evidence(mode, log_path)
    requests_valid = (len(requests) == len(prompt_pairs) * REPEATS
                      and all(item["validation"]["valid"] for item in requests))
    result = {
        **manifest,
        "status": ("passed" if error is None and requests_valid
                   and evidence["passed"] else "failed"),
        "duration_seconds": time.perf_counter() - started,
        "requests_valid": requests_valid,
        "activation_evidence": evidence,
        "requests": requests,
        "termination": termination,
        "error": error,
    }
    _atomic_write_json(mode_dir / "mode_result.json", result)
    return result, responses


def _choice(body: dict[str, Any]) -> dict[str, Any]:
    return body["choices"][0]


def _strict_compare(reference: dict[str, Any],
                    candidate: dict[str, Any]) -> dict[str, Any]:
    comparison = compare_responses(reference, candidate)
    ref_choice = _choice(reference)
    candidate_choice = _choice(candidate)
    text_exact = ref_choice.get("text") == candidate_choice.get("text")
    finish_reason_exact = (ref_choice.get("finish_reason")
                           == candidate_choice.get("finish_reason"))
    prompt_logprobs_exact = (ref_choice.get("prompt_logprobs")
                             == candidate_choice.get("prompt_logprobs"))
    generated_logprobs_exact = (ref_choice.get("logprobs")
                                == candidate_choice.get("logprobs"))
    comparison.update({
        "text_exact": text_exact,
        "finish_reason_exact": finish_reason_exact,
        "prompt_logprobs_exact": prompt_logprobs_exact,
        "generated_logprobs_exact": generated_logprobs_exact,
        "passed": bool(comparison["passed"] and text_exact
                       and finish_reason_exact and prompt_logprobs_exact
                       and generated_logprobs_exact),
    })
    return comparison


def _repeat_stability(
    prompts: list[dict[str, Any]],
    responses: dict[str, dict[str, list[dict[str, Any]]]],
    mode: str,
) -> dict[str, Any]:
    cases = {}
    for prompt in prompts:
        values = responses.get(mode, {}).get(prompt["name"], [])
        cases[prompt["name"]] = (
            _strict_compare(values[0], values[1])
            if len(values) == REPEATS else {
                "passed": False,
                "error": "two repeats were not collected",
            })
    return {
        "passed": all(case["passed"] for case in cases.values()),
        "cases": cases,
    }


def _compare_modes(
    prompts: list[dict[str, Any]],
    responses: dict[str, dict[str, list[dict[str, Any]]]],
    reference_mode: str,
    candidate_mode: str,
) -> dict[str, Any]:
    cases = {}
    for prompt in prompts:
        reference = responses.get(reference_mode, {}).get(prompt["name"], [])
        candidate = responses.get(candidate_mode, {}).get(prompt["name"], [])
        repeated = [
            _strict_compare(reference[index], candidate[index])
            for index in range(REPEATS)
        ] if len(reference) == len(candidate) == REPEATS else []
        cases[prompt["name"]] = {
            "passed": len(repeated) == REPEATS
            and all(item["passed"] for item in repeated),
            "repeats": repeated,
        }
    return {
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "passed": all(case["passed"] for case in cases.values()),
        "cases": cases,
    }


def _build_comparisons(
    prompts: list[dict[str, Any]],
    responses: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    repeat_stability = {
        mode: _repeat_stability(prompts, responses, mode) for mode in MODES
    }
    algorithm_reference = {
        mode: _compare_modes(prompts, responses, REFERENCE_MODE, mode)
        for mode in REQUEST_PARALLEL_MODES
    }
    graph_equivalence = _compare_modes(
        prompts,
        responses,
        "request_parallel_eager",
        "request_parallel_graph",
    )
    acceptance_passed = (
        all(repeat_stability[mode]["passed"] for mode in ACCEPTANCE_MODES)
        and all(result["passed"] for result in algorithm_reference.values())
        and graph_equivalence["passed"])
    characterization = {
        "numerical_comparison_gates_verdict": False,
        "execution_gates_verdict": True,
        "purpose": "measure standard TP2 multi-sequence packing sensitivity",
        "mode": CHARACTERIZATION_MODE,
        "repeat_stability": repeat_stability[CHARACTERIZATION_MODE],
        "against_single_seq_reference": _compare_modes(
            prompts,
            responses,
            REFERENCE_MODE,
            CHARACTERIZATION_MODE,
        ),
        "against_request_parallel_eager": _compare_modes(
            prompts,
            responses,
            "request_parallel_eager",
            CHARACTERIZATION_MODE,
        ),
    }
    return {
        "thresholds": THRESHOLDS,
        "exact_fields": [
            "prompt_token_ids",
            "generated_token_ids",
            "text",
            "finish_reason",
            "prompt_logprobs",
            "generated_logprobs",
        ],
        "acceptance": {
            "reference_mode": REFERENCE_MODE,
            "repeat_stability": {
                mode: repeat_stability[mode] for mode in ACCEPTANCE_MODES
            },
            "algorithm_reference": algorithm_reference,
            "graph_equivalence": graph_equivalence,
            "passed": acceptance_passed,
        },
        "characterization": characterization,
        "passed": acceptance_passed,
    }


def _protocol(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> dict[str,
                                                                            Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "qwen3_attention_request_parallel_semantics_gate",
        "modes": list(MODES),
        "reference_mode": REFERENCE_MODE,
        "characterization_mode": CHARACTERIZATION_MODE,
        "mode_contracts": {
            mode: {
                "max_num_seqs": _mode_max_num_seqs(mode),
                "request_parallel": mode in REQUEST_PARALLEL_MODES,
                "acl_graph": mode == "request_parallel_graph",
                "numerical_comparison_gates_verdict": (
                    mode in ACCEPTANCE_MODES),
                "execution_gates_verdict": True,
            }
            for mode in MODES
        },
        "seed": args.seed,
        "generated_tokens": GENERATED_TOKENS,
        "repeats": REPEATS,
        "graph_capture_size": GRAPH_CAPTURE_SIZE,
        "pairs": [{
            "name": pair["name"],
            "left": pair["left"]["name"],
            "right": pair["right"]["name"],
        } for pair in _pairs(prompts)],
        "prompts": [{
            key: value
            for key, value in prompt.items() if key != "token_ids"
        } for prompt in prompts],
        "thresholds": THRESHOLDS,
    }


def _build_summary(
    protocol: dict[str, Any],
    results: dict[str, Any],
    comparisons: dict[str, Any],
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    execution_passed = all(result.get("status") == "passed"
                           for result in results.values())
    passed = execution_passed and comparisons["passed"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "qwen3_attention_request_parallel_semantics_gate_summary",
        "model": protocol["model"],
        "python": protocol["python"],
        "ascend_env": protocol["ascend_env"],
        "pythonpath": protocol["pythonpath"],
        "runtime_sources": protocol["runtime_sources"],
        "device": protocol["device"],
        "started_at_utc": protocol["started_at_utc"],
        "completed_at_utc": completed_at_utc or _utc_now(),
        "execution_passed": execution_passed,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "mode_results": results,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    prompts = _fixed_prompts()
    _validate_comparator()
    protocol = _protocol(args, prompts)
    if args.validate_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2,
                         sort_keys=True))
        return 0

    assert args.output_dir is not None
    args.python = _resolved_python(args.python)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"--output-dir must be empty: {output_dir}")
    _preflight_runtime(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol.update({
        "device_execution": True,
        "started_at_utc": _utc_now(),
        "model": str(args.model.expanduser().resolve()),
        "python": args.python,
        "ascend_env": str(args.ascend_env.expanduser().resolve()),
        "pythonpath": [
            str(path.expanduser().resolve()) for path in args.pythonpath
        ],
        "runtime_sources": _runtime_source_records(args.pythonpath),
        "device": args.device,
    })
    _atomic_write_json(output_dir / "protocol.json", protocol)

    results: dict[str, Any] = {}
    responses: dict[str, dict[str, list[dict[str, Any]]]] = {}
    prompt_pairs = _pairs(prompts)
    for mode in MODES:
        try:
            result, mode_responses = _run_mode(args, mode, prompt_pairs,
                                               output_dir)
        except Exception:
            result = {
                "mode": mode,
                "status": "failed",
                "error": traceback.format_exc(),
            }
            mode_responses = {}
        results[mode] = result
        responses[mode] = mode_responses

    comparisons = _build_comparisons(prompts, responses)
    summary = _build_summary(protocol, results, comparisons)
    summary_path = output_dir / "agent_summary.json"
    _atomic_write_json(summary_path, summary)
    print(summary_path)
    print(summary["verdict"])
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
