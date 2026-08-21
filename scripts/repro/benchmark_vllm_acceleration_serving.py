#!/usr/bin/env python3
"""Launch vLLM OpenAI server and run vLLM online serving benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_PYTHON = sys.executable
FIXED_REQUEST_RUNNER = Path(__file__).with_name("benchmark_vllm_fixed_requests.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--mode", choices=["int4", "fp16", "w8a8"], required=True)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Server model/activation dtype. The historical default is float16.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served_model_name", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--output_len", type=int, required=True)
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--warmup_num_prompts", type=int, default=0)
    parser.add_argument("--warmup_max_concurrency", type=int, default=None)
    parser.add_argument("--request_rate", default="inf")
    parser.add_argument(
        "--request-file",
        "--request_file",
        dest="request_file",
        type=Path,
        default=None,
        help=(
            "Optional fixed JSONL file of complete /v1/completions request "
            "bodies. Uses the fixed-request client instead of random data."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        "--request_timeout",
        dest="request_timeout",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--fixed-after-warmup-check-command-json",
        default=None,
        help=(
            "Optional JSON string array passed to the fixed-request client as "
            "its after-warmup, before-formal fail-closed check."
        ),
    )
    parser.add_argument(
        "--fixed-synchronized-start",
        action="store_true",
        help="Release all fixed-request POST workers from one measured boundary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random-dataset seed passed to warmup and benchmark runs.",
    )
    parser.add_argument("--max_concurrency", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--num_gpu_blocks_override", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--enable_expert_parallel",
        action="store_true",
        help="Pass --enable-expert-parallel to vLLM serve for MoE models.",
    )
    parser.add_argument(
        "--additional_config",
        default=None,
        help="Optional JSON object passed to vLLM serve --additional-config.",
    )
    parser.add_argument(
        "--compilation_config",
        default=None,
        help="Optional JSON object passed to vLLM serve --compilation-config.",
    )
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument("--disable_chunked_prefill", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--async_scheduling",
        action="store_true",
        help="Pass --async-scheduling to vLLM serve.",
    )
    parser.add_argument("--aiv", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile only the formal benchmark through vLLM's /start_profile "
            "and /stop_profile endpoints. The server must receive "
            "VLLM_TORCH_PROFILER_DIR through --env."
        ),
    )
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--startup_timeout", type=int, default=900)
    parser.add_argument(
        "--quality_prompt",
        action="append",
        default=[],
        help=(
            "Optional fixed prompt sent after the timed benchmark. May be "
            "repeated; responses are archived for correctness review."
        ),
    )
    parser.add_argument("--quality_max_tokens", type=int, default=32)
    parser.add_argument("--quality_timeout", type=int, default=120)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Write the resolved server/client command contract without "
            "starting a server. The resulting summary is diagnostic-only."
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _wait_for_health(host: str, port: int, timeout: int, server_proc: subprocess.Popen) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {server_proc.returncode}.")
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}.")


def _check_health(host: str, port: int, timeout: float = 10.0) -> bool:
    url = f"http://{host}:{port}/health"
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def _run_quality_checks(
    host: str,
    port: int,
    model: str,
    prompts: list[str],
    max_tokens: int,
    timeout: int,
    seed: int | None,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts):
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "logprobs": 5,
        }
        if seed is not None:
            payload["seed"] = seed
        request = Request(
            f"http://{host}:{port}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices", [])
                text = choices[0].get("text", "") if choices else ""
                checks.append({
                    "index": index,
                    "prompt": prompt,
                    "status": response.status,
                    "elapsed_seconds": time.perf_counter() - started,
                    "text": text,
                    "nonempty": bool(text.strip()),
                    "usage": body.get("usage"),
                    "choice": choices[0] if choices else None,
                    "error": None,
                })
        except Exception as exc:
            checks.append({
                "index": index,
                "prompt": prompt,
                "status": None,
                "elapsed_seconds": time.perf_counter() - started,
                "text": "",
                "nonempty": False,
                "usage": None,
                "choice": None,
                "error": repr(exc),
            })
    completed = sum(check.get("status") == 200 for check in checks)
    nonempty = sum(bool(check.get("nonempty")) for check in checks)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "seed": seed,
        "completed": completed,
        "nonempty": nonempty,
        "failed": len(checks) - completed,
        "checks": checks,
    }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_tree(proc: subprocess.Popen) -> dict[str, object]:
    """Terminate the complete server process group and return audit evidence."""
    process_group_id = proc.pid
    term_sent = False
    kill_sent = False
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        proc.poll()
        if not _process_group_exists(process_group_id):
            break
        time.sleep(1)

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            proc.poll()
            if not _process_group_exists(process_group_id):
                break
            time.sleep(0.2)

    try:
        returncode = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        returncode = proc.poll()
    return {
        "process_group_id": process_group_id,
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
        "process_group_gone": not _process_group_exists(process_group_id),
        "server_returncode_after_teardown": returncode,
    }


def _wait_for_port_release(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_port_open(host, port):
            return True
        time.sleep(0.2)
    return not _is_port_open(host, port)


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finalize_single_server_summary(
    summary_path: Path,
    teardown: dict[str, object],
    *,
    port_released: bool,
) -> None:
    if not summary_path.is_file():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(summary, dict):
        return

    issues = summary.get("issues", [])
    if not isinstance(issues, list):
        issues = ["summary issues field was not a list"]
    else:
        issues = [str(issue) for issue in issues]
    returncode = summary.get("returncode")
    if returncode != 0:
        issues.append(f"benchmark returncode was {returncode!r}")
    failed = summary.get("failed")
    if failed != 0:
        issues.append(f"measured failed request count was {failed!r}")
    if summary.get("post_benchmark_health") is not True:
        issues.append("server health check failed after the timed benchmark")

    process_group_gone = teardown.get("process_group_gone") is True
    teardown_complete = process_group_gone and port_released
    if not process_group_gone:
        issues.append("server process group remained after teardown")
    if not port_released:
        issues.append("server port remained open after teardown")

    summary.update(
        {
            "status": "completed" if not issues and teardown_complete else "failed",
            "issues": issues,
            "teardown_complete": teardown_complete,
            "teardown_evidence": {
                **teardown,
                "host": summary.get("arguments", {}).get("host"),
                "port": summary.get("arguments", {}).get("port"),
                "port_released": port_released,
            },
        }
    )
    _write_json_atomic(summary_path, summary)


def _parse_server_metrics(server_log_path: Path) -> dict[str, object]:
    if not server_log_path.exists():
        return {}
    text = server_log_path.read_text(encoding="utf-8", errors="replace")

    def first(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def add_rank_vector(
        metrics: dict[str, object],
        name: str,
        values: list[int] | list[float],
        *,
        legacy_aggregate: str,
    ) -> None:
        if not values:
            return
        if legacy_aggregate not in {"min", "max"}:
            raise ValueError(
                f"Unsupported legacy aggregate: {legacy_aggregate!r}")
        minimum = min(values)
        maximum = max(values)
        metrics[f"{name}_vector"] = values
        metrics[f"{name}_min"] = minimum
        metrics[f"{name}_max"] = maximum
        metrics[name] = maximum if legacy_aggregate == "max" else minimum

    metrics: dict[str, object] = {}
    weights_gb = [
        float(match.group(1))
        for match in re.finditer(
            r"Loading model weights took ([0-9.]+) GB", text)
    ]
    memory_matches = list(re.finditer(
        r"Available memory: ([0-9]+), total memory: ([0-9]+)", text))
    available_memory = [int(match.group(1)) for match in memory_matches]
    total_memory = [int(match.group(2)) for match in memory_matches]
    kv_tokens = [
        int(match.group(1).replace(",", ""))
        for match in re.finditer(
            r"GPU KV cache size: ([0-9,]+) tokens", text)
    ]
    max_concurrency = [
        float(match.group(1))
        for match in re.finditer(
            r"Maximum concurrency for [^:]+: ([0-9.]+)x", text)
    ]
    graph_match = re.search(
        r"Graph capturing finished in ([0-9]+) secs, took ([0-9.]+) GiB",
        text,
    )
    chunked = first(r"chunked_prefill_enabled=(True|False)")
    quantization = first(r"quantization=([^,]+), enforce_eager=")
    engine_dtype = first(
        r"Initializing a V1 LLM engine[^\n]*?\bdtype="
        r"(torch\.(?:bfloat16|float16|float32))"
    )

    add_rank_vector(
        metrics,
        "weights_memory_gb",
        weights_gb,
        legacy_aggregate="max",
    )
    add_rank_vector(
        metrics,
        "available_kv_cache_memory_bytes",
        available_memory,
        legacy_aggregate="min",
    )
    add_rank_vector(
        metrics,
        "total_memory_bytes",
        total_memory,
        legacy_aggregate="min",
    )
    add_rank_vector(
        metrics,
        "kv_cache_tokens",
        kv_tokens,
        legacy_aggregate="min",
    )
    add_rank_vector(
        metrics,
        "max_concurrency_for_request",
        max_concurrency,
        legacy_aggregate="min",
    )
    metrics["rank_metric_legacy_aggregation"] = {
        "weights_memory_gb": "max",
        "available_kv_cache_memory_bytes": "min",
        "total_memory_bytes": "min",
        "kv_cache_tokens": "min",
        "max_concurrency_for_request": "min",
    }
    if graph_match is not None:
        metrics["graph_capture_seconds"] = int(graph_match.group(1))
        metrics["graph_capture_gib"] = float(graph_match.group(2))
    if chunked is not None:
        metrics["chunked_prefill_enabled"] = chunked == "True"
    if quantization is not None:
        metrics["engine_quantization"] = None if quantization == "None" else quantization
    metrics["engine_dtype"] = engine_dtype
    metrics["ascend_quantization_log"] = (
        "Using the vLLM Ascend Quantization now!" in text
    )
    return metrics


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize_latency_arrays(result: dict[str, object]) -> dict[str, object]:
    ttfts = result.get("ttfts")
    itls = result.get("itls")
    if not isinstance(ttfts, list) or not isinstance(itls, list):
        return {}

    e2els: list[float] = []
    ttft_values: list[float] = []
    for ttft, request_itls in zip(ttfts, itls):
        if not isinstance(ttft, (int, float)) or not isinstance(request_itls, list):
            continue
        numeric_itls = [
            value for value in request_itls if isinstance(value, (int, float))
        ]
        ttft_values.append(float(ttft))
        e2els.append(float(ttft) + sum(float(value) for value in numeric_itls))

    metrics: dict[str, object] = {}
    if ttft_values:
        sorted_ttfts = sorted(ttft_values)
        metrics.update({
            "ttft_seconds_min": sorted_ttfts[0],
            "ttft_seconds_p50": _percentile(sorted_ttfts, 50),
            "ttft_seconds_p90": _percentile(sorted_ttfts, 90),
            "ttft_seconds_p99": _percentile(sorted_ttfts, 99),
            "ttft_seconds_max": sorted_ttfts[-1],
            "ttft_gt_15s": sum(value > 15.0 for value in ttft_values),
        })
    if e2els:
        sorted_e2els = sorted(e2els)
        metrics.update({
            "e2el_seconds_min": sorted_e2els[0],
            "e2el_seconds_p25": _percentile(sorted_e2els, 25),
            "e2el_seconds_p50": _percentile(sorted_e2els, 50),
            "e2el_seconds_p75": _percentile(sorted_e2els, 75),
            "e2el_seconds_p90": _percentile(sorted_e2els, 90),
            "e2el_seconds_p99": _percentile(sorted_e2els, 99),
            "e2el_seconds_max": sorted_e2els[-1],
            "e2el_gt_15s": sum(value > 15.0 for value in e2els),
        })
    return metrics


def _build_shell_prefix(args: argparse.Namespace) -> str:
    exports = [f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(args.device)}"]
    if args.aiv:
        exports.append("export HCCL_OP_EXPANSION_MODE=AIV")
    for assignment in args.env:
        if "=" not in assignment:
            raise ValueError(f"Invalid --env assignment: {assignment!r}")
        key, value = assignment.split("=", 1)
        exports.append(f"export {key}={shlex.quote(value)}")
    return "source /usr/local/Ascend/ascend-toolkit/set_env.sh && " + " && ".join(exports)


def _append_json_object_arg(
    command: list[str],
    value: str | None,
    argument_name: str,
    cli_flag: str,
) -> None:
    if value is None:
        return
    parsed_value = json.loads(value)
    if not isinstance(parsed_value, dict):
        raise ValueError(f"{argument_name} must decode to a JSON object")
    command.extend([
        cli_flag,
        json.dumps(parsed_value, separators=(",", ":")),
    ])


def _append_profile_arg(command: list[str], enabled: bool) -> None:
    if enabled:
        command.append("--profile")


def _append_dtype_arg(command: list[str], dtype: str) -> None:
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    command.extend(["--dtype", dtype])


def _build_fixed_request_benchmark_command(
    args: argparse.Namespace,
    endpoint: str,
    output_dir: Path,
) -> tuple[list[str], Path]:
    if args.request_file is None:
        raise ValueError("request_file is required for the fixed-request client")
    if str(args.request_rate).lower() not in {"inf", "infinity"}:
        raise ValueError("Fixed JSONL mode currently requires --request_rate inf")
    request_file = args.request_file.expanduser().resolve()
    if not request_file.is_file():
        raise FileNotFoundError(f"Fixed request file does not exist: {request_file}")
    max_concurrency = args.max_concurrency or args.num_prompts
    client_tag = f"{args.tag}_{args.mode}"
    summary_path = output_dir / f"{client_tag}_fixed_summary.json"
    command = [
        args.python,
        str(FIXED_REQUEST_RUNNER),
        "--endpoint",
        endpoint,
        "--request-file",
        str(request_file),
        "--output-dir",
        str(output_dir),
        "--tag",
        client_tag,
        "--max-concurrency",
        str(max_concurrency),
        "--warmup-num-prompts",
        str(args.warmup_num_prompts),
        "--request-timeout",
        str(args.request_timeout),
        "--expected-num-prompts",
        str(args.num_prompts),
        "--rerun",
    ]
    if args.warmup_max_concurrency is not None:
        command.extend([
            "--warmup-max-concurrency",
            str(args.warmup_max_concurrency),
        ])
    if args.profile:
        command.append("--profile")
    if getattr(args, "fixed_synchronized_start", False):
        command.append("--synchronized-start")
    after_warmup_check = getattr(
        args, "fixed_after_warmup_check_command_json", None
    )
    if after_warmup_check is not None:
        command.extend([
            "--after-warmup-check-command-json",
            after_warmup_check,
        ])
    return command, summary_path


def _validate_profile_dir(enabled: bool, assignments: list[str]) -> Path | None:
    if not enabled:
        return None
    values = []
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if separator and key == "VLLM_TORCH_PROFILER_DIR":
            values.append(value)
    if len(values) != 1:
        raise ValueError(
            "--profile requires exactly one "
            "--env VLLM_TORCH_PROFILER_DIR=/absolute/path")
    profile_dir = Path(values[0]).expanduser()
    if not profile_dir.is_absolute():
        raise ValueError("VLLM_TORCH_PROFILER_DIR must be an absolute path")
    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise ValueError(f"Profiler directory is not empty: {profile_dir}")
    return profile_dir


def _build_server_command(
    args: argparse.Namespace,
    model: str,
    served_model_name: str,
) -> list[str]:
    max_model_len = args.max_model_len or args.input_len + args.output_len + 16
    server_cmd = [
        args.python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--disable-log-requests",
        "--disable-log-stats",
    ]
    _append_dtype_arg(server_cmd, args.dtype)
    if args.mode in {"int4", "w8a8"}:
        server_cmd.extend(["--quantization", "ascend"])
    if args.enable_expert_parallel:
        server_cmd.append("--enable-expert-parallel")
    _append_json_object_arg(
        server_cmd,
        args.compilation_config,
        "--compilation_config",
        "--compilation-config",
    )
    _append_json_object_arg(
        server_cmd,
        args.additional_config,
        "--additional_config",
        "--additional-config",
    )
    if args.enforce_eager:
        server_cmd.append("--enforce-eager")
    if args.async_scheduling:
        server_cmd.append("--async-scheduling")
    if args.disable_prefix_caching:
        server_cmd.append("--no-enable-prefix-caching")
    if args.disable_chunked_prefill:
        server_cmd.append("--no-enable-chunked-prefill")
    if args.max_num_batched_tokens is not None:
        server_cmd.extend([
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
        ])
    if args.max_num_seqs is not None:
        server_cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.num_gpu_blocks_override is not None:
        server_cmd.extend([
            "--num-gpu-blocks-override",
            str(args.num_gpu_blocks_override),
        ])
    return server_cmd


def main() -> int:
    args = parse_args()
    profile_dir = _validate_profile_dir(args.profile, args.env)
    if _is_port_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model
    served_model_name = args.served_model_name or f"{args.mode}-qwen25-7b"
    result_path = output_dir / f"{args.tag}_{args.mode}_serve.json"
    server_log_path = output_dir / f"{args.tag}_{args.mode}_server.log"
    bench_log_path = output_dir / f"{args.tag}_{args.mode}_bench.log"
    summary_path = output_dir / f"{args.tag}_{args.mode}_summary.json"
    quality_path = output_dir / f"{args.tag}_{args.mode}_quality.json"
    if result_path.exists() and not args.rerun:
        print(f"Existing result: {result_path}")
        return 0

    fixed_client = None
    if args.request_file is not None:
        fixed_client = _build_fixed_request_benchmark_command(
            args,
            f"http://{args.host}:{args.port}/v1/completions",
            output_dir,
        )

    server_cmd = _build_server_command(args, model, served_model_name)

    shell_prefix = _build_shell_prefix(args)
    server_shell_cmd = shell_prefix + " && " + shlex.join(server_cmd)
    bench_shell_cmd: str | None = None
    if args.dry_run:
        fixed_command = fixed_client[0] if fixed_client is not None else None
        fixed_summary_path = fixed_client[1] if fixed_client is not None else None
        bench_shell_cmd = (
            shell_prefix + " && " + shlex.join(fixed_command)
            if fixed_command is not None
            else None
        )
        summary = {
            "status": "dry_run",
            "dry_run": True,
            "diagnostic_only": True,
            "server_started": False,
            "mode": args.mode,
            "model": model,
            "served_model_name": served_model_name,
            "device": args.device,
            "dtype_requested": args.dtype,
            "server_command": server_shell_cmd,
            "bench_command": bench_shell_cmd,
            "fixed_request_client_summary_json": (
                str(fixed_summary_path) if fixed_summary_path is not None else None
            ),
            "server_log": str(server_log_path),
            "bench_log": str(bench_log_path),
            "profiler_dir": str(profile_dir) if profile_dir else None,
            "returncode": 0,
            "issues": [],
            "teardown_complete": True,
            "teardown_evidence": {
                "server_never_started": True,
                "port_preflight_free": True,
                "host": args.host,
                "port": args.port,
            },
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        _write_json_atomic(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server_proc = subprocess.Popen(
            ["bash", "-lc", server_shell_cmd],
            cwd=Path(__file__).resolve().parents[2],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            _wait_for_health(args.host, args.port, args.startup_timeout, server_proc)
            if fixed_client is not None:
                fixed_cmd, fixed_summary_path = fixed_client
                bench_shell_cmd = shell_prefix + " && " + shlex.join(fixed_cmd)
                started = time.perf_counter()
                bench_completed = subprocess.run(
                    ["bash", "-lc", bench_shell_cmd],
                    cwd=Path(__file__).resolve().parents[2],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                elapsed = time.perf_counter() - started
                bench_log_path.write_text(
                    bench_completed.stdout,
                    encoding="utf-8",
                    errors="replace",
                )
                if not fixed_summary_path.is_file():
                    raise RuntimeError(
                        f"Fixed-request benchmark failed with code "
                        f"{bench_completed.returncode}; see {bench_log_path}."
                    )
                fixed_summary = json.loads(
                    fixed_summary_path.read_text(encoding="utf-8"))
                fixed_arguments = fixed_summary.pop("arguments", None)
                fixed_result_json = fixed_summary.pop("result_json", None)
                post_benchmark_health = _check_health(args.host, args.port)

                quality_result = None
                if args.quality_prompt:
                    quality_result = _run_quality_checks(
                        args.host,
                        args.port,
                        served_model_name,
                        args.quality_prompt,
                        args.quality_max_tokens,
                        args.quality_timeout,
                        args.seed,
                    )
                    quality_path.write_text(
                        json.dumps(
                            quality_result,
                            ensure_ascii=False,
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )

                result = {
                    **fixed_summary,
                    "fixed_request_client_arguments": fixed_arguments,
                    "fixed_request_client_result_json": fixed_result_json,
                }
                result_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                summary = {
                    "mode": args.mode,
                    "model": model,
                    "served_model_name": served_model_name,
                    "python": args.python,
                    "device": args.device,
                    "dtype_requested": args.dtype,
                    "server_command": server_shell_cmd,
                    "bench_command": bench_shell_cmd,
                    "warmup_num_prompts": args.warmup_num_prompts,
                    "warmup_max_concurrency": args.warmup_max_concurrency,
                    "env_overrides": args.env,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "result_json": str(result_path),
                    "fixed_request_client_summary_json": str(fixed_summary_path),
                    "fixed_request_client_result_json": fixed_result_json,
                    "fixed_request_client_arguments": fixed_arguments,
                    "server_log": str(server_log_path),
                    "bench_log": str(bench_log_path),
                    "returncode": bench_completed.returncode,
                    "elapsed_seconds": elapsed,
                    "diagnostic_only": args.profile,
                    "profiler_dir": str(profile_dir) if profile_dir else None,
                    "post_benchmark_health": post_benchmark_health,
                }
                summary.update(fixed_summary)
                if quality_result is not None:
                    summary.update({
                        "quality_result_json": str(quality_path),
                        "quality_completed": quality_result["completed"],
                        "quality_nonempty": quality_result["nonempty"],
                        "quality_failed": quality_result["failed"],
                    })
                summary.update(_parse_server_metrics(server_log_path))
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
                return bench_completed.returncode

            if args.warmup_num_prompts > 0:
                warmup_result_path = output_dir / f"{args.tag}_{args.mode}_warmup.json"
                warmup_log_path = output_dir / f"{args.tag}_{args.mode}_warmup.log"
                warmup_cmd = [
                    args.python,
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "bench",
                    "serve",
                    "--backend",
                    "openai",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--endpoint",
                    "/v1/completions",
                    "--model",
                    served_model_name,
                    "--tokenizer",
                    model,
                    "--trust-remote-code",
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(args.input_len),
                    "--random-output-len",
                    str(args.output_len),
                    "--random-range-ratio",
                    "0",
                    "--num-prompts",
                    str(args.warmup_num_prompts),
                    "--request-rate",
                    str(args.request_rate),
                    "--ignore-eos",
                    "--save-result",
                    "--result-dir",
                    str(output_dir),
                    "--result-filename",
                    warmup_result_path.name,
                    "--disable-tqdm",
                    "--metadata",
                    f"mode={args.mode}",
                    f"tag={args.tag}",
                    "phase=warmup",
                ]
                warmup_max_concurrency = (
                    args.warmup_max_concurrency
                    if args.warmup_max_concurrency is not None
                    else args.max_concurrency
                )
                if warmup_max_concurrency is not None:
                    warmup_cmd.extend([
                        "--max-concurrency",
                        str(warmup_max_concurrency),
                    ])
                if args.seed is not None:
                    warmup_cmd.extend(["--seed", str(args.seed)])
                warmup_shell_cmd = shell_prefix + " && " + shlex.join(warmup_cmd)
                warmup_completed = subprocess.run(
                    ["bash", "-lc", warmup_shell_cmd],
                    cwd=Path(__file__).resolve().parents[2],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                warmup_log_path.write_text(
                    warmup_completed.stdout,
                    encoding="utf-8",
                    errors="replace",
                )
                if warmup_completed.returncode != 0:
                    raise RuntimeError(
                        f"Warmup failed with code {warmup_completed.returncode}. "
                        f"See {warmup_log_path}."
                    )
            bench_cmd = [
                args.python,
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "serve",
                "--backend",
                "openai",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--endpoint",
                "/v1/completions",
                "--model",
                served_model_name,
                "--tokenizer",
                model,
                "--trust-remote-code",
                "--dataset-name",
                "random",
                "--random-input-len",
                str(args.input_len),
                "--random-output-len",
                str(args.output_len),
                "--random-range-ratio",
                "0",
                "--num-prompts",
                str(args.num_prompts),
                "--request-rate",
                str(args.request_rate),
                "--ignore-eos",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(output_dir),
                "--result-filename",
                result_path.name,
                "--disable-tqdm",
                "--metric-percentiles",
                "50,90,99",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--metadata",
                f"mode={args.mode}",
                f"tag={args.tag}",
            ]
            if args.max_concurrency is not None:
                bench_cmd.extend(["--max-concurrency", str(args.max_concurrency)])
            if args.seed is not None:
                bench_cmd.extend(["--seed", str(args.seed)])
            _append_profile_arg(bench_cmd, args.profile)
            bench_shell_cmd = shell_prefix + " && " + shlex.join(bench_cmd)
            started = time.perf_counter()
            bench_completed = subprocess.run(
                ["bash", "-lc", bench_shell_cmd],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed = time.perf_counter() - started
            bench_log_path.write_text(bench_completed.stdout, encoding="utf-8", errors="replace")
            post_benchmark_health = _check_health(args.host, args.port)
            quality_result = None
            if args.quality_prompt:
                quality_result = _run_quality_checks(
                    args.host,
                    args.port,
                    served_model_name,
                    args.quality_prompt,
                    args.quality_max_tokens,
                    args.quality_timeout,
                    args.seed,
                )
                quality_path.write_text(
                    json.dumps(quality_result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            summary = {
                "mode": args.mode,
                "model": model,
                "served_model_name": served_model_name,
                "python": args.python,
                "device": args.device,
                "dtype_requested": args.dtype,
                "server_command": server_shell_cmd,
                "bench_command": bench_shell_cmd,
                "warmup_num_prompts": args.warmup_num_prompts,
                "warmup_max_concurrency": args.warmup_max_concurrency,
                "env_overrides": args.env,
                "arguments": vars(args),
                "result_json": str(result_path),
                "server_log": str(server_log_path),
                "bench_log": str(bench_log_path),
                "returncode": bench_completed.returncode,
                "elapsed_seconds": elapsed,
                "diagnostic_only": args.profile,
                "profiler_dir": str(profile_dir) if profile_dir else None,
                "post_benchmark_health": post_benchmark_health,
            }
            if quality_result is not None:
                summary.update({
                    "quality_result_json": str(quality_path),
                    "quality_completed": quality_result["completed"],
                    "quality_nonempty": quality_result["nonempty"],
                    "quality_failed": quality_result["failed"],
                })
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    for key in [
                        "duration",
                        "completed",
                        "input_tokens",
                        "output_tokens",
                        "failed",
                        "request_throughput",
                        "total_token_throughput",
                        "output_token_throughput",
                        "max_output_tokens_per_s",
                        "max_concurrency",
                        "max_concurrent_requests",
                        "mean_ttft_ms",
                        "median_ttft_ms",
                        "p90_ttft_ms",
                        "p99_ttft_ms",
                        "mean_tpot_ms",
                        "median_tpot_ms",
                        "mean_e2el_ms",
                        "median_e2el_ms",
                        "p90_e2el_ms",
                        "p99_e2el_ms",
                        "std_e2el_ms",
                    ]:
                        if key in result:
                            summary[key] = result[key]
                    for result_key, summary_key in (
                            ("total_input_tokens", "input_tokens"),
                            ("total_output_tokens", "output_tokens"),
                            ("output_throughput", "output_token_throughput")):
                        if result_key in result:
                            summary[summary_key] = result[result_key]
                    if "failed" not in summary and isinstance(
                            summary.get("completed"), int):
                        summary["failed"] = max(
                            0, args.num_prompts - summary["completed"])
                    summary.update(_summarize_latency_arrays(result))
                except Exception as exc:
                    summary["result_parse_error"] = repr(exc)
            summary["seed"] = args.seed
            summary.update(_parse_server_metrics(server_log_path))
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return bench_completed.returncode
        except Exception as exc:
            failure: dict[str, object] = {}
            if summary_path.is_file():
                try:
                    existing = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        failure.update(existing)
                except (OSError, json.JSONDecodeError):
                    pass
            issue = f"orchestration error: {exc!r}"
            existing_issues = failure.get("issues", [])
            if not isinstance(existing_issues, list):
                existing_issues = []
            failure.update(
                {
                    "status": "failed",
                    "mode": args.mode,
                    "model": model,
                    "served_model_name": served_model_name,
                    "python": args.python,
                    "device": args.device,
                    "dtype_requested": args.dtype,
                    "server_command": server_shell_cmd,
                    "bench_command": bench_shell_cmd,
                    "env_overrides": args.env,
                    "warmup_num_prompts": args.warmup_num_prompts,
                    "warmup_max_concurrency": args.warmup_max_concurrency,
                    "returncode": failure.get("returncode", server_proc.poll()),
                    "diagnostic_only": False,
                    "error": repr(exc),
                    "issues": [*existing_issues, issue],
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "server_log": str(server_log_path),
                    "bench_log": str(bench_log_path),
                }
            )
            _write_json_atomic(summary_path, failure)
            raise
        finally:
            teardown = _terminate_process_tree(server_proc)
            port_released = _wait_for_port_release(args.host, args.port)
            _finalize_single_server_summary(
                summary_path,
                teardown,
                port_released=port_released,
            )


if __name__ == "__main__":
    raise SystemExit(main())
