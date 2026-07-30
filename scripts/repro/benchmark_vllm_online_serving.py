#!/usr/bin/env python3
"""Launch vLLM OpenAI server and run vLLM online serving benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_PYTHON = sys.executable
DEFAULT_ASCEND_ENV = Path("/usr/local/Ascend/ascend-toolkit/set_env.sh")
ENVIRONMENT_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--runtime_pythonpath",
        "--runtime-pythonpath",
        dest="runtime_pythonpath",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Runtime source directory to prepend to PYTHONPATH and fingerprint. "
            "May be repeated."
        ),
    )
    parser.add_argument("--mode", choices=["int4", "fp16", "w8a8"], required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Explicit local model/checkpoint directory for the selected mode.",
    )
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
        "--skip_initial_test",
        action="store_true",
        help=(
            "Skip vLLM bench serve's initial single-request endpoint test. "
            "Use this only when the server requires a multi-request batch."
        ),
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
    parser.add_argument(
        "--ascend-env",
        "--ascend_env",
        dest="ascend_env",
        type=Path,
        default=DEFAULT_ASCEND_ENV,
        help="Absolute path to the trusted Ascend environment setup script.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Environment override; may be repeated.",
    )
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


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_tree(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    pgid = proc.pid
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while _group_exists(pgid) and time.monotonic() < deadline:
        proc.poll()
        time.sleep(0.25)
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + min(10.0, timeout)
        while _group_exists(pgid) and time.monotonic() < kill_deadline:
            proc.poll()
            time.sleep(0.25)
    try:
        proc.wait(timeout=min(10.0, timeout))
    except subprocess.TimeoutExpired:
        proc.poll()


def _parse_server_metrics(server_log_path: Path) -> dict[str, object]:
    if not server_log_path.exists():
        return {}
    text = server_log_path.read_text(encoding="utf-8", errors="replace")

    def first(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match else None

    metrics: dict[str, object] = {}
    weights_gb = first(r"Loading model weights took ([0-9.]+) GB")
    available_memory = first(r"Available memory: ([0-9]+), total memory: ([0-9]+)")
    total_memory = None
    if available_memory is not None:
        match = re.search(r"Available memory: ([0-9]+), total memory: ([0-9]+)", text)
        if match:
            available_memory, total_memory = match.groups()
    kv_tokens = first(r"GPU KV cache size: ([0-9,]+) tokens")
    max_concurrency = first(r"Maximum concurrency for [^:]+: ([0-9.]+)x")
    graph_match = re.search(
        r"Graph capturing finished in ([0-9]+) secs, took ([0-9.]+) GiB",
        text,
    )
    chunked = first(r"chunked_prefill_enabled=(True|False)")
    quantization = first(r"quantization=([^,]+), enforce_eager=")

    if weights_gb is not None:
        metrics["weights_memory_gb"] = float(weights_gb)
    if available_memory is not None:
        metrics["available_kv_cache_memory_bytes"] = int(available_memory)
    if total_memory is not None:
        metrics["total_memory_bytes"] = int(total_memory)
    if kv_tokens is not None:
        metrics["kv_cache_tokens"] = int(kv_tokens.replace(",", ""))
    if max_concurrency is not None:
        metrics["max_concurrency_for_request"] = float(max_concurrency)
    if graph_match is not None:
        metrics["graph_capture_seconds"] = int(graph_match.group(1))
        metrics["graph_capture_gib"] = float(graph_match.group(2))
    if chunked is not None:
        metrics["chunked_prefill_enabled"] = chunked == "True"
    if quantization is not None:
        metrics["engine_quantization"] = None if quantization == "None" else quantization
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


def _parse_env(assignments: list[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if (
            not separator
            or ENVIRONMENT_KEY_PATTERN.fullmatch(key) is None
            or "\0" in value
        ):
            raise ValueError(f"Invalid --env assignment: {assignment!r}")
        if key in environment:
            raise ValueError(f"Duplicate --env key: {key}")
        environment[key] = value
    return environment


def _resolve_ascend_env(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("--ascend-env must be an absolute path")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Ascend environment file does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Ascend environment file is not a file: {resolved}")
    return resolved


def _build_shell_prefix(
    args: argparse.Namespace, ascend_env: Path | None = None
) -> str:
    environment = {"ASCEND_RT_VISIBLE_DEVICES": str(args.device)}
    runtime_pythonpath = [
        str(Path(path).expanduser().resolve())
        for path in getattr(args, "runtime_pythonpath", [])
    ]
    if runtime_pythonpath:
        inherited_pythonpath = os.environ.get("PYTHONPATH")
        if inherited_pythonpath:
            runtime_pythonpath.append(inherited_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(runtime_pythonpath)
    overrides = _parse_env(args.env)
    if "ASCEND_RT_VISIBLE_DEVICES" in overrides:
        raise ValueError("Set Ascend devices with --device, not --env")
    if args.aiv:
        requested_aiv = overrides.pop("HCCL_OP_EXPANSION_MODE", "AIV")
        if requested_aiv != "AIV":
            raise ValueError("--aiv conflicts with HCCL_OP_EXPANSION_MODE override")
        environment["HCCL_OP_EXPANSION_MODE"] = "AIV"
    environment.update(overrides)
    exports = [
        f"export {key}={shlex.quote(value)}"
        for key, value in environment.items()
    ]
    environment_file = ascend_env or _resolve_ascend_env(args.ascend_env)
    return (
        f"source {shlex.quote(str(environment_file))} && "
        + " && ".join(exports)
    )


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


def _append_skip_initial_test_arg(command: list[str], enabled: bool) -> None:
    if enabled:
        command.extend(["--ready-check-timeout-sec", "0"])


def _append_quantization_arg(command: list[str], mode: str) -> None:
    if mode in {"int4", "w8a8"}:
        command.extend(["--quantization", "ascend"])


def _validate_profile_dir(enabled: bool, assignments: list[str]) -> Path | None:
    if not enabled:
        return None
    environment = _parse_env(assignments)
    value = environment.get("VLLM_TORCH_PROFILER_DIR")
    if value is None:
        raise ValueError(
            "--profile requires exactly one "
            "--env VLLM_TORCH_PROFILER_DIR=/absolute/path")
    profile_dir = Path(value).expanduser()
    if not profile_dir.is_absolute():
        raise ValueError("VLLM_TORCH_PROFILER_DIR must be an absolute path")
    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise ValueError(f"Profiler directory is not empty: {profile_dir}")
    return profile_dir


def _validate_runtime_args(args: argparse.Namespace) -> tuple[Path, Path]:
    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model}")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Model config does not exist: {model / 'config.json'}")
    ascend_env = _resolve_ascend_env(args.ascend_env)
    _parse_env(args.env)
    if not (Path(args.python).is_file() or shutil.which(args.python)):
        raise FileNotFoundError(f"Python executable does not exist: {args.python}")
    for runtime_path in args.runtime_pythonpath:
        if not runtime_path.expanduser().is_dir():
            raise FileNotFoundError(
                f"Runtime PYTHONPATH directory does not exist: {runtime_path}"
            )
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("--input_len and --output_len must be positive")
    if args.num_prompts <= 0 or args.warmup_num_prompts < 0:
        raise ValueError("Prompt counts must be positive (warmup may be zero)")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu_memory_utilization must be in (0, 1]")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor_parallel_size must be positive")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag) is None:
        raise ValueError("--tag must be a safe filename component")
    return model, ascend_env


def _jsonable_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return value


def _jsonable_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: _jsonable_value(value)
        for key, value in vars(args).items()
    }


def _capture_runtime_sources(paths: list[Path]) -> list[dict[str, object]]:
    if not paths:
        return []
    try:
        from scripts.repro.gate_qwen3_attention_request_parallel import (
            _runtime_source_records,
        )
    except ModuleNotFoundError:
        from gate_qwen3_attention_request_parallel import (  # type: ignore
            _runtime_source_records,
        )
    return _runtime_source_records(paths)


def main() -> int:
    args = parse_args()
    model_path, ascend_env = _validate_runtime_args(args)
    runtime_sources = _capture_runtime_sources(args.runtime_pythonpath)
    profile_dir = _validate_profile_dir(args.profile, args.env)
    if _is_port_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = str(model_path)
    served_model_name = args.served_model_name or f"{args.mode}-{model_path.name}"
    result_path = output_dir / f"{args.tag}_{args.mode}_serve.json"
    server_log_path = output_dir / f"{args.tag}_{args.mode}_server.log"
    bench_log_path = output_dir / f"{args.tag}_{args.mode}_bench.log"
    summary_path = output_dir / f"{args.tag}_{args.mode}_summary.json"
    quality_path = output_dir / f"{args.tag}_{args.mode}_quality.json"
    if result_path.exists() and not args.rerun:
        print(f"Existing result: {result_path}")
        return 0

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
        "--dtype",
        "float16",
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
    _append_quantization_arg(server_cmd, args.mode)
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
        server_cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.max_num_seqs is not None:
        server_cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.num_gpu_blocks_override is not None:
        server_cmd.extend([
            "--num-gpu-blocks-override",
            str(args.num_gpu_blocks_override),
        ])

    shell_prefix = _build_shell_prefix(args, ascend_env)
    server_shell_cmd = shell_prefix + " && " + shlex.join(server_cmd)
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
                _append_skip_initial_test_arg(
                    warmup_cmd, args.skip_initial_test)
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
            _append_skip_initial_test_arg(bench_cmd,
                                          args.skip_initial_test)
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
                "runtime_sources": runtime_sources,
                "server_command": server_shell_cmd,
                "bench_command": bench_shell_cmd,
                "warmup_num_prompts": args.warmup_num_prompts,
                "warmup_max_concurrency": args.warmup_max_concurrency,
                "ascend_environment_file": str(ascend_env),
                "env_overrides": args.env,
                "arguments": _jsonable_arguments(args),
                "result_json": str(result_path),
                "server_log": str(server_log_path),
                "bench_log": str(bench_log_path),
                "returncode": bench_completed.returncode,
                "elapsed_seconds": elapsed,
                "diagnostic_only": args.profile,
                "profiler_dir": str(profile_dir) if profile_dir else None,
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
                            ("total_output_tokens", "output_tokens")):
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
        finally:
            _terminate_process_tree(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
