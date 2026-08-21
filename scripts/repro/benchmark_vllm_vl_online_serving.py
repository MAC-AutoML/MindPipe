#!/usr/bin/env python3
"""Launch vLLM OpenAI server and benchmark multimodal chat completions."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PYTHON = sys.executable
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--mode", choices=["fp16", "w8a8"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served_model_name", default="qwen25-vl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument(
        "--api_server_count",
        type=int,
        default=1,
        help="Number of vLLM OpenAI API server processes.",
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--warmup_num_prompts", type=int, default=0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile only the formal benchmark through vLLM's /start_profile "
            "and /stop_profile endpoints. The server must receive "
            "VLLM_TORCH_PROFILER_DIR through --env."
        ),
    )
    parser.add_argument("--max_concurrency", type=int, default=16)
    parser.add_argument(
        "--dispatch_wave_size",
        type=int,
        default=0,
        help=(
            "Submit requests in synchronized waves of this size. 0 keeps the "
            "historical submit-all behavior."
        ),
    )
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Use SSE streaming Chat Completions and record first-token and "
            "per-chunk arrival timestamps."
        ),
    )
    parser.add_argument("--question", default="请根据图片内容完成结构化理解任务，并用简短中文回答。")
    parser.add_argument("--text_repetitions", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--num_gpu_blocks_override", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--limit_mm_per_prompt", default=None, help='Example: "image=1"')
    parser.add_argument("--disable_chunked_prefill", action="store_true")
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--generation_config",
        default=None,
        help="vLLM generation config mode, for example 'vllm' to ignore model defaults.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Optional request-level repetition penalty. Omit to preserve the model default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional request-level sampling seed for reproducible comparisons.",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Request output logprobs from the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--top_logprobs",
        type=int,
        default=None,
        help="Optional number of output alternatives to return with --logprobs.",
    )
    parser.add_argument(
        "--additional_config",
        default=None,
        help="JSON passed through to vLLM --additional-config.",
    )
    parser.add_argument(
        "--compilation_config",
        default=None,
        help="JSON passed through to vLLM --compilation-config.",
    )
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--startup_timeout", type=int, default=900)
    parser.add_argument("--request_timeout", type=int, default=600)
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
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}.")


def _post_profile_control(
    host: str,
    port: int,
    action: str,
    timeout: int,
) -> dict[str, object]:
    endpoint = f"http://{host}:{port}/{action}_profile"
    started = time.perf_counter()
    event: dict[str, object] = {
        "action": action,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    request = Request(endpoint, data=b"", method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            event["status"] = response.status
            event["success"] = 200 <= response.status < 300
            if body:
                event["response_body"] = body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        event["status"] = exc.code
        event["success"] = False
        event["error"] = body or repr(exc)
    except Exception as exc:
        event["status"] = None
        event["success"] = False
        event["error"] = repr(exc)
    event["elapsed_seconds"] = time.perf_counter() - started
    print(f"Profiler {action}: {json.dumps(event, ensure_ascii=False)}", flush=True)
    return event


def _write_profile_events(path: Path, events: list[dict[str, object]]) -> None:
    try:
        path.write_text(
            json.dumps({"events": events}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"WARNING: failed to write profiler events to {path}: {exc}", file=sys.stderr)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc.poll()
        if not _process_group_exists(proc.pid):
            return True
        time.sleep(0.1)
    proc.poll()
    return not _process_group_exists(proc.pid)


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_for_process_group_exit(proc, 30):
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if not _wait_for_process_group_exit(proc, 5):
        print(
            f"WARNING: server process group {proc.pid} did not exit after SIGKILL.",
            file=sys.stderr,
        )


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


def _parse_server_metrics(server_log_path: Path) -> dict[str, object]:
    if not server_log_path.exists():
        return {}
    text = server_log_path.read_text(encoding="utf-8", errors="replace")

    def first(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match else None

    metrics: dict[str, object] = {}
    weights_gb = first(r"Loading model weights took ([0-9.]+) GB")
    memory_match = re.search(r"Available memory: ([0-9]+), total memory: ([0-9]+)", text)
    kv_tokens = first(r"GPU KV cache size: ([0-9,]+) tokens")
    max_concurrency = first(r"Maximum concurrency for [^:]+: ([0-9.]+)x")
    chunked = first(r"chunked_prefill_enabled=(True|False)")
    quantization = first(r"quantization=([^,]+), enforce_eager=")

    if weights_gb is not None:
        metrics["weights_memory_gb"] = float(weights_gb)
    if memory_match:
        metrics["available_kv_cache_memory_bytes"] = int(memory_match.group(1))
        metrics["total_memory_bytes"] = int(memory_match.group(2))
    if kv_tokens is not None:
        metrics["kv_cache_tokens"] = int(kv_tokens.replace(",", ""))
    if max_concurrency is not None:
        metrics["max_concurrency_for_request"] = float(max_concurrency)
    if chunked is not None:
        metrics["chunked_prefill_enabled"] = chunked == "True"
    if quantization is not None:
        metrics["engine_quantization"] = None if quantization == "None" else quantization
    metrics["ascend_quantization_log"] = "Using the vLLM Ascend Quantization now!" in text
    return metrics


def _build_shell_prefix(args: argparse.Namespace) -> str:
    exports = [f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(args.device)}"]
    for assignment in args.env:
        if "=" not in assignment:
            raise ValueError(f"Invalid --env assignment: {assignment!r}")
        key, value = assignment.split("=", 1)
        exports.append(f"export {key}={shlex.quote(value)}")
    return "source /usr/local/Ascend/ascend-toolkit/set_env.sh && " + " && ".join(exports)


def _list_images(image_dir: Path) -> list[Path]:
    images = [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_payloads(args: argparse.Namespace, images: list[Path]) -> list[dict[str, object]]:
    repeated_context = (
        "请综合观察图片中的主体、文字、数量、空间关系、颜色、动作和可能含义。"
        "回答时保持简洁，优先给出可直接核验的事实。"
    )
    text = args.question + repeated_context * max(args.text_repetitions - 1, 0)
    payloads: list[dict[str, object]] = []
    for index in range(args.num_prompts):
        selected_images = [
            images[(index * args.images_per_prompt + offset) % len(images)]
            for offset in range(args.images_per_prompt)
        ]
        image_path = selected_images[0]
        content: list[dict[str, object]] = []
        for selected_image in selected_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(selected_image)},
            })
        content.append({
            "type": "text",
            "text": f"{text}\n样本编号：{index}",
        })
        payload = {
            "model": args.served_model_name,
            "messages": [{
                "role": "user",
                "content": content,
            }],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": args.stream,
        }
        if args.stream:
            payload["stream_options"] = {"include_usage": True}
        if args.repetition_penalty is not None:
            payload["repetition_penalty"] = args.repetition_penalty
        if args.seed is not None:
            payload["seed"] = args.seed
        if args.logprobs:
            payload["logprobs"] = True
            if args.top_logprobs is not None:
                payload["top_logprobs"] = args.top_logprobs
        if args.min_tokens > 0:
            payload["min_tokens"] = args.min_tokens
        if args.ignore_eos:
            payload["ignore_eos"] = True
        payloads.append({
            "request_id": index,
            "image_path": str(image_path),
            "payload": payload,
        })
    return payloads


def _post_chat_completion(
    endpoint: str,
    request_item: dict[str, object],
    timeout: int,
    dispatch_barrier: threading.Barrier | None = None,
) -> dict[str, object]:
    data = json.dumps(request_item["payload"], ensure_ascii=False).encode("utf-8")
    if dispatch_barrier is not None:
        dispatch_barrier.wait()
    started = time.perf_counter()
    request = Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result: dict[str, object] = {
        "request_id": request_item["request_id"],
        "image_path": request_item["image_path"],
        "stream": bool(request_item["payload"].get("stream", False)),
    }
    try:
        with urlopen(request, timeout=timeout) as response:
            result["status"] = response.status
            if result["stream"]:
                result.update(_read_stream_response(response, started))
            else:
                body = response.read().decode("utf-8", errors="replace")
                result["response"] = json.loads(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result["status"] = exc.code
        result["error"] = body
    except URLError as exc:
        result["status"] = None
        result["error"] = repr(exc)
    except Exception as exc:
        result["status"] = None
        result["error"] = repr(exc)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _stream_text(delta: object) -> str:
    """Return user-visible text from either content or reasoning chunks."""
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    parts = [value for value in (content, reasoning) if isinstance(value, str)]
    return "".join(parts)


def _read_stream_response(response: object, started: float) -> dict[str, object]:
    """Read OpenAI-compatible SSE chunks while preserving arrival timing."""
    chunks: list[dict[str, object]] = []
    text_parts: list[str] = []
    first_token_seconds: float | None = None
    last_token_seconds: float | None = None
    usage: dict[str, object] | None = None
    finish_reason: object = None

    for raw_line in response:  # type: ignore[union-attr]
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        received_seconds = time.perf_counter() - started
        if data == "[DONE]":
            chunks.append({
                "received_seconds": received_seconds,
                "done": True,
            })
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            chunks.append({
                "received_seconds": received_seconds,
                "parse_error": True,
                "raw": data,
            })
            continue
        event_usage = event.get("usage") if isinstance(event, dict) else None
        if isinstance(event_usage, dict):
            usage = event_usage
        choices = event.get("choices", []) if isinstance(event, dict) else []
        choice = choices[0] if isinstance(choices, list) and choices else {}
        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
        text = _stream_text(delta)
        if text and first_token_seconds is None:
            first_token_seconds = received_seconds
        if text:
            last_token_seconds = received_seconds
            text_parts.append(text)
        if isinstance(choice, dict) and choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        chunks.append({
            "received_seconds": received_seconds,
            "text": text,
            "has_text": bool(text),
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "usage": event_usage,
        })

    response_body: dict[str, object] = {
        "choices": [{
            "message": {"role": "assistant", "content": "".join(text_parts)},
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        response_body["usage"] = usage
    return {
        "response": response_body,
        "stream_chunks": chunks,
        "first_token_seconds": first_token_seconds,
        "last_token_seconds": last_token_seconds,
        "usage_from_stream": usage,
    }


def _run_requests(
    endpoint: str,
    payloads: list[dict[str, object]],
    max_concurrency: int,
    timeout: int,
    dispatch_wave_size: int = 0,
) -> tuple[float, list[dict[str, object]]]:
    if dispatch_wave_size > max_concurrency:
        raise ValueError(
            "dispatch_wave_size cannot exceed max_concurrency because each "
            "wave is synchronized by a barrier"
        )
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        if dispatch_wave_size <= 0:
            waves = [payloads]
        else:
            waves = [
                payloads[index:index + dispatch_wave_size]
                for index in range(0, len(payloads), dispatch_wave_size)
            ]
        for wave in waves:
            barrier = (
                threading.Barrier(len(wave))
                if dispatch_wave_size > 0 and len(wave) > 1
                else None
            )
            futures = [
                executor.submit(
                    _post_chat_completion,
                    endpoint,
                    payload,
                    timeout,
                    barrier,
                )
                for payload in wave
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    return time.perf_counter() - started, sorted(results, key=lambda item: int(item["request_id"]))


def _summarize_results(
    args: argparse.Namespace,
    elapsed: float,
    results: list[dict[str, object]],
) -> dict[str, object]:
    completed = [item for item in results if item.get("status") == 200 and "response" in item]
    latencies = sorted(
        float(item.get("last_token_seconds") or item["elapsed_seconds"])
        for item in completed
    )
    ttfts = sorted(
        float(item["first_token_seconds"])
        for item in completed
        if isinstance(item.get("first_token_seconds"), (int, float))
    )
    tpots: list[float] = []
    stream_chunk_intervals: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for item in completed:
        response = item.get("response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        request_completion_tokens = int(usage.get("completion_tokens") or 0)
        completion_tokens += request_completion_tokens
        total_tokens += int(usage.get("total_tokens") or 0)
        first_token_seconds = item.get("first_token_seconds")
        last_token_seconds = item.get("last_token_seconds")
        if (
            request_completion_tokens > 1
            and isinstance(first_token_seconds, (int, float))
            and isinstance(last_token_seconds, (int, float))
            and last_token_seconds >= first_token_seconds
        ):
            tpots.append(
                (float(last_token_seconds) - float(first_token_seconds))
                / (request_completion_tokens - 1)
            )
        chunks = item.get("stream_chunks")
        if isinstance(chunks, list):
            text_arrivals = [
                float(chunk["received_seconds"])
                for chunk in chunks
                if isinstance(chunk, dict)
                and chunk.get("has_text") is True
                and isinstance(chunk.get("received_seconds"), (int, float))
            ]
            stream_chunk_intervals.extend(
                later - earlier
                for earlier, later in zip(text_arrivals, text_arrivals[1:])
                if later >= earlier
            )

    summary: dict[str, object] = {
        "mode": args.mode,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "device": args.device,
        "num_prompts": args.num_prompts,
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "duration": elapsed,
        "request_throughput": len(completed) / elapsed if elapsed > 0 else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_token_throughput": prompt_tokens / elapsed if elapsed > 0 else 0.0,
        "output_token_throughput": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "total_token_throughput": total_tokens / elapsed if elapsed > 0 else 0.0,
        "max_concurrency": args.max_concurrency,
        "dispatch_wave_size": args.dispatch_wave_size,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "ignore_eos": args.ignore_eos,
        "generation_config": args.generation_config,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "logprobs": args.logprobs,
        "top_logprobs": args.top_logprobs,
        "stream": args.stream,
        "images_per_prompt": args.images_per_prompt,
        "text_repetitions": args.text_repetitions,
    }
    if latencies:
        summary.update({
            "e2el_seconds_min": latencies[0],
            "e2el_seconds_p50": _percentile(latencies, 50),
            "e2el_seconds_p90": _percentile(latencies, 90),
            "e2el_seconds_p99": _percentile(latencies, 99),
            "e2el_seconds_max": latencies[-1],
        })
    if args.stream:
        summary.update({
            "ttft_completed": len(ttfts),
            "ttft_missing": len(completed) - len(ttfts),
            "usage_completed": sum(
                isinstance(item.get("usage_from_stream"), dict)
                for item in completed
            ),
            "stream_chunk_interval_is_token_itl": False,
        })
    if ttfts:
        summary.update({
            "ttft_ms_mean": sum(ttfts) / len(ttfts) * 1000.0,
            "ttft_ms_p50": _percentile(ttfts, 50) * 1000.0,
            "ttft_ms_p90": _percentile(ttfts, 90) * 1000.0,
            "ttft_ms_p99": _percentile(ttfts, 99) * 1000.0,
        })
    if tpots:
        tpots.sort()
        summary.update({
            "tpot_ms_mean": sum(tpots) / len(tpots) * 1000.0,
            "tpot_ms_p50": _percentile(tpots, 50) * 1000.0,
            "tpot_ms_p90": _percentile(tpots, 90) * 1000.0,
            "tpot_ms_p99": _percentile(tpots, 99) * 1000.0,
        })
    if stream_chunk_intervals:
        stream_chunk_intervals.sort()
        summary.update({
            "stream_chunk_interval_ms_p50": (
                _percentile(stream_chunk_intervals, 50) * 1000.0
            ),
            "stream_chunk_interval_ms_p90": (
                _percentile(stream_chunk_intervals, 90) * 1000.0
            ),
            "stream_chunk_interval_ms_p99": (
                _percentile(stream_chunk_intervals, 99) * 1000.0
            ),
        })
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_server_command(args: argparse.Namespace) -> list[str]:
    max_model_len = args.max_model_len or 2048
    server_cmd = [
        args.python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
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
        str(max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--disable-log-requests",
        "--disable-log-stats",
        "--api-server-count",
        str(args.api_server_count),
    ]
    if args.mode == "w8a8":
        server_cmd.extend(["--quantization", "ascend"])
    if args.enforce_eager:
        server_cmd.append("--enforce-eager")
    if args.generation_config is not None:
        server_cmd.extend(["--generation-config", args.generation_config])
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
    if args.limit_mm_per_prompt is not None:
        server_cmd.extend(["--limit-mm-per-prompt", args.limit_mm_per_prompt])
    if args.compilation_config is not None:
        server_cmd.extend(["--compilation-config", args.compilation_config])
    if args.additional_config is not None:
        server_cmd.extend(["--additional-config", args.additional_config])
    return server_cmd


def main() -> int:
    args = parse_args()
    if args.api_server_count < 1:
        raise ValueError("api_server_count must be at least 1")
    if _is_port_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir).expanduser().resolve()
    images = _list_images(image_dir)

    result_path = output_dir / f"{args.tag}_{args.mode}_serve.json"
    summary_path = output_dir / f"{args.tag}_{args.mode}_summary.json"
    requests_path = output_dir / f"{args.tag}_{args.mode}_requests.jsonl"
    responses_path = output_dir / f"{args.tag}_{args.mode}_responses.jsonl"
    warmup_responses_path = output_dir / f"{args.tag}_{args.mode}_warmup_responses.jsonl"
    server_log_path = output_dir / f"{args.tag}_{args.mode}_server.log"
    bench_log_path = output_dir / f"{args.tag}_{args.mode}_bench.log"
    profile_path = output_dir / f"{args.tag}_{args.mode}_profile.json"
    if result_path.exists() and not args.rerun:
        print(f"Existing result: {result_path}")
        return 0

    server_cmd = _build_server_command(args)

    shell_prefix = _build_shell_prefix(args)
    server_shell_cmd = shell_prefix + " && " + shlex.join(server_cmd)
    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"

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
            warmup_summary = None
            warmup_responses: list[dict[str, object]] = []
            if args.warmup_num_prompts > 0:
                warmup_payloads = _build_payloads(args, images)[:args.warmup_num_prompts]
                warmup_wave_size = (
                    min(args.dispatch_wave_size, len(warmup_payloads))
                    if args.dispatch_wave_size > 0
                    else 0
                )
                warmup_elapsed, warmup_responses = _run_requests(
                    endpoint,
                    warmup_payloads,
                    min(args.max_concurrency, args.warmup_num_prompts),
                    args.request_timeout,
                    warmup_wave_size,
                )
                warmup_summary = _summarize_results(
                    args,
                    warmup_elapsed,
                    warmup_responses,
                )
                warmup_summary["num_prompts"] = len(warmup_payloads)
                if warmup_summary["failed"] != 0:
                    raise RuntimeError(
                        "Warmup requests failed: "
                        f"{warmup_summary['failed']}/{len(warmup_payloads)}"
                    )

            payloads = _build_payloads(args, images)
            _write_jsonl(requests_path, payloads)
            bench_started = time.strftime("%Y-%m-%d %H:%M:%S")
            profile_events: list[dict[str, object]] = []
            try:
                if args.profile:
                    profile_events.append(
                        _post_profile_control(
                            args.host,
                            args.port,
                            "start",
                            args.request_timeout,
                        ))
                    _write_profile_events(profile_path, profile_events)
                elapsed, responses = _run_requests(
                    endpoint,
                    payloads,
                    args.max_concurrency,
                    args.request_timeout,
                    args.dispatch_wave_size,
                )
            finally:
                if args.profile:
                    profile_events.append(
                        _post_profile_control(
                            args.host,
                            args.port,
                            "stop",
                            args.request_timeout,
                        ))
                    _write_profile_events(profile_path, profile_events)
            bench_log_path.write_text(
                f"started={bench_started}\nelapsed_seconds={elapsed:.6f}\nresponses={len(responses)}\n",
                encoding="utf-8",
            )
            _write_jsonl(responses_path, responses)
            if warmup_responses:
                _write_jsonl(warmup_responses_path, warmup_responses)

            result = {
                "arguments": vars(args),
                "server_command": server_shell_cmd,
                "endpoint": endpoint,
                "image_count": len(images),
                "requests_jsonl": str(requests_path),
                "responses_jsonl": str(responses_path),
                "responses": responses,
            }
            if warmup_summary is not None:
                result["warmup"] = {
                    "summary": warmup_summary,
                    "responses_jsonl": str(warmup_responses_path),
                }
            if args.profile:
                result["profile"] = {
                    "events_json": str(profile_path),
                    "events": profile_events,
                }
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            summary = _summarize_results(args, elapsed, responses)
            summary.update({
                "python": args.python,
                "server_command": server_shell_cmd,
                "result_json": str(result_path),
                "requests_jsonl": str(requests_path),
                "responses_jsonl": str(responses_path),
                "server_log": str(server_log_path),
                "bench_log": str(bench_log_path),
                "env_overrides": args.env,
                "arguments": vars(args),
            })
            if warmup_summary is not None:
                summary["warmup"] = {
                    "summary": warmup_summary,
                    "responses_jsonl": str(warmup_responses_path),
                }
            if args.profile:
                summary["profile"] = {
                    "events_json": str(profile_path),
                    "events": profile_events,
                }
            summary.update(_parse_server_metrics(server_log_path))
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return 0 if summary["failed"] == 0 else 1
        finally:
            _terminate_process_tree(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
