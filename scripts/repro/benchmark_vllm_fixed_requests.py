#!/usr/bin/env python3
"""Benchmark one or more OpenAI completion endpoints with fixed JSONL bodies.

Each non-empty JSONL line is sent verbatim as one ``/v1/completions`` request.
Requests are assigned to endpoints by their stable line-order round robin.  The
aggregate throughput always uses one wall-clock interval around the complete
multi-endpoint run; endpoint-local throughput values are deliberately not
computed or summed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        action="append",
        required=True,
        help=(
            "Full OpenAI completions URL. May be repeated for deterministic "
            "round-robin multi-endpoint dispatch."
        ),
    )
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument(
        "--max-concurrency-per-endpoint",
        type=int,
        default=None,
        help=(
            "Per-endpoint cap. Defaults to ceil(global concurrency / endpoint "
            "count)."
        ),
    )
    parser.add_argument("--warmup-num-prompts", type=int, default=0)
    parser.add_argument("--warmup-max-concurrency", type=int, default=None)
    parser.add_argument(
        "--warmup-max-concurrency-per-endpoint",
        type=int,
        default=None,
        help=(
            "Optional per-endpoint cap used only during warmup. Defaults to "
            "--max-concurrency-per-endpoint for backward compatibility."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--expected-num-prompts", type=int, default=None)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile only the formal fixed-request run. Warmup completes "
            "before POST /start_profile, and POST /stop_profile is always "
            "attempted after the formal run."
        ),
    )
    parser.add_argument(
        "--after-warmup-check-command-json",
        default=None,
        help=(
            "Optional JSON string array executed after a successful warmup and "
            "before the formal request window. A nonzero return code prevents "
            "the formal run."
        ),
    )
    parser.add_argument(
        "--synchronized-start",
        action="store_true",
        help=(
            "Wait until every request worker is ready, then release all POSTs "
            "from one measured start boundary. Requires the concurrency caps "
            "to admit the complete fixed request set."
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _parse_command_json(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "after-warmup-check-command-json must be valid JSON"
        ) from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(
            "after-warmup-check-command-json must decode to a non-empty "
            "string array"
        )
    return value


def _run_after_warmup_check(command: list[str]) -> dict[str, object]:
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    evidence: dict[str, object] = {
        "phase_boundary": "after_warmup_before_formal",
        "command": command,
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": time.time_ns(),
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "output": completed.stdout,
    }
    print(
        "After-warmup check: " + json.dumps(evidence, ensure_ascii=False),
        flush=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "After-warmup check failed with code "
            f"{completed.returncode}: {completed.stdout.rstrip()}"
        )
    return evidence


def _validate_endpoints(endpoints: list[str]) -> list[str]:
    if not endpoints:
        raise ValueError("At least one --endpoint is required")
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Endpoints must be unique")
    for endpoint in endpoints:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid endpoint URL: {endpoint!r}")
        if parsed.query or parsed.fragment:
            raise ValueError(f"Endpoint must not contain query or fragment: {endpoint!r}")
    return endpoints


def _profile_targets(endpoints: list[str]) -> list[str]:
    """Return one stable profiler control target per API server origin."""
    targets: list[str] = []
    seen: set[tuple[str, str]] = set()
    for endpoint in _validate_endpoints(endpoints):
        parsed = urlparse(endpoint)
        origin = (parsed.scheme, parsed.netloc)
        if origin in seen:
            continue
        seen.add(origin)
        targets.append(f"{parsed.scheme}://{parsed.netloc}")
    return targets


def _post_profile_control(
    target: str,
    action: str,
    timeout: float,
) -> dict[str, object]:
    if action not in {"start", "stop"}:
        raise ValueError(f"Unsupported profiler action: {action!r}")
    endpoint = f"{target}/{action}_profile"
    started = time.perf_counter()
    event: dict[str, object] = {
        "action": action,
        "target": target,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    request = Request(endpoint, data=b"", method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            response_bytes = response.read()
            event["status"] = response.status
            event["success"] = 200 <= response.status < 300
            if response_bytes:
                event["response_body"] = response_bytes.decode(
                    "utf-8", errors="replace")
    except HTTPError as exc:
        response_bytes = exc.read()
        event["status"] = exc.code
        event["success"] = False
        event["error"] = (
            response_bytes.decode("utf-8", errors="replace") or repr(exc)
        )
    except Exception as exc:
        event["status"] = None
        event["success"] = False
        event["error"] = repr(exc)
    event["elapsed_seconds"] = time.perf_counter() - started
    print(
        f"Profiler {action}: {json.dumps(event, ensure_ascii=False)}",
        flush=True,
    )
    return event


def _write_profile_events(
    path: Path,
    events: list[dict[str, object]],
) -> None:
    path.write_text(
        json.dumps({"events": events}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _profile_evidence(
    profile_path: Path,
    targets: list[str],
    events: list[dict[str, object]],
) -> dict[str, object]:
    starts = [event for event in events if event.get("action") == "start"]
    stops = [event for event in events if event.get("action") == "stop"]
    start_succeeded = (
        len(starts) == len(targets)
        and all(event.get("success") is True for event in starts)
    )
    stop_succeeded = (
        len(stops) == len(targets)
        and all(event.get("success") is True for event in stops)
    )
    return {
        "enabled": True,
        "scope": "formal_fixed_requests_only",
        "warmup_included": False,
        "formal_run_wrapped": True,
        "targets": targets,
        "events_json": str(profile_path),
        "events": events,
        "start": starts,
        "stop": stops,
        "start_succeeded": start_succeeded,
        "stop_succeeded": stop_succeeded,
        "control_succeeded": start_succeeded and stop_succeeded,
    }


def _load_fixed_requests(path: Path) -> list[dict[str, object]]:
    raw_file = path.read_bytes()
    if not raw_file:
        raise ValueError(f"Request file is empty: {path}")

    requests: list[dict[str, object]] = []
    for line_number, body in enumerate(raw_file.splitlines(), start=1):
        if not body.strip():
            raise ValueError(
                f"Blank JSONL line is not allowed: {path}:{line_number}")
        try:
            body_text = body.decode("utf-8")
            payload = json.loads(body_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid UTF-8 JSON request at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Request must be a JSON object: {path}:{line_number}")
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise ValueError(
                f"Request must contain a non-empty model: {path}:{line_number}")
        if not isinstance(payload.get("prompt"), str):
            raise ValueError(
                f"Request must contain a string prompt: {path}:{line_number}")
        max_tokens = payload.get("max_tokens")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError(
                f"Request must contain positive integer max_tokens: "
                f"{path}:{line_number}")
        if payload.get("stream", False) is not False:
            raise ValueError(
                f"Streaming requests are not supported: {path}:{line_number}")
        requests.append({
            "request_id": len(requests),
            "line_number": line_number,
            "body": body,
            "body_text": body_text,
            "payload": payload,
        })

    if not requests:
        raise ValueError(f"Request file has no JSONL records: {path}")
    return requests


def _assign_requests(
    requests: list[dict[str, object]],
    endpoints: list[str],
) -> list[dict[str, object]]:
    _validate_endpoints(endpoints)
    return [
        {
            **request,
            "endpoint_index": request_id % len(endpoints),
            "endpoint": endpoints[request_id % len(endpoints)],
        }
        for request_id, request in enumerate(requests)
    ]


def _post_fixed_request(
    assigned: dict[str, object],
    timeout: float,
    global_started: float,
) -> dict[str, object]:
    body = assigned["body"]
    if not isinstance(body, bytes):
        raise TypeError("Internal request body must be bytes")
    endpoint = str(assigned["endpoint"])
    started = time.perf_counter()
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result: dict[str, object] = {
        "request_id": int(assigned["request_id"]),
        "line_number": int(assigned["line_number"]),
        "endpoint_index": int(assigned["endpoint_index"]),
        "endpoint": endpoint,
        "started_offset_seconds": started - global_started,
    }
    response_bytes = b""
    try:
        with urlopen(request, timeout=timeout) as response:
            response_bytes = response.read()
            result["status"] = response.status
    except HTTPError as exc:
        response_bytes = exc.read()
        result["status"] = exc.code
        result["error"] = repr(exc)
    except URLError as exc:
        result["status"] = None
        result["error"] = repr(exc)
    except Exception as exc:
        result["status"] = None
        result["error"] = repr(exc)

    if response_bytes:
        response_text = response_bytes.decode("utf-8", errors="replace")
        result["response_body"] = response_text
        try:
            result["response"] = json.loads(response_text)
        except json.JSONDecodeError as exc:
            result["response_parse_error"] = repr(exc)
    finished = time.perf_counter()
    result["elapsed_seconds"] = finished - started
    result["finished_offset_seconds"] = finished - global_started
    return result


def _run_fixed_requests(
    endpoints: list[str],
    requests: list[dict[str, object]],
    max_concurrency: int,
    request_timeout: float,
    max_concurrency_per_endpoint: int | None = None,
    synchronized_start: bool = False,
) -> tuple[float, list[dict[str, object]], list[dict[str, object]]]:
    endpoints = _validate_endpoints(endpoints)
    if not requests:
        raise ValueError("At least one request is required")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive")
    if max_concurrency_per_endpoint is None:
        max_concurrency_per_endpoint = (
            max_concurrency + len(endpoints) - 1
        ) // len(endpoints)
    if max_concurrency_per_endpoint <= 0:
        raise ValueError("max_concurrency_per_endpoint must be positive")

    assignments = _assign_requests(requests, endpoints)
    if synchronized_start:
        if max_concurrency < len(assignments):
            raise ValueError(
                "synchronized start requires max_concurrency >= request count"
            )
        endpoint_counts = [
            sum(int(item["endpoint_index"]) == endpoint_index
                for item in assignments)
            for endpoint_index in range(len(endpoints))
        ]
        if any(count > max_concurrency_per_endpoint
               for count in endpoint_counts):
            raise ValueError(
                "synchronized start requires every per-endpoint cap to admit "
                "all requests assigned to that endpoint"
            )
    semaphores = [
        threading.Semaphore(max_concurrency_per_endpoint)
        for _ in endpoints
    ]

    ready = (threading.Barrier(len(assignments) + 1)
             if synchronized_start else None)
    release = threading.Event() if synchronized_start else None
    # In the normal path workers may begin immediately after submit. Start the
    # measured interval before creating them so no request work is omitted.
    global_started = 0.0 if synchronized_start else time.perf_counter()

    def run_one(assigned: dict[str, object]) -> dict[str, object]:
        semaphore = semaphores[int(assigned["endpoint_index"])]
        with semaphore:
            if ready is not None and release is not None:
                ready.wait()
                release.wait()
            return _post_fixed_request(
                assigned,
                timeout=request_timeout,
                global_started=global_started,
            )

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_concurrency, len(assignments))
    ) as executor:
        futures = [
            executor.submit(run_one, assigned)
            for assigned in assignments
        ]
        if ready is not None and release is not None:
            ready.wait()
            global_started = time.perf_counter()
            release.set()
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    global_elapsed = time.perf_counter() - global_started
    results.sort(key=lambda item: int(item["request_id"]))
    return global_elapsed, results, assignments


def _valid_usage(result: dict[str, object]) -> tuple[int, int, int] | None:
    if result.get("status") != 200:
        return None
    response = result.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    values = (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        return None
    prompt_tokens, completion_tokens, total_tokens = values
    if total_tokens != prompt_tokens + completion_tokens:
        return None
    return prompt_tokens, completion_tokens, total_tokens


def _summarize_fixed_results(
    global_elapsed: float,
    results: list[dict[str, object]],
    endpoints: list[str],
) -> dict[str, object]:
    if global_elapsed <= 0:
        raise ValueError("global_elapsed must be positive")
    endpoints = _validate_endpoints(endpoints)
    ordered = sorted(results, key=lambda item: int(item["request_id"]))
    expected_ids = list(range(len(ordered)))
    actual_ids = [int(item["request_id"]) for item in ordered]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Request results must contain each id exactly once: {actual_ids!r}")

    usages = [_valid_usage(result) for result in ordered]
    completed = sum(usage is not None for usage in usages)
    http_completed = sum(result.get("status") == 200 for result in ordered)
    prompt_tokens = sum(usage[0] for usage in usages if usage is not None)
    completion_tokens = sum(usage[1] for usage in usages if usage is not None)
    total_tokens = sum(usage[2] for usage in usages if usage is not None)

    endpoint_summaries: list[dict[str, object]] = []
    for endpoint_index, endpoint in enumerate(endpoints):
        endpoint_results = [
            result for result in ordered
            if int(result["endpoint_index"]) == endpoint_index
        ]
        endpoint_usages = [_valid_usage(result) for result in endpoint_results]
        endpoint_prompt_tokens = sum(
            usage[0] for usage in endpoint_usages if usage is not None)
        endpoint_completion_tokens = sum(
            usage[1] for usage in endpoint_usages if usage is not None)
        endpoint_total_tokens = sum(
            usage[2] for usage in endpoint_usages if usage is not None)
        endpoint_completed = sum(
            usage is not None for usage in endpoint_usages)
        endpoint_summaries.append({
            "endpoint_index": endpoint_index,
            "endpoint": endpoint,
            "assigned": len(endpoint_results),
            "completed": endpoint_completed,
            "failed": len(endpoint_results) - endpoint_completed,
            "prompt_tokens": endpoint_prompt_tokens,
            "completion_tokens": endpoint_completion_tokens,
            "total_tokens": endpoint_total_tokens,
            "global_wall_request_throughput_contribution": (
                endpoint_completed / global_elapsed),
            "global_wall_total_token_throughput_contribution": (
                endpoint_total_tokens / global_elapsed),
        })

    return {
        "duration": global_elapsed,
        "global_wall_seconds": global_elapsed,
        "num_prompts": len(ordered),
        "http_completed": http_completed,
        "completed": completed,
        "failed": len(ordered) - completed,
        "usage_failed": http_completed - completed,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_throughput": completed / global_elapsed,
        "prompt_token_throughput": prompt_tokens / global_elapsed,
        "output_token_throughput": completion_tokens / global_elapsed,
        "total_token_throughput": total_tokens / global_elapsed,
        "prompt_token_vector": [
            usage[0] if usage is not None else None for usage in usages
        ],
        "completion_token_vector": [
            usage[1] if usage is not None else None for usage in usages
        ],
        "endpoint_assignment_vector": [
            int(result["endpoint_index"]) for result in ordered
        ],
        "endpoint_summaries": endpoint_summaries,
        "aggregation": {
            "basis": "single_global_wall_clock",
            "formula": "sum(valid_response.total_tokens) / global_wall_seconds",
            "endpoint_local_throughputs_summed": False,
        },
    }


def _request_artifact_rows(
    assignments: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "request_id": int(item["request_id"]),
            "line_number": int(item["line_number"]),
            "endpoint_index": int(item["endpoint_index"]),
            "endpoint": str(item["endpoint"]),
            "request_body": str(item["body_text"]),
        }
        for item in assignments
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_formal_fixed_requests(
    endpoints: list[str],
    requests: list[dict[str, object]],
    max_concurrency: int,
    request_timeout: float,
    max_concurrency_per_endpoint: int | None,
    profile: bool,
    profile_path: Path,
    synchronized_start: bool = False,
) -> tuple[
    float,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object] | None,
]:
    if not profile:
        run_kwargs = {}
        if synchronized_start:
            run_kwargs["synchronized_start"] = True
        elapsed, results, assignments = _run_fixed_requests(
            endpoints,
            requests,
            max_concurrency=max_concurrency,
            request_timeout=request_timeout,
            max_concurrency_per_endpoint=max_concurrency_per_endpoint,
            **run_kwargs,
        )
        return elapsed, results, assignments, None

    targets = _profile_targets(endpoints)
    events: list[dict[str, object]] = []
    formal_result: tuple[
        float,
        list[dict[str, object]],
        list[dict[str, object]],
    ] | None = None
    try:
        for target in targets:
            events.append(
                _post_profile_control(target, "start", request_timeout))
        _write_profile_events(profile_path, events)
        if not all(event.get("success") is True for event in events):
            raise RuntimeError(
                "Failed to start profiling on every fixed-request endpoint")
        run_kwargs = {}
        if synchronized_start:
            run_kwargs["synchronized_start"] = True
        formal_result = _run_fixed_requests(
            endpoints,
            requests,
            max_concurrency=max_concurrency,
            request_timeout=request_timeout,
            max_concurrency_per_endpoint=max_concurrency_per_endpoint,
            **run_kwargs,
        )
    finally:
        for target in targets:
            events.append(
                _post_profile_control(target, "stop", request_timeout))
        _write_profile_events(profile_path, events)

    if formal_result is None:
        raise RuntimeError("Formal fixed-request run did not complete")
    elapsed, results, assignments = formal_result
    return (
        elapsed,
        results,
        assignments,
        _profile_evidence(profile_path, targets, events),
    )


def main() -> int:
    args = parse_args()
    after_warmup_command = _parse_command_json(
        args.after_warmup_check_command_json
    )
    if after_warmup_command is not None and args.warmup_num_prompts <= 0:
        raise ValueError(
            "after-warmup check requires --warmup-num-prompts greater than zero"
        )
    endpoints = _validate_endpoints(args.endpoint)
    request_path = args.request_file.expanduser().resolve()
    requests = _load_fixed_requests(request_path)
    if (
        args.expected_num_prompts is not None
        and len(requests) != args.expected_num_prompts
    ):
        raise ValueError(
            f"Expected {args.expected_num_prompts} requests, found {len(requests)}")
    if not 0 <= args.warmup_num_prompts <= len(requests):
        raise ValueError("warmup_num_prompts must be within the request file")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.tag}_fixed_summary.json"
    result_path = output_dir / f"{args.tag}_fixed_result.json"
    requests_path = output_dir / f"{args.tag}_requests.jsonl"
    responses_path = output_dir / f"{args.tag}_responses.jsonl"
    warmup_responses_path = output_dir / f"{args.tag}_warmup_responses.jsonl"
    profile_path = output_dir / f"{args.tag}_profile.json"
    if summary_path.exists() and not args.rerun:
        print(summary_path.read_text(encoding="utf-8"), end="")
        return 0

    warmup: dict[str, object] | None = None
    phase_boundaries: dict[str, int] = {}
    after_warmup_check: dict[str, object] | None = None
    if args.warmup_num_prompts:
        warmup_concurrency = (
            args.warmup_max_concurrency
            if args.warmup_max_concurrency is not None
            else args.max_concurrency
        )
        warmup_concurrency_per_endpoint = (
            args.warmup_max_concurrency_per_endpoint
            if args.warmup_max_concurrency_per_endpoint is not None
            else args.max_concurrency_per_endpoint
        )
        run_kwargs = {}
        if args.synchronized_start:
            run_kwargs["synchronized_start"] = True
        warmup_elapsed, warmup_results, _ = _run_fixed_requests(
            endpoints,
            requests[:args.warmup_num_prompts],
            max_concurrency=warmup_concurrency,
            request_timeout=args.request_timeout,
            max_concurrency_per_endpoint=warmup_concurrency_per_endpoint,
            **run_kwargs,
        )
        _write_jsonl(warmup_responses_path, warmup_results)
        warmup_summary = _summarize_fixed_results(
            warmup_elapsed, warmup_results, endpoints)
        warmup = {
            "summary": warmup_summary,
            "responses_jsonl": str(warmup_responses_path),
        }
        if warmup_summary["failed"] != 0:
            raise RuntimeError(
                f"Warmup failed for {warmup_summary['failed']} requests")
        phase_boundaries["warmup_completed_unix_ns"] = time.time_ns()

    if after_warmup_command is not None:
        after_warmup_check = _run_after_warmup_check(after_warmup_command)
        phase_boundaries["after_warmup_check_completed_unix_ns"] = int(
            after_warmup_check["finished_unix_ns"]
        )

    phase_boundaries["formal_started_unix_ns"] = time.time_ns()
    elapsed, results, assignments, profile_evidence = _run_formal_fixed_requests(
        endpoints,
        requests,
        max_concurrency=args.max_concurrency,
        request_timeout=args.request_timeout,
        max_concurrency_per_endpoint=args.max_concurrency_per_endpoint,
        profile=args.profile,
        profile_path=profile_path,
        synchronized_start=args.synchronized_start,
    )
    _write_jsonl(requests_path, _request_artifact_rows(assignments))
    _write_jsonl(responses_path, results)
    summary = _summarize_fixed_results(elapsed, results, endpoints)
    summary.update({
        "request_file": str(request_path),
        "endpoints": endpoints,
        "requests_jsonl": str(requests_path),
        "responses_jsonl": str(responses_path),
        "diagnostic_only": args.profile,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    })
    if warmup is not None:
        summary["warmup"] = warmup
    if after_warmup_check is not None:
        summary["after_warmup_check"] = after_warmup_check
    summary["phase_boundaries"] = phase_boundaries
    if profile_evidence is not None:
        summary["profile"] = profile_evidence
    result = {
        "summary": summary,
        "responses": results,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["result_json"] = str(result_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    profile_controls_succeeded = (
        profile_evidence is None
        or profile_evidence["control_succeeded"] is True
    )
    return 0 if summary["failed"] == 0 and profile_controls_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
