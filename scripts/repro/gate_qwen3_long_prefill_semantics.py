#!/usr/bin/env python3
"""Gate Qwen3-30B-A3B long-prefill semantics across three MoE paths.

The gate launches one eager TP2+EP2 vLLM service at a time, submits the same
fixed token-ID prompts twice, archives every request and response, and compares
the experimental paths with forced AllGather.  It is intentionally a quality
gate, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODES = ("forced_allgather", "replicated_local", "default_naive")
REFERENCE_MODE = "forced_allgather"
VOCAB_MODULUS = 151643
REPEATS = 2
GENERATED_TOKENS = 16
DEFAULT_ASCEND_ENV = Path("/usr/local/Ascend/ascend-toolkit/set_env.sh")

PROMPT_SPECS = (
    {
        "name": "offset_50207_len_2048",
        "offset": 50207,
        "length": 2048,
        "sha256_le_u32":
        "c31c0fa53ee1d52d19f0e664b1d9c9516d266e83007e325e3789c7b48a4ee86f",
    },
    {
        "name": "offset_105014_len_2048",
        "offset": 105014,
        "length": 2048,
        "sha256_le_u32":
        "cd700f310d6dd0642384da46e250fe8fcb43143eb53452acd17d006e25eb6da8",
    },
    {
        "name": "offset_53400_len_2048",
        "offset": 53400,
        "length": 2048,
        "sha256_le_u32":
        "71701a88deca1ff7a748f4e1d17c43ae825628222cb11ba885696c944ca351e5",
    },
    {
        "name": "offset_50207_len_1000_control",
        "offset": 50207,
        "length": 1000,
        "sha256_le_u32":
        "8260e436ccdd796781c316ef57fd82ba8fd72c75cbb99e4a932ec8b218506356",
    },
)

THRESHOLDS = {
    "generated_ids_exact": True,
    "prompt_top1_agreement_min": 0.999,
    "prompt_top5_set_agreement_min": 0.990,
    "prompt_actual_logprob_mae_max": 0.020,
    "prompt_actual_logprob_p99_abs_max": 0.100,
    "prompt_actual_logprob_max_abs_max": 0.500,
    "generated_actual_logprob_mae_max": 0.020,
    "generated_actual_logprob_max_abs_max": 0.100,
}

MANAGED_ENV = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "VLLM_LOGGING_LEVEL": "DEBUG",
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "1",
    "VLLM_ASCEND_ENABLE_FLASHCOMM_DYNAMIC_OUTPUT_SHAPES": "0",
    "MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING": "0",
    "MINDPIPE_QWEN3_MOE_GLOBAL_ROUTING_QUANT": "0",
    "MINDPIPE_QWEN3_MOE_QUANTIZED_PEER_REDUCE_SCATTER": "0",
    "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS": "0",
    "VLLM_ASCEND_MOE_PREFILL_COMM_METHOD": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18985)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--termination-timeout", type=float, default=30.0)
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=list(MODES))
    parser.add_argument("--served-model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-model-len", type=int, default=2304)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--execution-mode",
                        choices=("eager", "piecewise"),
                        default="eager")
    parser.add_argument("--compilation-config")
    parser.add_argument("--additional-config")
    parser.add_argument(
        "--forced-allgather-expert-map-path",
        type=Path,
        help=("Load this static expert map only in the forced-AllGather "
              "reference. This lets a replicated-local half map be compared "
              "against the same physical partition without making the "
              "replicated-local mode enable Ascend EPLB."),
    )
    parser.add_argument("--flashcomm-dynamic-output-shapes",
                        action="store_true")
    parser.add_argument("--ascend-env", type=Path, default=DEFAULT_ASCEND_ENV)
    parser.add_argument(
        "--pythonpath",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="Prepend a runtime source directory to PYTHONPATH. May be repeated.",
    )
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE",
        help="Extra server environment. Gate-managed variables take precedence.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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


def _validate_json_object(value: str | None, name: str) -> None:
    if value is None:
        return
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object")


def _json_object(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("configuration must decode to a JSON object")
    return parsed


def _selected_modes(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError("--modes must not contain duplicates")
    selected = [mode for mode in MODES if mode in values]
    if any(mode != REFERENCE_MODE for mode in selected) and REFERENCE_MODE not in selected:
        raise ValueError("candidate modes require forced_allgather as reference")
    return selected


def validate_args(args: argparse.Namespace) -> None:
    _selected_modes(args.modes)
    devices = [part.strip() for part in args.device.split(",")]
    if len(devices) != 2 or any(not part for part in devices):
        raise ValueError("--device must identify exactly two devices, e.g. 0,1")
    if len(set(devices)) != 2:
        raise ValueError("--device entries must be distinct")
    if not (1 <= args.port <= 65535):
        raise ValueError("--port must be in [1, 65535]")
    for name in ("startup_timeout", "request_timeout", "termination_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_model_len < max(spec["length"] for spec in PROMPT_SPECS) + GENERATED_TOKENS:
        raise ValueError("--max-model-len is too small for the fixed protocol")
    if args.max_num_batched_tokens < max(spec["length"] for spec in PROMPT_SPECS):
        raise ValueError("--max-num-batched-tokens must be at least 2048")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    _validate_json_object(args.compilation_config, "--compilation-config")
    _validate_json_object(args.additional_config, "--additional-config")
    additional_config = _json_object(args.additional_config)
    if (args.forced_allgather_expert_map_path is not None
            and "expert_map_path" in additional_config):
        raise ValueError(
            "--forced-allgather-expert-map-path cannot be combined with "
            "expert_map_path in --additional-config")
    if (args.forced_allgather_expert_map_path is not None
            and not args.forced_allgather_expert_map_path.is_file()):
        raise FileNotFoundError(args.forced_allgather_expert_map_path)
    if args.execution_mode == "piecewise" and args.compilation_config is None:
        raise ValueError(
            "--execution-mode piecewise requires --compilation-config")
    _parse_env(args.env)
    if args.validate_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-only is used")
    if not args.model.expanduser().is_dir():
        raise FileNotFoundError(args.model)
    if not args.ascend_env.expanduser().is_file():
        raise FileNotFoundError(args.ascend_env)
    if not (Path(args.python).expanduser().is_file() or shutil.which(args.python)):
        raise FileNotFoundError(args.python)
    for path in args.pythonpath:
        if not path.expanduser().is_dir():
            raise FileNotFoundError(f"PYTHONPATH directory does not exist: {path}")


def _runtime_environment(
    args: argparse.Namespace, overrides: dict[str, str]
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(overrides)
    configured = [str(path.expanduser().resolve()) for path in args.pythonpath]
    inherited = environment.get("PYTHONPATH")
    if configured:
        environment["PYTHONPATH"] = os.pathsep.join(
            configured + ([inherited] if inherited else [])
        )
    return environment


def _sourced_command(args: argparse.Namespace, command: list[str]) -> str:
    ascend_env = args.ascend_env.expanduser().resolve()
    return f"source {shlex.quote(str(ascend_env))} && exec {shlex.join(command)}"


def _preflight_runtime(args: argparse.Namespace) -> None:
    command = [
        args.python,
        "-c",
        "import vllm; import vllm_ascend",
    ]
    completed = subprocess.run(
        ["bash", "-lc", _sourced_command(args, command)],
        cwd=Path(__file__).resolve().parents[2],
        env=_runtime_environment(args, _parse_env(args.env)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=min(args.startup_timeout, 120.0),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Runtime preflight could not import vllm and vllm_ascend "
            f"with the configured Python/PYTHONPATH:\n{completed.stdout}"
        )


def _prompt_token_ids(offset: int, length: int) -> list[int]:
    return [(offset + index) % VOCAB_MODULUS for index in range(length)]


def _token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if not isinstance(token_id, int) or not (0 <= token_id <= 0xFFFFFFFF):
            raise ValueError(f"Invalid uint32 token ID: {token_id!r}")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def _fixed_prompts() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for spec in PROMPT_SPECS:
        token_ids = _prompt_token_ids(spec["offset"], spec["length"])
        actual_hash = _token_ids_sha256(token_ids)
        if actual_hash != spec["sha256_le_u32"]:
            raise RuntimeError(
                f"Fixed prompt hash mismatch for {spec['name']}: {actual_hash}")
        prompts.append({**spec, "token_ids": token_ids})
    return prompts


def _mode_environment(mode: str, args: argparse.Namespace) -> dict[str, str]:
    overrides = _parse_env(args.env)
    overrides.update(MANAGED_ENV)
    overrides["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    overrides["VLLM_ASCEND_ENABLE_FLASHCOMM_DYNAMIC_OUTPUT_SHAPES"] = (
        "1" if args.flashcomm_dynamic_output_shapes else "0")
    if mode == "forced_allgather":
        overrides["VLLM_ASCEND_MOE_PREFILL_COMM_METHOD"] = "allgather"
    elif mode == "replicated_local":
        overrides["MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS"] = "1"
    elif mode != "default_naive":
        raise ValueError(f"Unknown mode: {mode}")
    return overrides


def _mode_additional_config(args: argparse.Namespace,
                            mode: str) -> str | None:
    config = _json_object(args.additional_config)
    if (mode == REFERENCE_MODE
            and args.forced_allgather_expert_map_path is not None):
        config["expert_map_path"] = str(
            args.forced_allgather_expert_map_path.resolve())
    if not config:
        return None
    return json.dumps(config, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True)


def _server_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        args.python,
        "-m", "vllm.entrypoints.cli.main", "serve",
        str(args.model.expanduser().resolve()),
        "--host", args.host,
        "--port", str(args.port),
        "--trust-remote-code",
        "--dtype", "float16",
        "--served-model-name", args.served_model_name,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", "2",
        "--quantization", "ascend",
        "--enable-expert-parallel",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", "1",
        "--disable-log-requests",
        "--disable-log-stats",
    ]
    if args.execution_mode == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(["--compilation-config", args.compilation_config])
    additional_config = _mode_additional_config(args, mode)
    if additional_config is not None:
        command.extend(["--additional-config", additional_config])
    return command


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _tail(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        try:
            handle.seek(-limit, os.SEEK_END)
        except OSError:
            handle.seek(0)
        return handle.read().decode("utf-8", errors="replace")


def _wait_for_health(args: argparse.Namespace, proc: subprocess.Popen[Any],
                     log_path: Path) -> None:
    url = f"http://{args.host}:{args.port}/health"
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with code {proc.returncode}; "
                f"log tail:\n{_tail(log_path)}")
        try:
            with urlopen(url, timeout=min(5.0, args.request_timeout)) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {url}; log tail:\n{_tail(log_path)}")


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_tree(proc: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pid": proc.pid,
        "term_sent": False,
        "kill_sent": False,
        "returncode_before": proc.poll(),
    }
    if _group_exists(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            record["term_sent"] = True
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while _group_exists(proc.pid) and time.monotonic() < deadline:
        proc.poll()  # Reap an exited group leader before probing its PGID.
        time.sleep(0.25)
    if _group_exists(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            record["kill_sent"] = True
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + min(10.0, timeout)
        while _group_exists(proc.pid) and time.monotonic() < kill_deadline:
            proc.poll()
            time.sleep(0.25)
    try:
        record["returncode_after"] = proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        record["returncode_after"] = proc.poll()
    record["process_group_gone"] = not _group_exists(proc.pid)
    return record


def _request_payload(args: argparse.Namespace, prompt: dict[str, Any],
                     mode: str, repeat: int) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "prompt": prompt["token_ids"],
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
        "request_id": f"long-prefill-{mode}-{prompt['name']}-r{repeat}",
    }


def _logprob(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("logprob")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _ranked_tokens(entry: Any, limit: int = 5) -> list[int]:
    if not isinstance(entry, dict):
        return []
    rows: list[tuple[int | None, float, int]] = []
    for raw_token, value in entry.items():
        try:
            token_id = int(raw_token)
        except (TypeError, ValueError):
            continue
        probability = _logprob(value)
        if probability is None:
            continue
        rank = value.get("rank") if isinstance(value, dict) else None
        rank = int(rank) if isinstance(rank, int) and rank > 0 else None
        rows.append((rank, probability, token_id))
    ranked = [row for row in rows if row[0] is not None and row[0] <= limit]
    if ranked:
        ranked.sort(key=lambda row: (int(row[0]), -row[1], row[2]))
    else:
        ranked = sorted(rows, key=lambda row: (-row[1], row[2]))[:limit]
    return [row[2] for row in ranked]


def _actual_prompt_logprob(entry: Any, token_id: int) -> float | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get(str(token_id), entry.get(token_id))
    return _logprob(value)


def _choice(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    return choices[0]


def _validate_response(body: Any, prompt: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    choice = _choice(body)
    if choice is None:
        return {"valid": False, "errors": ["response must contain exactly one choice"]}
    prompt_ids = choice.get("prompt_token_ids")
    generated_ids = choice.get("token_ids")
    prompt_logprobs = choice.get("prompt_logprobs")
    generation_logprobs = choice.get("logprobs")
    if prompt_ids != prompt["token_ids"]:
        errors.append("returned prompt_token_ids do not equal the fixed prompt")
    if not isinstance(generated_ids, list) or len(generated_ids) != GENERATED_TOKENS:
        errors.append(f"expected exactly {GENERATED_TOKENS} generated token IDs")
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != prompt["length"]:
        errors.append("prompt_logprobs length does not equal prompt length")
    else:
        missing_actual = 0
        incomplete_top5 = 0
        for index in range(1, prompt["length"]):
            if _actual_prompt_logprob(prompt_logprobs[index], prompt["token_ids"][index]) is None:
                missing_actual += 1
            if len(_ranked_tokens(prompt_logprobs[index], 5)) != 5:
                incomplete_top5 += 1
        if missing_actual:
            errors.append(f"actual-token prompt logprob missing at {missing_actual} positions")
        if incomplete_top5:
            errors.append(f"top-5 prompt set incomplete at {incomplete_top5} positions")
    token_logprobs = (generation_logprobs.get("token_logprobs")
                      if isinstance(generation_logprobs, dict) else None)
    if (not isinstance(token_logprobs, list)
            or len(token_logprobs) != GENERATED_TOKENS
            or any(_logprob(value) is None for value in token_logprobs)):
        errors.append("generated actual-token logprobs are missing or malformed")
    usage = body.get("usage") if isinstance(body, dict) else None
    if isinstance(usage, dict) and usage.get("prompt_tokens") != prompt["length"]:
        errors.append("usage.prompt_tokens does not equal prompt length")
    return {
        "valid": not errors,
        "errors": errors,
        "prompt_sha256_le_u32": (_token_ids_sha256(prompt_ids)
                                  if isinstance(prompt_ids, list)
                                  and all(isinstance(item, int) for item in prompt_ids)
                                  else None),
        "generated_token_count": len(generated_ids) if isinstance(generated_ids, list) else None,
        "finish_reason": choice.get("finish_reason"),
    }


def _send_request(args: argparse.Namespace, mode: str, prompt: dict[str, Any],
                  repeat: int, path: Path) -> tuple[dict[str, Any], Any | None]:
    payload = _request_payload(args, prompt, mode, repeat)
    request = Request(
        f"http://{args.host}:{args.port}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status: int | None = None
    body: Any | None = None
    error: str | None = None
    try:
        with urlopen(request, timeout=args.request_timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw_error_body": raw}
        error = repr(exc)
    except Exception as exc:
        error = repr(exc)
    validation = (_validate_response(body, prompt) if status == 200 else {
        "valid": False,
        "errors": [error or f"HTTP status {status}"],
    })
    record = {
        "schema_version": 1,
        "mode": mode,
        "case": prompt["name"],
        "offset": prompt["offset"],
        "length": prompt["length"],
        "expected_prompt_sha256_le_u32": prompt["sha256_le_u32"],
        "repeat": repeat,
        "requested_at_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "http_status": status,
        "request": payload,
        "response": body,
        "validation": validation,
        "error": error,
    }
    _atomic_write_json(path, record)
    return record, body


def _abs_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mae": None, "p99_abs": None, "max_abs": None}
    ordered = sorted(values)
    p99_index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mae": sum(values) / len(values),
        "p99_abs": ordered[p99_index],
        "max_abs": ordered[-1],
    }


def compare_responses(reference: Any, candidate: Any) -> dict[str, Any]:
    ref_choice = _choice(reference)
    cand_choice = _choice(candidate)
    if ref_choice is None or cand_choice is None:
        return {"passed": False, "error": "missing response choice"}
    ref_prompt_ids = ref_choice.get("prompt_token_ids")
    cand_prompt_ids = cand_choice.get("prompt_token_ids")
    ref_generated_ids = ref_choice.get("token_ids")
    cand_generated_ids = cand_choice.get("token_ids")
    ref_prompt_lp = ref_choice.get("prompt_logprobs")
    cand_prompt_lp = cand_choice.get("prompt_logprobs")
    shapes_valid = (
        isinstance(ref_prompt_ids, list)
        and ref_prompt_ids == cand_prompt_ids
        and isinstance(ref_prompt_lp, list)
        and isinstance(cand_prompt_lp, list)
        and len(ref_prompt_lp) == len(cand_prompt_lp) == len(ref_prompt_ids)
        and isinstance(ref_generated_ids, list)
        and isinstance(cand_generated_ids, list)
        and len(ref_generated_ids) == len(cand_generated_ids) == GENERATED_TOKENS)
    top1_equal = 0
    top5_equal = 0
    prompt_positions = max(0, len(ref_prompt_ids) - 1) if isinstance(ref_prompt_ids, list) else 0
    prompt_differences: list[float] = []
    missing_prompt_values = 0
    if shapes_valid:
        for index in range(1, len(ref_prompt_ids)):
            ref_ranked = _ranked_tokens(ref_prompt_lp[index], 5)
            cand_ranked = _ranked_tokens(cand_prompt_lp[index], 5)
            top1_equal += bool(ref_ranked and cand_ranked and ref_ranked[0] == cand_ranked[0])
            top5_equal += bool(len(ref_ranked) == len(cand_ranked) == 5
                               and set(ref_ranked) == set(cand_ranked))
            token_id = ref_prompt_ids[index]
            left = _actual_prompt_logprob(ref_prompt_lp[index], token_id)
            right = _actual_prompt_logprob(cand_prompt_lp[index], token_id)
            if left is None or right is None:
                missing_prompt_values += 1
            else:
                prompt_differences.append(abs(left - right))
    ref_generation_lp = ref_choice.get("logprobs")
    cand_generation_lp = cand_choice.get("logprobs")
    ref_token_lp = (ref_generation_lp.get("token_logprobs")
                    if isinstance(ref_generation_lp, dict) else None)
    cand_token_lp = (cand_generation_lp.get("token_logprobs")
                     if isinstance(cand_generation_lp, dict) else None)
    generated_differences: list[float] = []
    missing_generated_values = 0
    if isinstance(ref_token_lp, list) and isinstance(cand_token_lp, list):
        for left_raw, right_raw in zip(ref_token_lp, cand_token_lp):
            left = _logprob(left_raw)
            right = _logprob(right_raw)
            if left is None or right is None:
                missing_generated_values += 1
            else:
                generated_differences.append(abs(left - right))
    else:
        missing_generated_values = GENERATED_TOKENS
    prompt_stats = _abs_stats(prompt_differences)
    generated_stats = _abs_stats(generated_differences)
    top1_agreement = top1_equal / prompt_positions if prompt_positions else 0.0
    top5_agreement = top5_equal / prompt_positions if prompt_positions else 0.0
    generated_ids_exact = ref_generated_ids == cand_generated_ids
    passed = bool(
        shapes_valid
        and generated_ids_exact
        and missing_prompt_values == 0
        and len(prompt_differences) == prompt_positions
        and missing_generated_values == 0
        and len(generated_differences) == GENERATED_TOKENS
        and top1_agreement >= THRESHOLDS["prompt_top1_agreement_min"]
        and top5_agreement >= THRESHOLDS["prompt_top5_set_agreement_min"]
        and prompt_stats["mae"] <= THRESHOLDS["prompt_actual_logprob_mae_max"]
        and prompt_stats["p99_abs"] <= THRESHOLDS["prompt_actual_logprob_p99_abs_max"]
        and prompt_stats["max_abs"] <= THRESHOLDS["prompt_actual_logprob_max_abs_max"]
        and generated_stats["mae"] <= THRESHOLDS["generated_actual_logprob_mae_max"]
        and generated_stats["max_abs"] <= THRESHOLDS["generated_actual_logprob_max_abs_max"])
    return {
        "passed": passed,
        "shapes_valid": shapes_valid,
        "generated_ids_exact": generated_ids_exact,
        "prompt_positions": prompt_positions,
        "prompt_top1_agreement": top1_agreement,
        "prompt_top5_set_agreement": top5_agreement,
        "prompt_actual_logprob_abs": prompt_stats,
        "prompt_actual_logprob_missing": missing_prompt_values,
        "generated_actual_logprob_abs": generated_stats,
        "generated_actual_logprob_missing": missing_generated_values,
    }


def _activation_evidence(mode: str, log_path: Path,
                         args: argparse.Namespace) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    patterns = {
        "forced_allgather": (
            r"num_tokens:\s*(?:1000|1024|2048),\s*moe_comm_type:\s*"
            r"MoECommType\.ALLGATHER"),
        "default_naive": (
            r"num_tokens:\s*(?:1000|1024|2048),\s*moe_comm_type:\s*"
            r"MoECommType\.NAIVE_MULTICAST"),
        "replicated_local": (
            r"Using experimental Qwen3 W8A8 replicated-local MoE"),
    }
    matches = re.findall(patterns[mode], text)
    result = {
        "passed": bool(matches),
        "expected_pattern": patterns[mode],
        "match_count": len(matches),
    }
    if (mode == REFERENCE_MODE
            and args.forced_allgather_expert_map_path is not None):
        resolved_path = str(args.forced_allgather_expert_map_path.resolve())
        path_matches = text.count(resolved_path)
        result["expert_map_path"] = resolved_path
        result["expert_map_path_match_count"] = path_matches
        result["passed"] = result["passed"] and path_matches > 0
    return result


def _run_mode(args: argparse.Namespace, mode: str, prompts: list[dict[str, Any]],
              output_dir: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    log_path = mode_dir / "server.log"
    command = _server_command(args, mode)
    shell_command = _sourced_command(args, command)
    overrides = _mode_environment(mode, args)
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "created_at_utc": _utc_now(),
        "server_command": command,
        "server_shell_command": shell_command,
        "environment_overrides": overrides,
        "forced_allgather_expert_map_path": (
            str(args.forced_allgather_expert_map_path.resolve())
            if mode == REFERENCE_MODE
            and args.forced_allgather_expert_map_path is not None else None),
        "server_log": str(log_path.resolve()),
    }
    _atomic_write_json(mode_dir / "mode_manifest.json", manifest)
    responses: dict[str, list[Any]] = {prompt["name"]: [] for prompt in prompts}
    request_summaries: list[dict[str, Any]] = []
    proc: subprocess.Popen[Any] | None = None
    termination: dict[str, Any] | None = None
    error: str | None = None
    started = time.perf_counter()
    if _is_port_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use")
    environment = _runtime_environment(args, overrides)
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
                for prompt in prompts:
                    for repeat in range(REPEATS):
                        request_path = mode_dir / (
                            f"request_{prompt['name']}_repeat_{repeat}.json")
                        record, body = _send_request(
                            args, mode, prompt, repeat, request_path)
                        responses[prompt["name"]].append(body)
                        request_summaries.append({
                            "case": prompt["name"],
                            "repeat": repeat,
                            "path": str(request_path.resolve()),
                            "http_status": record["http_status"],
                            "elapsed_seconds": record["elapsed_seconds"],
                            "validation": record["validation"],
                            "error": record["error"],
                        })
            finally:
                termination = _terminate_process_tree(proc, args.termination_timeout)
    except Exception:
        error = traceback.format_exc()
        if proc is not None and termination is None:
            termination = _terminate_process_tree(proc, args.termination_timeout)
    evidence = _activation_evidence(mode, log_path, args)
    requests_valid = (
        len(request_summaries) == len(prompts) * REPEATS
        and all(item["validation"].get("valid") for item in request_summaries))
    result = {
        **manifest,
        "status": "passed" if error is None and requests_valid and evidence["passed"] else "failed",
        "duration_seconds": time.perf_counter() - started,
        "requests_valid": requests_valid,
        "activation_evidence": evidence,
        "requests": request_summaries,
        "termination": termination,
        "error": error,
    }
    _atomic_write_json(mode_dir / "mode_result.json", result)
    return result, responses


def _build_comparisons(modes: list[str], prompts: list[dict[str, Any]],
                       responses: dict[str, dict[str, list[Any]]]) -> dict[str, Any]:
    repeat_stability: dict[str, Any] = {}
    for mode in modes:
        cases: dict[str, Any] = {}
        for prompt in prompts:
            values = responses.get(mode, {}).get(prompt["name"], [])
            cases[prompt["name"]] = (
                compare_responses(values[0], values[1]) if len(values) == REPEATS
                else {"passed": False, "error": "two valid repeats were not collected"})
        repeat_stability[mode] = {
            "passed": all(value["passed"] for value in cases.values()),
            "cases": cases,
        }
    against_reference: dict[str, Any] = {}
    for mode in modes:
        if mode == REFERENCE_MODE:
            continue
        cases: dict[str, Any] = {}
        for prompt in prompts:
            ref_values = responses.get(REFERENCE_MODE, {}).get(prompt["name"], [])
            candidate_values = responses.get(mode, {}).get(prompt["name"], [])
            pairs: list[dict[str, Any]] = []
            if len(ref_values) == len(candidate_values) == REPEATS:
                for repeat in range(REPEATS):
                    pairs.append({
                        "repeat": repeat,
                        **compare_responses(ref_values[repeat], candidate_values[repeat]),
                    })
            cases[prompt["name"]] = {
                "passed": len(pairs) == REPEATS and all(pair["passed"] for pair in pairs),
                "paired_repeats": pairs,
            }
        against_reference[mode] = {
            "passed": all(value["passed"] for value in cases.values()),
            "cases": cases,
        }
    reference_stable = repeat_stability.get(
        REFERENCE_MODE, {"passed": False})["passed"]
    return {
        "thresholds": THRESHOLDS,
        "reference_mode": REFERENCE_MODE,
        "reference_repeat_stable": reference_stable,
        "repeat_stability": repeat_stability,
        "against_reference": against_reference,
    }


def _static_contract(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "qwen3_long_prefill_semantics_gate",
        "device_execution": False,
        "modes": _selected_modes(args.modes),
        "reference_mode": REFERENCE_MODE,
        "service": {
            "execution_mode": args.execution_mode,
            "eager": args.execution_mode == "eager",
            "compilation_config": args.compilation_config,
            "additional_config": args.additional_config,
            "forced_allgather_expert_map_path": (
                str(args.forced_allgather_expert_map_path.resolve())
                if args.forced_allgather_expert_map_path is not None else None),
            "tensor_parallel_size": 2,
            "expert_parallel": True,
            "flashcomm": True,
            "prefix_caching": False,
            "chunked_prefill": False,
            "generated_tokens": GENERATED_TOKENS,
            "repeats": REPEATS,
            "logging_level": "DEBUG",
            "flashcomm_dynamic_output_shapes": (
                args.flashcomm_dynamic_output_shapes),
        },
        "disabled_candidates": [
            "MINDPIPE_QWEN3_MOE_PREQUANT_ROUTING",
            "MINDPIPE_QWEN3_MOE_GLOBAL_ROUTING_QUANT",
            "MINDPIPE_QWEN3_MOE_QUANTIZED_PEER_REDUCE_SCATTER",
        ],
        "prompt_formula": "[(offset + i) % 151643 for i in range(length)]",
        "prompt_hash_encoding": "concatenated little-endian uint32",
        "prompts": [{key: value for key, value in prompt.items()
                     if key != "token_ids"} for prompt in prompts],
        "thresholds": THRESHOLDS,
    }


def _validate_comparator() -> None:
    prompt_ids = list(range(10, 18))
    prompt_logprobs: list[Any] = [None]
    for token_id in prompt_ids[1:]:
        prompt_logprobs.append({
            str(token_id): {"logprob": -0.1, "rank": 1},
            str(token_id + 100): {"logprob": -1.0, "rank": 2},
            str(token_id + 200): {"logprob": -2.0, "rank": 3},
            str(token_id + 300): {"logprob": -3.0, "rank": 4},
            str(token_id + 400): {"logprob": -4.0, "rank": 5},
        })
    body = {
        "choices": [{
            "prompt_token_ids": prompt_ids,
            "token_ids": list(range(GENERATED_TOKENS)),
            "prompt_logprobs": prompt_logprobs,
            "logprobs": {"token_logprobs": [-0.2] * GENERATED_TOKENS},
        }],
    }
    if not compare_responses(body, json.loads(json.dumps(body)))["passed"]:
        raise RuntimeError("identical synthetic responses failed comparator self-test")
    changed = json.loads(json.dumps(body))
    changed["choices"][0]["token_ids"][0] = 999
    if compare_responses(body, changed)["passed"]:
        raise RuntimeError("generated-token mismatch passed comparator self-test")


def main() -> int:
    args = parse_args()
    validate_args(args)
    prompts = _fixed_prompts()
    _validate_comparator()
    contract = _static_contract(args, prompts)
    if args.validate_only:
        print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    assert args.output_dir is not None
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"--output-dir must be empty: {output_dir}")
    _preflight_runtime(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract["device_execution"] = True
    contract["started_at_utc"] = _utc_now()
    contract["model"] = str(args.model.expanduser().resolve())
    contract["python"] = args.python
    contract["ascend_env"] = str(args.ascend_env.expanduser().resolve())
    contract["pythonpath"] = [
        str(path.expanduser().resolve()) for path in args.pythonpath
    ]
    contract["device"] = args.device
    _atomic_write_json(output_dir / "protocol.json", contract)

    modes = _selected_modes(args.modes)
    mode_results: dict[str, Any] = {}
    responses: dict[str, dict[str, list[Any]]] = {}
    for mode in modes:
        try:
            result, mode_responses = _run_mode(args, mode, prompts, output_dir)
        except Exception:
            result = {
                "mode": mode,
                "status": "failed",
                "error": traceback.format_exc(),
            }
            mode_responses = {}
            _atomic_write_json(output_dir / f"{mode}_launch_failure.json", result)
        mode_results[mode] = result
        responses[mode] = mode_responses

    comparisons = _build_comparisons(modes, prompts, responses)
    all_modes_passed = all(result.get("status") == "passed"
                           for result in mode_results.values())
    all_repeat_stable = all(
        result["passed"] for result in comparisons["repeat_stability"].values())
    candidates_match = all(
        result["passed"] for result in comparisons["against_reference"].values())
    passed = bool(all_modes_passed and comparisons["reference_repeat_stable"]
                  and all_repeat_stable and candidates_match)
    summary = {
        "schema_version": 1,
        "kind": "qwen3_long_prefill_semantics_gate_summary",
        "completed_at_utc": _utc_now(),
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "protocol": str((output_dir / "protocol.json").resolve()),
        "mode_results": mode_results,
        "comparisons": comparisons,
        "notes": [
            "forced_allgather is the correctness reference",
            "all four fixed prompts and both repeats must pass",
            "activation evidence is parsed from DEBUG server logs",
        ],
    }
    summary_path = output_dir / "agent_summary.json"
    _atomic_write_json(summary_path, summary)
    print(summary_path)
    print(summary["verdict"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
