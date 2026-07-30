#!/usr/bin/env python3
"""Run human-readable prompts against the accepted accelerated Qwen3 W8A8 path."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


DEFAULT_ASCEND_ENV = Path("/usr/local/Ascend/ascend-toolkit/set_env.sh")
SERVER_ENV = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MINDPIPE_ENGINE_IDLE_COALESCE_US": "30000",
    "MINDPIPE_ENGINE_IDLE_COALESCE_TARGET_ADDS": "31",
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
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "0",
    "VLLM_ASCEND_ENABLE_QWEN3_MOE_CHUNKED_OVERLAP": "0",
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "0",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE_W8A8": "0",
}


def _case(name: str, category: str, prompt: str, check: str, expected: Any) -> dict[str, Any]:
    return {"name": name, "category": category, "prompt": prompt, "check": check, "expected": expected}


def cases() -> list[dict[str, Any]]:
    return [
        _case("math_multiply", "数学", "只回答最终数字：17乘以23等于多少？", "exact", "391"),
        _case("math_divide", "数学", "只回答最终数字：144除以12等于多少？", "exact", "12"),
        _case("math_power", "数学", "只回答最终数字：2的10次方是多少？", "exact", "1024"),
        _case("math_sum", "数学", "只回答最终数字：从1到100的整数之和是多少？", "exact", "5050"),
        _case("instruction_exact", "指令遵循", "只输出 MINDPIPE_OK，不要添加标点或解释。", "exact", "MINDPIPE_OK"),
        _case("instruction_lines", "指令遵循", "严格输出三行，依次是 alpha、beta、gamma，不要添加其他内容。", "exact", "alpha\nbeta\ngamma"),
        _case("instruction_keywords", "指令遵循", "从这句话提取三个技术词，只用英文逗号连接：系统采用W8A8量化，在C32并发下运行TP2。", "exact", "W8A8,C32,TP2"),
        _case("instruction_sort", "指令遵循", "把数字9、2、5、3升序排列，只输出英文逗号分隔的数字。", "exact", "2,3,5,9"),
        _case("json_object", "结构化输出", "只输出合法JSON，不要代码块：status为ok，count为3。", "json", {"status": "ok", "count": 3}),
        _case("json_array", "结构化输出", "只输出合法JSON数组，不要代码块，内容是前四个正偶数。", "json", [2, 4, 6, 8]),
        _case("json_boolean", "结构化输出", "只输出合法JSON，不要代码块：name为MindPipe，enabled为true。", "json", {"name": "MindPipe", "enabled": True}),
        _case("json_nested", "结构化输出", "只输出合法JSON，不要代码块，结构为metrics.speedup，值是1.608009。", "json", {"metrics": {"speedup": 1.608009}}),
        _case("knowledge_capital", "基础知识", "中国的首都是哪里？只用一个城市名回答。", "contains", ["北京"]),
        _case("knowledge_formula", "基础知识", "水的化学式是什么？只回答化学式。", "contains", ["H2O"]),
        _case("knowledge_colors", "基础知识", "光的三原色是什么？用顿号分隔，简短回答。", "contains", ["红", "绿", "蓝"]),
        _case("knowledge_ocean", "基础知识", "世界上面积最大的海洋是什么？只回答名称。", "contains", ["太平洋"]),
        _case("translation_en_zh", "翻译", "把 Machine learning changes software development. 翻译成中文，只给译文。", "contains", ["机器学习", "软件开发"]),
        _case("translation_zh_en", "翻译", "把“今天天气很好”翻译成英文，只给译文。", "contains_ci", ["weather", "today"]),
        _case("translation_term", "翻译", "把“端到端吞吐”翻译成英文，只给译文。", "contains_ci", ["end-to-end", "throughput"]),
        _case("translation_sentence", "翻译", "把 The test passed without errors. 翻译成中文，只给译文。", "contains", ["测试", "通过", "错误"]),
        _case("code_python", "代码", "只输出一行Python代码，定义函数add(a,b)并返回两数之和。", "contains_ci", ["def add", "return", "a + b"]),
        _case("code_sql", "代码", "只输出一条SQL：从users表查询active等于1的用户id和name。", "contains_ci", ["select", "id", "name", "from users", "active"]),
        _case("code_bash", "代码", "只输出用于打印当前工作目录的bash命令。", "exact", "pwd"),
        _case("code_python_list", "代码", "只输出Python表达式：生成1到5的平方列表。", "contains_ci", ["range(1, 6)", "**2"]),
        _case("classify_sentiment", "分类与摘要", "判断情感，只回答正面或负面：这个版本速度更快，结果也稳定。", "exact", "正面"),
        _case("classify_topic", "分类与摘要", "判断主题，只回答科技、体育或金融：新芯片提升了模型推理吞吐。", "exact", "科技"),
        _case("summary_system", "分类与摘要", "用不超过15个汉字概括：优化降低了推理延迟，同时提高了系统吞吐。", "contains", ["延迟", "吞吐"]),
        _case("extract_date", "分类与摘要", "从句子提取日期，只输出YYYY-MM-DD：项目将在2026年7月26日完成复核。", "exact", "2026-07-26"),
        _case("logic_sequence", "逻辑", "数列2、4、8、16的下一项是什么？只回答数字。", "exact", "32"),
        _case("logic_deduction", "逻辑", "所有A都是B，所有B都是C，能否推出所有A都是C？只回答能或不能。", "exact", "能"),
        _case("logic_odd", "逻辑", "猫、狗、汽车中哪一个不同类？只回答词语。", "exact", "汽车"),
        _case("logic_weekday", "逻辑", "如果今天是星期一，三天后是星期几？只回答星期几。", "exact", "星期四"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
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
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra server environment. Acceleration variables take precedence.",
    )
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19077)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260712)
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


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.expanduser().is_dir():
        raise FileNotFoundError(args.model)
    if not args.ascend_env.expanduser().is_file():
        raise FileNotFoundError(args.ascend_env)
    if not (Path(args.python).expanduser().is_file() or shutil.which(args.python)):
        raise FileNotFoundError(args.python)
    for path in args.pythonpath:
        if not path.expanduser().is_dir():
            raise FileNotFoundError(f"PYTHONPATH directory does not exist: {path}")
    _parse_env(args.env)
    devices = [part.strip() for part in args.device.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2 or any(not part for part in devices):
        raise ValueError("--device must identify two distinct devices")
    if not (1 <= args.port <= 65535):
        raise ValueError("--port must be in [1, 65535]")


def _runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_parse_env(args.env))
    environment.update(SERVER_ENV)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = args.device
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


def _preflight_runtime(args: argparse.Namespace, environment: dict[str, str]) -> None:
    command = [args.python, "-c", "import vllm; import vllm_ascend"]
    completed = subprocess.run(
        ["bash", "-lc", _sourced_command(args, command)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
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


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python, "-m", "vllm.entrypoints.cli.main", "serve",
        str(args.model.expanduser().resolve()),
        "--host", args.host, "--port", str(args.port), "--trust-remote-code",
        "--dtype", "float16", "--served-model-name", "qwen3-30b-a3b",
        "--max-model-len", "2304", "--gpu-memory-utilization", "0.8",
        "--tensor-parallel-size", "2", "--quantization", "ascend",
        "--enable-expert-parallel", "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill", "--max-num-batched-tokens", "65536",
        "--max-num-seqs", "32", "--disable-log-requests", "--disable-log-stats",
        "--additional-config", '{"torchair_graph_config":{"enabled":false},"ascend_scheduler_config":{"enabled":true},"refresh":true}',
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}',
    ]


def _wait_health(args: argparse.Namespace, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + args.startup_timeout
    url = f"http://{args.host}:{args.port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {args.startup_timeout}s")


def _post(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"http://{args.host}:{args.port}/v1/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    match = re.fullmatch(r"```(?:json|python|sql|bash)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _normalize_text(text: str, *, strip_terminal_punctuation: bool = False) -> str:
    value = _strip_code_fence(text).translate(
        str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    )
    value = value.strip()
    if strip_terminal_punctuation:
        value = re.sub(r"[。.!！?？]+$", "", value).rstrip()
    return value


def _validate(case: dict[str, Any], text: str) -> tuple[bool, str]:
    value = _normalize_text(text)
    check = case["check"]
    expected = case["expected"]
    if check == "exact":
        normalized = _normalize_text(value, strip_terminal_punctuation=True)
        normalized = re.sub(r"[，,]\s*", ",", normalized)
        return normalized == expected, f"expected exact {expected!r}"
    if check == "json":
        try:
            actual = json.loads(value)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
        return actual == expected, f"expected JSON {expected!r}"
    haystack = value if check == "contains" else value.lower()
    needles = expected if check == "contains" else [item.lower() for item in expected]
    missing = [item for item in needles if item not in haystack]
    return not missing, f"missing required fragments: {missing!r}"


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate(process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    pgid = process.pid
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while _group_exists(pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.25)
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + min(10.0, timeout)
        while _group_exists(pgid) and time.monotonic() < kill_deadline:
            process.poll()
            time.sleep(0.25)
    try:
        process.wait(timeout=min(10.0, timeout))
    except subprocess.TimeoutExpired:
        process.poll()


def _all_finish_reasons_stop(case_results: list[dict[str, Any]], expected: int) -> bool:
    return len(case_results) == expected and all(
        item.get("finish_reason") == "stop" for item in case_results
    )


def _evidence(log_text: str) -> dict[str, Any]:
    patterns = {
        "request_parallel_enabled": r"Using experimental Qwen3 attention request parallelism \(decode_aclgraph=True\)",
        "replicated_local_moe_enabled": r"Using experimental Qwen3 W8A8 replicated-local MoE",
        "replicated_single_pass_used": r"Using runtime Qwen3 replicated-local single-pass 128-expert MLP",
        "c32_requests_coalesced": r"collected 31 additional ADD request\(s\)",
        "acl_graph_replayed": r"Replaying aclgraph",
    }
    return {
        name: {"passed": bool(re.search(pattern, log_text)), "matches": len(re.findall(pattern, log_text))}
        for name, pattern in patterns.items()
    }


def main() -> int:
    args = parse_args()
    _validate_args(args)
    environment = _runtime_environment(args)
    _preflight_runtime(args, environment)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = cases()
    if len(test_cases) != 32:
        raise RuntimeError(f"expected 32 cases, got {len(test_cases)}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.expanduser().resolve(), trust_remote_code=True
    )
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for case in test_cases
    ]
    payload = {
        "model": "qwen3-30b-a3b", "prompt": rendered,
        "max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 1.0,
        "seed": args.seed, "stream": False, "echo": False,
    }
    request_path = args.output_dir / "request.json"
    response_path = args.output_dir / "response.json"
    result_path = args.output_dir / "RESULT.json"
    log_path = args.output_dir / "server.log"
    request_path.write_text(json.dumps({"cases": test_cases, "payload": payload}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    command = _server_command(args)
    shell_command = _sourced_command(args, command)
    started_at = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            ["bash", "-lc", shell_command], cwd=Path(__file__).resolve().parents[2],
            env=environment, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        try:
            _wait_health(args, process)
            request_started = time.perf_counter()
            response = _post(args, payload)
            elapsed = time.perf_counter() - request_started
            response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            _terminate(process)

    choices = response.get("choices", []) if isinstance(response, dict) else []
    by_index = {choice.get("index"): choice for choice in choices if isinstance(choice, dict)}
    case_results = []
    for index, case in enumerate(test_cases):
        choice = by_index.get(index, {})
        text = choice.get("text", "") if isinstance(choice, dict) else ""
        content_ok = isinstance(text, str) and bool(text.strip())
        check_ok, detail = _validate(case, text) if content_ok else (False, "empty output")
        case_results.append({
            **case, "index": index, "output": text,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "output_nonempty": content_ok, "check_passed": check_ok, "check_detail": detail,
        })
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    evidence = _evidence(log_text)
    all_outputs_present = len(choices) == 32 and all(item["output_nonempty"] for item in case_results)
    all_checks_passed = all(item["check_passed"] for item in case_results)
    all_finish_reasons_stop = _all_finish_reasons_stop(case_results, len(test_cases))
    all_paths_evidenced = all(item["passed"] for item in evidence.values())
    result = {
        "schema_version": 1,
        "kind": "qwen3_w8a8_accelerated_prompt_functional_validation",
        "started_at_utc": started_at,
        "request_elapsed_seconds": elapsed,
        "model": str(args.model.expanduser().resolve()),
        "runtime": {
            "python": args.python,
            "ascend_env": str(args.ascend_env.expanduser().resolve()),
            "pythonpath": [
                str(path.expanduser().resolve()) for path in args.pythonpath
            ],
        },
        "profile": {"device": args.device, "batch_size": 32, "max_tokens": args.max_tokens, "temperature": 0.0, "seed": args.seed},
        "server_command": shell_command,
        "response_usage": response.get("usage") if isinstance(response, dict) else None,
        "case_count": len(test_cases),
        "nonempty_count": sum(item["output_nonempty"] for item in case_results),
        "check_pass_count": sum(item["check_passed"] for item in case_results),
        "finish_reasons": {reason: sum(item["finish_reason"] == reason for item in case_results) for reason in sorted({str(item["finish_reason"]) for item in case_results})},
        "acceleration_evidence": evidence,
        "all_outputs_present": all_outputs_present,
        "all_checks_passed": all_checks_passed,
        "all_finish_reasons_stop": all_finish_reasons_stop,
        "all_acceleration_paths_evidenced": all_paths_evidenced,
        "passed": (
            all_outputs_present
            and all_checks_passed
            and all_finish_reasons_stop
            and all_paths_evidenced
        ),
        "cases": case_results,
        "artifacts": {"request": str(request_path.resolve()), "response": str(response_path.resolve()), "server_log": str(log_path.resolve())},
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "nonempty_count", "check_pass_count", "finish_reasons", "acceleration_evidence", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
