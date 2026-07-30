import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "gate_qwen3_attention_request_parallel.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "gate_qwen3_attention_request_parallel", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _args(**overrides):
    values = {
        "python": "/env/bin/python",
        "model": Path("/models/qwen3-30b-a3b"),
        "host": "127.0.0.1",
        "port": 19058,
        "served_model_name": "qwen3-30b-a3b",
        "gpu_memory_utilization": 0.8,
        "additional_config": GATE.DEFAULT_ADDITIONAL_CONFIG,
        "device": "0,1",
        "env": [
            "CUSTOM_SWITCH=enabled",
            "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL=caller-value",
        ],
        "seed": 20260712,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _prompt(name, token_ids):
    return {
        "name": name,
        "length": len(token_ids),
        "token_ids": token_ids,
        "sha256_le_u32": "unused-in-unit-test",
    }


def _choice(index, prompt, generated_start=100):
    prompt_logprobs = [None]
    for token_id in prompt["token_ids"][1:]:
        prompt_logprobs.append(
            {
                str(token_id): {"logprob": -0.1, "rank": 1},
                str(token_id + 100): {"logprob": -1.0, "rank": 2},
                str(token_id + 200): {"logprob": -2.0, "rank": 3},
                str(token_id + 300): {"logprob": -3.0, "rank": 4},
                str(token_id + 400): {"logprob": -4.0, "rank": 5},
            }
        )
    return {
        "index": index,
        "text": f"generated-{prompt['name']}",
        "finish_reason": "length",
        "prompt_token_ids": prompt["token_ids"],
        "token_ids": list(
            range(generated_start, generated_start + GATE.GENERATED_TOKENS)
        ),
        "prompt_logprobs": prompt_logprobs,
        "logprobs": {"token_logprobs": [-0.2] * GATE.GENERATED_TOKENS},
    }


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _git_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "runtime-repo"
    runtime = repo / "runtime"
    runtime.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Unit Test")
    _git(repo, "config", "user.email", "unit-test@example.invalid")
    tracked = runtime / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    _git(repo, "add", "runtime/tracked.py")
    _git(repo, "commit", "-q", "-m", "initial")
    return runtime, tracked


def _single_response(prompt):
    return {
        "choices": [_choice(0, prompt)],
        "usage": {
            "prompt_tokens": prompt["length"],
            "completion_tokens": GATE.GENERATED_TOKENS,
        },
    }


def test_request_payload_batches_two_prompts_and_swaps_repeat_order():
    left = _prompt("left", [11, 12, 13])
    right = _prompt("right", [21, 22])
    pair = {"name": "pair", "left": left, "right": right}

    first_payload, first_order = GATE._request_payload(
        _args(), "request_parallel_graph", pair, 0
    )
    second_payload, second_order = GATE._request_payload(
        _args(), "request_parallel_graph", pair, 1
    )

    assert first_payload["prompt"] == [[11, 12, 13], [21, 22]]
    assert second_payload["prompt"] == [[21, 22], [11, 12, 13]]
    assert [prompt["name"] for prompt in first_order] == ["left", "right"]
    assert [prompt["name"] for prompt in second_order] == ["right", "left"]
    assert first_payload["request_id"].endswith("-r0")
    assert second_payload["request_id"].endswith("-r1")
    assert first_payload["max_tokens"] == first_payload["min_tokens"] == 16


def test_two_choice_response_is_validated_and_split_with_per_prompt_usage():
    left = _prompt("left", [11, 12, 13])
    right = _prompt("right", [21, 22])
    body = {
        "choices": [
            _choice(0, left, generated_start=100),
            _choice(1, right, generated_start=200),
        ],
        "usage": {
            "prompt_tokens": left["length"] + right["length"],
            "completion_tokens": 2 * GATE.GENERATED_TOKENS,
        },
    }

    validation, split = GATE._split_and_validate_response(body, [left, right])

    assert validation == {"valid": True, "errors": []}
    assert set(split) == {"left", "right"}
    assert split["left"]["choices"] == [body["choices"][0]]
    assert split["right"]["choices"] == [body["choices"][1]]
    assert split["left"]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 16,
    }
    assert split["right"]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 16,
    }

    bad_usage = copy.deepcopy(body)
    bad_usage["usage"]["completion_tokens"] -= 1
    invalid, _ = GATE._split_and_validate_response(bad_usage, [left, right])
    assert invalid["valid"] is False
    assert "usage.completion_tokens must equal 32" in invalid["errors"]


def test_four_modes_have_distinct_reference_characterization_and_graph_contracts():
    args = _args()
    environments = {
        mode: GATE._mode_environment(args, mode) for mode in GATE.MODES
    }
    commands = {mode: GATE._server_command(args, mode) for mode in GATE.MODES}

    assert {
        mode: (
            environment["MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL"],
            environment["MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH"],
        )
        for mode, environment in environments.items()
    } == {
        "standard_single_seq": ("0", "0"),
        "standard_batched_eager": ("0", "0"),
        "request_parallel_eager": ("1", "0"),
        "request_parallel_graph": ("1", "1"),
    }
    assert all(
        environment["CUSTOM_SWITCH"] == "enabled"
        for environment in environments.values()
    )

    for mode in (
        "standard_single_seq",
        "standard_batched_eager",
        "request_parallel_eager",
    ):
        assert "--enforce-eager" in commands[mode]
        assert "--compilation-config" not in commands[mode]

    assert {
        mode: command[command.index("--max-num-seqs") + 1]
        for mode, command in commands.items()
    } == {
        "standard_single_seq": "1",
        "standard_batched_eager": "2",
        "request_parallel_eager": "2",
        "request_parallel_graph": "2",
    }

    graph_command = commands["request_parallel_graph"]
    assert "--enforce-eager" not in graph_command
    config_index = graph_command.index("--compilation-config")
    assert json.loads(graph_command[config_index + 1]) == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [2],
    }


def test_runtime_environment_prepends_configured_pythonpath(tmp_path: Path):
    vllm_path = tmp_path / "vllm"
    ascend_path = tmp_path / "vllm-ascend"
    vllm_path.mkdir()
    ascend_path.mkdir()
    args = _args(pythonpath=[vllm_path, ascend_path])

    environment = GATE._runtime_environment(
        args, {"PYTHONPATH": "/caller/runtime", "CUSTOM": "enabled"}
    )

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(vllm_path.resolve()),
        str(ascend_path.resolve()),
        "/caller/runtime",
    ]
    assert environment["CUSTOM"] == "enabled"


def test_runtime_source_fingerprint_is_stable_and_detects_tracked_changes(tmp_path):
    runtime, tracked = _git_repo(tmp_path)

    clean = GATE._runtime_source_record(runtime)
    repeated = GATE._runtime_source_record(runtime)

    assert clean == repeated
    assert clean["pythonpath"] == str(runtime.resolve())
    assert clean["git_root"] == str(runtime.parent.resolve())
    assert clean["head_commit"] == _git(runtime, "rev-parse", "HEAD")
    assert clean["dirty"] is False
    assert clean["tracked_diff_bytes"] == 0
    assert len(clean["source_fingerprint_sha256"]) == 64

    tracked.write_text("VALUE = 2\n")
    changed = GATE._runtime_source_record(runtime)

    assert changed["dirty"] is True
    assert changed["tracked_diff_bytes"] > 0
    assert changed["source_fingerprint_sha256"] != clean["source_fingerprint_sha256"]


def test_runtime_source_fingerprint_covers_sorted_untracked_source(tmp_path):
    runtime, _ = _git_repo(tmp_path)
    clean = GATE._runtime_source_record(runtime)
    (runtime / "z_last.py").write_text("VALUE = 'z'\n")
    include_dir = runtime / "native"
    include_dir.mkdir()
    (include_dir / "a_first.cpp").write_text("int value = 1;\n")

    first = GATE._runtime_source_record(runtime)
    (runtime / "z_last.py").write_text("VALUE = 'changed'\n")
    second = GATE._runtime_source_record(runtime)

    assert first["dirty"] is True
    assert first["untracked_source_files"] == [
        "runtime/native/a_first.cpp",
        "runtime/z_last.py",
    ]
    assert first["source_fingerprint_sha256"] != clean["source_fingerprint_sha256"]
    assert second["source_fingerprint_sha256"] != first["source_fingerprint_sha256"]


def test_runtime_source_fingerprint_excludes_generated_artifacts(tmp_path):
    runtime, _ = _git_repo(tmp_path)
    baseline = GATE._runtime_source_record(runtime)
    artifacts = {
        "build/generated.py": "VALUE = 'build'\n",
        "build_candidate/generated.cpp": "int generated = 1;\n",
        "results/output.cpp": "int output = 1;\n",
        "generated/config.h": "#define GENERATED 1\n",
        "__pycache__/cache.py": "VALUE = 'cache'\n",
    }
    for relative_path, contents in artifacts.items():
        path = runtime.parent / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    with_artifacts = GATE._runtime_source_record(runtime)

    assert with_artifacts == baseline


def test_summary_exposes_resolved_runtime_provenance_and_keeps_mode_results():
    runtime_sources = [{
        "pythonpath": "/runtime/vllm",
        "git_root": "/runtime/vllm",
        "head_commit": "a" * 40,
        "dirty": False,
        "source_fingerprint_sha256": "b" * 64,
    }]
    protocol = {
        "model": "/models/qwen3",
        "python": "/env/bin/python3",
        "ascend_env": "/opt/ascend/set_env.sh",
        "pythonpath": ["/runtime/vllm"],
        "runtime_sources": runtime_sources,
        "device": "0,1",
        "started_at_utc": "2026-07-30T01:02:03Z",
    }
    results = {mode: {"status": "passed"} for mode in GATE.MODES}

    summary = GATE._build_summary(
        protocol,
        results,
        {"passed": True},
        completed_at_utc="2026-07-30T01:20:00Z",
    )

    for field in (
        "model",
        "python",
        "ascend_env",
        "pythonpath",
        "runtime_sources",
        "device",
        "started_at_utc",
    ):
        assert summary[field] == protocol[field]
    assert summary["completed_at_utc"] == "2026-07-30T01:20:00Z"
    assert summary["mode_results"] == results
    assert summary["passed"] is True
    assert summary["verdict"] == "PASS"


def test_strict_comparator_rejects_generated_token_and_text_mismatches():
    prompt = _prompt("left", [11, 12, 13])
    reference = _single_response(prompt)

    assert GATE._strict_compare(reference, copy.deepcopy(reference))["passed"] is True

    token_mismatch = copy.deepcopy(reference)
    token_mismatch["choices"][0]["token_ids"][0] += 1
    token_result = GATE._strict_compare(reference, token_mismatch)
    assert token_result["passed"] is False
    assert token_result["generated_ids_exact"] is False
    assert token_result["text_exact"] is True

    text_mismatch = copy.deepcopy(reference)
    text_mismatch["choices"][0]["text"] = "different text"
    text_result = GATE._strict_compare(reference, text_mismatch)
    assert text_result["passed"] is False
    assert text_result["generated_ids_exact"] is True
    assert text_result["text_exact"] is False

    logprob_mismatch = copy.deepcopy(reference)
    logprob_mismatch["choices"][0]["logprobs"]["token_logprobs"][0] += 1e-6
    logprob_result = GATE._strict_compare(reference, logprob_mismatch)
    assert logprob_result["passed"] is False
    assert logprob_result["generated_ids_exact"] is True
    assert logprob_result["generated_logprobs_exact"] is False


def test_standard_batched_instability_is_reported_but_does_not_fail_acceptance():
    prompt = _prompt("left", [11, 12, 13])
    stable = _single_response(prompt)
    unstable = copy.deepcopy(stable)
    unstable["choices"][0]["text"] = "packing-sensitive"
    responses = {
        "standard_single_seq": {"left": [stable, copy.deepcopy(stable)]},
        "standard_batched_eager": {"left": [stable, unstable]},
        "request_parallel_eager": {"left": [stable, copy.deepcopy(stable)]},
        "request_parallel_graph": {"left": [stable, copy.deepcopy(stable)]},
    }

    comparisons = GATE._build_comparisons([prompt], responses)

    assert comparisons["passed"] is True
    assert comparisons["acceptance"]["passed"] is True
    characterization = comparisons["characterization"]
    assert characterization["numerical_comparison_gates_verdict"] is False
    assert characterization["execution_gates_verdict"] is True
    assert characterization["repeat_stability"]["passed"] is False
    assert characterization["against_single_seq_reference"]["passed"] is False


def test_request_parallel_mismatch_fails_acceptance():
    prompt = _prompt("left", [11, 12, 13])
    stable = _single_response(prompt)
    mismatch = copy.deepcopy(stable)
    mismatch["choices"][0]["text"] = "incorrect-graph-output"
    responses = {
        "standard_single_seq": {"left": [stable, copy.deepcopy(stable)]},
        "standard_batched_eager": {"left": [stable, copy.deepcopy(stable)]},
        "request_parallel_eager": {"left": [stable, copy.deepcopy(stable)]},
        "request_parallel_graph": {"left": [mismatch, copy.deepcopy(mismatch)]},
    }

    comparisons = GATE._build_comparisons([prompt], responses)

    assert comparisons["passed"] is False
    assert comparisons["acceptance"]["passed"] is False
    assert comparisons["acceptance"]["graph_equivalence"]["passed"] is False


def test_activation_evidence_requires_the_mode_specific_aclgraph_marker(tmp_path):
    coalescing = "\n".join(
        ["Engine idle request coalescing collected 1 additional ADD"] * 4
    )
    eager_log = tmp_path / "eager.log"
    eager_log.write_text(
        "\n".join(
            [
                "Using experimental Qwen3 attention request parallelism "
                "(decode_aclgraph=False)",
                "Using experimental Qwen3 attention request parallelism "
                "(decode_aclgraph=False)",
                coalescing,
            ]
        )
    )
    graph_log = tmp_path / "graph.log"
    graph_log.write_text(
        eager_log.read_text()
        + "\nGraph capturing finished\nGraph capturing finished"
        + "\nReplaying aclgraph\nReplaying aclgraph"
    )

    eager = GATE._activation_evidence("request_parallel_eager", eager_log)
    wrong_graph = GATE._activation_evidence("request_parallel_graph", graph_log)

    assert eager["passed"] is True
    assert eager["request_parallel_configured_marker_count"] == 2
    assert wrong_graph["passed"] is False
    assert wrong_graph["request_parallel_marker_count"] == 2
    assert wrong_graph["request_parallel_configured_marker_count"] == 0
