from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "summarize_qwen3_strict_alternating_c32_20260723.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_qwen3_strict_alternating_c32_20260723", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SUMMARIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARIZER)


INPUT_TOKENS = 130_048
OUTPUT_TOKENS = 64 * 16
TOTAL_TOKENS = INPUT_TOKENS + OUTPUT_TOKENS
PASSING_DURATIONS = {
    (1, "fp16"): 10.0,
    (1, "w8a8"): 6.0,
    (2, "fp16"): 10.0,
    (2, "w8a8"): 6.0,
    (3, "fp16"): 10.0,
    (3, "w8a8"): 6.0,
}
ARITHMETIC_MEAN_FALSE_POSITIVE_DURATIONS = {
    (1, "fp16"): 10.0,
    (1, "w8a8"): 3.0,
    (2, "fp16"): 10.0,
    (2, "w8a8"): 10.0,
    (3, "fp16"): 10.0,
    (3, "w8a8"): 15.0,
}
RUN_DATES = {
    run: f"20260723-1200{index:02d}"
    for index, run in enumerate(SUMMARIZER.EXPECTED_SEQUENCE, start=1)
}
FP16_MODEL = "/models/qwen3-30b-a3b-fp16"
W8A8_MODEL = "/models/qwen3-30b-a3b-w8a8"
PYTHON = "/env/bin/python3"
DEVICE = SUMMARIZER.EXPECTED_PROFILE["device"]
RUNTIME_SOURCES = [
    {
        "pythonpath": "/runtime/vllm",
        "git_root": "/runtime/vllm",
        "head_commit": SUMMARIZER.PINNED_RUNTIME_HEADS["vllm"],
        "source_fingerprint_sha256": "a" * 64,
    },
    {
        "pythonpath": "/runtime/vllm-ascend",
        "git_root": "/runtime/vllm-ascend",
        "head_commit": SUMMARIZER.PINNED_RUNTIME_HEADS["vllm_ascend"],
        "source_fingerprint_sha256": "b" * 64,
    },
]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(root: Path, pair: int, mode: str) -> Path:
    return next((root / f"pair{pair}_{mode}").glob("*_summary.json"))


def _result_path(summary_path: Path) -> Path:
    summary = _read_json(summary_path)
    result_path = Path(summary["result_json"])
    if not result_path.is_absolute():
        result_path = summary_path.parent / result_path
    return result_path


def _arguments(mode: str) -> dict[str, Any]:
    arguments = copy.deepcopy(SUMMARIZER.EXPECTED_PROFILE)
    environment = {**SUMMARIZER.COMMON_ENV, **SUMMARIZER.MODE_ENV[mode]}
    arguments.update(
        {
            "mode": mode,
            "model": FP16_MODEL if mode == "fp16" else W8A8_MODEL,
            "python": PYTHON,
            "env": [f"{key}={value}" for key, value in environment.items()],
            "quality_prompt": [],
        }
    )
    return arguments


def _write_evidence(
    root: Path,
    durations: dict[tuple[int, str], float] | None = None,
) -> None:
    durations = durations or PASSING_DURATIONS
    for pair in (1, 2, 3):
        for mode in ("fp16", "w8a8"):
            directory = root / f"pair{pair}_{mode}"
            directory.mkdir(parents=True)
            duration = durations[(pair, mode)]
            throughput = TOTAL_TOKENS / duration
            result_path = directory / f"pair{pair}_{mode}_serve.json"
            summary_path = directory / f"pair{pair}_{mode}_summary.json"
            _write_json(
                result_path,
                {
                    "date": RUN_DATES[(pair, mode)],
                    "mode": mode,
                    "duration": duration,
                    "completed": 64,
                    "total_input_tokens": INPUT_TOKENS,
                    "total_output_tokens": OUTPUT_TOKENS,
                    "total_token_throughput": throughput,
                },
            )
            _write_json(
                summary_path,
                {
                    "mode": mode,
                    "model": FP16_MODEL if mode == "fp16" else W8A8_MODEL,
                    "python": PYTHON,
                    "device": DEVICE,
                    "runtime_sources": copy.deepcopy(RUNTIME_SOURCES),
                    "seed": SUMMARIZER.EXPECTED_PROFILE["seed"],
                    "arguments": _arguments(mode),
                    "duration": duration,
                    "completed": 64,
                    "failed": 0,
                    "returncode": 0,
                    "input_tokens": INPUT_TOKENS,
                    "output_tokens": OUTPUT_TOKENS,
                    "total_token_throughput": throughput,
                    "result_json": result_path.name,
                },
            )


def _semantic_summary() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "qwen3_attention_request_parallel_semantics_gate_summary",
        "passed": True,
        "verdict": "PASS",
        "execution_passed": True,
        "model": W8A8_MODEL,
        "python": PYTHON,
        "device": DEVICE,
        "runtime_sources": copy.deepcopy(RUNTIME_SOURCES),
        "comparisons": {
            "acceptance": {
                "passed": True,
                "reference_mode": "standard_single_seq",
                "algorithm_reference": {
                    "request_parallel_eager": {"passed": True},
                    "request_parallel_graph": {"passed": True},
                },
                "graph_equivalence": {"passed": True},
                "repeat_stability": {
                    "standard_single_seq": {"passed": True},
                    "request_parallel_eager": {"passed": True},
                    "request_parallel_graph": {"passed": True},
                },
            }
        },
    }


def _write_semantic_summary(path: Path, value: dict[str, Any] | None = None) -> None:
    _write_json(path, value or _semantic_summary())


def _set_nested(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    output: Path,
    semantic_summary: Path,
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(root),
            "--output",
            str(output),
            "--semantic-summary",
            str(semantic_summary),
        ],
    )
    return SUMMARIZER.main(), _read_json(output)


def test_arithmetic_mean_1_667x_false_positive_fails_strict_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    output_path = tmp_path / "summary.json"
    _write_evidence(root, ARITHMETIC_MEAN_FALSE_POSITIVE_DURATIONS)
    _write_semantic_summary(semantic_path)

    returncode, result = _run_main(
        monkeypatch, root, output_path, semantic_path
    )

    assert returncode == 1
    assert result["schema_version"] == 2
    assert result["kind"] == "qwen3_strict_alternating_c32_summary"
    assert result["profile"] == SUMMARIZER.EXPECTED_PROFILE
    assert len(result["profile_fingerprint_sha256"]) == 64
    assert result["fp16_mean_total_token_throughput"] == pytest.approx(
        TOTAL_TOKENS / 10
    )
    assert result["w8a8_mean_total_token_throughput"] == pytest.approx(
        TOTAL_TOKENS / 6
    )
    assert result["arithmetic_mean_throughput_ratio_context"] == pytest.approx(
        5 / 3
    )
    assert result["fp16_aggregated_total_token_throughput"] == pytest.approx(
        3 * TOTAL_TOKENS / 30
    )
    assert result["w8a8_aggregated_total_token_throughput"] == pytest.approx(
        3 * TOTAL_TOKENS / 28
    )
    assert result["throughput_ratio_primary"] == pytest.approx(30 / 28)
    assert result["aggregated_throughput_at_least_1_5x"] is False
    assert result["all_pair_speedups_at_least_1_5x"] is False
    assert [pair["speedup"] for pair in result["pairs"]] == pytest.approx(
        [10 / 3, 1, 2 / 3]
    )
    assert result["wall_time_ratio_context"] == pytest.approx(30 / 28)
    assert result["wall_time_ratio_context"] < 1.5
    assert result["fixed_prompt_throughput_ratio"] == pytest.approx(5 / 3)
    assert result["execution_order"] == {
        "expected": [list(run) for run in SUMMARIZER.EXPECTED_SEQUENCE],
        "observed": [list(run) for run in SUMMARIZER.EXPECTED_SEQUENCE],
        "valid": True,
    }
    assert result["token_counts_equal"] is True
    assert len(result["raw_rows"]) == 6
    assert all(pair["valid"] for pair in result["pairs"])
    assert result["semantic_gate"]["provided"] is True
    assert result["semantic_gate"]["passed"] is True
    assert all(result["semantic_gate"]["checks"].values())
    assert result["all_six_measurements_valid"] is True
    assert result["all_six_valid"] is True
    assert result["validation_errors"] == []
    assert result["strict_1_5x_pass"] is False


def test_boolean_only_legacy_semantic_gate_fails_provenance(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    semantic = _semantic_summary()
    for field in ("model", "python", "device", "runtime_sources"):
        semantic.pop(field)
    _write_evidence(root)
    _write_semantic_summary(semantic_path, semantic)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["semantic_gate"]["passed"] is False
    assert result["semantic_gate"]["checks"]["runtime_sources_complete"] is False
    assert result["provenance_binding"]["passed"] is False
    assert any(
        "runtime_sources must be a non-empty list" in error
        for error in result["validation_errors"]
    )
    assert result["strict_1_5x_pass"] is False


def test_semantic_model_must_match_w8a8_performance_model(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    semantic = _semantic_summary()
    semantic["model"] = "/models/forged-old-w8a8"
    _write_evidence(root)
    _write_semantic_summary(semantic_path, semantic)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["semantic_gate"]["passed"] is True
    assert result["provenance_binding"]["passed"] is False
    assert any(
        "semantic/performance provenance mismatch: model=" in error
        for error in result["validation_errors"]
    )
    assert result["strict_1_5x_pass"] is False


def test_semantic_runtime_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    semantic = _semantic_summary()
    semantic["runtime_sources"][1]["source_fingerprint_sha256"] = "c" * 64
    _write_evidence(root)
    _write_semantic_summary(semantic_path, semantic)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["semantic_gate"]["passed"] is True
    assert result["provenance_binding"]["passed"] is False
    assert any(
        "runtime fingerprint does not match semantic gate" in error
        for error in result["validation_errors"]
    )
    assert result["strict_1_5x_pass"] is False


def test_w8a8_performance_runs_must_share_model(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 2, "w8a8")
    summary = _read_json(summary_path)
    summary["model"] = "/models/forged-pair2-w8a8"
    summary["arguments"]["model"] = summary["model"]
    _write_json(summary_path, summary)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["provenance_binding"]["passed"] is False
    assert any(
        "w8a8 runs do not share one model" in error
        for error in result["validation_errors"]
    )
    assert result["strict_1_5x_pass"] is False


def test_all_performance_runs_must_share_python(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 3, "fp16")
    summary = _read_json(summary_path)
    summary["python"] = "/env/bin/other-python"
    summary["arguments"]["python"] = summary["python"]
    _write_json(summary_path, summary)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["provenance_binding"]["passed"] is False
    assert any(
        "six runs do not share one Python" in error
        for error in result["validation_errors"]
    )
    assert result["strict_1_5x_pass"] is False


def test_schema_v2_passes_aggregated_and_all_pair_speedup_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    output_path = tmp_path / "summary.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)

    returncode, result = _run_main(
        monkeypatch, root, output_path, semantic_path
    )

    assert returncode == 0
    assert result["throughput_ratio_primary"] == pytest.approx(5 / 3)
    assert result["arithmetic_mean_throughput_ratio_context"] == pytest.approx(
        5 / 3
    )
    assert result["aggregated_throughput_at_least_1_5x"] is True
    assert result["all_pair_speedups_at_least_1_5x"] is True
    assert all(
        pair["speedup_at_least_1_5x"] for pair in result["pairs"]
    )
    assert all(
        row["total_token_throughput_consistent"]
        for row in result["raw_rows"]
    )
    assert result["all_six_valid"] is True
    assert result["semantic_gate"]["passed"] is True
    assert result["validation_errors"] == []
    assert result["strict_1_5x_pass"] is True


def test_missing_semantic_summary_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_evidence(root)

    result = SUMMARIZER.summarize(root, None)

    assert result["all_six_measurements_valid"] is True
    assert result["all_six_valid"] is False
    assert result["semantic_gate"]["provided"] is False
    assert result["semantic_gate"]["passed"] is False
    assert "semantic gate summary is required" in result["validation_errors"]
    assert result["strict_1_5x_pass"] is False


def test_aggregated_ratio_above_threshold_does_not_override_pair_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durations = {
        (1, "fp16"): 3.0,
        (1, "w8a8"): 5.0,
        (2, "fp16"): 10.0,
        (2, "w8a8"): 5.0,
        (3, "fp16"): 15.0,
        (3, "w8a8"): 5.0,
    }
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    output_path = tmp_path / "summary.json"
    _write_evidence(root, durations)
    _write_semantic_summary(semantic_path)

    returncode, result = _run_main(
        monkeypatch, root, output_path, semantic_path
    )

    assert returncode == 1
    assert result["all_six_valid"] is True
    assert result["semantic_gate"]["passed"] is True
    assert result["validation_errors"] == []
    assert result["throughput_ratio_primary"] == pytest.approx(28 / 15)
    assert result["arithmetic_mean_throughput_ratio_context"] == pytest.approx(
        6 / 5
    )
    assert result["wall_time_ratio_context"] == pytest.approx(28 / 15)
    assert result["wall_time_ratio_context"] > 1.5
    assert result["aggregated_throughput_at_least_1_5x"] is True
    assert result["all_pair_speedups_at_least_1_5x"] is False
    assert result["strict_1_5x_pass"] is False


@pytest.mark.parametrize(
    ("path", "replacement", "failed_check"),
    [
        (("kind",), "wrong_summary_kind", "summary_kind"),
        (("schema_version",), 1, "schema_version"),
        (("passed",), False, "summary_passed"),
        (("verdict",), "FAIL", "verdict_pass"),
        (("execution_passed",), False, "execution_passed"),
        (("comparisons", "acceptance", "passed"), False, "acceptance_passed"),
        (
            ("comparisons", "acceptance", "reference_mode"),
            "standard_batched_eager",
            "reference_is_standard_single_seq",
        ),
        (
            (
                "comparisons",
                "acceptance",
                "algorithm_reference",
                "request_parallel_eager",
                "passed",
            ),
            False,
            "request_parallel_eager_exact",
        ),
        (
            (
                "comparisons",
                "acceptance",
                "algorithm_reference",
                "request_parallel_graph",
                "passed",
            ),
            False,
            "request_parallel_graph_exact",
        ),
        (
            ("comparisons", "acceptance", "graph_equivalence", "passed"),
            False,
            "graph_equivalence_exact",
        ),
        *[
            (
                (
                    "comparisons",
                    "acceptance",
                    "repeat_stability",
                    mode,
                    "passed",
                ),
                False,
                "repeat_stability_exact",
            )
            for mode in (
                "standard_single_seq",
                "request_parallel_eager",
                "request_parallel_graph",
            )
        ],
    ],
)
def test_semantic_gate_rejects_each_required_check(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    failed_check: str,
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    semantic = _semantic_summary()
    _set_nested(semantic, path, replacement)
    _write_evidence(root)
    _write_semantic_summary(semantic_path, semantic)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["all_six_valid"] is True
    assert result["semantic_gate"]["passed"] is False
    assert result["semantic_gate"]["checks"][failed_check] is False
    assert f"semantic gate check failed: {failed_check}" in result["validation_errors"]
    assert result["strict_1_5x_pass"] is False


@pytest.mark.parametrize(
    ("document", "field", "replacement", "error_fragment"),
    [
        ("summary", "completed", 63, "completed=63"),
        ("summary", "failed", 1, "failed=1"),
        ("summary", "returncode", 1, "returncode=1"),
        ("summary", "mode", "w8a8", "summary mode='w8a8'"),
        ("arguments", "mode", "w8a8", "arguments.mode='w8a8'"),
        ("result", "mode", "w8a8", "result mode='w8a8'"),
        ("summary", "device", "2,3", "summary device='2,3'"),
        ("summary", "seed", 7, "summary seed=7"),
        ("arguments", "device", "2,3", "device='2,3'"),
        ("arguments", "seed", 7, "seed=7"),
        ("arguments", "tensor_parallel_size", 1, "tensor_parallel_size=1"),
        ("arguments", "enable_expert_parallel", False, "enable_expert_parallel=False"),
        (
            "arguments",
            "compilation_config",
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [16],
            },
            "compilation_config=",
        ),
        ("result", "completed", 63, "result completed=63"),
        (
            "result",
            "total_token_throughput",
            1.0,
            "result total_token_throughput=1.0",
        ),
    ],
)
def test_invalid_run_status_mode_or_core_profile_fails(
    tmp_path: Path,
    document: str,
    field: str,
    replacement: Any,
    error_fragment: str,
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 1, "fp16")
    target_path = summary_path if document != "result" else _result_path(summary_path)
    target = _read_json(target_path)
    if document == "arguments":
        target["arguments"][field] = replacement
    else:
        target[field] = replacement
    _write_json(target_path, target)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["all_six_valid"] is False
    assert any(error_fragment in error for error in result["validation_errors"])
    assert result["strict_1_5x_pass"] is False


def test_self_consistent_reported_throughput_must_match_tokens_and_duration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 1, "fp16")
    result_path = _result_path(summary_path)
    summary = _read_json(summary_path)
    raw_result = _read_json(result_path)
    forged_throughput = summary["total_token_throughput"] * (5 / 3)
    summary["total_token_throughput"] = forged_throughput
    raw_result["total_token_throughput"] = forged_throughput
    _write_json(summary_path, summary)
    _write_json(result_path, raw_result)

    result = SUMMARIZER.summarize(root, semantic_path)

    forged_row = next(
        row
        for row in result["raw_rows"]
        if row["pair"] == 1 and row["mode"] == "fp16"
    )
    assert forged_row["total_token_throughput"] == pytest.approx(
        forged_throughput
    )
    assert forged_row["derived_total_token_throughput"] == pytest.approx(
        TOTAL_TOKENS / PASSING_DURATIONS[(1, "fp16")]
    )
    assert forged_row["total_token_throughput_consistent"] is False
    assert any(
        "total_token_throughput=" in error
        and "does not match (input_tokens + output_tokens) / duration=" in error
        for error in result["validation_errors"]
    )
    assert result["all_six_valid"] is False
    assert result["strict_1_5x_pass"] is False


def test_reported_throughput_allows_normal_float_rounding(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 1, "fp16")
    result_path = _result_path(summary_path)
    summary = _read_json(summary_path)
    raw_result = _read_json(result_path)
    rounded_throughput = round(summary["total_token_throughput"], 3)
    summary["total_token_throughput"] = rounded_throughput
    raw_result["total_token_throughput"] = rounded_throughput
    _write_json(summary_path, summary)
    _write_json(result_path, raw_result)

    result = SUMMARIZER.summarize(root, semantic_path)

    rounded_row = next(
        row
        for row in result["raw_rows"]
        if row["pair"] == 1 and row["mode"] == "fp16"
    )
    assert rounded_row["total_token_throughput_consistent"] is True
    assert result["validation_errors"] == []
    assert result["strict_1_5x_pass"] is True


def test_w8a8_request_parallel_graph_environment_is_required(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 1, "w8a8")
    summary = _read_json(summary_path)
    summary["arguments"]["env"] = [
        (
            "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH=0"
            if item == "MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH=1"
            else item
        )
        for item in summary["arguments"]["env"]
    ]
    _write_json(summary_path, summary)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["all_six_valid"] is False
    assert any(
        "env MINDPIPE_QWEN3_ATTENTION_REQUEST_PARALLEL_ACLGRAPH='0'" in error
        for error in result["validation_errors"]
    )
    assert result["strict_1_5x_pass"] is False


@pytest.mark.parametrize(
    ("summary_field", "result_field", "replacement"),
    [
        ("input_tokens", "total_input_tokens", INPUT_TOKENS + 1),
        ("output_tokens", "total_output_tokens", OUTPUT_TOKENS + 1),
    ],
)
def test_cross_run_token_mismatch_fails(
    tmp_path: Path,
    summary_field: str,
    result_field: str,
    replacement: int,
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    summary_path = _summary_path(root, 2, "w8a8")
    result_path = _result_path(summary_path)
    summary = _read_json(summary_path)
    raw_result = _read_json(result_path)
    summary[summary_field] = replacement
    raw_result[result_field] = replacement
    _write_json(summary_path, summary)
    _write_json(result_path, raw_result)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["token_counts_equal"] is False
    assert result["all_six_valid"] is False
    assert any("token counts differ" in error for error in result["validation_errors"])
    assert result["strict_1_5x_pass"] is False


def test_non_alternating_result_timestamps_fail_execution_order(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    fp16_result_path = _result_path(_summary_path(root, 2, "fp16"))
    w8a8_result_path = _result_path(_summary_path(root, 2, "w8a8"))
    fp16_result = _read_json(fp16_result_path)
    w8a8_result = _read_json(w8a8_result_path)
    fp16_result["date"], w8a8_result["date"] = (
        w8a8_result["date"],
        fp16_result["date"],
    )
    _write_json(fp16_result_path, fp16_result)
    _write_json(w8a8_result_path, w8a8_result)

    result = SUMMARIZER.summarize(root, semantic_path)

    assert result["execution_order"]["valid"] is False
    assert result["all_six_valid"] is False
    assert any("execution order=" in error for error in result["validation_errors"])
    assert result["strict_1_5x_pass"] is False


def test_cli_rejects_missing_result_json_with_schema_v2_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    semantic_path = tmp_path / "semantic.json"
    output_path = tmp_path / "summary.json"
    _write_evidence(root)
    _write_semantic_summary(semantic_path)
    _result_path(_summary_path(root, 3, "w8a8")).unlink()

    returncode, result = _run_main(
        monkeypatch, root, output_path, semantic_path
    )

    assert returncode == 1
    assert result["schema_version"] == 2
    assert result["all_six_valid"] is False
    assert result["strict_1_5x_pass"] is False
    assert any("result_json does not exist" in error for error in result["validation_errors"])
