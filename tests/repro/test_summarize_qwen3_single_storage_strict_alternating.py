from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "summarize_qwen3_single_storage_strict_alternating.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_qwen3_single_storage_strict_alternating", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


FP16_MODEL = Path("/models/qwen3-fp16")
W8A8_MODEL = Path("/models/qwen3-w8a8")


def _request_bodies() -> list[dict[str, object]]:
    return [
        {
            "model": "qwen3-30b-a3b",
            "prompt": f"fixed prompt {index}",
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 20260712,
        }
        for index in range(64)
    ]


def _phase_shape() -> dict[str, object]:
    return {
        "num_prompts": 64,
        "http_completed": 64,
        "completed": 64,
        "failed": 0,
        "usage_failed": 0,
        "input_tokens": 131072,
        "output_tokens": 1024,
        "prompt_tokens": 131072,
        "completion_tokens": 1024,
        "total_tokens": 132096,
        "prompt_token_vector": [2048] * 64,
        "completion_token_vector": [16] * 64,
    }


def _write_request_artifact(
    path: Path, requests: list[dict[str, object]]
) -> None:
    rows = [
        {"request_id": index, "request_body": body}
        for index, body in enumerate(requests)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_response_artifact(path: Path, requests: list[dict[str, object]]) -> None:
    rows = [
        {"request_id": index, "line_number": index + 1, "status": 200}
        for index, _ in enumerate(requests)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_quality(path: Path, mode: str) -> None:
    value = {
        "model": "qwen3-30b-a3b",
        "seed": 20260712,
        "completed": 2,
        "failed": 0,
        "checks": [
            {
                "index": 0,
                "elapsed_seconds": 0.2,
                "text": f"{mode}-answer-0",
            },
            {
                "index": 1,
                "elapsed_seconds": 0.3,
                "text": f"{mode}-answer-1",
            },
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _environment(mode: str) -> list[str]:
    values = {
        "PYTHONPATH": "/runtime",
        "LD_LIBRARY_PATH": "/runtime/lib",
        "MINDPIPE_QWEN3_MOE_SINGLE_STORAGE_AUDIT": "0",
        **{name: "0" for name in SUMMARY.REPLICATION_SWITCHES},
        **{name: "0" for name in SUMMARY.DISABLED_EXPERIMENTS},
        **(
            SUMMARY.W8A8_SWITCHES
            if mode == "w8a8"
            else SUMMARY.FP16_SWITCHES
        ),
        "VLLM_DISABLE_COMPILE_CACHE": "1" if mode == "w8a8" else "0",
    }
    return [f"{name}={value}" for name, value in values.items()]


def _server_log(mode: str) -> str:
    if mode == "fp16":
        return ""
    markers = (
        "Using the vLLM Ascend Quantization now!",
        "Sequence-parallel fast path hit",
        "MindPipe Qwen3 quantized QKV gather fast path hit",
        "MindPipe Qwen3 fast RoPE path hit",
        "MindPipe Qwen3 quantized EP2 finalize path hit",
    )
    return "".join(f"rank {rank}: {marker}\n" for marker in markers for rank in range(2))


def _storage_audit(path: Path) -> None:
    workers = []
    for rank in range(2):
        workers.append({
            "local_global_expert_ids": list(
                range(rank * 64, (rank + 1) * 64)
            ),
            "storage_summary": {
                "main_weight_unique_bytes": 14_495_514_624,
                "unique_main_weight_storage_count": 96,
                "aliased_main_weights": {},
            },
            "load_audit": {
                "duplicate_source_count": 0,
                "loaded_count": 27_648,
                "skipped_nonlocal_count": 27_648,
            },
        })
    path.write_text(
        json.dumps({
            "passed": True,
            "worker_count": 2,
            "workers": workers,
        }),
        encoding="utf-8",
    )


def _profile_audits(qkv_path: Path, mechanism_path: Path) -> None:
    qkv_path.write_text(
        json.dumps({
            "kind": "qwen3_sp_quantized_qkv_profile_audit",
            "passed": True,
            "ranks": [
                {
                    "rank": rank,
                    "dynamic_quant_call_delta": 0,
                    "quantized_qkv_gather_calls": 282,
                    "checks": {"actual_profile_events": True},
                }
                for rank in range(2)
            ],
        }),
        encoding="utf-8",
    )
    mechanism_path.write_text(
        json.dumps({
            "kind": "qwen3_single_storage_mechanism_audit",
            "passed": True,
            "mechanisms": {
                "quantized_ep2_finalize": {
                    "ranks": [
                        {
                            "rank": rank,
                            "quantized_finalize_pairs": [{
                                "token_count": 32768,
                                "fp32_scale_all_reduce_calls": 96,
                                "int8_payload_all_reduce_calls": 96,
                            }],
                            "checks": {"actual_profile_events": True},
                        }
                        for rank in range(2)
                    ]
                },
                "fast_rope": {
                    "ranks": [
                        {
                            "rank": rank,
                            "fast_rope_calls": 240,
                            "checks": {"actual_profile_events": True},
                        }
                        for rank in range(2)
                    ]
                },
            },
        }),
        encoding="utf-8",
    )


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "runs"
    request_file = tmp_path / "requests.jsonl"
    storage_audit = tmp_path / "storage.json"
    qkv_audit = tmp_path / "qkv_profile_audit.json"
    mechanism_audit = tmp_path / "mechanism_audit.json"
    requests = _request_bodies()
    request_file.write_text(
        "".join(json.dumps(body) + "\n" for body in requests),
        encoding="utf-8",
    )
    _storage_audit(storage_audit)
    _profile_audits(qkv_audit, mechanism_audit)

    for sequence_index, (pair, mode) in enumerate(SUMMARY.SEQUENCE):
        directory = root / f"pair{pair}_{mode}"
        directory.mkdir(parents=True)
        prefix = directory / f"pair{pair}_{mode}"
        server_log = prefix.with_name(prefix.name + "_server.log")
        formal_requests = prefix.with_name(prefix.name + "_requests.jsonl")
        formal_responses = prefix.with_name(prefix.name + "_responses.jsonl")
        warmup_responses = prefix.with_name(
            prefix.name + "_warmup_responses.jsonl"
        )
        quality = prefix.with_name(prefix.name + "_quality.json")
        server_log.write_text(_server_log(mode), encoding="utf-8")
        _write_request_artifact(formal_requests, requests)
        _write_response_artifact(formal_responses, requests)
        _write_response_artifact(warmup_responses, requests)
        _write_quality(quality, mode)

        model = W8A8_MODEL if mode == "w8a8" else FP16_MODEL
        duration = 6.5 if mode == "w8a8" else 10.0
        arguments = {
            **SUMMARY.EXPECTED_ARGUMENTS,
            "python": "/usr/bin/python",
            "mode": mode,
            "model": str(model),
            "port": 19700 + sequence_index,
            "tag": f"pair{pair}_{mode}",
            "output_dir": str(directory),
            "request_file": str(request_file),
            "env": _environment(mode),
        }
        value = {
            **_phase_shape(),
            "arguments": arguments,
            "env_overrides": _environment(mode),
            "acceptance_sequence_index": sequence_index,
            "returncode": 0,
            "status": "completed",
            "diagnostic_only": False,
            "issues": [],
            "teardown_complete": True,
            "post_benchmark_health": True,
            "teardown_evidence": {
                "process_group_gone": True,
                "port_released": True,
            },
            "aggregation": {
                "basis": "single_global_wall_clock",
                "endpoint_local_throughputs_summed": False,
            },
            "quality_completed": 2,
            "quality_failed": 0,
            "engine_dtype": "torch.float16",
            "model": str(model),
            "weights_memory_gb_vector": (
                [14.6194, 14.6194]
                if mode == "w8a8"
                else [28.4573, 28.4573]
            ),
            "weights_memory_gb": 14.6194 if mode == "w8a8" else 28.4573,
            "engine_quantization": "ascend" if mode == "w8a8" else None,
            "ascend_quantization_log": mode == "w8a8",
            "duration": duration,
            "total_token_throughput": 132_096 / duration,
            "server_log": str(server_log),
            "requests_jsonl": str(formal_requests),
            "responses_jsonl": str(formal_responses),
            "quality_result_json": str(quality),
            "warmup": {
                "summary": _phase_shape(),
                "responses_jsonl": str(warmup_responses),
            },
        }
        (directory / f"pair{pair}_{mode}_summary.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
    return root, request_file, storage_audit


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[int, dict[str, object], Path]:
    root, request_file, storage_audit = _build_fixture(tmp_path)
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        root=root,
        request_file=request_file,
        fp16_model=FP16_MODEL,
        w8a8_model=W8A8_MODEL,
        storage_audit=storage_audit,
        qkv_profile_audit=tmp_path / "qkv_profile_audit.json",
        mechanism_audit=tmp_path / "mechanism_audit.json",
        output=output,
    )
    monkeypatch.setattr(SUMMARY, "parse_args", lambda: args)
    code = SUMMARY.main()
    return code, json.loads(output.read_text(encoding="utf-8")), root


def _summary(root: Path, label: str) -> Path:
    paths = list((root / label).glob("*_summary.json"))
    assert len(paths) == 1
    return paths[0]


def _mutate_summary(root: Path, label: str, mutation) -> None:
    path = _summary(root, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_schema_v4_passes_without_sha_or_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, result, _ = _run(monkeypatch, tmp_path)

    assert code == 0
    assert result["schema_version"] == 4
    assert result["passed"] is True
    assert result["failures"] == []
    assert "sha256" not in json.dumps(result).lower()
    assert "provenance" not in json.dumps(result).lower()
    assert all(pair["speedup"] > 1.5 for pair in result["pairs"])


def test_quality_signature_ignores_latency_but_retains_output(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_quality(first, "same")
    value = json.loads(first.read_text(encoding="utf-8"))
    value["checks"][0]["elapsed_seconds"] = 99.0
    second.write_text(json.dumps(value), encoding="utf-8")
    assert SUMMARY.quality_signature(first) == SUMMARY.quality_signature(second)

    value["checks"][0]["text"] = "changed"
    second.write_text(json.dumps(value), encoding="utf-8")
    assert SUMMARY.quality_signature(first) != SUMMARY.quality_signature(second)


def test_direct_request_body_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _, root = _run(monkeypatch, tmp_path)
    assert code == 0
    summary = json.loads(_summary(root, "pair1_fp16").read_text(encoding="utf-8"))
    artifact = Path(summary["requests_jsonl"])
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    rows[0]["request_body"]["prompt"] = "changed"
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "rerun.json"
    monkeypatch.setattr(
        SUMMARY,
        "parse_args",
        lambda: argparse.Namespace(
            root=root,
            request_file=tmp_path / "requests.jsonl",
            fp16_model=FP16_MODEL,
            w8a8_model=W8A8_MODEL,
            storage_audit=tmp_path / "storage.json",
            qkv_profile_audit=tmp_path / "qkv_profile_audit.json",
            mechanism_audit=tmp_path / "mechanism_audit.json",
            output=output,
        ),
    )
    assert SUMMARY.main() == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert any("formal_requests_matches_request_file" in item for item in result["failures"])


def test_requires_all_two_rank_mechanism_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _, root = _run(monkeypatch, tmp_path)
    assert code == 0
    qkv_path = tmp_path / "qkv_profile_audit.json"
    qkv = json.loads(qkv_path.read_text(encoding="utf-8"))
    qkv["ranks"][0]["quantized_qkv_gather_calls"] = 0
    qkv_path.write_text(json.dumps(qkv), encoding="utf-8")
    mechanism_path = tmp_path / "mechanism_audit.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    mechanism["mechanisms"]["quantized_ep2_finalize"]["ranks"][0][
        "quantized_finalize_pairs"
    ] = []
    mechanism["mechanisms"]["fast_rope"]["ranks"][0]["fast_rope_calls"] = 0
    mechanism_path.write_text(json.dumps(mechanism), encoding="utf-8")
    output = tmp_path / "mechanism_result.json"
    monkeypatch.setattr(
        SUMMARY,
        "parse_args",
        lambda: argparse.Namespace(
            root=root,
            request_file=tmp_path / "requests.jsonl",
            fp16_model=FP16_MODEL,
            w8a8_model=W8A8_MODEL,
            storage_audit=tmp_path / "storage.json",
            qkv_profile_audit=qkv_path,
            mechanism_audit=mechanism_path,
            output=output,
        ),
    )
    assert SUMMARY.main() == 1
    failures = "\n".join(json.loads(output.read_text())["failures"])
    assert "QKV profiler audit" in failures
    assert "actual_hits" in failures
    assert "quantized_ep2_actual_hits" in failures
    assert "fast_rope_actual_hits" in failures


def test_replication_switch_must_be_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _, root = _run(monkeypatch, tmp_path)
    assert code == 0

    def enable_replication(value: dict[str, object]) -> None:
        values = value["env_overrides"]
        values[:] = [
            "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS=1"
            if item.startswith("MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS=")
            else item
            for item in values
        ]

    _mutate_summary(root, "pair1_w8a8", enable_replication)
    output = tmp_path / "replication_result.json"
    monkeypatch.setattr(
        SUMMARY,
        "parse_args",
        lambda: argparse.Namespace(
            root=root,
            request_file=tmp_path / "requests.jsonl",
            fp16_model=FP16_MODEL,
            w8a8_model=W8A8_MODEL,
            storage_audit=tmp_path / "storage.json",
            qkv_profile_audit=tmp_path / "qkv_profile_audit.json",
            mechanism_audit=tmp_path / "mechanism_audit.json",
            output=output,
        ),
    )
    assert SUMMARY.main() == 1
    failures = "\n".join(json.loads(output.read_text())["failures"])
    assert "replication_disabled" in failures


def test_unrounded_pair_and_aggregate_thresholds_are_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _, root = _run(monkeypatch, tmp_path)
    assert code == 0
    for pair in (1, 2, 3):
        _mutate_summary(
            root,
            f"pair{pair}_w8a8",
            lambda value: value.update({
                "duration": 10.0 / 1.499999999,
                "total_token_throughput": 132_096 / (10.0 / 1.499999999),
            }),
        )
    output = tmp_path / "threshold_result.json"
    monkeypatch.setattr(
        SUMMARY,
        "parse_args",
        lambda: argparse.Namespace(
            root=root,
            request_file=tmp_path / "requests.jsonl",
            fp16_model=FP16_MODEL,
            w8a8_model=W8A8_MODEL,
            storage_audit=tmp_path / "storage.json",
            qkv_profile_audit=tmp_path / "qkv_profile_audit.json",
            mechanism_audit=tmp_path / "mechanism_audit.json",
            output=output,
        ),
    )
    assert SUMMARY.main() == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert all(pair["passed"] is False for pair in result["pairs"])
    assert result["aggregate"]["passed"] is False


def test_storage_audit_requires_no_duplicate_checkpoint_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _, _ = _run(monkeypatch, tmp_path)
    assert code == 0
    storage_path = tmp_path / "storage.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage["workers"][0]["load_audit"]["duplicate_source_count"] = 1
    storage_path.write_text(json.dumps(storage), encoding="utf-8")
    output = tmp_path / "storage_result.json"
    root = tmp_path / "runs"
    monkeypatch.setattr(
        SUMMARY,
        "parse_args",
        lambda: argparse.Namespace(
            root=root,
            request_file=tmp_path / "requests.jsonl",
            fp16_model=FP16_MODEL,
            w8a8_model=W8A8_MODEL,
            storage_audit=storage_path,
            qkv_profile_audit=tmp_path / "qkv_profile_audit.json",
            mechanism_audit=tmp_path / "mechanism_audit.json",
            output=output,
        ),
    )
    assert SUMMARY.main() == 1
    failures = "\n".join(json.loads(output.read_text())["failures"])
    assert "single_load" in failures
