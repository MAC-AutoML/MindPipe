import argparse
import importlib.util
import shlex
import signal
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "benchmark_vllm_online_serving.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_vllm_online_serving", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_append_compilation_config_to_serve_command():
    command = ["vllm", "serve", "model"]

    BENCHMARK._append_json_object_arg(
        command,
        '{"cudagraph_capture_sizes": [32, 16, 1], "level": 3}',
        "--compilation_config",
        "--compilation-config",
    )

    assert command[-2:] == [
        "--compilation-config",
        '{"cudagraph_capture_sizes":[32,16,1],"level":3}',
    ]


def test_append_json_object_arg_rejects_non_object():
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        BENCHMARK._append_json_object_arg(
            [],
            "[1, 2]",
            "--compilation_config",
            "--compilation-config",
        )


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        (False, ["vllm", "bench", "serve"]),
        (True, ["vllm", "bench", "serve", "--profile"]),
    ],
)
def test_append_profile_arg(enabled, expected):
    command = ["vllm", "bench", "serve"]

    BENCHMARK._append_profile_arg(command, enabled)

    assert command == expected


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        (False, ["vllm", "bench", "serve"]),
        (
            True,
            [
                "vllm",
                "bench",
                "serve",
                "--ready-check-timeout-sec",
                "0",
            ],
        ),
    ],
)
def test_append_skip_initial_test_arg(enabled, expected):
    command = ["vllm", "bench", "serve"]

    BENCHMARK._append_skip_initial_test_arg(command, enabled)

    assert command == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("fp16", ["vllm", "serve", "model"]),
        (
            "int4",
            ["vllm", "serve", "model", "--quantization", "ascend"],
        ),
        (
            "w8a8",
            ["vllm", "serve", "model", "--quantization", "ascend"],
        ),
    ],
)
def test_append_quantization_arg(mode, expected):
    command = ["vllm", "serve", "model"]

    BENCHMARK._append_quantization_arg(command, mode)

    assert command == expected


def test_validate_profile_dir_requires_absolute_empty_dir(tmp_path):
    profile_dir = tmp_path / "profile"

    result = BENCHMARK._validate_profile_dir(
        True,
        [f"VLLM_TORCH_PROFILER_DIR={profile_dir}"],
    )

    assert result == profile_dir


def test_validate_profile_dir_rejects_nonempty_dir(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "old-trace").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        BENCHMARK._validate_profile_dir(
            True,
            [f"VLLM_TORCH_PROFILER_DIR={profile_dir}"],
        )


def test_validate_profile_dir_requires_exactly_one_assignment():
    with pytest.raises(ValueError, match="exactly one"):
        BENCHMARK._validate_profile_dir(True, [])


def test_parse_env_accepts_equals_in_value():
    assert BENCHMARK._parse_env(["SETTING=left=right", "EMPTY="]) == {
        "SETTING": "left=right",
        "EMPTY": "",
    }


@pytest.mark.parametrize(
    "assignment",
    [
        "MISSING_SEPARATOR",
        "BAD-NAME=value",
        "NAME; touch /tmp/injected=value",
        "1INVALID=value",
        "NULL=value\0suffix",
    ],
)
def test_parse_env_rejects_unsafe_assignments(assignment):
    with pytest.raises(ValueError, match="Invalid --env assignment"):
        BENCHMARK._parse_env([assignment])


def test_parse_env_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="Duplicate --env key"):
        BENCHMARK._parse_env(["SETTING=first", "SETTING=second"])


def test_build_shell_prefix_quotes_file_and_values_and_deduplicates_aiv(tmp_path):
    ascend_env = tmp_path / "Ascend environment.sh"
    ascend_env.write_text("export TEST_ASCEND_ENV=1\n", encoding="utf-8")
    args = argparse.Namespace(
        ascend_env=ascend_env,
        device="0,1",
        aiv=True,
        env=[
            "HCCL_OP_EXPANSION_MODE=AIV",
            "SAFE_VALUE=words; touch /tmp/not-created",
        ],
    )

    prefix = BENCHMARK._build_shell_prefix(args)

    assert prefix.startswith(f"source {shlex.quote(str(ascend_env))} && ")
    assert prefix.count("HCCL_OP_EXPANSION_MODE") == 1
    assert (
        f"export SAFE_VALUE={shlex.quote('words; touch /tmp/not-created')}"
        in prefix
    )


def test_build_shell_prefix_rejects_device_override(tmp_path):
    ascend_env = tmp_path / "set_env.sh"
    ascend_env.write_text("", encoding="utf-8")
    args = argparse.Namespace(
        ascend_env=ascend_env,
        device="0",
        aiv=False,
        env=["ASCEND_RT_VISIBLE_DEVICES=1"],
    )

    with pytest.raises(ValueError, match="Set Ascend devices with --device"):
        BENCHMARK._build_shell_prefix(args)


def test_resolve_ascend_env_requires_absolute_path():
    with pytest.raises(ValueError, match="absolute path"):
        BENCHMARK._resolve_ascend_env(Path("relative/set_env.sh"))


def test_parse_args_requires_explicit_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--mode",
            "w8a8",
            "--output_dir",
            str(tmp_path),
            "--tag",
            "explicit-model-required",
            "--input_len",
            "16",
            "--output_len",
            "4",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        BENCHMARK.parse_args()

    assert exc_info.value.code == 2


def test_jsonable_arguments_serializes_path_values(tmp_path):
    args = argparse.Namespace(
        model=tmp_path / "model",
        ascend_env=tmp_path / "set_env.sh",
        env=["SETTING=value"],
    )

    result = BENCHMARK._jsonable_arguments(args)

    assert result == {
        "model": str(tmp_path / "model"),
        "ascend_env": str(tmp_path / "set_env.sh"),
        "env": ["SETTING=value"],
    }


def test_terminate_process_tree_cleans_group_after_leader_exits(monkeypatch):
    class ExitedLeader:
        pid = 12345

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    group_states = iter([True, False, False])
    sent = []
    monkeypatch.setattr(
        BENCHMARK, "_group_exists", lambda pgid: next(group_states)
    )
    monkeypatch.setattr(BENCHMARK.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    BENCHMARK._terminate_process_tree(ExitedLeader(), timeout=0.1)

    assert sent == [(12345, signal.SIGTERM)]
