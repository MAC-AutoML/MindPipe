import importlib.util
import os
import signal
from argparse import Namespace
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "validate_qwen3_w8a8_accelerated_prompts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_qwen3_w8a8_accelerated_prompts", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
_validate = VALIDATOR._validate


def test_exact_check_accepts_terminal_punctuation() -> None:
    case = {"check": "exact", "expected": "能"}
    assert _validate(case, "能。")[0] is True


def test_contains_check_normalizes_unicode_subscripts() -> None:
    case = {"check": "contains", "expected": ["H2O"]}
    assert _validate(case, "H₂O")[0] is True


def test_exact_check_still_rejects_wrong_answer() -> None:
    case = {"check": "exact", "expected": "汽车"}
    assert _validate(case, "猫")[0] is False


def test_json_check_accepts_code_fence() -> None:
    case = {"check": "json", "expected": {"status": "ok"}}
    assert _validate(case, '```json\n{"status":"ok"}\n```')[0] is True


def test_contains_check_preserves_code_spacing() -> None:
    case = {"check": "contains_ci", "expected": ["range(1, 6)", "**2"]}
    assert _validate(case, "[x**2 for x in range(1, 6)]")[0] is True


def test_runtime_environment_uses_configured_pythonpath(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_path = tmp_path / "runtime"
    runtime_path.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/inherited/runtime")
    args = Namespace(
        env=["CUSTOM_SWITCH=enabled"],
        pythonpath=[runtime_path],
        device="0,1",
    )

    environment = VALIDATOR._runtime_environment(args)

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(runtime_path.resolve()),
        "/inherited/runtime",
    ]
    assert environment["CUSTOM_SWITCH"] == "enabled"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"


def test_terminate_cleans_group_after_leader_exits(monkeypatch) -> None:
    class ExitedLeader:
        pid = 12345

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    group_states = iter([True, False, False])
    sent = []
    monkeypatch.setattr(
        VALIDATOR, "_group_exists", lambda pgid: next(group_states)
    )
    monkeypatch.setattr(VALIDATOR.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))

    VALIDATOR._terminate(ExitedLeader(), timeout=0.1)

    assert sent == [(12345, signal.SIGTERM)]


def test_all_finish_reasons_stop_rejects_length_termination() -> None:
    results = [{"finish_reason": "stop"} for _ in range(31)]
    results.append({"finish_reason": "length"})

    assert VALIDATOR._all_finish_reasons_stop(results, 32) is False
    assert VALIDATOR._all_finish_reasons_stop(
        [{"finish_reason": "stop"} for _ in range(32)], 32
    ) is True
