import json
import sys
from pathlib import Path

import pytest

import workflow.executor as executor
from workflow.builder import build_run_config
from workflow.builder import build_run_parser
from workflow.builder import validate_workflow_config
from workflow.schema import WorkflowConfig
from workflow.schema import WorkflowStage

from test_modelslim_exporter import FAKE_MODELSLIM


def _export_args(tmp_path: Path, script: Path) -> list[str]:
    return [
        "--model_path",
        str(tmp_path / "source"),
        "--device",
        "cpu",
        "--output_dir",
        str(tmp_path / "runs"),
        "--export_real_quant",
        "true",
        "--export_backend",
        "modelslim",
        "--modelslim_quant_script",
        str(script),
        "--modelslim_python",
        sys.executable,
        "--modelslim_device_type",
        "cpu",
        "--weight_bits",
        "8",
        "--activation_bits",
        "8",
    ]


def test_workflow_runs_modelslim_without_loading_hf_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "source").mkdir()
    script = tmp_path / "quant_qwen.py"
    script.write_text(FAKE_MODELSLIM, encoding="utf-8")
    args = build_run_parser().parse_args(_export_args(tmp_path, script))
    config = build_run_config(args)

    def fail_model_load(*args, **kwargs):
        raise AssertionError("export-only workflow must not load the HF model")

    monkeypatch.setattr(executor, "load_model_and_tokenizer", fail_model_load)
    result = executor.run_workflow(config)

    assert result.metrics["export_precision"] == "w8a8_dynamic"
    artifacts = json.loads(Path(result.artifacts_path).read_text(encoding="utf-8"))
    assert artifacts["real_quant_export"]["backend"] == "modelslim"
    status = json.loads((Path(result.output_dir) / "export_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert Path(status["export_path"]).parent == Path(result.output_dir).parent


def test_workflow_missing_script_is_explicit_and_does_not_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_run_parser().parse_args(_export_args(tmp_path, tmp_path / "missing.py"))
    config = build_run_config(args)
    loaded = False

    def track_model_load(*args, **kwargs):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(executor, "load_model_and_tokenizer", track_model_load)
    with pytest.raises(FileNotFoundError, match="No model was loaded or exported"):
        executor.run_workflow(config)
    assert loaded is False


def test_builder_rejects_evaluation_combined_with_modelslim(tmp_path: Path) -> None:
    args = build_run_parser().parse_args(
        _export_args(tmp_path, tmp_path / "quant_qwen.py") + ["--eval_ppl", "true"]
    )

    with pytest.raises(ValueError, match="disable in-process evaluation"):
        build_run_config(args)


def test_builder_accepts_legacy_namespace_without_export_fields(tmp_path: Path) -> None:
    args = build_run_parser().parse_args(
        [
            "--model_path",
            str(tmp_path / "source"),
            "--output_dir",
            str(tmp_path / "runs"),
            "--eval_ppl",
            "true",
        ]
    )
    delattr(args, "export_real_quant")
    delattr(args, "export_backend")

    config = build_run_config(args)

    assert config.stages == []
    assert config.common_args["export_real_quant"] is False
    assert config.common_args["export_backend"] == "modelslim"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("stages", "export-only ModelSlim operation"),
        ("evaluation", "disable in-process evaluation"),
        ("save_model", "save_model is not applicable"),
        ("backend", "Unsupported export-only backend"),
    ],
)
def test_programmatic_export_config_is_centrally_validated(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    common_args = {
        "export_real_quant": True,
        "export_backend": "modelslim",
        "eval_ppl": False,
        "eval_zero_shot": False,
        "eval_vlm": False,
        "save_model": False,
    }
    stages = []
    save_model = False
    if case == "stages":
        stages = [WorkflowStage(stage_type="quantization", algorithm_name="gptq")]
    elif case == "evaluation":
        common_args["eval_ppl"] = True
    elif case == "save_model":
        save_model = True
    elif case == "backend":
        common_args["export_backend"] = "unsupported"

    config = WorkflowConfig(
        model_path=str(tmp_path / "source"),
        common_args=common_args,
        stages=stages,
        output_dir=tmp_path / "workflow",
        save_model=save_model,
    )

    with pytest.raises(ValueError, match=message):
        validate_workflow_config(config)


@pytest.mark.parametrize("relation", ["same", "ancestor", "descendant"])
def test_workflow_rejects_checkpoint_and_metadata_path_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relation: str,
) -> None:
    args = build_run_parser().parse_args(_export_args(tmp_path, tmp_path / "quant_qwen.py"))
    config = build_run_config(args)
    workflow_dir = Path(config.output_dir)
    if relation == "same":
        export_dir = workflow_dir
    elif relation == "ancestor":
        export_dir = workflow_dir.parent
    else:
        export_dir = workflow_dir / "checkpoint"
    config.common_args["export_quantized_model_dir"] = str(export_dir)

    called = False

    def track_export(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(executor, "export_modelslim_ascend_quant", track_export)
    with pytest.raises(ValueError, match="must not be identical or nested"):
        executor.run_workflow(config)

    assert called is False
    status = json.loads((workflow_dir / "export_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"
    assert not (workflow_dir / "metrics.json").exists()
    assert not (workflow_dir / "artifacts.json").exists()


def test_failed_retry_clears_previous_success_metadata(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    script = tmp_path / "quant_qwen.py"
    script.write_text(FAKE_MODELSLIM, encoding="utf-8")
    args = build_run_parser().parse_args(_export_args(tmp_path, script))
    config = build_run_config(args)

    first_result = executor.run_workflow(config)
    workflow_dir = Path(first_result.output_dir)
    assert Path(first_result.metrics_path).is_file()
    assert Path(first_result.artifacts_path).is_file()

    config.common_args["modelslim_quant_script"] = str(tmp_path / "missing.py")
    with pytest.raises(FileNotFoundError, match="No model was loaded or exported"):
        executor.run_workflow(config)

    assert not Path(first_result.metrics_path).exists()
    assert not Path(first_result.artifacts_path).exists()
    status = json.loads((workflow_dir / "export_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "FileNotFoundError"


def test_non_export_workflow_still_uses_original_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_run_parser().parse_args(
        [
            "--model_path",
            str(tmp_path / "source"),
            "--device",
            "cpu",
            "--output_dir",
            str(tmp_path / "runs"),
            "--eval_ppl",
            "true",
        ]
    )
    config = build_run_config(args)
    loaded_paths = []

    class FakeModel:
        pass

    def fake_load(model_path, **kwargs):
        loaded_paths.append(model_path)
        return FakeModel(), object()

    monkeypatch.setattr(executor, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(executor, "run_evaluations", lambda **kwargs: {"perplexity": 1.0})

    result = executor.run_workflow(config)

    assert loaded_paths == [config.model_path]
    assert result.metrics["perplexity"] == 1.0
    assert "export_backend" not in result.metrics
