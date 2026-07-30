import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "export_fp16_to_ascend_w8a8_dynamic_probe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_fp16_to_ascend_w8a8_dynamic_probe", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def _write_source(path: Path, *, model_type: str = "qwen2") -> None:
    path.mkdir(parents=True)
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": model_type,
        "num_hidden_layers": 1,
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float16).reshape(4, 3),
        "model.layers.0.input_layernorm.weight": torch.ones(3, dtype=torch.float16),
        "lm_head.weight": torch.arange(12, dtype=torch.float16).reshape(4, 3),
    }
    for index, suffix in enumerate(EXPORTER.LINEAR_SUFFIXES, start=1):
        tensors[f"model.layers.0.{suffix}.weight"] = (
            torch.arange(6, dtype=torch.float16).reshape(2, 3) - index
        )
    save_file(tensors, path / "model.safetensors")


def _run_exporter(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *(str(value) for value in arguments)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_cli_converts_and_verifies_tiny_dense_qwen2_checkpoint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    converted = _run_exporter("--source", source, "--output", output)

    assert converted.returncode == 0, converted.stderr
    summary = json.loads(converted.stdout)
    assert summary["converted_linear_modules"] == 7
    assert summary["validation"]["validated_linear_weights"] == 7
    assert summary["validation"]["output_index_present"] is True
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert "model.layers.0.self_attn.q_proj.weight_scale" in index["weight_map"]
    with safe_open(
        output / "model.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert str(handle.get_slice(
            "model.layers.0.self_attn.q_proj.weight").get_dtype()).upper() in {
                "I8",
                "INT8",
            }
        assert handle.get_slice(
            "model.layers.0.self_attn.q_proj.weight_scale"
        ).get_shape() == [2, 1]

    verified = _run_exporter(
        "--source", source, "--output", output, "--verify-only"
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["output_index_present"] is True

    # Historical accepted Dense exports did not contain an index. Keep their
    # verify-only path usable while all new exports write a complete index.
    (output / "model.safetensors.index.json").unlink()
    verified_legacy = _run_exporter(
        "--source", source, "--output", output, "--verify-only"
    )
    assert verified_legacy.returncode == 0, verified_legacy.stderr
    assert json.loads(verified_legacy.stdout)["output_index_present"] is False


@pytest.mark.parametrize("model_type", ["qwen2_vl", "qwen3", "qwen3_moe"])
def test_read_config_rejects_non_dense_qwen2_model_types(tmp_path, model_type):
    source = tmp_path / model_type
    _write_source(source, model_type=model_type)

    with pytest.raises(ValueError, match="Expected dense Qwen2/Qwen2.5"):
        EXPORTER.read_config(source)


def test_validate_source_layout_rejects_missing_projection(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    tensors = {}
    with safe_open(
        source / "model.safetensors", framework="pt", device="cpu"
    ) as handle:
        for name in handle.keys():
            if not name.endswith("mlp.down_proj.weight"):
                tensors[name] = handle.get_tensor(name)
    save_file(tensors, source / "replacement.safetensors")
    (source / "model.safetensors").unlink()
    (source / "replacement.safetensors").rename(source / "model.safetensors")
    config = EXPORTER.read_config(source)
    _, weight_map = EXPORTER.get_weight_index(source)

    with pytest.raises(ValueError, match="missing 1 required"):
        EXPORTER.validate_source_layout(config, weight_map)


def test_resolve_paths_rejects_equal_or_nested_directories(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="separate, non-nested"):
        EXPORTER.resolve_paths(str(source), str(source))
    with pytest.raises(ValueError, match="separate, non-nested"):
        EXPORTER.resolve_paths(str(source), str(source / "output"))
    with pytest.raises(ValueError, match="separate, non-nested"):
        EXPORTER.resolve_paths(str(source), str(tmp_path))


def test_quantize_weight_rejects_non_floating_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="floating-point"):
        EXPORTER.quantize_weight_per_channel(torch.ones((2, 2), dtype=torch.int8))
    with pytest.raises(ValueError, match="non-finite"):
        EXPORTER.quantize_weight_per_channel(
            torch.tensor([[1.0, float("nan")]], dtype=torch.float16)
        )


def test_verify_only_rejects_scale_dtype_that_differs_from_source(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    converted = _run_exporter("--source", source, "--output", output)
    assert converted.returncode == 0, converted.stderr

    shard = output / "model.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    scale_name = "model.layers.0.self_attn.q_proj.weight_scale"
    tensors[scale_name] = tensors[scale_name].to(torch.bfloat16)
    save_file(tensors, shard)

    verified = _run_exporter(
        "--source", source, "--output", output, "--verify-only"
    )

    assert verified.returncode != 0
    assert "expected source weight dtype F16" in verified.stderr
