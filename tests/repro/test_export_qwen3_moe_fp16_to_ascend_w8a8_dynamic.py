import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "export_qwen3_moe_fp16_to_ascend_w8a8_dynamic.py"
)
SPEC = importlib.util.spec_from_file_location("export_qwen3_moe_w8a8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def _minimal_layout():
    config = {"num_hidden_layers": 1, "num_experts": 1}
    attention, experts, routers = EXPORTER.expected_weight_names(config)
    source_names = attention | experts | routers | {"model.embed_tokens.weight"}
    source_by_file = {"model.safetensors": sorted(source_names)}
    return config, attention, experts, routers, source_names, source_by_file


def _write_output(
    output: Path,
    *,
    omitted: set[str] | None = None,
    source_dtype: torch.dtype = torch.float16,
    scale_dtype: torch.dtype | None = None,
    index_overrides: dict[str, str] | None = None,
) -> tuple[dict, set[str], set[str], set[str], dict[str, list[str]], Path]:
    config, attention, experts, routers, source_names, source_by_file = (
        _minimal_layout()
    )
    omitted = omitted or set()
    scale_dtype = scale_dtype or source_dtype
    quantized = attention | experts
    source = output.with_name(f"{output.name}_source")
    source.mkdir()
    save_file(
        {
            name: torch.ones((2, 2), dtype=source_dtype)
            for name in sorted(source_names)
        },
        source / "model.safetensors",
    )
    tensors = {}
    for name in sorted(quantized):
        tensors[name] = torch.ones((2, 2), dtype=torch.int8)
        tensors[f"{name}_scale"] = torch.ones((2, 1), dtype=scale_dtype)
        tensors[f"{name}_offset"] = torch.zeros((2, 1), dtype=scale_dtype)
    for name in sorted(source_names - quantized):
        tensors[name] = torch.ones((2, 2), dtype=torch.float16)
    for name in omitted:
        tensors.pop(name)

    output.mkdir()
    save_file(tensors, output / "model.safetensors")
    expected_names = source_names | {
        f"{name}{suffix}"
        for name in quantized
        for suffix in ("_scale", "_offset")
    }
    weight_map = {name: "model.safetensors" for name in expected_names}
    weight_map.update(index_overrides or {})
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    description = {name: EXPORTER.QUANT_TYPE for name in quantized}
    description.update({name: EXPORTER.FLOAT_TYPE for name in routers})
    (output / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )
    return config, attention, experts, routers, source_by_file, source


def test_validate_paths_rejects_equal_or_nested_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="different directories"):
        EXPORTER.validate_paths(source, source)
    with pytest.raises(ValueError, match="must not contain"):
        EXPORTER.validate_paths(source, source / "quantized")
    with pytest.raises(ValueError, match="must not contain"):
        EXPORTER.validate_paths(source, tmp_path)


def test_read_config_accepts_only_qwen3_moe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config_path = source / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
            }
        ),
        encoding="utf-8",
    )

    assert EXPORTER.read_config(source)["model_type"] == "qwen3_moe"

    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen2_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected model_type"):
        EXPORTER.read_config(source)


def test_validate_output_checks_index_against_physical_shard_keys(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    config, attention, experts, routers, source_by_file, source = _write_output(output)

    report = EXPORTER.validate_output(
        output, source, source_by_file, config, attention, experts, routers
    )

    assert report["validated_expert_projection_weights"] == 3
    assert report["validated_attention_projection_weights"] == 4


def test_validate_output_rejects_indexed_source_tensor_missing_from_shard(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    missing_name = "model.embed_tokens.weight"
    config, attention, experts, routers, source_by_file, source = _write_output(
        output, omitted={missing_name}
    )

    with pytest.raises(ValueError, match="Physical checkpoint misses"):
        EXPORTER.validate_output(
            output, source, source_by_file, config, attention, experts, routers
        )


def test_validate_output_rejects_index_mapping_to_wrong_shard(tmp_path: Path) -> None:
    output = tmp_path / "output"
    mismatched_name = "model.embed_tokens.weight"
    config, attention, experts, routers, source_by_file, source = _write_output(
        output, index_overrides={mismatched_name: "wrong.safetensors"}
    )

    with pytest.raises(ValueError, match="wrong shard"):
        EXPORTER.validate_output(
            output, source, source_by_file, config, attention, experts, routers
        )


def test_validate_output_rejects_nonfloating_scale_and_offset(tmp_path: Path) -> None:
    output = tmp_path / "output"
    config, attention, experts, routers, source_by_file, source = _write_output(
        output, scale_dtype=torch.int8
    )

    with pytest.raises(ValueError, match="expected source weight dtype"):
        EXPORTER.validate_output(
            output, source, source_by_file, config, attention, experts, routers
        )


def test_validate_output_requires_scale_and_offset_to_match_source_dtype(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    config, attention, experts, routers, source_by_file, source = _write_output(
        output,
        source_dtype=torch.bfloat16,
        scale_dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="expected source weight dtype BF16"):
        EXPORTER.validate_output(
            output, source, source_by_file, config, attention, experts, routers
        )
