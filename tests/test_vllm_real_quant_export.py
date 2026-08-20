import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import workflow.executor as executor
from algorithm.quantization.exporters.schema import RealQuantLinearArtifact
from algorithm.quantization.exporters.vllm import export_vllm_gptq_w4a16
from algorithm.quantization.exporters.vllm import pack_int4_for_vllm
from workflow.builder import build_run_config
from workflow.builder import build_run_parser
from workflow.builder import validate_workflow_config
from workflow.schema import WorkflowStage


def _unpack_int4(packed: torch.Tensor, columns: int) -> torch.Tensor:
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * 8), dtype=torch.int16)
    for offset in range(8):
        unpacked[:, offset::8] = ((packed >> (4 * offset)) & 0xF).to(torch.int16)
    return (unpacked[:, :columns] - 8).to(torch.int8)


def test_pack_int4_round_trip_handles_signed_values_and_padding():
    values = torch.tensor(
        [[-8, -7, -1, 0, 1, 6, 7, -3, 4, 2]],
        dtype=torch.int8,
    )

    packed = pack_int4_for_vllm(values)

    assert packed.dtype == torch.int32
    assert packed.shape == (1, 2)
    assert torch.equal(_unpack_int4(packed, values.shape[1]), values)


def test_gptq_real_quant_artifact_reconstructs_fake_quant_weight(monkeypatch):
    source_root = (
        Path(__file__).resolve().parents[1]
        / "algorithm"
        / "quantization"
        / "ptq"
        / "gptq"
        / "source"
    )
    monkeypatch.syspath_prepend(str(source_root))
    GPTQ = importlib.import_module("gptq").GPTQ
    Quantizer = importlib.import_module("quant").Quantizer
    torch.manual_seed(0)
    linear = nn.Linear(8, 3, bias=False)
    gptq = GPTQ(linear)
    quantizer = Quantizer()
    quantizer.configure(4, perchannel=True, sym=True, mse=False)
    gptq.quantizer = quantizer
    inputs = torch.randn(2, 16, 8)
    gptq.add_batch(inputs, linear(inputs))

    result = gptq.fasterquant(
        blocksize=4,
        percdamp=0.01,
        groupsize=4,
        return_real_quant=True,
    )
    reconstructed = result["int_weight"].float() * result["scale"].repeat_interleave(
        4,
        dim=1,
    )

    assert result["int_weight"].min().item() >= -8
    assert result["int_weight"].max().item() <= 7
    assert result["scale"].shape == (3, 2)
    assert torch.allclose(reconstructed, linear.weight.float(), atol=1e-6, rtol=0)


class _TinyExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 2, bias=True)
        self.ignored = nn.Linear(8, 2, bias=False)

    def save_pretrained(self, output_dir, *, safe_serialization):
        assert safe_serialization is True
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), output / "model.safetensors")
        (output / "config.json").write_text(
            json.dumps({"model_type": "tiny", "rope_theta": 10000.0}),
            encoding="utf-8",
        )


class _TinyTokenizerBundle:
    source_path = None

    def save_pretrained(self, output_dir):
        Path(output_dir, "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_export_vllm_writes_packed_checkpoint_and_restores_model(tmp_path: Path):
    model = _TinyExportModel()
    original_proj = model.proj
    int_weight = torch.tensor(
        [
            [-8, -7, -1, 0, 1, 6, 7, -3],
            [7, 6, 1, 0, -1, -7, -8, 3],
        ],
        dtype=torch.int8,
    )
    artifact = RealQuantLinearArtifact(
        name="proj",
        bits=4,
        group_size=4,
        symmetric=True,
        original_shape=(2, 8),
        int_weight=int_weight,
        scale=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float16),
    )

    summary = export_vllm_gptq_w4a16(
        model=model,
        tokenizer_bundle=_TinyTokenizerBundle(),
        artifacts={"proj": artifact},
        export_dir=tmp_path,
        group_size=4,
    )

    assert model.proj is original_proj
    assert summary["backend"] == "vllm"
    assert summary["quantized_linear_count"] == 1
    assert summary["ignored_layers"] == ["ignored"]
    with safe_open(
        tmp_path / "model.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert torch.equal(
            _unpack_int4(handle.get_tensor("proj.weight_packed"), 8),
            int_weight,
        )
        assert tuple(handle.get_tensor("proj.weight_scale").shape) == (2, 2)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["compression_config"]["format"] == "pack-quantized"
    assert (
        config["compression_config"]["config_groups"]["group_0"]["weights"][
            "group_size"
        ]
        == 4
    )


def _vllm_export_args(tmp_path: Path) -> argparse.Namespace:
    return build_run_parser().parse_args(
        [
            "--model_path",
            str(tmp_path / "model"),
            "--output_dir",
            str(tmp_path / "runs"),
            "--quantization",
            "gptq",
            "--weight_bits",
            "4",
            "--activation_bits",
            "16",
            "--weight_group_size",
            "4",
            "--export_real_quant",
            "true",
            "--export_backend",
            "vllm",
        ]
    )


def test_builder_accepts_single_stage_vllm_gptq_export(tmp_path: Path):
    config = build_run_config(_vllm_export_args(tmp_path))

    validate_workflow_config(config)
    assert [(stage.stage_type, stage.algorithm_name) for stage in config.stages] == [
        ("quantization", "gptq")
    ]
    assert config.common_args["export_backend"] == "vllm"
    assert config.result_metadata["export_backend"] == "vllm"


def test_builder_rejects_activation_order_for_vllm_export(tmp_path: Path):
    args = _vllm_export_args(tmp_path)
    args.use_activation_order = True

    with pytest.raises(ValueError, match="does not support --use_activation_order"):
        build_run_config(args)


def test_run_stage_separates_nonserializable_real_quant_artifacts(tmp_path: Path):
    artifacts = {"model.layers.0.proj": object()}

    class FakeGPTQMethod:
        def resolve_output_dir(self, _args):
            return tmp_path

        def apply_fake_quantization(self, _model, _tokenizer_bundle, _args):
            return {
                "quantized_linear_count": 1,
                "_real_quant_artifacts": artifacts,
            }

    record, next_model, next_tokenizer = executor._run_stage(
        FakeGPTQMethod(),
        WorkflowStage(stage_type="quantization", algorithm_name="gptq"),
        model=object(),
        tokenizer_bundle=object(),
        stage_args=argparse.Namespace(hf_token="secret"),
    )

    assert record["artifacts"] == {"quantized_linear_count": 1}
    assert record["internal_artifacts"] == {"_real_quant_artifacts": artifacts}
    assert "hf_token" not in record["parameters"]
    assert next_model is not None
    assert next_tokenizer is not None


def test_executor_dispatches_vllm_artifacts_without_serializing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact = SimpleNamespace(group_size=4)
    observed = {}

    def fake_export(**kwargs):
        observed.update(kwargs)
        return {"backend": "vllm", "path": str(kwargs["export_dir"])}

    monkeypatch.setattr(executor, "export_vllm_gptq_w4a16", fake_export)
    result = executor._export_vllm_real_quant_model(
        model="model",
        tokenizer_bundle="tokenizer",
        common_args={"export_real_quant": True, "export_backend": "vllm"},
        stage_records=[
            {
                "stage_type": "quantization",
                "algorithm_name": "gptq",
                "internal_artifacts": {"_real_quant_artifacts": {"proj": artifact}},
            }
        ],
        final_output_dir=tmp_path,
    )

    assert result["backend"] == "vllm"
    assert observed["group_size"] == 4
    assert observed["export_dir"] == tmp_path / "real_quant_vllm_model"
