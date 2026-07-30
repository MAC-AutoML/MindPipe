import json
import sys
from pathlib import Path

import pytest

from algorithm.quantization.exporters.modelslim import export_modelslim_ascend_quant
from algorithm.quantization.exporters.modelslim import resolve_modelslim_precision


FAKE_MODELSLIM = r'''import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--save_directory", required=True)
parser.add_argument("--w_bit", required=True)
parser.add_argument("--a_bit", required=True)
parser.add_argument("--is_dynamic", default="False")
args, extra = parser.parse_known_args()
output = Path(args.save_directory)
output.mkdir(parents=True, exist_ok=True)
quant_type = f"W{args.w_bit}A{args.a_bit}"
if args.is_dynamic == "True":
    quant_type += "_DYNAMIC"
(output / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
(output / "quant_model_description.json").write_text(
    json.dumps({"model.layers.0.self_attn.q_proj.weight": quant_type}),
    encoding="utf-8",
)
(output / "model.safetensors").write_bytes(b"test checkpoint")
(output / "observed.json").write_text(
    json.dumps({"w_bit": args.w_bit, "a_bit": args.a_bit, "is_dynamic": args.is_dynamic, "extra": extra}),
    encoding="utf-8",
)
'''


def _write_script(tmp_path: Path, content: str = FAKE_MODELSLIM) -> Path:
    script = tmp_path / "quant_qwen.py"
    script.write_text(content, encoding="utf-8")
    return script


def _common_args(script: Path, **overrides) -> dict:
    args = {
        "modelslim_quant_script": str(script),
        "modelslim_python": sys.executable,
        "modelslim_precision": "auto",
        "modelslim_device_type": "cpu",
        "modelslim_overwrite": False,
        "weight_bits": 8,
        "activation_bits": 8,
    }
    args.update(overrides)
    return args


def test_auto_w8a8_exports_dynamic_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = _write_script(tmp_path)
    output = tmp_path / "quantized"

    summary = export_modelslim_ascend_quant(
        model_path=str(source),
        export_dir=output,
        common_args=_common_args(script),
    )

    observed = json.loads((output / "observed.json").read_text(encoding="utf-8"))
    assert summary["precision"] == "w8a8_dynamic"
    assert summary["quant_description_counts"] == {"W8A8_DYNAMIC": 1}
    assert observed["is_dynamic"] == "True"
    assert (output / "modelslim_export_summary.json").is_file()


def test_precision_must_match_weight_and_activation_bits() -> None:
    with pytest.raises(ValueError, match="requires W4A8"):
        resolve_modelslim_precision(
            requested_precision="w4a8_dynamic",
            weight_bits=8,
            activation_bits=8,
        )


def test_missing_quant_script_fails_before_creating_export(tmp_path: Path) -> None:
    output = tmp_path / "quantized"

    with pytest.raises(FileNotFoundError, match="No model was loaded or exported"):
        export_modelslim_ascend_quant(
            model_path=str(tmp_path / "source"),
            export_dir=output,
            common_args=_common_args(tmp_path / "missing.py"),
        )

    assert not output.exists()


def test_failed_overwrite_cannot_pass_using_stale_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "quantized"
    output.mkdir()
    (output / "stale.marker").write_text("keep", encoding="utf-8")
    no_output_script = _write_script(tmp_path, "# successful process that emits no checkpoint\n")

    with pytest.raises(FileNotFoundError, match="quant_model_description.json"):
        export_modelslim_ascend_quant(
            model_path=str(source),
            export_dir=output,
            common_args=_common_args(no_output_script, modelslim_overwrite=True),
        )

    assert (output / "stale.marker").read_text(encoding="utf-8") == "keep"


def test_description_must_match_requested_precision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "quantized"
    static_output_script = _write_script(
        tmp_path,
        FAKE_MODELSLIM.replace('if args.is_dynamic == "True":', "if False:"),
    )

    with pytest.raises(ValueError, match="contains no W8A8_DYNAMIC"):
        export_modelslim_ascend_quant(
            model_path=str(source),
            export_dir=output,
            common_args=_common_args(static_output_script),
        )

    assert not output.exists()


def test_successful_overwrite_replaces_old_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "quantized"
    output.mkdir()
    (output / "stale.marker").write_text("remove", encoding="utf-8")
    script = _write_script(tmp_path)

    export_modelslim_ascend_quant(
        model_path=str(source),
        export_dir=output,
        common_args=_common_args(script, modelslim_overwrite=True),
    )

    assert not (output / "stale.marker").exists()
    assert (output / "model.safetensors").is_file()


@pytest.mark.parametrize("relative_output", [".", "nested", "parent"])
def test_source_and_output_must_not_overlap(tmp_path: Path, relative_output: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = _write_script(tmp_path)
    if relative_output == ".":
        output = source
    elif relative_output == "parent":
        output = tmp_path
    else:
        output = source / relative_output

    with pytest.raises(ValueError, match="must not be identical or nested"):
        export_modelslim_ascend_quant(
            model_path=str(source),
            export_dir=output,
            common_args=_common_args(script),
        )
