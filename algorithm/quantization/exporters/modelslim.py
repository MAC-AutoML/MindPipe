"""ModelSlim exporter for vLLM Ascend quantized checkpoints."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


_PRECISION_TO_BITS = {
    "w4a8_dynamic": (4, 8),
    "w8a8": (8, 8),
    "w8a8_dynamic": (8, 8),
}

_PRECISION_TO_DESC_VALUE = {
    "w4a8_dynamic": "W4A8_DYNAMIC",
    "w8a8": "W8A8",
    "w8a8_dynamic": "W8A8_DYNAMIC",
}


def resolve_modelslim_precision(
    *,
    requested_precision: str | None,
    weight_bits: int,
    activation_bits: int,
) -> str:
    """Resolve and validate the ModelSlim precision from W/A bits."""

    requested = (requested_precision or "auto").strip().lower()
    bits = (weight_bits, activation_bits)
    if requested != "auto":
        expected_bits = _PRECISION_TO_BITS.get(requested)
        if expected_bits is None:
            raise ValueError(
                f"Unsupported ModelSlim precision {requested_precision!r}; "
                f"expected one of auto, {', '.join(sorted(_PRECISION_TO_BITS))}."
            )
        if bits != expected_bits:
            raise ValueError(
                f"ModelSlim precision {requested} requires "
                f"W{expected_bits[0]}A{expected_bits[1]}, got W{weight_bits}A{activation_bits}."
            )
        return requested

    if bits == (4, 8):
        return "w4a8_dynamic"
    if bits == (8, 8):
        return "w8a8_dynamic"
    raise ValueError(
        "Cannot infer ModelSlim precision from "
        f"W{weight_bits}A{activation_bits}. Set --modelslim_precision to "
        "w4a8_dynamic, w8a8, or w8a8_dynamic."
    )


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _optional_bool(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _append_optional_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text:
        cmd.extend([f"--{name}", text])


def _defaulted_options(precision: str, common_args: dict[str, Any]) -> dict[str, Any]:
    if precision == "w4a8_dynamic":
        return {
            "anti_method": common_args.get("modelslim_anti_method") or "m6",
            "group_size": common_args.get("modelslim_group_size") or 256,
            "is_dynamic": _optional_bool(common_args.get("modelslim_is_dynamic"), True),
            "is_lowbit": _optional_bool(common_args.get("modelslim_is_lowbit"), True),
            "open_outlier": _optional_bool(common_args.get("modelslim_open_outlier"), False),
            "w_method": common_args.get("modelslim_w_method") or "HQQ",
        }
    if precision == "w8a8_dynamic":
        return {
            "anti_method": common_args.get("modelslim_anti_method") or "m1",
            "group_size": common_args.get("modelslim_group_size"),
            "is_dynamic": _optional_bool(common_args.get("modelslim_is_dynamic"), True),
            "is_lowbit": common_args.get("modelslim_is_lowbit"),
            "open_outlier": common_args.get("modelslim_open_outlier"),
            "w_method": common_args.get("modelslim_w_method"),
        }
    return {
        "anti_method": common_args.get("modelslim_anti_method") or "m1",
        "group_size": common_args.get("modelslim_group_size"),
        "is_dynamic": _optional_bool(common_args.get("modelslim_is_dynamic"), False),
        "is_lowbit": common_args.get("modelslim_is_lowbit"),
        "open_outlier": common_args.get("modelslim_open_outlier"),
        "w_method": common_args.get("modelslim_w_method"),
    }


def _validate_precision_options(precision: str, common_args: dict[str, Any]) -> None:
    is_dynamic = common_args.get("modelslim_is_dynamic")
    if precision.endswith("_dynamic") and is_dynamic is False:
        raise ValueError(f"ModelSlim precision {precision} requires --modelslim_is_dynamic true.")
    if precision == "w8a8" and is_dynamic is True:
        raise ValueError("ModelSlim precision w8a8 requires --modelslim_is_dynamic false.")
    if precision == "w4a8_dynamic" and common_args.get("modelslim_is_lowbit") is False:
        raise ValueError("ModelSlim precision w4a8_dynamic requires --modelslim_is_lowbit true.")


def _resolve_script(script_path: str | None) -> Path:
    candidate = script_path or os.environ.get("MODELSLIM_QUANT_SCRIPT")
    if not candidate:
        raise FileNotFoundError(
            "ModelSlim quantization script is unavailable. Pass "
            "--modelslim_quant_script /path/to/msit/msmodelslim/example/Qwen/quant_qwen.py "
            "or set MODELSLIM_QUANT_SCRIPT. No model was loaded or exported."
        )
    script = Path(candidate).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(
            f"ModelSlim quantization script not found: {script}. "
            "No model was loaded or exported."
        )
    return script


def _build_modelslim_command(
    *,
    model_path: str,
    export_path: Path,
    precision: str,
    script_path: Path,
    common_args: dict[str, Any],
) -> list[str]:
    weight_bits, activation_bits = _PRECISION_TO_BITS[precision]
    options = _defaulted_options(precision, common_args)
    python_executable = common_args.get("modelslim_python") or sys.executable
    model_path_arg = common_args.get("modelslim_model_path_arg_name") or "model_path"
    save_arg = common_args.get("modelslim_save_arg_name") or "save_directory"

    cmd = [
        str(python_executable),
        str(script_path),
        f"--{model_path_arg}",
        str(model_path),
        f"--{save_arg}",
        str(export_path),
        "--w_bit",
        str(weight_bits),
        "--a_bit",
        str(activation_bits),
        "--device_type",
        str(common_args.get("modelslim_device_type") or "npu"),
        "--trust_remote_code",
        _bool_text(_optional_bool(common_args.get("modelslim_trust_remote_code"), True)),
    ]

    _append_optional_arg(cmd, "model_type", common_args.get("modelslim_model_type"))
    _append_optional_arg(cmd, "calib_file", common_args.get("modelslim_calib_file"))
    _append_optional_arg(cmd, "anti_method", options["anti_method"])
    _append_optional_arg(cmd, "anti_calib_file", common_args.get("modelslim_anti_calib_file"))
    _append_optional_arg(cmd, "group_size", options["group_size"])

    for name in ("is_dynamic", "is_lowbit", "open_outlier"):
        value = options[name]
        if value is not None:
            cmd.extend([f"--{name}", _bool_text(bool(value))])
    _append_optional_arg(cmd, "w_method", options["w_method"])

    extra_args = common_args.get("modelslim_extra_args")
    if extra_args:
        if isinstance(extra_args, str):
            cmd.extend(shlex.split(extra_args))
        else:
            cmd.extend(str(item) for item in extra_args)
    return cmd


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _validate_weight_indexes(export_path: Path, weight_files: list[Path]) -> list[str]:
    index_paths = sorted(export_path.glob("*.safetensors.index.json"))
    available = {path.name for path in weight_files}
    for index_path in index_paths:
        index = _load_json_object(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no non-empty weight_map: {index_path}")
        missing = sorted({str(name) for name in weight_map.values()} - available)
        if missing:
            raise FileNotFoundError(
                f"Safetensors index {index_path} references missing shards: {', '.join(missing)}"
            )
    return [path.name for path in index_paths]


def _validate_output(export_path: Path, precision: str) -> dict[str, Any]:
    description_path = export_path / "quant_model_description.json"
    if not description_path.is_file():
        raise FileNotFoundError(
            f"ModelSlim export did not produce {description_path}. "
            "This is not a vLLM Ascend quantized checkpoint."
        )
    config_path = export_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"ModelSlim export did not produce required config.json under {export_path}.")
    config = _load_json_object(config_path, "model config")
    if not config:
        raise ValueError(f"Model config is empty: {config_path}")

    weight_paths = sorted(
        path
        for pattern in ("model*.safetensors", "quant_model_weight*.safetensors")
        for path in export_path.glob(pattern)
        if path.is_file()
    )
    weight_paths = list(dict.fromkeys(weight_paths))
    if not weight_paths:
        raise FileNotFoundError(
            f"ModelSlim export did not produce model*.safetensors or "
            f"quant_model_weight*.safetensors under {export_path}."
        )
    empty_weights = [path.name for path in weight_paths if path.stat().st_size <= 0]
    if empty_weights:
        raise ValueError(f"ModelSlim export produced empty weight files: {', '.join(empty_weights)}")

    description = _load_json_object(description_path, "quantization description")
    if not description:
        raise ValueError(f"Quantization description is empty: {description_path}")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in description.items()):
        raise ValueError(f"Quantization description keys and values must be strings: {description_path}")
    counts = Counter(description.values())
    expected = _PRECISION_TO_DESC_VALUE[precision]
    if counts.get(expected, 0) == 0:
        raise ValueError(
            f"quant_model_description.json contains no {expected} layers. "
            f"Found quantization types: {dict(sorted(counts.items()))}."
        )
    unexpected_quant_types = sorted(value for value in counts if value not in {"FLOAT", expected})
    if unexpected_quant_types:
        raise ValueError(
            f"quant_model_description.json contains types incompatible with {precision}: "
            f"{', '.join(unexpected_quant_types)}."
        )

    index_files = _validate_weight_indexes(export_path, weight_paths)
    return {
        "config_path": str(config_path),
        "quant_description_path": str(description_path),
        "quant_description_entries": len(description),
        "quant_description_counts": dict(sorted(counts.items())),
        "weight_files": [path.name for path in weight_paths],
        "weight_index_files": index_files,
    }


def _paths_overlap(source_path: str, export_path: Path) -> bool:
    source = Path(source_path).expanduser()
    if not source.exists():
        return False
    source = source.resolve()
    return (
        source == export_path
        or source in export_path.parents
        or export_path in source.parents
    )


def _persist_failed_log(staging_path: Path, export_path: Path) -> Path | None:
    source = staging_path / "modelslim_export.log"
    if not source.is_file():
        return None
    destination = export_path.parent / f"{export_path.name}.modelslim_export_failed.log"
    shutil.copy2(source, destination)
    return destination


def _commit_staging_output(staging_path: Path, export_path: Path) -> None:
    backup_path: Path | None = None
    if export_path.exists():
        backup_path = export_path.with_name(
            f".{export_path.name}.modelslim-backup-{uuid.uuid4().hex}"
        )
        export_path.rename(backup_path)
    try:
        staging_path.rename(export_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not export_path.exists():
            backup_path.rename(export_path)
        raise
    if backup_path is not None:
        if backup_path.is_dir():
            shutil.rmtree(backup_path)
        else:
            backup_path.unlink()


def export_modelslim_ascend_quant(
    *,
    model_path: str,
    export_dir: str | Path,
    common_args: dict[str, Any],
) -> dict[str, Any]:
    """Run ModelSlim, validate its output, and publish it transactionally."""

    precision = resolve_modelslim_precision(
        requested_precision=common_args.get("modelslim_precision"),
        weight_bits=int(common_args.get("weight_bits", 0)),
        activation_bits=int(common_args.get("activation_bits", 0)),
    )
    _validate_precision_options(precision, common_args)
    script_path = _resolve_script(common_args.get("modelslim_quant_script"))
    export_path = Path(export_dir).expanduser().resolve()
    if _paths_overlap(model_path, export_path):
        raise ValueError(
            f"ModelSlim source and output paths must not be identical or nested: "
            f"source={Path(model_path).expanduser().resolve()}, output={export_path}."
        )
    if export_path.exists() and not export_path.is_dir():
        raise FileExistsError(f"ModelSlim export path exists and is not a directory: {export_path}")
    if export_path.exists() and any(export_path.iterdir()) and not common_args.get("modelslim_overwrite", False):
        raise FileExistsError(
            f"{export_path} already exists and is not empty; pass --modelslim_overwrite true."
        )

    export_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{export_path.name}.modelslim-staging-", dir=export_path.parent)
    )
    cmd = _build_modelslim_command(
        model_path=model_path,
        export_path=staging_path,
        precision=precision,
        script_path=script_path,
        common_args=common_args,
    )
    log_path = staging_path / "modelslim_export.log"
    env = os.environ.copy()
    visible_devices = common_args.get("modelslim_visible_devices")
    if visible_devices:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(visible_devices)
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("$ " + shlex.join(cmd) + "\n\n")
            log_file.flush()
            result = subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
        if result.returncode != 0:
            failed_log = _persist_failed_log(staging_path, export_path)
            raise RuntimeError(
                f"ModelSlim export failed with exit code {result.returncode}; "
                f"see {failed_log or log_path}."
            )
        _validate_output(staging_path, precision)
        _commit_staging_output(staging_path, export_path)
    except BaseException:
        if staging_path.exists():
            try:
                _persist_failed_log(staging_path, export_path)
            except OSError:
                pass
            finally:
                shutil.rmtree(staging_path, ignore_errors=True)
        raise

    validation = _validate_output(export_path, precision)
    summary = {
        "backend": "modelslim",
        "format": "ascend/modelslim",
        "precision": precision,
        "path": str(export_path),
        "script": str(script_path),
        "command": cmd,
        "log_path": str(export_path / "modelslim_export.log"),
        "vllm_runtime_quantization": "ascend",
        **validation,
    }
    summary_path = export_path / "modelslim_export_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
