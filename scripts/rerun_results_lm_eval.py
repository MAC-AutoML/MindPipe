#!/usr/bin/env python3
"""Replay existing result configs and rerun them with lm-eval zero-shot tasks."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
MAIN_PATH = REPO_ROOT / "main.py"
DEFAULT_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
)
STATUS_ROOT = RESULTS_ROOT / "lm_eval_batch"
STATUS_PATH = STATUS_ROOT / "status.json"
MANIFEST_PATH = STATUS_ROOT / "manifest.json"
LOG_ROOT = STATUS_ROOT / "logs"

QUANTIZATION_DEFAULTS = {
    "awq": {
        "calibration_dataset": "pileval",
        "calibration_samples": 128,
        "weight_method": "gptq",
        "group_size": 128,
        "weight_group_size": 128,
        "activation_group_size": 128,
        "kv_group_size": 128,
        "weight_symmetric": False,
        "activation_symmetric": True,
        "query_symmetric": True,
        "key_symmetric": True,
        "value_symmetric": True,
        "awq_search": False,
    },
    "gptq": {
        "calibration_dataset": "pileval",
        "calibration_samples": 128,
        "weight_method": "gptq",
        "damp_percent": 0.05,
        "group_size": 128,
        "weight_group_size": 128,
        "activation_group_size": 128,
        "kv_group_size": 128,
        "weight_symmetric": False,
        "activation_symmetric": True,
        "query_symmetric": True,
        "key_symmetric": True,
        "value_symmetric": True,
    },
    "quarot": {
        "calibration_dataset": "pileval",
        "calibration_samples": 128,
        "weight_method": "gptq",
        "rotation_mode": "hadamard",
        "group_size": -1,
        "weight_group_size": -1,
        "activation_group_size": -1,
        "kv_group_size": -1,
        "weight_symmetric": True,
        "activation_symmetric": True,
        "query_symmetric": True,
        "key_symmetric": True,
        "value_symmetric": True,
    },
    "spinquant": {
        "calibration_dataset": "pileval",
        "calibration_samples": 128,
        "weight_method": "gptq",
        "rotation_mode": "hadamard",
        "group_size": 128,
        "weight_group_size": 128,
        "activation_group_size": 128,
        "kv_group_size": 128,
        "weight_symmetric": False,
        "activation_symmetric": True,
        "query_symmetric": True,
        "key_symmetric": True,
        "value_symmetric": True,
    },
    "flatquant": {
        "calibration_dataset": "pileval",
        "group_size": 128,
        "weight_group_size": None,
        "activation_group_size": None,
        "kv_group_size": 128,
        "weight_symmetric": True,
        "activation_symmetric": True,
        "query_symmetric": True,
        "key_symmetric": True,
        "value_symmetric": True,
    },
    "splitquant": {
        "calibration_dataset": "pileval",
        "calibration_samples": 128,
        "weight_method": "rtn",
        "group_size": 128,
        "weight_group_size": None,
        "activation_group_size": None,
        "kv_group_size": 128,
        "weight_symmetric": True,
        "activation_symmetric": True,
        "query_symmetric": True,
        "key_symmetric": True,
        "value_symmetric": True,
        "splitquant_epochs": 15,
        "splitquant_calibration_batch_size": 32,
        "splitquant_lr": 5e-3,
        "splitquant_diag_init": "sq_style",
        "splitquant_diag_alpha": 0.3,
        "splitquant_cali_trans": True,
        "splitquant_add_diag": True,
        "splitquant_lwc": True,
        "splitquant_lac": True,
        "splitquant_warmup": False,
        "splitquant_deactive_amp": True,
        "splitquant_separate_vtrans": False,
        "splitquant_save_matrix": False,
    },
}

PRUNING_DEFAULTS = {
    "wanda": {
        "calibration_dataset": "c4",
        "calibration_samples": 128,
        "structure_pattern": "unstructured",
    },
    "sparsegpt": {
        "calibration_dataset": "c4",
        "calibration_samples": 128,
        "structure_pattern": "unstructured",
        "block_size": 64,
        "damp_percent": 0.01,
    },
    "flap": {
        "calibration_dataset": "c4",
        "calibration_samples": 128,
        "structure_pattern": "AL-AM",
        "flap_metrics": "WIFV",
        "flap_remove_heads": 8,
        "pseudo_pruning": True,
    },
    "wanda_sp": {
        "calibration_dataset": "c4",
        "calibration_samples": 128,
        "structure_pattern": "unstructured",
        "flap_remove_heads": 8,
        "pseudo_pruning": True,
    },
}

BOOLEAN_OPTIONAL_KEYS = (
    "weight_symmetric",
    "activation_symmetric",
    "query_symmetric",
    "key_symmetric",
    "value_symmetric",
    "use_activation_order",
    "flatquant_cali_trans",
    "flatquant_add_diag",
    "flatquant_lwc",
    "flatquant_lac",
    "flatquant_warmup",
    "flatquant_deactive_amp",
    "flatquant_direct_inv",
    "flatquant_separate_vtrans",
    "flatquant_save_matrix",
    "splitquant_cali_trans",
    "splitquant_add_diag",
    "splitquant_lwc",
    "splitquant_lac",
    "splitquant_warmup",
    "splitquant_deactive_amp",
    "splitquant_separate_vtrans",
    "splitquant_save_matrix",
    "static_groups",
    "awq_search",
    "use_variant",
    "pseudo_pruning",
)


@dataclass
class ReplayJob:
    metrics_path: Path
    task_type: str
    cli_args: list[str]
    model_path: str
    zero_shot_batch_size: int
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay result configs with lm-eval zero-shot tasks.")
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:4", "cuda:7"])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--text_zero_shot_batch_size", type=int, default=4)
    parser.add_argument("--vl_zero_shot_batch_size", type=int, default=2)
    parser.add_argument("--hf_endpoint", default="https://hf-mirror.com")
    parser.add_argument("--max_jobs", type=int, default=None)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _append_bool_flag(args: list[str], key: str, value: bool | None) -> None:
    if value is None:
        return
    args.append(f"--{key}" if value else f"--no-{key}")


def _append_value(args: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    args.extend([f"--{key}", str(value)])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifacts(payload: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    inline_artifacts = payload.get("artifacts")
    if isinstance(inline_artifacts, dict):
        return inline_artifacts

    artifacts_ref = payload.get("artifacts_path")
    candidate_paths: list[Path] = []
    if isinstance(artifacts_ref, str) and artifacts_ref:
        artifacts_path = Path(artifacts_ref)
        candidate_paths.append(artifacts_path if artifacts_path.is_absolute() else metrics_path.parent / artifacts_path)
    candidate_paths.append(metrics_path.parent / "artifacts.json")

    for artifacts_path in candidate_paths:
        if artifacts_path.exists():
            loaded_artifacts = _load_json(artifacts_path)
            if isinstance(loaded_artifacts, dict):
                return loaded_artifacts
    return {}


def _infer_output_root(results_root: Path, metrics_path: Path) -> Path:
    rel_parts = metrics_path.relative_to(results_root).parts
    if rel_parts[0] == "refactor_validation":
        return results_root / rel_parts[0] / rel_parts[1]
    return results_root / rel_parts[0]


def _infer_task_type(results_root: Path, metrics_path: Path) -> str:
    rel_parts = metrics_path.relative_to(results_root).parts
    if rel_parts[0] == "refactor_validation":
        return rel_parts[1]
    if rel_parts[0] == "pruning_dtype_check":
        return "pruning"
    if rel_parts[0] == "quantization_full_eval":
        return "quantization"
    return rel_parts[0]


def _parse_quant_run_spec(run_spec: str) -> dict[str, int]:
    match = re.match(
        r"^(?P<algorithm>.+)_w(?P<weight_bits>\d+)a(?P<activation_bits>\d+)"
        r"(?:_q(?P<query_bits>\d+)k(?P<key_bits>\d+)v(?P<value_bits>\d+))?"
        r"_seq(?P<sequence_length>\d+)$",
        run_spec,
    )
    if not match:
        raise ValueError(f"Unsupported quantization run spec: {run_spec}")
    payload = {
        key: int(value)
        for key, value in match.groupdict(default="0").items()
        if key != "algorithm"
    }
    payload.setdefault("query_bits", 16)
    payload.setdefault("key_bits", 16)
    payload.setdefault("value_bits", 16)
    if payload["query_bits"] == 0:
        payload["query_bits"] = 16
    if payload["key_bits"] == 0:
        payload["key_bits"] = 16
    if payload["value_bits"] == 0:
        payload["value_bits"] = 16
    return payload


def _estimate_zero_shot_batch_size(model_path: str, args: argparse.Namespace) -> int:
    return args.vl_zero_shot_batch_size if "VL" in Path(model_path).name else args.text_zero_shot_batch_size


def _matches_include(results_root: Path, metrics_path: Path, include_patterns: list[str] | None) -> bool:
    if not include_patterns:
        return True
    rel = str(metrics_path.relative_to(results_root))
    return any(fnmatch.fnmatch(rel, pattern) for pattern in include_patterns)


def _build_quantization_args(results_root: Path, metrics_path: Path, payload: dict[str, Any]) -> list[str]:
    algorithm_name = payload["algorithm_name"]
    defaults = QUANTIZATION_DEFAULTS[algorithm_name].copy()
    output_root = _infer_output_root(results_root, metrics_path)
    run_spec = metrics_path.parent.name
    quant_spec = _parse_quant_run_spec(run_spec)
    artifacts = _load_artifacts(payload, metrics_path)
    if algorithm_name == "flatquant":
        q_config = artifacts.get("flatquant_config")
    elif algorithm_name == "splitquant":
        q_config = artifacts.get("splitquant_config") or artifacts.get("flatquant_config")
    else:
        q_config = artifacts.get("quarot_config") or artifacts.get("spinquant_config")

    if algorithm_name == "awq":
        quant_cfg = artifacts.get("quantization_config", {})
        defaults["weight_group_size"] = quant_cfg.get("q_group_size", defaults["weight_group_size"])
        defaults["group_size"] = quant_cfg.get("q_group_size", defaults["group_size"])
        defaults["weight_symmetric"] = not bool(quant_cfg.get("zero_point", False))
    elif algorithm_name == "gptq":
        sample_quant = next(iter(artifacts.get("quantized_linear_layers", {}).values()), {})
        defaults["weight_group_size"] = sample_quant.get("group_size", defaults["weight_group_size"])
        defaults["group_size"] = sample_quant.get("group_size", defaults["group_size"])
        defaults["weight_symmetric"] = sample_quant.get("symmetric", defaults["weight_symmetric"])
    elif q_config:
        defaults["calibration_samples"] = q_config.get("calibration_samples", defaults.get("calibration_samples"))
        defaults["weight_method"] = q_config.get("weight_quantizer", defaults.get("weight_method"))
        defaults["rotation_mode"] = q_config.get("rotation_mode", defaults.get("rotation_mode"))
        defaults["rotation_checkpoint"] = q_config.get("rotation_checkpoint")
        defaults["weight_group_size"] = q_config.get("weight_group_size", defaults.get("weight_group_size"))
        defaults["activation_group_size"] = q_config.get("activation_group_size", defaults.get("activation_group_size"))
        defaults["kv_group_size"] = q_config.get("kv_group_size", defaults.get("kv_group_size"))
        if algorithm_name == "quarot":
            sample_quant = next(iter(artifacts.get("quantized_linear_layers", {}).values()), {})
            defaults["weight_symmetric"] = sample_quant.get("symmetric", defaults["weight_symmetric"])
        if algorithm_name == "spinquant":
            sample_quant = next(iter(artifacts.get("quantized_linear_layers", {}).values()), {})
            defaults["weight_symmetric"] = sample_quant.get("symmetric", defaults["weight_symmetric"])
        if algorithm_name == "flatquant":
            defaults["calibration_samples"] = q_config.get("calibration_samples", defaults.get("calibration_samples"))
            defaults["flatquant_epochs"] = q_config.get("epochs", 15)
            defaults["flatquant_calibration_batch_size"] = q_config.get("calibration_batch_size", 4)
            defaults["flatquant_lr"] = q_config.get("flat_lr", 1e-5)
            defaults["flatquant_diag_init"] = q_config.get("diag_init", defaults.get("flatquant_diag_init", "sq_style"))
            defaults["flatquant_diag_alpha"] = q_config.get("diag_alpha", defaults.get("flatquant_diag_alpha", 0.3))
            defaults["flatquant_cali_trans"] = q_config.get("cali_trans", False)
            defaults["flatquant_add_diag"] = q_config.get("add_diag", False)
            defaults["flatquant_lwc"] = q_config.get("lwc", False)
            defaults["flatquant_lac"] = q_config.get("lac", False)
            defaults["flatquant_warmup"] = q_config.get("warmup", defaults.get("flatquant_warmup"))
            defaults["flatquant_deactive_amp"] = q_config.get("deactive_amp", defaults.get("flatquant_deactive_amp"))
            defaults["flatquant_separate_vtrans"] = q_config.get("separate_vtrans", defaults.get("flatquant_separate_vtrans"))
            defaults["flatquant_save_matrix"] = q_config.get("save_matrix", defaults.get("flatquant_save_matrix"))
            if q_config.get("weight_quantizer") in {"gptq", "rtn"}:
                defaults["weight_method"] = q_config["weight_quantizer"]
        if algorithm_name == "splitquant":
            defaults["calibration_samples"] = q_config.get("calibration_samples", defaults.get("calibration_samples"))
            defaults["splitquant_epochs"] = q_config.get("epochs", defaults.get("splitquant_epochs", 15))
            defaults["splitquant_calibration_batch_size"] = q_config.get("calibration_batch_size", defaults.get("splitquant_calibration_batch_size", 32))
            defaults["splitquant_lr"] = q_config.get("lr", q_config.get("flat_lr", defaults.get("splitquant_lr", 5e-3)))
            defaults["splitquant_diag_init"] = q_config.get("diag_init", defaults.get("splitquant_diag_init", "sq_style"))
            defaults["splitquant_diag_alpha"] = q_config.get("diag_alpha", defaults.get("splitquant_diag_alpha", 0.3))
            defaults["splitquant_cali_trans"] = q_config.get("cali_trans", defaults.get("splitquant_cali_trans", True))
            defaults["splitquant_add_diag"] = q_config.get("add_diag", defaults.get("splitquant_add_diag", True))
            defaults["splitquant_lwc"] = q_config.get("lwc", defaults.get("splitquant_lwc", True))
            defaults["splitquant_lac"] = q_config.get("lac", defaults.get("splitquant_lac", True))
            defaults["splitquant_warmup"] = q_config.get("warmup", defaults.get("splitquant_warmup", False))
            defaults["splitquant_deactive_amp"] = q_config.get("deactive_amp", defaults.get("splitquant_deactive_amp", True))
            defaults["splitquant_separate_vtrans"] = q_config.get("separate_vtrans", defaults.get("splitquant_separate_vtrans", False))
            defaults["splitquant_save_matrix"] = q_config.get("save_matrix", defaults.get("splitquant_save_matrix", False))
            if q_config.get("weight_quantizer") in {"gptq", "rtn"}:
                defaults["weight_method"] = q_config["weight_quantizer"]

    cli_args = [
        "--algorithm",
        algorithm_name,
        "--model_path",
        payload["model_path"],
        "--output_dir",
        str(output_root),
        "--dtype",
        payload["dtype"],
        "--evaluation_dataset",
        payload["evaluation_dataset"],
        "--calibration_dataset",
        defaults["calibration_dataset"],
        "--calibration_samples",
        str(defaults["calibration_samples"]),
        "--sequence_length",
        str(payload["sequence_length"]),
        "--batch_size",
        str(payload["batch_size"]),
        "--max_eval_chunks",
        str(payload.get("evaluated_chunks", 64)),
        "--weight_bits",
        str(quant_spec["weight_bits"]),
        "--activation_bits",
        str(quant_spec["activation_bits"]),
        "--query_bits",
        str(quant_spec["query_bits"]),
        "--key_bits",
        str(quant_spec["key_bits"]),
        "--value_bits",
        str(quant_spec["value_bits"]),
    ]
    _append_value(cli_args, "group_size", defaults.get("group_size"))
    _append_value(cli_args, "weight_group_size", defaults.get("weight_group_size"))
    _append_value(cli_args, "activation_group_size", defaults.get("activation_group_size"))
    _append_value(cli_args, "kv_group_size", defaults.get("kv_group_size"))
    weight_method = defaults.get("weight_method")
    if weight_method in {"gptq", "rtn"}:
        _append_value(cli_args, "weight_method", weight_method)
    if algorithm_name == "gptq":
        _append_value(cli_args, "damp_percent", defaults.get("damp_percent"))
    if algorithm_name in {"quarot", "spinquant"}:
        _append_value(cli_args, "rotation_mode", defaults.get("rotation_mode"))
        if defaults.get("rotation_checkpoint") and str(REPO_ROOT) in str(defaults["rotation_checkpoint"]):
            _append_value(cli_args, "rotation_checkpoint", defaults.get("rotation_checkpoint"))
    if algorithm_name == "flatquant":
        _append_value(cli_args, "flatquant_epochs", defaults.get("flatquant_epochs"))
        _append_value(cli_args, "flatquant_calibration_batch_size", defaults.get("flatquant_calibration_batch_size"))
        _append_value(cli_args, "flatquant_lr", defaults.get("flatquant_lr"))
        _append_value(cli_args, "flatquant_diag_init", defaults.get("flatquant_diag_init"))
        _append_value(cli_args, "flatquant_diag_alpha", defaults.get("flatquant_diag_alpha"))
        for key in (
            "flatquant_cali_trans",
            "flatquant_add_diag",
            "flatquant_lwc",
            "flatquant_lac",
            "flatquant_warmup",
            "flatquant_deactive_amp",
            "flatquant_separate_vtrans",
            "flatquant_save_matrix",
        ):
            _append_bool_flag(cli_args, key, defaults.get(key))
    if algorithm_name == "splitquant":
        _append_value(cli_args, "splitquant_epochs", defaults.get("splitquant_epochs"))
        _append_value(cli_args, "splitquant_calibration_batch_size", defaults.get("splitquant_calibration_batch_size"))
        _append_value(cli_args, "splitquant_lr", defaults.get("splitquant_lr"))
        _append_value(cli_args, "splitquant_diag_init", defaults.get("splitquant_diag_init"))
        _append_value(cli_args, "splitquant_diag_alpha", defaults.get("splitquant_diag_alpha"))
        for key in (
            "splitquant_cali_trans",
            "splitquant_add_diag",
            "splitquant_lwc",
            "splitquant_lac",
            "splitquant_warmup",
            "splitquant_deactive_amp",
            "splitquant_separate_vtrans",
            "splitquant_save_matrix",
        ):
            _append_bool_flag(cli_args, key, defaults.get(key))
    for key in (
        "weight_symmetric",
        "activation_symmetric",
        "query_symmetric",
        "key_symmetric",
        "value_symmetric",
        "awq_search",
    ):
        if key in defaults:
            _append_bool_flag(cli_args, key, defaults.get(key))
    return cli_args


def _build_pruning_args(results_root: Path, metrics_path: Path, payload: dict[str, Any]) -> list[str]:
    algorithm_name = payload["algorithm_name"]
    defaults = PRUNING_DEFAULTS[algorithm_name].copy()
    artifacts = _load_artifacts(payload, metrics_path)
    output_root = _infer_output_root(results_root, metrics_path)

    structure_pattern = payload.get("structure_pattern") or artifacts.get("structure_pattern") or defaults["structure_pattern"]
    if algorithm_name == "flap":
        defaults["flap_metrics"] = artifacts.get("flap_metrics", payload.get("flap_metrics", defaults["flap_metrics"]))
        defaults["flap_remove_heads"] = artifacts.get("flap_remove_heads", payload.get("flap_remove_heads", defaults["flap_remove_heads"]))
        defaults["pseudo_pruning"] = artifacts.get("pseudo_pruning", payload.get("pseudo_pruning", defaults["pseudo_pruning"]))
    elif algorithm_name == "wanda_sp":
        defaults["calibration_samples"] = artifacts.get("calibration_samples", defaults["calibration_samples"])
        defaults["calibration_dataset"] = artifacts.get("calibration_dataset", defaults["calibration_dataset"])
        defaults["pseudo_pruning"] = artifacts.get("pseudo_pruning", defaults["pseudo_pruning"])
        structure_pattern = "unstructured"

    cli_args = [
        "--algorithm",
        algorithm_name,
        "--model_path",
        payload["model_path"],
        "--output_dir",
        str(output_root),
        "--dtype",
        payload["dtype"],
        "--evaluation_dataset",
        payload["evaluation_dataset"],
        "--calibration_dataset",
        defaults["calibration_dataset"],
        "--calibration_samples",
        str(defaults["calibration_samples"]),
        "--sequence_length",
        str(payload["sequence_length"]),
        "--batch_size",
        str(payload["batch_size"]),
        "--max_eval_chunks",
        str(payload.get("evaluated_chunks", 64)),
        "--sparsity_ratio",
        str(payload["sparsity_ratio"]),
        "--structure_pattern",
        structure_pattern,
    ]
    if algorithm_name == "sparsegpt":
        _append_value(cli_args, "block_size", defaults.get("block_size"))
        _append_value(cli_args, "damp_percent", defaults.get("damp_percent"))
    if algorithm_name in {"flap", "wanda_sp"}:
        _append_value(cli_args, "flap_metrics", defaults.get("flap_metrics"))
        _append_value(cli_args, "flap_remove_heads", defaults.get("flap_remove_heads"))
        _append_bool_flag(cli_args, "pseudo_pruning", defaults.get("pseudo_pruning"))
    return cli_args


def _build_workflow_args(results_root: Path, metrics_path: Path, payload: dict[str, Any]) -> list[str]:
    artifacts = _load_artifacts(payload, metrics_path)
    stages = artifacts.get("stages", [])
    quantization_algorithm = payload["quantization_algorithm"]
    pruning_algorithm = payload["pruning_algorithm"]
    shared_parameters = {}
    if stages and isinstance(stages[0].get("parameters"), dict):
        shared_parameters = stages[0]["parameters"]
    quant_stage = next((stage for stage in stages if stage["stage_type"] == "quantization"), None)
    pruning_stage = next((stage for stage in stages if stage["stage_type"] == "pruning"), None)
    output_root = _infer_output_root(results_root, metrics_path)
    quant_defaults = QUANTIZATION_DEFAULTS.get(quantization_algorithm, {})
    pruning_defaults = PRUNING_DEFAULTS.get(pruning_algorithm, {})

    cli_args = [
        "--quantization",
        quantization_algorithm,
        "--pruning",
        pruning_algorithm,
        "--execution_order",
        payload["execution_order"],
        "--model_path",
        payload["model_path"],
        "--output_dir",
        str(output_root),
        "--dtype",
        payload["dtype"],
        "--evaluation_dataset",
        payload["evaluation_dataset"],
        "--sequence_length",
        str(payload["sequence_length"]),
        "--batch_size",
        str(payload["batch_size"]),
        "--max_eval_chunks",
        str(payload.get("evaluated_chunks", 64)),
        "--seed",
        str(shared_parameters.get("seed", 0)),
        "--quantization_calibration_dataset",
        payload.get("quantization_calibration_dataset", quant_stage.get("calibration_dataset") if quant_stage else quant_defaults.get("calibration_dataset", "pileval")),
        "--pruning_calibration_dataset",
        payload.get("pruning_calibration_dataset", pruning_stage.get("calibration_dataset") if pruning_stage else pruning_defaults.get("calibration_dataset", "c4")),
        "--quantization_calibration_samples",
        str(payload.get("quantization_calibration_samples", quant_stage.get("calibration_samples") if quant_stage else quant_defaults.get("calibration_samples", 128))),
        "--pruning_calibration_samples",
        str(payload.get("pruning_calibration_samples", pruning_stage.get("calibration_samples") if pruning_stage else pruning_defaults.get("calibration_samples", 128))),
        "--weight_bits",
        str(payload.get("weight_bits", shared_parameters.get("weight_bits", 4))),
        "--activation_bits",
        str(payload.get("activation_bits", shared_parameters.get("activation_bits", 16))),
        "--query_bits",
        str(payload.get("query_bits", shared_parameters.get("query_bits", 16))),
        "--key_bits",
        str(payload.get("key_bits", shared_parameters.get("key_bits", 16))),
        "--value_bits",
        str(payload.get("value_bits", shared_parameters.get("value_bits", 16))),
        "--sparsity_ratio",
        str(payload["sparsity_ratio"]),
        "--structure_pattern",
        payload.get("structure_pattern", shared_parameters.get("structure_pattern", pruning_defaults.get("structure_pattern", "unstructured"))),
    ]

    _append_value(cli_args, "group_size", shared_parameters.get("group_size", quant_defaults.get("group_size")))
    _append_value(cli_args, "weight_group_size", shared_parameters.get("weight_group_size", quant_defaults.get("weight_group_size")))
    _append_value(cli_args, "activation_group_size", shared_parameters.get("activation_group_size", quant_defaults.get("activation_group_size")))
    _append_value(cli_args, "kv_group_size", shared_parameters.get("kv_group_size", quant_defaults.get("kv_group_size")))
    _append_value(cli_args, "weight_method", payload.get("weight_method", shared_parameters.get("weight_method")))
    _append_value(
        cli_args,
        "quantization_damp_percent",
        (quant_stage.get("parameters", {}) if quant_stage else {}).get("damp_percent", quant_stage.get("damp_percent") if quant_stage else quant_defaults.get("damp_percent")),
    )
    _append_value(
        cli_args,
        "pruning_damp_percent",
        (pruning_stage.get("parameters", {}) if pruning_stage else {}).get("damp_percent", pruning_stage.get("damp_percent") if pruning_stage else pruning_defaults.get("damp_percent")),
    )
    _append_value(cli_args, "flatquant_epochs", shared_parameters.get("flatquant_epochs"))
    _append_value(cli_args, "flatquant_calibration_batch_size", shared_parameters.get("flatquant_calibration_batch_size"))
    _append_value(cli_args, "flatquant_lr", shared_parameters.get("flatquant_lr"))
    _append_value(cli_args, "flatquant_diag_init", shared_parameters.get("flatquant_diag_init"))
    _append_value(cli_args, "flatquant_diag_alpha", shared_parameters.get("flatquant_diag_alpha"))
    _append_value(cli_args, "splitquant_epochs", shared_parameters.get("splitquant_epochs"))
    _append_value(cli_args, "splitquant_calibration_batch_size", shared_parameters.get("splitquant_calibration_batch_size"))
    _append_value(cli_args, "splitquant_lr", shared_parameters.get("splitquant_lr"))
    _append_value(cli_args, "splitquant_diag_init", shared_parameters.get("splitquant_diag_init"))
    _append_value(cli_args, "splitquant_diag_alpha", shared_parameters.get("splitquant_diag_alpha"))
    _append_value(cli_args, "rotation_mode", shared_parameters.get("rotation_mode", quant_defaults.get("rotation_mode")))
    if shared_parameters.get("rotation_checkpoint") and str(REPO_ROOT) in str(shared_parameters["rotation_checkpoint"]):
        _append_value(cli_args, "rotation_checkpoint", shared_parameters.get("rotation_checkpoint"))
    _append_value(cli_args, "block_size", shared_parameters.get("block_size"))
    _append_value(cli_args, "flap_metrics", payload.get("flap_metrics", shared_parameters.get("flap_metrics", pruning_defaults.get("flap_metrics"))))
    _append_value(cli_args, "flap_remove_heads", payload.get("flap_remove_heads", shared_parameters.get("flap_remove_heads", pruning_defaults.get("flap_remove_heads"))))
    for key in BOOLEAN_OPTIONAL_KEYS:
        if key in shared_parameters:
            _append_bool_flag(cli_args, key, shared_parameters.get(key))
        elif key in quant_defaults:
            _append_bool_flag(cli_args, key, quant_defaults.get(key))
        elif key in pruning_defaults:
            _append_bool_flag(cli_args, key, pruning_defaults.get(key))
    return cli_args


def _build_job(results_root: Path, metrics_path: Path, args: argparse.Namespace) -> ReplayJob:
    payload = _load_json(metrics_path)
    task_type = _infer_task_type(results_root, metrics_path)
    if task_type == "quantization":
        cli_args = _build_quantization_args(results_root, metrics_path, payload)
    elif task_type == "pruning":
        cli_args = _build_pruning_args(results_root, metrics_path, payload)
    elif task_type == "workflow":
        cli_args = _build_workflow_args(results_root, metrics_path, payload)
    else:
        raise ValueError(f"Unsupported task type for replay: {task_type}")
    zero_shot_batch_size = _estimate_zero_shot_batch_size(payload["model_path"], args)
    log_rel = metrics_path.relative_to(results_root).with_suffix(".log")
    log_path = LOG_ROOT / log_rel
    return ReplayJob(
        metrics_path=metrics_path,
        task_type=task_type,
        cli_args=cli_args,
        model_path=payload["model_path"],
        zero_shot_batch_size=zero_shot_batch_size,
        log_path=log_path,
    )


def _estimate_cost(job: ReplayJob) -> int:
    parts = job.metrics_path.parts
    score = 0
    if job.task_type == "workflow":
        score += 100
    elif job.task_type == "quantization":
        score += 50
    elif job.task_type == "pruning":
        score += 30
    if "flatquant" in parts:
        score += 40
    if "splitquant" in parts:
        score += 40
    if "wanda_sp" in parts:
        score += 20
    if "VL" in parts[-4]:
        score += 10
    return score


def _discover_jobs(results_root: Path, args: argparse.Namespace) -> list[ReplayJob]:
    jobs: list[ReplayJob] = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        rel_parts = metrics_path.relative_to(results_root).parts
        if not rel_parts:
            continue
        if rel_parts[0] in {"logs", "lm_eval", "lm_eval_batch"}:
            continue
        if not _matches_include(results_root, metrics_path, args.include):
            continue
        payload = _load_json(metrics_path)
        if not args.force and payload.get("zero_shot"):
            continue
        jobs.append(_build_job(results_root, metrics_path, args))
    jobs.sort(key=_estimate_cost, reverse=True)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs


def _summarize_jobs(results_root: Path, jobs: list[ReplayJob], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_root": str(results_root),
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "devices": args.devices,
        "job_count": len(jobs),
        "jobs": [str(job.metrics_path.relative_to(results_root)) for job in jobs],
    }


def _status_stub(results_root: Path, jobs: list[ReplayJob], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_root": str(results_root),
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "devices": args.devices,
        "total": len(jobs),
        "completed": [],
        "failed": [],
        "pending": [str(job.metrics_path.relative_to(results_root)) for job in jobs],
        "running": {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _command_for_job(job: ReplayJob, device: str, args: argparse.Namespace, zero_shot_batch_size: int) -> list[str]:
    # Determine the flag name based on task type
    algo_flag = "--quantization" if job.task_type == "quantization" else "--pruning"
    # Find and replace --algorithm <name> with --<quantization|pruning> <name>
    cli_args = list(job.cli_args)
    algo_idx = cli_args.index("--algorithm")
    algo_name = cli_args[algo_idx + 1]
    cli_args[algo_idx:algo_idx + 2] = [algo_flag, algo_name]
    # Replace --output_root with --output_dir
    for i, arg in enumerate(cli_args):
        if arg == "--output_root":
            cli_args[i] = "--output_dir"
    return [
        os.environ.get("PYTHON", os.sys.executable),
        str(MAIN_PATH),
        *cli_args,
        "--device",
        device,
        "--eval_zero_shot",
        "true",
        "--zero_shot_tasks",
        *args.tasks,
        "--zero_shot_num_fewshot",
        str(args.num_fewshot),
        "--zero_shot_batch_size",
        str(zero_shot_batch_size),
    ]


def _run_one_job(job: ReplayJob, device: str, args: argparse.Namespace) -> tuple[bool, float, str | None]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    for zero_shot_batch_size in dict.fromkeys((job.zero_shot_batch_size, 1)):
        command = _command_for_job(job, device, args, zero_shot_batch_size)
        env = os.environ.copy()
        env["HF_ENDPOINT"] = args.hf_endpoint
        start_time = time.perf_counter()
        with job.log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"$ {shlex.join(command)}\n")
            handle.write(f"HF_ENDPOINT={args.hf_endpoint}\n")
            handle.flush()
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                check=False,
            )
        elapsed_seconds = time.perf_counter() - start_time
        if completed.returncode == 0:
            payload = _load_json(job.metrics_path)
            zero_shot = payload.get("zero_shot")
            if zero_shot is None:
                return False, elapsed_seconds, "metrics_missing_zero_shot"
            return True, elapsed_seconds, None
        log_text = job.log_path.read_text(encoding="utf-8", errors="ignore").lower()
        if zero_shot_batch_size > 1 and "out of memory" in log_text:
            continue
        return False, elapsed_seconds, f"exit_{completed.returncode}"
    return False, 0.0, "oom_after_retry"


def _worker(device: str, job_queue: queue.Queue[ReplayJob], args: argparse.Namespace, status: dict[str, Any], lock: threading.Lock) -> None:
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return
        rel_path = str(job.metrics_path.relative_to(Path(args.results_root)))
        with lock:
            status["running"][device] = rel_path
            if rel_path in status["pending"]:
                status["pending"].remove(rel_path)
            _write_json(STATUS_PATH, status)
        ok, elapsed_seconds, error = _run_one_job(job, device, args)
        with lock:
            status["running"].pop(device, None)
            record = {
                "metrics_path": rel_path,
                "device": device,
                "elapsed_seconds": elapsed_seconds,
                "log_path": str(job.log_path),
            }
            if ok:
                status["completed"].append(record)
            else:
                record["error"] = error
                status["failed"].append(record)
            status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_json(STATUS_PATH, status)
        job_queue.task_done()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    jobs = _discover_jobs(results_root, args)
    _write_json(MANIFEST_PATH, _summarize_jobs(results_root, jobs, args))
    status = _status_stub(results_root, jobs, args)
    _write_json(STATUS_PATH, status)

    if args.dry_run:
        print(json.dumps(_summarize_jobs(results_root, jobs, args), ensure_ascii=False, indent=2))
        return 0

    job_queue: queue.Queue[ReplayJob] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    lock = threading.Lock()
    threads = [
        threading.Thread(target=_worker, args=(device, job_queue, args, status, lock), daemon=False)
        for device in args.devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(json.dumps(_load_json(STATUS_PATH), ensure_ascii=False, indent=2))
    return 0 if not status["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
# Maintenance touch for repository metadata refresh.
