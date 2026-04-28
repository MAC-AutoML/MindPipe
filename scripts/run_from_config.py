#!/usr/bin/env python3
"""Run experiments from layered YAML configs without changing existing shell entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.common.logging import setup_logging
from algorithm.common.reproducibility import set_global_seed
from workflow.builder import build_run_config
from workflow.builder import build_run_parser
from workflow.executor import run_workflow


CONFIG_SCHEMA = "mindpipe/config/v1"
RESOLVED_SCHEMA = "mindpipe/resolved/v1"
DEFAULT_CONFIG_ROOT = REPO_ROOT / "configs"
DEFAULT_LOCAL_CONFIG = Path("common") / "local.yaml"
DEFAULT_COMMON_CONFIG = Path("common") / "base.yaml"
SENSITIVE_KEYS = {"hf_token"}
PATH_LIKE_KEYS = {"data_path", "output_dir", "vlm_eval_kit_root", "model_path"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MindPipe experiments from layered YAML configs.")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--local-config", default=str(DEFAULT_LOCAL_CONFIG))
    parser.add_argument("--algorithm", required=True, help="Algorithm folder name under configs/algorithms.")
    parser.add_argument("--model", required=True, help="Model config filename without the .yaml suffix.")
    parser.add_argument("--recipe", required=True, help="Recipe name under the model config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a resolved config key. VALUE is parsed with YAML semantics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print the planned run without executing it.")
    return parser


def _timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    schema = payload.get("schema")
    if schema not in {None, CONFIG_SCHEMA}:
        raise ValueError(f"Unsupported config schema in {path}: {schema!r}")
    return _normalize_config_payload(payload)


def _expand_path_string(value: str) -> str:
    env = dict(os.environ)
    env.setdefault("REPO_ROOT", str(REPO_ROOT))
    return os.path.expanduser(Template(value).safe_substitute(env))


def _normalize_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)

    defaults = normalized.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, dict):
            raise ValueError("'defaults' must be a mapping")
        normalized["defaults"] = {
            key: (_expand_path_string(value) if key in PATH_LIKE_KEYS and isinstance(value, str) else value)
            for key, value in defaults.items()
        }

    model_meta = normalized.get("model")
    if model_meta is not None:
        if not isinstance(model_meta, dict):
            raise ValueError("'model' must be a mapping")
        updated_model_meta = dict(model_meta)
        model_path = updated_model_meta.get("model_path")
        if isinstance(model_path, str):
            updated_model_meta["model_path"] = _expand_path_string(model_path)
        normalized["model"] = updated_model_meta

    model_paths = normalized.get("model_paths")
    if model_paths is not None:
        if not isinstance(model_paths, dict):
            raise ValueError("'model_paths' must be a mapping")
        normalized["model_paths"] = {
            str(model_name): (_expand_path_string(path) if isinstance(path, str) else path)
            for model_name, path in model_paths.items()
        }

    return normalized


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_override(entry: str) -> tuple[str, Any]:
    if "=" not in entry:
        raise ValueError(f"Invalid override {entry!r}. Expected KEY=VALUE.")
    key, raw_value = entry.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override {entry!r}. Override key cannot be empty.")
    value = yaml.safe_load(raw_value)
    if key in PATH_LIKE_KEYS and isinstance(value, str):
        value = _expand_path_string(value)
    return key, value


def _extract_model_paths(payload: dict[str, Any], path: Path) -> dict[str, str]:
    model_paths = payload.get("model_paths") or {}
    if not isinstance(model_paths, dict):
        raise ValueError(f"'model_paths' must be a mapping in {path}")
    return {str(model_name): str(model_path) for model_name, model_path in model_paths.items()}


def _build_cli_args(parser: argparse.ArgumentParser, resolved_inputs: dict[str, Any]) -> list[str]:
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest != "help"
    }
    unknown = sorted(set(resolved_inputs) - set(action_by_dest))
    if unknown:
        raise ValueError(f"Unknown config keys for main parser: {', '.join(unknown)}")

    zero_shot_option_dests = {
        "zero_shot_tasks",
        "zero_shot_num_fewshot",
        "zero_shot_batch_size",
        "zero_shot_limit",
    }
    vlm_option_dests = {
        "vlm_datasets",
        "vlm_mode",
        "vlm_work_dir",
        "vlm_eval_kit_root",
        "vlm_judge",
        "vlm_api_nproc",
        "vlm_verbose",
        "vlm_ignore_failed",
        "vlm_pred_format",
        "vlm_use_cache",
    }

    cli_args: list[str] = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        if action.dest not in resolved_inputs:
            continue
        if action.dest in zero_shot_option_dests and not bool(resolved_inputs.get("eval_zero_shot", False)):
            continue
        if action.dest in vlm_option_dests and not bool(resolved_inputs.get("eval_vlm", False)):
            continue

        value = resolved_inputs[action.dest]
        if value is None:
            continue

        option = next((opt for opt in action.option_strings if opt.startswith("--")), action.option_strings[0])
        if isinstance(value, list):
            if not value:
                continue
            cli_args.append(option)
            cli_args.extend(str(item) for item in value)
            continue

        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        cli_args.extend([option, rendered])
    return cli_args


def _sanitize_cli_args(cli_args: list[str]) -> list[str]:
    masked: list[str] = []
    index = 0
    while index < len(cli_args):
        current = cli_args[index]
        masked.append(current)
        if current == "--hf_token" and index + 1 < len(cli_args):
            masked.append("<redacted>")
            index += 2
            continue
        index += 1
    return masked


def _select_resolved_args(
    parser: argparse.ArgumentParser,
    parsed_args: argparse.Namespace,
    resolved_inputs: dict[str, Any],
) -> dict[str, Any]:
    always_include = {
        "model_path",
        "device",
        "dtype",
        "attn_implementation",
        "seed",
        "output_dir",
        "quantization",
        "pruning",
    }
    included_keys = set(resolved_inputs) | always_include
    parsed_values = vars(parsed_args)
    selected: dict[str, Any] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        dest = action.dest
        if dest not in included_keys:
            continue
        if dest in resolved_inputs:
            value = resolved_inputs.get(dest)
        else:
            value = parsed_values.get(dest)
        if value is None:
            continue
        selected[dest] = "<redacted>" if dest in SENSITIVE_KEYS else value
    return selected


def _sanitize_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ("<redacted>" if key in SENSITIVE_KEYS and value is not None else value)
        for key, value in values.items()
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _update_artifacts_with_resolved_config(artifacts_path: Path, resolved_config_path: Path) -> None:
    if not artifacts_path.exists():
        return
    payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    payload["resolved_config_path"] = resolved_config_path.name
    artifacts_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_layered_config(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    config_root = Path(args.config_root).resolve()
    common_base_path = config_root / DEFAULT_COMMON_CONFIG
    local_config_path = Path(args.local_config)
    if not local_config_path.is_absolute():
        local_config_path = config_root / local_config_path
    algorithm_base_path = config_root / "algorithms" / args.algorithm / "base.yaml"
    model_config_path = config_root / "algorithms" / args.algorithm / "models" / f"{args.model}.yaml"

    merged: dict[str, Any] = {}
    sources: list[str] = []
    model_path_overrides: dict[str, str] = {}
    for path, allow_missing in (
        (common_base_path, False),
        (local_config_path, local_config_path == config_root / DEFAULT_LOCAL_CONFIG),
        (algorithm_base_path, False),
    ):
        payload = _load_yaml(path, allow_missing=allow_missing)
        has_content = False
        defaults = payload.get("defaults") or {}
        if defaults:
            if not isinstance(defaults, dict):
                raise ValueError(f"'defaults' must be a mapping in {path}")
            merged = _deep_merge(merged, defaults)
            has_content = True
        path_overrides = _extract_model_paths(payload, path)
        if path_overrides:
            model_path_overrides.update(path_overrides)
            has_content = True
        if has_content:
            sources.append(_relative_path(path))

    model_payload = _load_yaml(model_config_path)
    path_overrides = _extract_model_paths(model_payload, model_config_path)
    if path_overrides:
        model_path_overrides.update(path_overrides)
    model_meta = model_payload.get("model") or {}
    if not isinstance(model_meta, dict):
        raise ValueError(f"'model' must be a mapping in {model_config_path}")
    if model_meta.get("name") and model_meta["name"] != args.model:
        raise ValueError(
            f"Model config name mismatch: expected {args.model!r}, got {model_meta['name']!r} in {model_config_path}"
        )
    model_defaults = model_payload.get("defaults") or {}
    if model_defaults:
        if not isinstance(model_defaults, dict):
            raise ValueError(f"'defaults' must be a mapping in {model_config_path}")
        merged = _deep_merge(merged, model_defaults)
    recipes = model_payload.get("recipes") or {}
    if not isinstance(recipes, dict):
        raise ValueError(f"'recipes' must be a mapping in {model_config_path}")
    if args.recipe not in recipes:
        raise KeyError(f"Recipe {args.recipe!r} not found in {model_config_path}")
    recipe_payload = recipes[args.recipe] or {}
    if not isinstance(recipe_payload, dict):
        raise ValueError(f"Recipe {args.recipe!r} must resolve to a mapping in {model_config_path}")
    merged = _deep_merge(merged, recipe_payload)
    sources.append(_relative_path(model_config_path))

    model_path = model_path_overrides.get(args.model, model_meta.get("model_path"))
    if not model_path:
        raise ValueError(f"Model config must define model.model_path: {model_config_path}")
    merged["model_path"] = model_path

    quantization = merged.get("quantization")
    if quantization is None:
        merged["quantization"] = args.algorithm
    elif quantization != args.algorithm:
        raise ValueError(
            f"Config quantization={quantization!r} does not match requested algorithm={args.algorithm!r}"
        )

    overrides: dict[str, Any] = {}
    for entry in args.overrides:
        key, value = _parse_override(entry)
        overrides[key] = value
        merged[key] = value

    return merged, sources, overrides


def _build_resolved_payload(
    *,
    algorithm: str,
    model: str,
    recipe: str,
    sources: list[str],
    overrides: dict[str, Any],
    sanitized_cli_args: list[str],
    selected_resolved_args: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    return {
        "schema": RESOLVED_SCHEMA,
        "algorithm": algorithm,
        "model": model,
        "recipe": recipe,
        "config_sources": sources,
        "cli_overrides": _sanitize_mapping(overrides),
        "runtime": {
            "created_at": _timestamp(),
            "cwd": str(Path.cwd()),
            "predicted_output_dir": output_dir,
            "status": "pending",
        },
        "command": {
            "entrypoint": "main.py",
            "argv": sanitized_cli_args,
        },
        "resolved": selected_resolved_args,
    }


def _build_result_payload(parsed_args: argparse.Namespace, result, resolved_config_path: Path) -> dict[str, Any]:
    payload = {
        "model_path": result.model_path,
        "output_dir": result.output_dir,
        "metrics_path": result.metrics_path,
        "artifacts_path": result.artifacts_path,
        "resolved_config_path": str(resolved_config_path),
        "metrics": result.metrics,
    }
    if parsed_args.pruning:
        payload["pruning_algorithm"] = parsed_args.pruning
    if parsed_args.quantization:
        payload["quantization_algorithm"] = parsed_args.quantization
    return payload


def main(argv: list[str] | None = None) -> int:
    wrapper_args = _build_parser().parse_args(argv)
    resolved_inputs, sources, overrides = _resolve_layered_config(wrapper_args)

    run_parser = build_run_parser()
    cli_args = _build_cli_args(run_parser, resolved_inputs)
    parsed_args = run_parser.parse_args(cli_args)
    sanitized_cli_args = _sanitize_cli_args(cli_args)
    selected_resolved_args = _select_resolved_args(run_parser, parsed_args, resolved_inputs)
    workflow_config = build_run_config(parsed_args)
    output_dir = str(workflow_config.output_dir)
    resolved_config_path = Path(output_dir) / "resolved_config.yaml"

    resolved_payload = _build_resolved_payload(
        algorithm=wrapper_args.algorithm,
        model=wrapper_args.model,
        recipe=wrapper_args.recipe,
        sources=sources,
        overrides=overrides,
        sanitized_cli_args=sanitized_cli_args,
        selected_resolved_args=selected_resolved_args,
        output_dir=output_dir,
    )
    _write_yaml(resolved_config_path, resolved_payload)

    if wrapper_args.dry_run:
        print(json.dumps(
            {
                "resolved_config_path": str(resolved_config_path),
                "command": resolved_payload["command"],
                "resolved": resolved_payload["resolved"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    setup_logging(parsed_args.log_level)
    set_global_seed(parsed_args.seed, device=parsed_args.device)

    try:
        result = run_workflow(workflow_config)
    except Exception as exc:
        resolved_payload["runtime"]["status"] = "failed"
        resolved_payload["runtime"]["completed_at"] = _timestamp()
        resolved_payload["runtime"]["error"] = f"{exc.__class__.__name__}: {exc}"
        _write_yaml(resolved_config_path, resolved_payload)
        raise

    resolved_payload["runtime"]["status"] = "completed"
    resolved_payload["runtime"]["completed_at"] = _timestamp()
    resolved_payload["run_result"] = {
        "output_dir": result.output_dir,
        "metrics_path": result.metrics_path,
        "artifacts_path": result.artifacts_path,
    }
    _write_yaml(resolved_config_path, resolved_payload)
    _update_artifacts_with_resolved_config(Path(result.artifacts_path), resolved_config_path)

    print(json.dumps(_build_result_payload(parsed_args, result, resolved_config_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# Maintenance touch for repository metadata refresh.
