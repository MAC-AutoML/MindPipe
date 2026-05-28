"""LM Eval harness helpers."""

from __future__ import annotations

import fnmatch
import time

import torch


DEFAULT_ZERO_SHOT_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
)


def _resolve_zero_shot_tasks(requested_tasks, all_tasks, lm_eval_utils):
    requested = list(requested_tasks or [])
    if not requested:
        return []
    if lm_eval_utils is not None and hasattr(lm_eval_utils, "pattern_match") and all_tasks is not None:
        return sorted(lm_eval_utils.pattern_match(requested, all_tasks))
    if all_tasks is None:
        return requested

    resolved = set()
    for pattern in requested:
        matches = fnmatch.filter(all_tasks, pattern)
        if matches:
            resolved.update(matches)
        elif pattern in all_tasks:
            resolved.add(pattern)
    return sorted(resolved)


def _select_accuracy_metric(result_payload):
    for metric_name in ("acc_norm,none", "acc,none", "exact_match,none", "exact_match"):
        metric_value = result_payload.get(metric_name)
        if metric_value is not None:
            return metric_name, float(metric_value)
    for metric_name, metric_value in result_payload.items():
        if metric_name.startswith("acc") and isinstance(metric_value, (int, float)):
            return metric_name, float(metric_value)
    return None, None


def _has_device_map(model) -> bool:
    for module in (model, getattr(model, "text_model", None), getattr(model, "model", None), getattr(model, "language_model", None)):
        if module is not None and getattr(module, "hf_device_map", None):
            return True
    return False


def _module_local_device(module):
    for param in module.parameters(recurse=False):
        if param.device.type != "meta":
            return param.device
    for buffer in module.buffers(recurse=False):
        if buffer.device.type != "meta":
            return buffer.device
    return None


def _model_devices(model) -> set[torch.device]:
    devices = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "meta":
            devices.add(tensor.device)
    return devices


def _is_multi_device_model(model) -> bool:
    return len(_model_devices(model)) > 1


def _model_output_device(model, fallback_device: torch.device) -> torch.device:
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
        if weight is not None and weight.device.type != "meta":
            return weight.device
        if output_embeddings is not None:
            try:
                param = next(output_embeddings.parameters())
                if param.device.type != "meta":
                    return param.device
            except StopIteration:
                pass
    for name in ("lm_head", "embed_out", "output"):
        module = getattr(model, name, None)
        if module is not None:
            module_device = _module_local_device(module)
            if module_device is not None:
                return module_device
    devices = list(_model_devices(model))
    return devices[-1] if devices else fallback_device


def _move_tensors_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device) if value.device != device else value
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    return value


def _register_module_device_align_hooks(model):
    handles = []

    def pre_hook(module, args, kwargs):
        module_device = _module_local_device(module)
        if module_device is None:
            return args, kwargs
        return _move_tensors_to_device(args, module_device), _move_tensors_to_device(kwargs, module_device)

    for module in model.modules():
        if _module_local_device(module) is not None:
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
    return handles


def _build_hflm_class(base_hflm_class):
    class DeviceAlignedHFLM(base_hflm_class):
        def _model_call(self, *args, **kwargs):
            logits = super()._model_call(*args, **kwargs)
            return logits.to(self.device) if logits.device != self.device else logits

    return DeviceAlignedHFLM


def evaluate_zero_shot(
    model,
    tokenizer,
    task_names,
    batch_size: int,
    device: str,
    num_fewshot: int = 0,
    num_samples: int | None = None,
):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError(
            "`lm_eval` is required for zero-shot evaluation. Install lm-evaluation-harness first."
        ) from exc

    try:
        from lm_eval import utils as lm_eval_utils
    except ImportError:
        lm_eval_utils = None

    all_tasks = None
    task_manager = None
    try:
        from lm_eval.api.registry import ALL_TASKS

        all_tasks = ALL_TASKS
    except ImportError:
        try:
            from lm_eval.tasks import TaskManager

            task_manager = TaskManager()
            all_tasks = getattr(task_manager, "all_tasks", None)
        except ImportError:
            try:
                from lm_eval import tasks as lm_eval_tasks

                all_tasks = getattr(lm_eval_tasks, "ALL_TASKS", None)
            except ImportError:
                all_tasks = None

    resolved_tasks = _resolve_zero_shot_tasks(task_names, all_tasks, lm_eval_utils)
    if not resolved_tasks:
        requested = ", ".join(task_names or [])
        raise ValueError(f"No zero-shot tasks matched: {requested}")

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    is_multi_device = _is_multi_device_model(model)
    if not is_multi_device and not _has_device_map(model):
        model.to(device)

    start_time = time.perf_counter()
    hooks = _register_module_device_align_hooks(model) if is_multi_device else []
    try:
        hflm_device = _model_output_device(model, torch.device(device)) if is_multi_device else torch.device(device)
        hflm_class = _build_hflm_class(HFLM) if is_multi_device else HFLM
        hflm = hflm_class(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=str(hflm_device))
        results = lm_eval.simple_evaluate(
            model=hflm,
            tasks=resolved_tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=num_samples,
            log_samples=False,
            task_manager=task_manager,
        )["results"]
    finally:
        for handle in hooks:
            handle.remove()
    elapsed_seconds = time.perf_counter() - start_time

    task_results = {}
    score_values = []
    for task_name in resolved_tasks:
        metric_name, metric_value = _select_accuracy_metric(results.get(task_name, {}))
        if metric_name is None:
            task_results[task_name] = {"metric": None, "value": None}
            continue
        rounded_value = round(metric_value, 4)
        task_results[task_name] = {"metric": metric_name, "value": rounded_value}
        score_values.append(rounded_value)

    acc_avg = round(sum(score_values) / len(score_values), 4) if score_values else None
    return {
        "tasks": resolved_tasks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "results": task_results,
        "acc_avg": acc_avg,
        "elapsed_seconds": elapsed_seconds,
    }
