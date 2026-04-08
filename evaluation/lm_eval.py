"""LM Eval harness helpers."""

from __future__ import annotations

import fnmatch
import time


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
    if not getattr(model, "hf_device_map", None):
        model.to(device)

    start_time = time.perf_counter()
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=resolved_tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=num_samples,
        log_samples=False,
        task_manager=task_manager,
    )["results"]
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
