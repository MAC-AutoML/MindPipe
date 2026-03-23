#!/usr/bin/env python3
"""Serial resume-safe runner for official NPU quantization and pruning tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path("/home/ma-user/work/algorithm-v1")
RESULTS_ROOT = REPO_ROOT / "results"
RESULTS_NPU_ROOT = REPO_ROOT / "results-npu"
COMPARE_SCRIPT = REPO_ROOT / "tools" / "compare_results.py"
MODELS = {
    "Qwen2.5-7B-Instruct": "/home/ma-user/work/models/Qwen2.5-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "/home/ma-user/work/models/Qwen2.5-VL-7B-Instruct",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    metrics_relative_path: str
    argv: tuple[str, ...]

    @property
    def metrics_path(self) -> Path:
        return RESULTS_NPU_ROOT / self.metrics_relative_path

    @property
    def log_path(self) -> Path:
        return RESULTS_NPU_ROOT / "logs" / f"{self.task_id}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--task", action="append", default=[], help="Only run tasks whose id contains this value.")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run tasks even if metrics already exist.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--compare-after-each", action="store_true")
    parser.add_argument("--status-path", default=str(RESULTS_NPU_ROOT / "runner-status.json"))
    return parser.parse_args()


def q_model(name: str) -> str:
    return MODELS[name]


def build_quant_task(
    model_name: str,
    run_name: str,
    *,
    algorithm: str,
    dtype: str,
    calibration_dataset: str = "pileval",
    evaluation_dataset: str = "wikitext2",
    calibration_samples: int,
    sequence_length: int,
    max_eval_chunks: int,
    extra_args: Sequence[str],
) -> Task:
    metrics_relative_path = f"quantization/{model_name}/{algorithm}/{run_name}/metrics.json"
    argv = (
        "main.py",
        "quantization",
        "--algorithm",
        algorithm,
        "--model_path",
        q_model(model_name),
        "--device",
        "{device}",
        "--dtype",
        dtype,
        "--calibration_dataset",
        calibration_dataset,
        "--evaluation_dataset",
        evaluation_dataset,
        "--calibration_samples",
        str(calibration_samples),
        "--sequence_length",
        str(sequence_length),
        "--batch_size",
        "1",
        "--max_eval_chunks",
        str(max_eval_chunks),
        *extra_args,
        "--output_root",
        str(RESULTS_NPU_ROOT / "quantization"),
    )
    return Task(
        task_id=f"quant-{model_name}-{algorithm}-{run_name}",
        metrics_relative_path=metrics_relative_path,
        argv=tuple(argv),
    )


def build_prune_task(
    model_name: str,
    run_name: str,
    *,
    algorithm: str,
    dtype: str,
    calibration_dataset: str,
    evaluation_dataset: str = "wikitext2",
    calibration_samples: int,
    sequence_length: int,
    max_eval_chunks: int,
    extra_args: Sequence[str],
) -> Task:
    metrics_relative_path = f"pruning/{model_name}/{algorithm}/{run_name}/metrics.json"
    argv = (
        "main.py",
        "pruning",
        "--algorithm",
        algorithm,
        "--model_path",
        q_model(model_name),
        "--device",
        "{device}",
        "--dtype",
        dtype,
        "--calibration_dataset",
        calibration_dataset,
        "--evaluation_dataset",
        evaluation_dataset,
        "--calibration_samples",
        str(calibration_samples),
        "--sequence_length",
        str(sequence_length),
        "--batch_size",
        "1",
        "--max_eval_chunks",
        str(max_eval_chunks),
        *extra_args,
        "--output_root",
        str(RESULTS_NPU_ROOT / "pruning"),
    )
    return Task(
        task_id=f"prune-{model_name}-{algorithm}-{run_name}",
        metrics_relative_path=metrics_relative_path,
        argv=tuple(argv),
    )


def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    for model_name in MODELS:
        tasks.append(
            build_quant_task(
                model_name,
                "awq_w4a16_seq512",
                algorithm="awq",
                dtype="float16",
                calibration_samples=8,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--no-awq_search",
                    "--weight_bits",
                    "4",
                    "--group_size",
                    "128",
                    "--no-weight_symmetric",
                ),
            )
        )

    for model_name in MODELS:
        tasks.append(
            build_quant_task(
                model_name,
                "gptq_w4a16_seq512",
                algorithm="gptq",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "4",
                    "--group_size",
                    "128",
                    "--damp_percent",
                    "0.05",
                    "--no-weight_symmetric",
                ),
            )
        )

    for model_name in MODELS:
        tasks.append(
            build_quant_task(
                model_name,
                "quarot_w16a16_seq512",
                algorithm="quarot",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "16",
                    "--activation_bits",
                    "16",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "16",
                    "--value_bits",
                    "16",
                    "--group_size",
                    "-1",
                    "--weight_group_size",
                    "-1",
                    "--activation_group_size",
                    "-1",
                    "--kv_group_size",
                    "-1",
                    "--weight_method",
                    "rtn",
                ),
            )
        )
        tasks.append(
            build_quant_task(
                model_name,
                "quarot_w4a16_seq512",
                algorithm="quarot",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "4",
                    "--activation_bits",
                    "16",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "16",
                    "--value_bits",
                    "16",
                    "--group_size",
                    "-1",
                    "--weight_group_size",
                    "-1",
                    "--activation_group_size",
                    "-1",
                    "--kv_group_size",
                    "-1",
                    "--weight_method",
                    "gptq",
                ),
            )
        )
    tasks.append(
        build_quant_task(
            "Qwen2.5-7B-Instruct",
            "quarot_w4a4_seq512",
            algorithm="quarot",
            dtype="float16",
            calibration_samples=4,
            sequence_length=512,
            max_eval_chunks=64,
            extra_args=(
                "--weight_bits",
                "4",
                "--activation_bits",
                "4",
                "--query_bits",
                "16",
                "--key_bits",
                "4",
                "--value_bits",
                "4",
                "--group_size",
                "-1",
                "--weight_group_size",
                "-1",
                "--activation_group_size",
                "-1",
                "--kv_group_size",
                "-1",
                "--weight_method",
                "gptq",
            ),
        )
    )

    for model_name in MODELS:
        tasks.append(
            build_quant_task(
                model_name,
                "spinquant_w16a16_seq512",
                algorithm="spinquant",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "16",
                    "--activation_bits",
                    "16",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "16",
                    "--value_bits",
                    "16",
                    "--group_size",
                    "128",
                    "--kv_group_size",
                    "128",
                    "--weight_method",
                    "rtn",
                ),
            )
        )
        tasks.append(
            build_quant_task(
                model_name,
                "spinquant_w4a16_seq512",
                algorithm="spinquant",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "4",
                    "--activation_bits",
                    "16",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "16",
                    "--value_bits",
                    "16",
                    "--group_size",
                    "128",
                    "--kv_group_size",
                    "128",
                    "--weight_method",
                    "gptq",
                    "--no-weight_symmetric",
                ),
            )
        )
        tasks.append(
            build_quant_task(
                model_name,
                "spinquant_w4a4_seq512",
                algorithm="spinquant",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "4",
                    "--activation_bits",
                    "4",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "4",
                    "--value_bits",
                    "4",
                    "--group_size",
                    "-1",
                    "--weight_group_size",
                    "-1",
                    "--activation_group_size",
                    "-1",
                    "--kv_group_size",
                    "128",
                    "--weight_method",
                    "gptq",
                ),
            )
        )

    for model_name in MODELS:
        tasks.append(
            build_quant_task(
                model_name,
                "flatquant_w16a16_seq512",
                algorithm="flatquant",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "16",
                    "--activation_bits",
                    "16",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "16",
                    "--value_bits",
                    "16",
                    "--group_size",
                    "128",
                    "--kv_group_size",
                    "128",
                    "--weight_method",
                    "rtn",
                ),
            )
        )

    tasks.append(
        build_quant_task(
            "Qwen2.5-7B-Instruct",
            "flatquant_w4a16_seq128",
            algorithm="flatquant",
            dtype="float16",
            calibration_samples=4,
            sequence_length=128,
            max_eval_chunks=4,
            extra_args=(
                "--weight_bits",
                "4",
                "--activation_bits",
                "16",
                "--query_bits",
                "16",
                "--key_bits",
                "16",
                "--value_bits",
                "16",
                "--weight_method",
                "rtn",
                "--flatquant_lr",
                "0.005",
                "--flatquant_lwc",
                "--flatquant_lac",
                "--flatquant_cali_trans",
                "--flatquant_add_diag",
                "--no-weight_symmetric",
            ),
        )
    )

    for model_name in MODELS:
        tasks.append(
            build_quant_task(
                model_name,
                "flatquant_w4a4_q16k16v16_seq512",
                algorithm="flatquant",
                dtype="bfloat16",
                calibration_samples=2,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "4",
                    "--activation_bits",
                    "4",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "16",
                    "--value_bits",
                    "16",
                    "--weight_method",
                    "rtn",
                ),
            )
        )
        tasks.append(
            build_quant_task(
                model_name,
                "flatquant_w4a4_seq512",
                algorithm="flatquant",
                dtype="float16",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--weight_bits",
                    "4",
                    "--activation_bits",
                    "4",
                    "--query_bits",
                    "16",
                    "--key_bits",
                    "4",
                    "--value_bits",
                    "4",
                    "--kv_group_size",
                    "128",
                    "--weight_method",
                    "rtn",
                    "--flatquant_lr",
                    "0.005",
                    "--flatquant_lwc",
                    "--flatquant_lac",
                    "--flatquant_cali_trans",
                    "--flatquant_add_diag",
                ),
            )
        )

    for model_name in MODELS:
        tasks.append(
            build_prune_task(
                model_name,
                "wanda_s0.5_seq512",
                algorithm="wanda",
                dtype="float16",
                calibration_dataset="c4",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--sparsity_ratio",
                    "0.5",
                    "--structure_pattern",
                    "unstructured",
                ),
            )
        )
        tasks.append(
            build_prune_task(
                model_name,
                "sparsegpt_s0.5_seq512",
                algorithm="sparsegpt",
                dtype="float16",
                calibration_dataset="c4",
                calibration_samples=4,
                sequence_length=512,
                max_eval_chunks=64,
                extra_args=(
                    "--sparsity_ratio",
                    "0.5",
                    "--structure_pattern",
                    "unstructured",
                    "--block_size",
                    "64",
                    "--damp_percent",
                    "0.05",
                ),
            )
        )

    tasks.append(
        build_prune_task(
            "Qwen2.5-7B-Instruct",
            "flap_s0.2_seq256",
            algorithm="flap",
            dtype="bfloat16",
            calibration_dataset="wikitext2",
            calibration_samples=1,
            sequence_length=256,
            max_eval_chunks=1,
            extra_args=(
                "--sparsity_ratio",
                "0.2",
                "--structure_pattern",
                "AL-AM",
                "--flap_metrics",
                "WIFN",
                "--flap_remove_heads",
                "-1",
                "--pseudo_pruning",
            ),
        )
    )
    tasks.append(
        build_prune_task(
            "Qwen2.5-VL-7B-Instruct",
            "flap_s0.2_seq256",
            algorithm="flap",
            dtype="bfloat16",
            calibration_dataset="wikitext2",
            calibration_samples=1,
            sequence_length=256,
            max_eval_chunks=1,
            extra_args=(
                "--sparsity_ratio",
                "0.2",
                "--structure_pattern",
                "AL-AM",
                "--flap_metrics",
                "WIFN",
                "--flap_remove_heads",
                "-1",
                "--pseudo_pruning",
            ),
        )
    )
    tasks.append(
        build_prune_task(
            "Qwen2.5-7B-Instruct",
            "flap_s0.5_seq128",
            algorithm="flap",
            dtype="float16",
            calibration_dataset="c4",
            calibration_samples=4,
            sequence_length=128,
            max_eval_chunks=4,
            extra_args=(
                "--sparsity_ratio",
                "0.5",
                "--structure_pattern",
                "AL-AM",
                "--flap_metrics",
                "WIFV",
                "--flap_remove_heads",
                "8",
                "--pseudo_pruning",
            ),
        )
    )
    tasks.append(
        build_prune_task(
            "Qwen2.5-7B-Instruct",
            "flap_s0.5_seq512",
            algorithm="flap",
            dtype="bfloat16",
            calibration_dataset="c4",
            calibration_samples=4,
            sequence_length=512,
            max_eval_chunks=64,
            extra_args=(
                "--sparsity_ratio",
                "0.5",
                "--structure_pattern",
                "AL-AM",
                "--flap_metrics",
                "WIFV",
                "--flap_remove_heads",
                "-1",
                "--pseudo_pruning",
            ),
        )
    )
    tasks.append(
        build_prune_task(
            "Qwen2.5-VL-7B-Instruct",
            "flap_s0.5_seq512",
            algorithm="flap",
            dtype="bfloat16",
            calibration_dataset="c4",
            calibration_samples=4,
            sequence_length=512,
            max_eval_chunks=64,
            extra_args=(
                "--sparsity_ratio",
                "0.5",
                "--structure_pattern",
                "AL-AM",
                "--flap_metrics",
                "WIFV",
                "--flap_remove_heads",
                "-1",
                "--pseudo_pruning",
            ),
        )
    )

    return tasks


def task_matches(task: Task, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    return any(value in task.task_id for value in filters)


def task_priority(task: Task) -> tuple[int, str]:
    task_id = task.task_id
    if "flatquant_w16a16" in task_id:
        return (0, task_id)
    if "wanda" in task_id:
        return (5, task_id)
    if "flap_s0.2" in task_id:
        return (10, task_id)
    if "flap_s0.5_seq128" in task_id:
        return (12, task_id)
    if "quarot_w16a16" in task_id or "spinquant_w16a16" in task_id:
        return (15, task_id)
    if (
        "quarot_w4a16" in task_id
        or "spinquant_w4a16" in task_id
        or "flatquant_w4a16" in task_id
        or "flatquant_w4a4_q16k16v16" in task_id
        or "flatquant_w4a4_seq512" in task_id
        or "quarot_w4a4" in task_id
        or "spinquant_w4a4" in task_id
    ):
        return (20, task_id)
    if "awq" in task_id:
        return (22, task_id)
    if "flap_s0.5_seq512" in task_id:
        return (30, task_id)
    if "sparsegpt" in task_id:
        return (40, task_id)
    if "gptq" in task_id:
        return (50, task_id)
    return (25, task_id)


def metrics_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return True


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_command(task: Task, python_exe: str, device: str) -> list[str]:
    argv = [part.format(device=device) for part in task.argv]
    return [python_exe, *argv]


def run_task(task: Task, python_exe: str, device: str, status_path: Path) -> tuple[bool, int | None]:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    command = render_command(task, python_exe, device)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with task.log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{started_at}] START {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return_code = process.wait()
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        handle.write(f"[{finished_at}] END return_code={return_code}\n")
    success = return_code == 0 and metrics_exists(task.metrics_path)
    write_status(
        status_path,
        {
            "updated_at": finished_at,
            "current_task": task.task_id,
            "last_return_code": return_code,
            "last_success": success,
            "last_metrics_path": str(task.metrics_path),
            "last_log_path": str(task.log_path),
        },
    )
    return success, return_code


def run_compare(python_exe: str) -> int:
    command = [python_exe, str(COMPARE_SCRIPT)]
    return subprocess.call(command, cwd=str(REPO_ROOT))


def main() -> int:
    args = parse_args()
    all_tasks = sorted(build_tasks(), key=task_priority)
    selected = [task for task in all_tasks if task_matches(task, args.task)]
    status_path = Path(args.status_path).resolve()

    run_queue: list[Task] = []
    skipped_existing = 0
    for task in selected:
        if not args.force and metrics_exists(task.metrics_path):
            skipped_existing += 1
            continue
        run_queue.append(task)

    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": len(all_tasks),
        "selected_tasks": len(selected),
        "skipped_existing": skipped_existing,
        "queued_tasks": len(run_queue),
        "device": args.device,
    }
    write_status(status_path, summary)

    completed = 0
    failed: list[dict[str, object]] = []

    for task in run_queue:
        if args.max_tasks is not None and completed >= args.max_tasks:
            break
        success, return_code = run_task(task, args.python_exe, args.device, status_path)
        completed += 1
        if args.compare_after_each:
            run_compare(args.python_exe)
        if not success:
            failed.append(
                {
                    "task_id": task.task_id,
                    "return_code": return_code,
                    "metrics_path": str(task.metrics_path),
                    "log_path": str(task.log_path),
                }
            )
            if args.stop_on_error:
                break
        time.sleep(args.sleep_seconds)

    run_compare(args.python_exe)
    final_payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "total_tasks": len(all_tasks),
        "selected_tasks": len(selected),
        "skipped_existing": skipped_existing,
        "attempted_tasks": completed,
        "remaining_queued_tasks": max(0, len(run_queue) - completed),
        "failed_tasks": failed,
    }
    write_status(status_path, final_payload)
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
