#!/usr/bin/env python3
"""Queue combined workflow experiments onto currently idle GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "main.py"
DEFAULT_MODEL_PATH = "/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "workflow"
DEFAULT_LOG_ROOT = REPO_ROOT / "results" / "logs" / "workflow"
DEFAULT_QUANTIZATION_ALGORITHMS = ("gptq", "awq", "quarot", "spinquant", "flatquant")
DEFAULT_PRUNING_ALGORITHMS = ("sparsegpt", "wanda", "flap")
EXECUTION_ORDERS = (
    "quantization_then_pruning",
    "pruning_then_quantization",
)
QUANTIZATION_PROFILES: dict[str, dict[str, Any]] = {
    "gptq": {},
    "awq": {
        "awq_search": False,
    },
    "quarot": {},
    "spinquant": {},
    "flatquant": {
        "activation_bits": 4,
        "query_bits": 16,
        "key_bits": 16,
        "value_bits": 16,
    },
}
PRUNING_PROFILES: dict[str, dict[str, Any]] = {
    "sparsegpt": {},
    "wanda": {},
    "flap": {
        "sparsity_ratio": 0.2,
        "flap_metrics": "WIFV",
        "flap_remove_heads": 8,
        "pseudo_pruning": True,
    },
}
BOOLEAN_FLAG_NAMES = (
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
    "static_groups",
    "awq_search",
    "use_variant",
    "pseudo_pruning",
)


@dataclass
class JobSpec:
    quantization_algorithm: str
    pruning_algorithm: str
    execution_order: str

    @property
    def job_name(self) -> str:
        prefix = "q_then_p" if self.execution_order == "quantization_then_pruning" else "p_then_q"
        return f"{prefix}__{self.quantization_algorithm}__{self.pruning_algorithm}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch workflow experiments on idle GPUs.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--quantization_algorithms", default=",".join(DEFAULT_QUANTIZATION_ALGORITHMS))
    parser.add_argument("--pruning_algorithms", default=",".join(DEFAULT_PRUNING_ALGORITHMS))
    parser.add_argument("--gpu_pool", default="0,3,4,7")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--idle_memory_threshold", type=int, default=2048)
    parser.add_argument("--idle_utilization_threshold", type=int, default=10)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--evaluation_dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=16)
    parser.add_argument("--query_bits", type=int, default=16)
    parser.add_argument("--key_bits", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--weight_symmetric", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activation_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--key_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight_method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--structure_pattern", default="unstructured")
    parser.add_argument("--quantization_calibration_dataset", default="pileval", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--pruning_calibration_dataset", default="c4", choices=["wikitext2", "c4", "pileval"])
    parser.add_argument("--quantization_calibration_samples", type=int, default=4)
    parser.add_argument("--pruning_calibration_samples", type=int, default=128)
    parser.add_argument("--quantization_damp_percent", type=float, default=0.05)
    parser.add_argument("--pruning_damp_percent", type=float, default=0.01)
    parser.add_argument("--use_activation_order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_epochs", type=int, default=15)
    parser.add_argument("--flatquant_calibration_batch_size", type=int, default=4)
    parser.add_argument("--flatquant_lr", type=float, default=1e-5)
    parser.add_argument("--flatquant_cali_trans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_add_diag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lwc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_lac", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_diag_init", default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--flatquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--flatquant_warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_deactive_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_direct_inv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flatquant_separate_vtrans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static_groups", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--awq_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation_mode", default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_checkpoint", default=None)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--use_variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flap_metrics", default="WIFV", choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument("--flap_remove_heads", type=int, default=8)
    parser.add_argument("--pseudo_pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude_jobs", default="")
    parser.add_argument("--rerun", action="store_true")
    return parser


def parse_gpu_pool(raw_gpu_pool: str) -> list[int]:
    return [int(part.strip()) for part in raw_gpu_pool.split(",") if part.strip()]


def parse_job_filter(raw_jobs: str) -> set[str]:
    return {part.strip() for part in raw_jobs.split(",") if part.strip()}


def parse_algorithm_list(raw_algorithms: str) -> list[str]:
    return [part.strip() for part in raw_algorithms.split(",") if part.strip()]


def get_gpu_status() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    status: dict[int, dict[str, int]] = {}
    for line in completed.stdout.strip().splitlines():
        index_text, memory_text, utilization_text = [part.strip() for part in line.split(",")]
        status[int(index_text)] = {
            "memory_used": int(memory_text),
            "utilization": int(utilization_text),
        }
    return status


def is_gpu_idle(
    gpu_index: int,
    status: dict[int, dict[str, int]],
    idle_memory_threshold: int,
    idle_utilization_threshold: int,
) -> bool:
    gpu_state = status[gpu_index]
    return (
        gpu_state["memory_used"] <= idle_memory_threshold
        and gpu_state["utilization"] <= idle_utilization_threshold
    )


def resolve_job_settings(args, job: JobSpec) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "log_level": args.log_level,
        "dtype": args.dtype,
        "evaluation_dataset": args.evaluation_dataset,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "max_eval_chunks": args.max_eval_chunks,
        "seed": args.seed,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "query_bits": args.query_bits,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "group_size": args.group_size,
        "weight_symmetric": args.weight_symmetric,
        "activation_symmetric": args.activation_symmetric,
        "query_symmetric": args.query_symmetric,
        "key_symmetric": args.key_symmetric,
        "value_symmetric": args.value_symmetric,
        "weight_method": args.weight_method,
        "quantization_damp_percent": args.quantization_damp_percent,
        "pruning_damp_percent": args.pruning_damp_percent,
        "use_activation_order": args.use_activation_order,
        "flatquant_epochs": args.flatquant_epochs,
        "flatquant_calibration_batch_size": args.flatquant_calibration_batch_size,
        "flatquant_lr": args.flatquant_lr,
        "flatquant_cali_trans": args.flatquant_cali_trans,
        "flatquant_add_diag": args.flatquant_add_diag,
        "flatquant_lwc": args.flatquant_lwc,
        "flatquant_lac": args.flatquant_lac,
        "flatquant_diag_init": args.flatquant_diag_init,
        "flatquant_diag_alpha": args.flatquant_diag_alpha,
        "flatquant_warmup": args.flatquant_warmup,
        "flatquant_deactive_amp": args.flatquant_deactive_amp,
        "flatquant_direct_inv": args.flatquant_direct_inv,
        "flatquant_separate_vtrans": args.flatquant_separate_vtrans,
        "static_groups": args.static_groups,
        "awq_search": args.awq_search,
        "rotation_mode": args.rotation_mode,
        "rotation_checkpoint": args.rotation_checkpoint,
        "sparsity_ratio": args.sparsity_ratio,
        "structure_pattern": args.structure_pattern,
        "quantization_calibration_dataset": args.quantization_calibration_dataset,
        "pruning_calibration_dataset": args.pruning_calibration_dataset,
        "quantization_calibration_samples": args.quantization_calibration_samples,
        "pruning_calibration_samples": args.pruning_calibration_samples,
        "block_size": args.block_size,
        "use_variant": args.use_variant,
        "flap_metrics": args.flap_metrics,
        "flap_remove_heads": args.flap_remove_heads,
        "pseudo_pruning": args.pseudo_pruning,
    }
    settings.update(QUANTIZATION_PROFILES.get(job.quantization_algorithm, {}))
    settings.update(PRUNING_PROFILES.get(job.pruning_algorithm, {}))
    return settings


def _build_quantization_run_spec(job: JobSpec, settings: dict[str, Any]) -> str:
    run_spec = f"{job.quantization_algorithm}_w{settings['weight_bits']}a{settings['activation_bits']}"
    if job.quantization_algorithm == "flatquant":
        run_spec += f"_q{settings['query_bits']}k{settings['key_bits']}v{settings['value_bits']}"
    return run_spec


def _build_pruning_run_spec(job: JobSpec, settings: dict[str, Any]) -> str:
    run_spec = f"{job.pruning_algorithm}_s{settings['sparsity_ratio']}"
    if job.pruning_algorithm == "flap":
        run_spec += f"_{settings['flap_metrics']}_h{settings['flap_remove_heads']}"
    return run_spec


def resolve_metrics_path(args, job: JobSpec) -> Path:
    settings = resolve_job_settings(args, job)
    run_spec = (
        f"{_build_quantization_run_spec(job, settings)}"
        f"__{_build_pruning_run_spec(job, settings)}"
        f"_seq{settings['sequence_length']}"
    )
    return (
        Path(args.output_root)
        / Path(args.model_path.rstrip("/")).name
        / job.execution_order
        / f"{job.quantization_algorithm}__{job.pruning_algorithm}"
        / run_spec
        / "metrics.json"
    )


def append_boolean_flag(command: list[str], name: str, value: bool) -> None:
    command.append(f"--{name}" if value else f"--no-{name}")


def build_command(args, job: JobSpec, gpu_index: int) -> list[str]:
    settings = resolve_job_settings(args, job)
    command = [
        "conda",
        "run",
        "-n",
        "mindpipe",
        "python",
        "-u",
        str(ENTRYPOINT),
        "workflow",
        "--model_path",
        args.model_path,
        "--output_root",
        args.output_root,
        "--device",
        f"cuda:{gpu_index}",
        "--log_level",
        settings["log_level"],
        "--dtype",
        settings["dtype"],
        "--quantization_algorithm",
        job.quantization_algorithm,
        "--pruning_algorithm",
        job.pruning_algorithm,
        "--execution_order",
        job.execution_order,
        "--evaluation_dataset",
        settings["evaluation_dataset"],
        "--sequence_length",
        str(settings["sequence_length"]),
        "--batch_size",
        str(settings["batch_size"]),
        "--max_eval_chunks",
        str(settings["max_eval_chunks"]),
        "--seed",
        str(settings["seed"]),
        "--weight_bits",
        str(settings["weight_bits"]),
        "--activation_bits",
        str(settings["activation_bits"]),
        "--query_bits",
        str(settings["query_bits"]),
        "--key_bits",
        str(settings["key_bits"]),
        "--value_bits",
        str(settings["value_bits"]),
        "--group_size",
        str(settings["group_size"]),
        "--weight_method",
        settings["weight_method"],
        "--sparsity_ratio",
        str(settings["sparsity_ratio"]),
        "--structure_pattern",
        settings["structure_pattern"],
        "--quantization_calibration_dataset",
        settings["quantization_calibration_dataset"],
        "--pruning_calibration_dataset",
        settings["pruning_calibration_dataset"],
        "--quantization_calibration_samples",
        str(settings["quantization_calibration_samples"]),
        "--pruning_calibration_samples",
        str(settings["pruning_calibration_samples"]),
        "--quantization_damp_percent",
        str(settings["quantization_damp_percent"]),
        "--pruning_damp_percent",
        str(settings["pruning_damp_percent"]),
        "--flatquant_epochs",
        str(settings["flatquant_epochs"]),
        "--flatquant_calibration_batch_size",
        str(settings["flatquant_calibration_batch_size"]),
        "--flatquant_lr",
        str(settings["flatquant_lr"]),
        "--flatquant_diag_init",
        settings["flatquant_diag_init"],
        "--flatquant_diag_alpha",
        str(settings["flatquant_diag_alpha"]),
        "--rotation_mode",
        settings["rotation_mode"],
        "--block_size",
        str(settings["block_size"]),
        "--flap_metrics",
        settings["flap_metrics"],
        "--flap_remove_heads",
        str(settings["flap_remove_heads"]),
    ]
    if settings["rotation_checkpoint"]:
        command.extend(["--rotation_checkpoint", str(settings["rotation_checkpoint"])])
    for name in BOOLEAN_FLAG_NAMES:
        append_boolean_flag(command, name, bool(settings[name]))
    return command


def load_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    excluded_jobs = parse_job_filter(args.exclude_jobs)
    quantization_algorithms = parse_algorithm_list(args.quantization_algorithms)
    pruning_algorithms = parse_algorithm_list(args.pruning_algorithms)

    jobs = [
        JobSpec(quantization_algorithm=quant_algo, pruning_algorithm=prune_algo, execution_order=order)
        for quant_algo in quantization_algorithms
        for prune_algo in pruning_algorithms
        for order in EXECUTION_ORDERS
    ]
    pending_jobs: list[JobSpec] = []
    for job in jobs:
        if job.job_name in excluded_jobs:
            print(f"[exclude] {job.job_name}")
            continue
        metrics_path = resolve_metrics_path(args, job)
        if metrics_path.exists() and not args.rerun:
            print(f"[skip] {job.job_name} -> {metrics_path}")
            continue
        pending_jobs.append(job)

    if not pending_jobs:
        print("No pending jobs.")
        return 0

    gpu_pool = parse_gpu_pool(args.gpu_pool)
    running: list[dict] = []
    failures = 0

    while pending_jobs or running:
        next_running: list[dict] = []
        for record in running:
            process = record["process"]
            return_code = process.poll()
            if return_code is None:
                next_running.append(record)
                continue
            metrics = load_metrics(record["metrics_path"])
            if return_code == 0 and metrics is not None:
                perplexity = metrics.get("perplexity")
                print(
                    f"[done] {record['job'].job_name} gpu={record['gpu_index']} "
                    f"ppl={perplexity} metrics={record['metrics_path']}"
                )
            else:
                failures += 1
                print(
                    f"[fail] {record['job'].job_name} gpu={record['gpu_index']} "
                    f"exit={return_code} log={record['log_path']}"
                )
        running = next_running

        while pending_jobs and len(running) < args.max_parallel:
            status = get_gpu_status()
            free_gpu = None
            for gpu_index in gpu_pool:
                if any(record["gpu_index"] == gpu_index for record in running):
                    continue
                if is_gpu_idle(
                    gpu_index=gpu_index,
                    status=status,
                    idle_memory_threshold=args.idle_memory_threshold,
                    idle_utilization_threshold=args.idle_utilization_threshold,
                ):
                    free_gpu = gpu_index
                    break
            if free_gpu is None:
                break

            job = pending_jobs.pop(0)
            command = build_command(args, job, free_gpu)
            metrics_path = resolve_metrics_path(args, job)
            log_path = log_root / f"{job.job_name}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            running.append(
                {
                    "job": job,
                    "gpu_index": free_gpu,
                    "process": process,
                    "log_path": log_path,
                    "metrics_path": metrics_path,
                }
            )
            print(
                f"[start] {job.job_name} gpu={free_gpu} pid={process.pid} "
                f"log={log_path}"
            )

        if pending_jobs or running:
            time.sleep(args.poll_interval)

    print(f"All jobs finished. failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
