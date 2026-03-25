#!/usr/bin/env python3
"""Continuously monitor workflow experiments and write status snapshots."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "workflow"
DEFAULT_STATUS_PATH = REPO_ROOT / "results" / "workflow_status.json"
DEFAULT_LOG_PATH = REPO_ROOT / "results" / "logs" / "workflow" / "monitor.log"
DEFAULT_MODEL_PATH = "/mnt/82_store/LLM-weights/Qwen2.5-VL-7B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor workflow experiments.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--status_path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--log_path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def collect_gpu_status() -> list[dict]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    statuses = []
    for line in output.strip().splitlines():
        index_text, memory_text, utilization_text = [part.strip() for part in line.split(",")]
        statuses.append(
            {
                "index": int(index_text),
                "memory_used_mib": int(memory_text),
                "utilization_gpu_percent": int(utilization_text),
            }
        )
    return statuses


def collect_running_jobs(model_path: str) -> list[dict]:
    pattern = f"{REPO_ROOT / 'main.py'} workflow --model_path {model_path}"
    completed = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    jobs = []
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        pid_text, command = line.split(" ", maxsplit=1)
        if "conda run -n mindpipe" in command:
            continue
        jobs.append({"pid": int(pid_text), "command": command})
    return jobs


def collect_finished_metrics(results_root: Path) -> list[dict]:
    metrics = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.append(
            {
                "metrics_path": str(metrics_path),
                "execution_order": payload.get("execution_order"),
                "quantization_algorithm": payload.get("quantization_algorithm"),
                "pruning_algorithm": payload.get("pruning_algorithm"),
                "perplexity": payload.get("perplexity"),
            }
        )
    return metrics


def build_snapshot(results_root: Path, model_path: str) -> dict:
    return {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results_root": str(results_root),
        "model_path": model_path,
        "running_jobs": collect_running_jobs(model_path),
        "gpu_status": collect_gpu_status(),
        "finished_metrics": collect_finished_metrics(results_root),
    }


def append_log(log_path: Path, snapshot: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    finished = len(snapshot["finished_metrics"])
    running = len(snapshot["running_jobs"])
    gpu_brief = ", ".join(
        f"cuda:{item['index']} mem={item['memory_used_mib']} util={item['utilization_gpu_percent']}"
        for item in snapshot["gpu_status"]
    )
    lines = [
        f"[{snapshot['updated_at']}] finished={finished} running={running}",
        gpu_brief,
    ]
    for item in snapshot["finished_metrics"]:
        combo = f"{item['execution_order']}::{item['quantization_algorithm']}__{item['pruning_algorithm']}"
        lines.append(f"done {combo} ppl={item['perplexity']}")
    for item in snapshot["running_jobs"]:
        lines.append(f"run pid={item['pid']} {item['command']}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    status_path = Path(args.status_path)
    log_path = Path(args.log_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = build_snapshot(results_root, args.model_path)
        status_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(log_path, snapshot)
        if args.once:
            break
        time.sleep(args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
