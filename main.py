"""Top-level entrypoint for mindpipe."""

from __future__ import annotations

import json
import sys

from algorithm.common.logging import setup_logging
from algorithm.common.reproducibility import set_global_seed
from workflow.builder import build_pruning_config
from workflow.builder import build_pruning_parser
from workflow.builder import build_quantization_config
from workflow.builder import build_quantization_parser
from workflow.builder import build_workflow_config
from workflow.builder import build_workflow_parser
from workflow.executor import run_workflow


TASKS = {
    "pruning": {
        "parser_builder": build_pruning_parser,
        "config_builder": build_pruning_config,
    },
    "quantization": {
        "parser_builder": build_quantization_parser,
        "config_builder": build_quantization_config,
    },
    "workflow": {
        "parser_builder": build_workflow_parser,
        "config_builder": build_workflow_config,
    },
}


def _build_result_payload(task_name: str, args, result) -> dict:
    payload = {
        "model_path": result.model_path,
        "output_dir": result.output_dir,
        "metrics_path": result.metrics_path,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }
    if task_name in {"quantization", "pruning"}:
        payload["algorithm_name"] = args.algorithm
    return payload


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        available = ", ".join(sorted(TASKS))
        print(f"Usage: python main.py <task> [args]\nTasks: {available}")
        return 1

    task_name = argv[0]
    task = TASKS.get(task_name)
    if task is None:
        available = ", ".join(sorted(TASKS))
        print(f"Unknown task '{task_name}'. Available tasks: {available}")
        return 1

    args = task["parser_builder"]().parse_args(argv[1:])
    setup_logging(args.log_level)
    set_global_seed(args.seed, device=args.device)
    result = run_workflow(task["config_builder"](args))
    print(json.dumps(_build_result_payload(task_name, args, result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
