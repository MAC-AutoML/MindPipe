"""Top-level entrypoint for algorithm-v1."""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pruning.main import main as pruning_main
    from quantization.main import main as quantization_main
    from workflow.main import main as workflow_main
else:
    from .pruning.main import main as pruning_main
    from .quantization.main import main as quantization_main
    from .workflow.main import main as workflow_main


TASKS = {
    "pruning": pruning_main,
    "quantization": quantization_main,
    "workflow": workflow_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        available = ", ".join(sorted(TASKS))
        print(
            "Usage: python /mnt/42_store/lcw/data2/Huawei/algorithm-workflow copy/main.py "
            f"<task> [args]\nTasks: {available}"
        )
        return 1
    task_name = argv[0]
    task_main = TASKS.get(task_name)
    if task_main is None:
        available = ", ".join(sorted(TASKS))
        print(f"Unknown task '{task_name}'. Available tasks: {available}")
        return 1
    return int(task_main(argv[1:]) or 0)
