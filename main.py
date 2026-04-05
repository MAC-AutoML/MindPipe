"""Top-level entrypoint for mindpipe."""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import json
import sys

from algorithm.common.logging import setup_logging
from algorithm.common.reproducibility import set_global_seed
from workflow.builder import build_run_parser
from workflow.builder import build_run_config
from workflow.executor import run_workflow

def _build_result_payload(args, result) -> dict:
    payload = {
        "model_path": result.model_path,
        "output_dir": result.output_dir,
        "metrics_path": result.metrics_path,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }
    if args.pruning:
        payload["pruning_algorithm"] = args.pruning
    if args.quantization:
        payload["quantization_algorithm"] = args.quantization
    return payload


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    args = build_run_parser().parse_args(argv)
    setup_logging(args.log_level)
    set_global_seed(args.seed, device=args.device)
    result = run_workflow(build_run_config(args))
    print(json.dumps(_build_result_payload(args, result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
