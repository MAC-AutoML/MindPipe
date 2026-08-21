#!/usr/bin/env python3
"""Gate a paired campaign by the ratio of mean token throughputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_run(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    throughput = data.get("total_token_throughput")
    if not isinstance(throughput, (int, float)) or not math.isfinite(throughput):
        raise ValueError(f"Invalid total_token_throughput in {resolved}")
    if throughput <= 0:
        raise ValueError(f"Non-positive total_token_throughput in {resolved}")
    completed = data.get("completed")
    failed = data.get("failed")
    expected = data.get("num_prompts")
    if not isinstance(completed, int) or completed <= 0 or failed != 0:
        raise ValueError(f"Incomplete benchmark in {resolved}")
    if isinstance(expected, int) and completed != expected:
        raise ValueError(f"Completed/request mismatch in {resolved}")
    if data.get("diagnostic_only") is True:
        raise ValueError(f"Diagnostic result cannot enter acceptance: {resolved}")
    return {
        "path": str(resolved),
        "throughput": float(throughput),
        "completed": completed,
        "failed": failed,
    }


def main() -> int:
    args = parse_args()
    if len(args.control) != len(args.candidate):
        raise ValueError("Control and candidate run counts must match")
    controls = [read_run(path) for path in args.control]
    candidates = [read_run(path) for path in args.candidate]
    mean_control = sum(run["throughput"] for run in controls) / len(controls)
    mean_candidate = sum(run["throughput"] for run in candidates) / len(candidates)
    speedup = mean_candidate / mean_control
    pair_speedups = [
        candidate["throughput"] / control["throughput"]
        for control, candidate in zip(controls, candidates)
    ]
    passed = speedup >= args.minimum_speedup
    report = {
        "model": args.model,
        "valid": True,
        "decision": "PASS" if passed else "FAIL",
        "minimum_speedup": args.minimum_speedup,
        "formula": "mean(candidate total_token_throughput) / mean(control total_token_throughput)",
        "rounding_policy": "unrounded comparison",
        "speedup_ratio_of_means": speedup,
        "mean_control_total_token_throughput": mean_control,
        "mean_candidate_total_token_throughput": mean_candidate,
        "pair_speedups": pair_speedups,
        "controls": controls,
        "candidates": candidates,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
