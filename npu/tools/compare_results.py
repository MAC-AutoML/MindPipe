#!/usr/bin/env python3
"""Compare official GPU and NPU metrics and regenerate NPU summaries."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricEntry:
    relative_metrics_path: str
    category: str
    model: str
    algorithm: str
    run_name: str
    perplexity: float | None
    sequence_length: int | None
    evaluated_chunks: int | None
    batch_size: int | None
    dtype: str | None
    device: str | None
    evaluation_dataset: str | None
    sparsity_ratio: float | None
    metrics_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-root",
        default="/home/ma-user/work/algorithm-v1/results",
        help="Root directory for official GPU results.",
    )
    parser.add_argument(
        "--npu-root",
        default="/home/ma-user/work/algorithm-v1/results-npu",
        help="Root directory for official NPU results.",
    )
    return parser.parse_args()


def _is_probe_path(path: Path) -> bool:
    return any(part.endswith("-probe") for part in path.parts)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return value
    return value


def scan_metrics(root: Path) -> dict[str, MetricEntry]:
    metrics: dict[str, MetricEntry] = {}
    for path in sorted(root.rglob("metrics.json")):
        if _is_probe_path(path):
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 4:
            continue
        payload = _load_json(path)
        metrics[str(relative)] = MetricEntry(
            relative_metrics_path=str(relative),
            category=relative.parts[0],
            model=relative.parts[1],
            algorithm=relative.parts[2],
            run_name=relative.parts[3],
            perplexity=_finite_or_none(payload.get("perplexity")),
            sequence_length=payload.get("sequence_length"),
            evaluated_chunks=payload.get("evaluated_chunks"),
            batch_size=payload.get("batch_size"),
            dtype=payload.get("dtype"),
            device=payload.get("device"),
            evaluation_dataset=payload.get("evaluation_dataset"),
            sparsity_ratio=payload.get("sparsity_ratio"),
            metrics_path=str(path),
        )
    return metrics


def build_npu_summary(npu_root: Path, metrics: dict[str, MetricEntry]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "official_metrics": len(metrics),
            "quantization": sum(1 for item in metrics.values() if item.category == "quantization"),
            "pruning": sum(1 for item in metrics.values() if item.category == "pruning"),
        },
        "quantization": {},
        "pruning": {},
    }
    for relative_path, item in sorted(metrics.items()):
        bucket = summary[item.category].setdefault(item.model, {})
        bucket[item.run_name] = {
            "perplexity": item.perplexity,
            "sequence_length": item.sequence_length,
            "evaluated_chunks": item.evaluated_chunks,
            "dtype": item.dtype,
            "metrics_path": str(Path("results-npu") / relative_path),
        }
    return summary


def compare_metrics(gpu: dict[str, MetricEntry], npu: dict[str, MetricEntry]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_on_npu = 0
    comparable = 0
    nonfinite_pairs = 0

    for relative_path in sorted(gpu):
        gpu_item = gpu[relative_path]
        npu_item = npu.get(relative_path)
        row: dict[str, Any] = {
            "relative_metrics_path": relative_path,
            "category": gpu_item.category,
            "model": gpu_item.model,
            "algorithm": gpu_item.algorithm,
            "run_name": gpu_item.run_name,
            "gpu": {
                "perplexity": gpu_item.perplexity,
                "dtype": gpu_item.dtype,
                "sequence_length": gpu_item.sequence_length,
                "evaluated_chunks": gpu_item.evaluated_chunks,
                "evaluation_dataset": gpu_item.evaluation_dataset,
                "sparsity_ratio": gpu_item.sparsity_ratio,
                "metrics_path": gpu_item.metrics_path,
            },
            "npu": None,
            "status": "missing_on_npu",
            "delta": None,
            "delta_percent": None,
        }
        if npu_item is None:
            missing_on_npu += 1
            rows.append(row)
            continue

        row["npu"] = {
            "perplexity": npu_item.perplexity,
            "dtype": npu_item.dtype,
            "sequence_length": npu_item.sequence_length,
            "evaluated_chunks": npu_item.evaluated_chunks,
            "evaluation_dataset": npu_item.evaluation_dataset,
            "sparsity_ratio": npu_item.sparsity_ratio,
            "metrics_path": npu_item.metrics_path,
        }

        gpu_ppl = gpu_item.perplexity
        npu_ppl = npu_item.perplexity
        if (
            isinstance(gpu_ppl, float)
            and isinstance(npu_ppl, float)
            and math.isfinite(gpu_ppl)
            and math.isfinite(npu_ppl)
        ):
            delta = npu_ppl - gpu_ppl
            row["status"] = "compared"
            row["delta"] = delta
            row["delta_percent"] = None if gpu_ppl == 0 else delta / gpu_ppl * 100.0
            comparable += 1
        else:
            row["status"] = "nonfinite_pair"
            nonfinite_pairs += 1
        rows.append(row)

    extras = [
        {
            "relative_metrics_path": relative_path,
            "category": item.category,
            "model": item.model,
            "algorithm": item.algorithm,
            "run_name": item.run_name,
            "perplexity": item.perplexity,
            "metrics_path": item.metrics_path,
        }
        for relative_path, item in sorted(npu.items())
        if relative_path not in gpu
    ]

    rows.sort(
        key=lambda row: (
            0 if row["status"] == "missing_on_npu" else 1,
            0 if row["status"] == "nonfinite_pair" else 1,
            -abs(row["delta"]) if isinstance(row["delta"], float) else -1.0,
            row["relative_metrics_path"],
        )
    )

    return {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "gpu_official_metrics": len(gpu),
            "npu_official_metrics": len(npu),
            "matched_paths": len(gpu) - missing_on_npu,
            "missing_on_npu": missing_on_npu,
            "comparable_finite_pairs": comparable,
            "nonfinite_pairs": nonfinite_pairs,
            "extra_on_npu": len(extras),
        },
        "rows": rows,
        "extras_on_npu": extras,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GPU vs NPU Compare",
        "",
        f"- Updated at: `{payload['updated_at']}`",
        f"- GPU official metrics: `{payload['counts']['gpu_official_metrics']}`",
        f"- NPU official metrics: `{payload['counts']['npu_official_metrics']}`",
        f"- Matched paths: `{payload['counts']['matched_paths']}`",
        f"- Missing on NPU: `{payload['counts']['missing_on_npu']}`",
        f"- Comparable finite pairs: `{payload['counts']['comparable_finite_pairs']}`",
        f"- Non-finite pairs: `{payload['counts']['nonfinite_pairs']}`",
        f"- Extra on NPU: `{payload['counts']['extra_on_npu']}`",
        "",
        "| Status | Path | GPU PPL | NPU PPL | Delta | Delta % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        gpu_ppl = row["gpu"]["perplexity"]
        npu_ppl = None if row["npu"] is None else row["npu"]["perplexity"]
        delta = row["delta"]
        delta_pct = row["delta_percent"]
        lines.append(
            "| {status} | `{path}` | {gpu} | {npu} | {delta} | {delta_pct} |".format(
                status=row["status"],
                path=row["relative_metrics_path"],
                gpu=format_number(gpu_ppl),
                npu=format_number(npu_ppl),
                delta=format_number(delta),
                delta_pct=format_percent(delta_pct),
            )
        )
    if payload["extras_on_npu"]:
        lines.extend(["", "## Extra On NPU", ""])
        for item in payload["extras_on_npu"]:
            lines.append(
                f"- `{item['relative_metrics_path']}`: ppl={format_number(item['perplexity'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{value:.6f}"
    return str(value)


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{value:.2f}%"
    return str(value)


def main() -> int:
    args = parse_args()
    gpu_root = Path(args.gpu_root).resolve()
    npu_root = Path(args.npu_root).resolve()

    gpu_metrics = scan_metrics(gpu_root)
    npu_metrics = scan_metrics(npu_root)

    summary = build_npu_summary(npu_root, npu_metrics)
    compare = compare_metrics(gpu_metrics, npu_metrics)

    summary_path = npu_root / "summary.json"
    compare_json_path = npu_root / "compare-vs-gpu.json"
    compare_md_path = npu_root / "compare-vs-gpu.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compare_json_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(compare_md_path, compare)

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "compare_json_path": str(compare_json_path),
                "compare_md_path": str(compare_md_path),
                "counts": compare["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
