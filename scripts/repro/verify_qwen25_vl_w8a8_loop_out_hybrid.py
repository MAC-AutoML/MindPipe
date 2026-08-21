#!/usr/bin/env python3
"""Verify runtime routing for the Qwen2.5-VL-7B loop-out custom op."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch
import torch_npu


HIDDEN_SIZE = 3584
CHUNKS = 4
CHUNK_SIZE = 4736
DEFAULT_SOURCE = Path(__file__).with_name(
    "aclnn_grouped_swiglu_out_bridge.cpp"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--small-tokens", type=int, default=68)
    parser.add_argument("--large-tokens", type=int, default=161415)
    parser.add_argument("--min-tokens", type=int, default=32768)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--vllm-ascend-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_weights(device: torch.device) -> dict[str, object]:
    torch.manual_seed(20260801)
    gate_up_nd = torch.randint(
        -4,
        5,
        (CHUNKS, HIDDEN_SIZE, 2 * CHUNK_SIZE),
        dtype=torch.int8,
        device=device,
    )
    batched_weight = torch_npu.npu_format_cast(gate_up_nd.contiguous(), 29)
    del gate_up_nd
    batched_scale = (
        torch.rand(
            (CHUNKS, 2 * CHUNK_SIZE),
            dtype=torch.float32,
            device=device,
        )
        + 0.01
    )
    down_nd = torch.randint(
        -4,
        5,
        (CHUNKS, CHUNK_SIZE, HIDDEN_SIZE),
        dtype=torch.int8,
        device=device,
    )
    down_weight = torch_npu.npu_format_cast(down_nd.contiguous(), 29)
    del down_nd
    down_scale = (
        torch.rand((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device) + 0.01
    )
    return {
        "batched_weight": batched_weight,
        "weight_views": [
            batched_weight[index : index + 1] for index in range(CHUNKS)
        ],
        "batched_scale": batched_scale,
        "scale_views": [
            batched_scale[index : index + 1] for index in range(CHUNKS)
        ],
        "down_weight": down_weight,
        "down_scale": down_scale,
    }


def make_activation(tokens: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.randn((tokens, HIDDEN_SIZE), dtype=torch.float16, device=device)
    quantized_x, x_scale = torch_npu.npu_dynamic_quant(x)
    del x
    return quantized_x, x_scale


def batched_gate(
    quantized_x: torch.Tensor,
    x_scale: torch.Tensor,
    weights: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = quantized_x.shape[0]
    repeated_x = quantized_x.repeat((CHUNKS, 1))
    repeated_scale = x_scale.repeat(CHUNKS)
    group_list = (
        torch.arange(
            1,
            CHUNKS + 1,
            device=quantized_x.device,
            dtype=torch.int64,
        )
        * token_count
    )
    quant_output, scale_output, _ = (
        torch_npu.npu_grouped_matmul_swiglu_quant(
            x=repeated_x,
            weight=weights["batched_weight"],
            group_list=group_list,
            weight_scale=weights["batched_scale"],
            x_scale=repeated_scale,
        )
    )
    return (
        quant_output.reshape(CHUNKS, token_count, CHUNK_SIZE),
        scale_output.reshape(CHUNKS, token_count),
    )


def reduce_down(
    gate_output: tuple[torch.Tensor, torch.Tensor],
    weights: dict[str, object],
) -> torch.Tensor:
    return torch_npu.npu_quant_matmul_reduce_sum(
        gate_output[0],
        weights["down_weight"],
        x1_scale=gate_output[1],
        x2_scale=weights["down_scale"],
    )


def capture(
    fn: Callable[[], tuple[torch.Tensor, ...]],
) -> tuple[torch.npu.NPUGraph, tuple[torch.Tensor, ...]]:
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = fn()
    torch.npu.synchronize()
    return graph, outputs


def all_equal(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]
) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right))


def main() -> int:
    args = parse_args()
    if not (args.small_tokens < args.min_tokens <= args.large_tokens):
        raise ValueError(
            "expected small_tokens < min_tokens <= large_tokens, got "
            f"{args.small_tokens}, {args.min_tokens}, {args.large_tokens}"
        )

    sys.path[:0] = [
        str(args.vllm_root.expanduser().resolve()),
        str(args.vllm_ascend_root.expanduser().resolve()),
    ]
    trace_path = args.output.with_name("routes.jsonl")
    os.environ["TASK_QUEUE_ENABLE"] = "1"
    os.environ["MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_SOURCE"] = str(
        args.source.resolve()
    )
    os.environ["MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_TRACE_PATH"] = str(
        trace_path.resolve()
    )
    os.environ["MINDPIPE_QWEN2_MLP_CHUNKED_LOOP_OUT_TRACE_LIMIT"] = "100"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.unlink(missing_ok=True)

    from vllm.model_executor.models import mindpipe_qwen2_loop_out as helper

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    op = helper.load_and_register()
    weights = make_weights(device)
    torch.npu.synchronize()

    compile_count = 0

    def backend(graph_module: torch.fx.GraphModule, example_inputs: list[object]):
        del example_inputs
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def hybrid(
        quantized_x: torch.Tensor, x_scale: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        group_list = (
            torch.ones_like(x_scale[:1], dtype=torch.int64) * x_scale.shape[0]
        )
        gate_output = op(
            quantized_x,
            weights["weight_views"],
            weights["batched_weight"],
            group_list,
            weights["scale_views"],
            weights["batched_scale"],
            x_scale,
            args.min_tokens,
            7,
        )
        return gate_output[0], gate_output[1], reduce_down(gate_output, weights)

    torch._dynamo.reset()
    compiled = torch.compile(
        hybrid,
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    cases = []
    for tokens, expected_route in (
        (args.small_tokens, "batched_repeat"),
        (args.large_tokens, "direct"),
    ):
        quantized_x, x_scale = make_activation(tokens, device)
        direct_group = torch.tensor([tokens], dtype=torch.int64, device=device)
        direct_gate = helper._BRIDGE.forward(
            quantized_x,
            weights["weight_views"],
            direct_group,
            weights["scale_views"],
            x_scale,
        )
        direct_outputs = (
            direct_gate[0],
            direct_gate[1],
            reduce_down(direct_gate, weights),
        )
        batched_output = batched_gate(quantized_x, x_scale, weights)
        batched_outputs = (
            batched_output[0],
            batched_output[1],
            reduce_down(batched_output, weights),
        )
        torch.npu.synchronize()
        references_exact = all_equal(direct_outputs, batched_outputs)

        eager_outputs = compiled(quantized_x, x_scale)
        torch.npu.synchronize()
        expected_outputs = (
            direct_outputs if expected_route == "direct" else batched_outputs
        )
        eager_exact = all_equal(eager_outputs, expected_outputs)
        del direct_outputs, batched_outputs, direct_gate, batched_output

        graph, graph_outputs = capture(lambda: compiled(quantized_x, x_scale))
        graph_capture_exact = all_equal(graph_outputs, eager_outputs)
        for output in graph_outputs:
            output.fill_(17)
        torch.npu.synchronize()
        graph_outputs_poisoned = all(
            float(output.reshape(-1)[0].item()) == 17.0
            and float(output.reshape(-1)[-1].item()) == 17.0
            for output in graph_outputs
        ) and not all_equal(graph_outputs, eager_outputs)
        graph.replay()
        torch.npu.synchronize()
        graph_exact = all_equal(graph_outputs, eager_outputs)
        cases.append(
            {
                "tokens": tokens,
                "expected_route": expected_route,
                "references_exact": references_exact,
                "eager_exact": eager_exact,
                "graph_capture_exact": graph_capture_exact,
                "graph_outputs_poisoned": graph_outputs_poisoned,
                "graph_replay_exact": graph_exact,
                "quant_shape": list(eager_outputs[0].shape),
                "scale_shape": list(eager_outputs[1].shape),
                "final_shape": list(eager_outputs[2].shape),
            }
        )
        del graph, graph_outputs, eager_outputs, expected_outputs
        del quantized_x, x_scale, direct_group
        torch.npu.empty_cache()

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_routes = {
        str(tokens): sorted(
            {
                row["route"]
                for row in trace_rows
                if int(row["tokens"]) == tokens
            }
        )
        for tokens in (args.small_tokens, args.large_tokens)
    }
    result = {
        "small_tokens": args.small_tokens,
        "large_tokens": args.large_tokens,
        "min_tokens": args.min_tokens,
        "compile_count": compile_count,
        "cases": cases,
        "observed_routes": observed_routes,
        "weight_storage_offsets": [
            int(weight.storage_offset()) for weight in weights["weight_views"]
        ],
        "trace_path": str(trace_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    valid = compile_count == 1
    valid = valid and all(
        case["references_exact"]
        and case["eager_exact"]
        and case["graph_outputs_poisoned"]
        and case["graph_replay_exact"]
        for case in cases
    )
    valid = valid and observed_routes[str(args.small_tokens)] == [
        "batched_repeat"
    ]
    valid = valid and observed_routes[str(args.large_tokens)] == ["direct"]
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
