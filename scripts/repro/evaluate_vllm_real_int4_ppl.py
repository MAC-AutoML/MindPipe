#!/usr/bin/env python3
"""Evaluate PPL through vLLM on a real int4 compressed-tensors export."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from algorithm.common.datasets import get_evaluation_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Self-contained vLLM model export directory."
    )
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument(
        "--data_path", default="/mnt/42_store/lcw/data2/Huawei/datasets"
    )
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_chunks", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--runtime_quantization", default="compressed-tensors/int4")
    parser.add_argument(
        "--force_contiguous_int4_input",
        action="store_true",
        help="Diagnostic option: make vLLM WNA16 kernel inputs contiguous before GEMM.",
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _patch_vllm_int4_contiguous_inputs() -> None:
    from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
        ExllamaLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )

    if not getattr(
        ExllamaLinearKernel.apply_weights, "_mindpipe_contiguous_patch", False
    ):
        original_exllama_apply = ExllamaLinearKernel.apply_weights

        def exllama_apply_contiguous(self, layer, x, bias=None):
            return original_exllama_apply(self, layer, x.contiguous(), bias)

        exllama_apply_contiguous._mindpipe_contiguous_patch = True
        ExllamaLinearKernel.apply_weights = exllama_apply_contiguous

    if not getattr(
        MarlinLinearKernel.apply_weights, "_mindpipe_contiguous_patch", False
    ):
        original_marlin_apply = MarlinLinearKernel.apply_weights

        def marlin_apply_contiguous(self, layer, x, bias=None):
            return original_marlin_apply(self, layer, x.contiguous(), bias)

        marlin_apply_contiguous._mindpipe_contiguous_patch = True
        MarlinLinearKernel.apply_weights = marlin_apply_contiguous


def _token_logprob(token_id: int, prompt_logprob_entry) -> float:
    if prompt_logprob_entry is None:
        raise ValueError("Missing prompt logprob entry for a non-prefix token.")
    if token_id in prompt_logprob_entry:
        return float(prompt_logprob_entry[token_id].logprob)
    available = ", ".join(str(key) for key in list(prompt_logprob_entry)[:8])
    raise KeyError(
        f"Token id {token_id} not found in prompt logprobs; available keys: {available}"
    )


def main() -> int:
    args = parse_args()
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
    except ImportError as error:
        raise RuntimeError(
            "evaluate_vllm_real_int4_ppl.py requires vLLM in the active environment."
        ) from error

    if args.force_contiguous_int4_input:
        _patch_vllm_int4_contiguous_inputs()

    model_path = str(Path(args.model).expanduser())
    max_model_len = args.max_model_len or (int(args.sequence_length) + 1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    evaluation_tokens = get_evaluation_tokens(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        sequence_length=int(args.sequence_length),
        seed=0,
        data_path=args.data_path,
    )
    token_ids = evaluation_tokens.input_ids.reshape(-1)
    total_chunks = int(token_ids.numel()) // int(args.sequence_length)
    if args.max_eval_chunks is not None:
        total_chunks = min(total_chunks, int(args.max_eval_chunks))

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        enforce_eager=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=max_model_len,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
        detokenize=False,
    )

    total_nll = 0.0
    total_scored_tokens = 0
    start_time = time.perf_counter()
    for chunk_start in range(0, total_chunks, int(args.batch_size)):
        chunk_end = min(chunk_start + int(args.batch_size), total_chunks)
        chunk_token_ids: list[list[int]] = []
        prompts = []
        for chunk_idx in range(chunk_start, chunk_end):
            start = chunk_idx * int(args.sequence_length)
            end = start + int(args.sequence_length)
            ids = token_ids[start:end].tolist()
            chunk_token_ids.append(ids)
            prompts.append(TokensPrompt(prompt_token_ids=ids))

        outputs = llm.generate(prompts, sampling_params)
        for ids, output in zip(chunk_token_ids, outputs, strict=True):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("vLLM did not return prompt_logprobs.")
            if len(prompt_logprobs) != len(ids):
                raise RuntimeError(
                    f"prompt_logprobs length {len(prompt_logprobs)} does not match prompt length {len(ids)}."
                )
            for pos in range(1, len(ids)):
                total_nll -= _token_logprob(int(ids[pos]), prompt_logprobs[pos])
            total_scored_tokens += max(len(ids) - 1, 0)

    elapsed_seconds = time.perf_counter() - start_time
    ppl = math.exp(total_nll / max(total_scored_tokens, 1))
    result = {
        "perplexity": ppl,
        "evaluation_dataset": args.dataset,
        "sequence_length": int(args.sequence_length),
        "evaluated_chunks": total_chunks,
        "batch_size": int(args.batch_size),
        "total_scored_tokens": total_scored_tokens,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": total_scored_tokens / max(elapsed_seconds, 1e-6),
        "model": model_path,
        "backend": "vllm",
        "runtime_quantization": args.runtime_quantization,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
