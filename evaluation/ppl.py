"""Perplexity evaluation helpers."""

from __future__ import annotations

import math
import time

import torch

from algorithm.common.datasets import get_evaluation_tokens


@torch.inference_mode()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    batch_size: int,
    max_eval_chunks: int | None,
    device: str,
):
    evaluation_tokens = get_evaluation_tokens(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sequence_length=sequence_length,
        seed=0,
    )
    token_ids = evaluation_tokens.input_ids
    total_chunks = token_ids.numel() // sequence_length
    if max_eval_chunks is not None:
        total_chunks = min(total_chunks, max_eval_chunks)

    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    total_nll = 0.0
    total_tokens = 0
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    start_time = time.perf_counter()
    for chunk_start in range(0, total_chunks, batch_size):
        chunk_end = min(chunk_start + batch_size, total_chunks)
        batch = token_ids[
            :,
            chunk_start * sequence_length : chunk_end * sequence_length,
        ].to(device)
        batch = batch.reshape(chunk_end - chunk_start, sequence_length)
        outputs = model(input_ids=batch, use_cache=False)
        logits = torch.nan_to_num(outputs.logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        valid_tokens = (sequence_length - 1) * (chunk_end - chunk_start)
        total_nll += float(
            loss_function(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
        )
        total_tokens += valid_tokens

    elapsed_seconds = time.perf_counter() - start_time
    perplexity = math.exp(total_nll / max(total_tokens, 1))
    return {
        "perplexity": perplexity,
        "evaluation_dataset": dataset_name,
        "sequence_length": sequence_length,
        "evaluated_chunks": total_chunks,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": total_tokens / max(elapsed_seconds, 1e-6),
    }
