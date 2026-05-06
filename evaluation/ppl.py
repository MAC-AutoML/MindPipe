"""Perplexity evaluation helpers."""

from __future__ import annotations

import math
import time

import torch

from algorithm.common.datasets import get_evaluation_tokens
from algorithm.common.device import resolve_device


def _forward_for_ppl(model, batch: torch.Tensor):
    outputs = model(input_ids=batch, use_cache=False)
    return outputs.logits


def _first_tensor_device(module) -> torch.device | None:
    for tensor in module.parameters(recurse=True):
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers(recurse=True):
        if tensor.device.type != "meta":
            return tensor.device
    return None


def _resolve_input_device(model, fallback_device: torch.device) -> torch.device:
    if not getattr(model, "hf_device_map", None):
        return fallback_device

    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            device = _first_tensor_device(embeddings)
            if device is not None and device.type != "cpu":
                return device

    device = _first_tensor_device(model)
    if device is not None:
        return device
    return fallback_device


@torch.inference_mode()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    batch_size: int,
    max_eval_chunks: int | None,
    device: str,
    data_path: str = None,
):
    resolved_device = resolve_device(device)
    evaluation_tokens = get_evaluation_tokens(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sequence_length=sequence_length,
        seed=0,
        data_path=data_path,
    )
    token_ids = evaluation_tokens.input_ids
    total_chunks = token_ids.numel() // sequence_length
    if max_eval_chunks is not None:
        total_chunks = min(total_chunks, max_eval_chunks)

    # Free cached memory from pruning/calibration before evaluation
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    if not getattr(model, "hf_device_map", None):
        model.to(resolved_device)
    input_device = _resolve_input_device(model, resolved_device)
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
        ].to(input_device)
        batch = batch.reshape(chunk_end - chunk_start, sequence_length)
        logits = _forward_for_ppl(model, batch)
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(shift_logits.device).contiguous()
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
