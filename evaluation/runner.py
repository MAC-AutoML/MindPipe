"""Shared evaluation runner."""

from __future__ import annotations

from evaluation.lm_eval import evaluate_zero_shot
from evaluation.ppl import evaluate_perplexity


def run_evaluations(model, tokenizer, common_args: dict) -> dict:
    metrics = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name=common_args["evaluation_dataset"],
        sequence_length=int(common_args["sequence_length"]),
        batch_size=int(common_args["batch_size"]),
        max_eval_chunks=common_args["max_eval_chunks"],
        device=common_args["device"],
    )
    if common_args.get("eval_zero_shot", False):
        metrics["zero_shot"] = evaluate_zero_shot(
            model=model,
            tokenizer=tokenizer,
            task_names=common_args["zero_shot_tasks"],
            batch_size=int(common_args["zero_shot_batch_size"]),
            device=common_args["device"],
            num_fewshot=int(common_args["zero_shot_num_fewshot"]),
            limit=common_args.get("zero_shot_limit"),
        )
    return metrics
