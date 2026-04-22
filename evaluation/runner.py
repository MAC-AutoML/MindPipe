"""Shared evaluation runner."""

from __future__ import annotations

from evaluation.vlm_eval import evaluate_vlm
from evaluation.lm_eval import evaluate_zero_shot
from evaluation.ppl import evaluate_perplexity


def run_evaluations(model, tokenizer=None, tokenizer_bundle=None, common_args: dict | None = None) -> dict:
    common_args = {} if common_args is None else common_args
    if tokenizer_bundle is None:
        tokenizer_bundle = tokenizer
    resolved_tokenizer = getattr(tokenizer_bundle, "tokenizer", tokenizer_bundle)
    metrics = {}
    zero_shot_num_samples = common_args.get("num_samples")
    save_callback = common_args.get("evaluation_save_callback")
    if common_args.get("eval_ppl", True):
        metrics = evaluate_perplexity(
            model=model,
            tokenizer=resolved_tokenizer,
            dataset_name=common_args["evaluation_dataset"],
            sequence_length=int(common_args["sequence_length"]),
            batch_size=int(common_args["batch_size"]),
            max_eval_chunks=common_args["max_eval_chunks"],
            device=common_args["device"],
            data_path=common_args.get("data_path"),
        )
        if save_callback is not None:
            save_callback(metrics)
    if common_args.get("eval_zero_shot", False):
        metrics["zero_shot"] = evaluate_zero_shot(
            model=model,
            tokenizer=resolved_tokenizer,
            task_names=common_args["zero_shot_tasks"],
            batch_size=int(common_args["zero_shot_batch_size"]),
            device=common_args["device"],
            num_fewshot=int(common_args["zero_shot_num_fewshot"]),
            num_samples=zero_shot_num_samples,
        )
        if save_callback is not None:
            save_callback(metrics)
    if common_args.get("eval_vlm", False):
        metrics["vlm_eval"] = evaluate_vlm(
            model=model,
            tokenizer_bundle=tokenizer_bundle,
            common_args=common_args,
        )
        if save_callback is not None:
            save_callback(metrics)
    return metrics
