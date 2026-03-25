"""Shared quantization argument helpers."""

from __future__ import annotations


def normalize_args(args):
    if args.weight_group_size is None:
        args.weight_group_size = args.group_size
    if args.activation_group_size is None:
        args.activation_group_size = args.group_size
    if args.kv_group_size is None:
        args.kv_group_size = args.group_size
    return args
