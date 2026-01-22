#!/usr/bin/env python
"""
Unified Algorithm Framework for LLM Compression
Supports pruning and quantization algorithms.

Usage:
    # Pruning - Unstructured (wanda)
    python main.py --task pruning --algorithm wanda --prune_method wanda --model <model_path> --sparsity_ratio 0.5

    # Pruning - Structured (flap)
    python main.py --task pruning --algorithm flap --prune_method flap --model <model_path> --pruning_ratio 0.5

    # Quantization (placeholder)
    python main.py --task quantization --method gptq --model <model_path> --bits 4
"""

import argparse
import os
import sys

# Add current directory to path
ALGORITHM_ROOT = os.path.dirname(os.path.abspath(__file__))
if ALGORITHM_ROOT not in sys.path:
    sys.path.insert(0, ALGORITHM_ROOT)


def main():
    # First, parse only the task argument to determine which submodule to use
    parser = argparse.ArgumentParser(
        description="Unified Algorithm Framework for LLM Compression",
        add_help=False  # Disable help here, let submodule handle it
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=['pruning', 'quantization'],
                        help='Task type: pruning or quantization')

    # Parse known args to get task type
    args, remaining = parser.parse_known_args()

    if args.task == 'pruning':
        # Import and run pruning main
        from pruning.main import main as pruning_main
        # Re-inject the remaining arguments
        sys.argv = [sys.argv[0]] + remaining
        pruning_main()

    elif args.task == 'quantization':
        # Import and run quantization main
        from quantization.main import main as quantization_main
        sys.argv = [sys.argv[0]] + remaining
        quantization_main()


if __name__ == '__main__':
    main()
