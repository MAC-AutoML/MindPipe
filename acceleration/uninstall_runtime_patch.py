#!/usr/bin/env python3
"""Remove MindPipe's acceleration patch from runtime source trees."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from install_runtime_patch import git_apply, patch_state, runtime_patches


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--vllm-ascend-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    items = runtime_patches(args.vllm_root, args.vllm_ascend_root)
    try:
        states = {item.name: patch_state(item) for item in items}
        for item in items:
            print(f"{item.name}: {states[item.name]} ({item.root})")
        if args.check:
            return 0
        removed = []
        try:
            for item in reversed(items):
                if states[item.name] == "not-applied":
                    continue
                result = git_apply(item, "--reverse")
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to remove {item.name}: {result.stderr.strip()}"
                    )
                removed.append(item)
                print(f"{item.name}: removed")
        except Exception:
            for item in reversed(removed):
                git_apply(item)
            raise
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
