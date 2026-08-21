#!/usr/bin/env python3
"""Install MindPipe's four-model acceleration into compatible runtime trees."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATCH_DIR = ROOT / "runtime_patches"


@dataclass(frozen=True)
class RuntimePatch:
    name: str
    root: Path
    patch: Path
    anchors: tuple[tuple[str, str], ...]
    version_file: str
    version_pattern: str


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--vllm-ascend-root", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate compatibility and patch applicability without writing.",
    )
    return parser.parse_args()


def runtime_patches(vllm_root: Path, ascend_root: Path) -> tuple[RuntimePatch, ...]:
    return (
        RuntimePatch(
            name="vLLM",
            root=vllm_root.expanduser().resolve(),
            patch=PATCH_DIR / "vllm.patch",
            anchors=(
                ("vllm/model_executor/models/qwen2.py", "class Qwen2MLP"),
                (
                    "vllm/model_executor/models/qwen3_moe.py",
                    "class Qwen3MoeModel",
                ),
                ("vllm/v1/engine/core.py", "class EngineCore"),
            ),
            version_file="vllm/_version.py",
            version_pattern=r"0\.11\.0rc[0-9]+",
        ),
        RuntimePatch(
            name="vLLM-Ascend",
            root=ascend_root.expanduser().resolve(),
            patch=PATCH_DIR / "vllm_ascend.patch",
            anchors=(
                (
                    "vllm_ascend/quantization/w8a8_dynamic.py",
                    "class AscendW8A8DynamicLinearMethod",
                ),
                ("vllm_ascend/worker/model_runner_v1.py", "class NPUModelRunner"),
                ("vllm_ascend/ops/moe/token_dispatcher.py", "class MoETokenDispatcher"),
            ),
            version_file="vllm_ascend/_version.py",
            version_pattern=r"0\.11\.0rc[0-9]+",
        ),
    )


def validate_layout(item: RuntimePatch) -> None:
    if not item.root.is_dir():
        raise RuntimeError(f"{item.name} root is not a directory: {item.root}")
    if not item.patch.is_file():
        raise RuntimeError(f"Missing packaged patch: {item.patch}")
    for relative, anchor in item.anchors:
        path = item.root / relative
        if not path.is_file():
            raise RuntimeError(f"{item.name} required file is absent: {path}")
        if anchor not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"{item.name} API anchor {anchor!r} is absent from {path}")
    version_path = item.root / item.version_file
    if version_path.is_file():
        text = version_path.read_text(encoding="utf-8")
        if re.search(item.version_pattern, text) is None:
            raise RuntimeError(
                f"{item.name} generated version is outside the supported 0.11.0rc line: "
                f"{version_path}"
            )


def git_apply(item: RuntimePatch, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *options, str(item.patch)],
        cwd=item.root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def patch_state(item: RuntimePatch) -> str:
    if git_apply(item, "--check").returncode == 0:
        return "not-applied"
    if git_apply(item, "--reverse", "--check").returncode == 0:
        return "applied"
    forward = git_apply(item, "--check").stderr.strip()
    reverse = git_apply(item, "--reverse", "--check").stderr.strip()
    raise RuntimeError(
        f"{item.name} is neither a compatible clean tree nor an already patched tree.\n"
        f"Forward check: {forward}\nReverse check: {reverse}"
    )


def install(items: tuple[RuntimePatch, ...], check_only: bool) -> None:
    states: dict[str, str] = {}
    for item in items:
        validate_layout(item)
        states[item.name] = patch_state(item)
        print(f"{item.name}: {states[item.name]} ({item.root})")
    if check_only:
        return

    applied: list[RuntimePatch] = []
    try:
        for item in items:
            if states[item.name] == "applied":
                continue
            result = git_apply(item)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to apply {item.name}: {result.stderr.strip()}")
            applied.append(item)
            print(f"{item.name}: installed")
    except Exception:
        for item in reversed(applied):
            git_apply(item, "--reverse")
        raise


def main() -> int:
    args = _args()
    try:
        install(runtime_patches(args.vllm_root, args.vllm_ascend_root), args.check)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
