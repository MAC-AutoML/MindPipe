#!/usr/bin/env python3
"""Fail closed when a MindPipe acceleration delivery contains unsafe files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_MAX_TOTAL_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_FILE_BYTES = 1024 * 1024
LOCAL_HOME = b"/home/" + b"ma-user"
FORBIDDEN_DIRS = {
    "checkpoints",
    "extra-info",
    "kernel_meta",
    "logs",
    "models",
    "my_results",
    "new_results",
    "outputs",
    "profiles",
    "profiler",
    "results",
    "tokenizer",
    "tokenizers",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".gz",
    ".h5",
    ".hdf5",
    ".log",
    ".model",
    ".npy",
    ".npz",
    ".onnx",
    ".prof",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".trace",
    ".zip",
}
FORBIDDEN_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "profiler_info.json",
    "spiece.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "assigned secret": re.compile(
        rb"(?i)\b(?:access[_-]?key|secret[_-]?key|api[_-]?key)\s*=\s*['\"]?[^\s'\"]+"
    ),
}


class AuditError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _nul_paths(payload: bytes) -> set[Path]:
    return {Path(item.decode("utf-8")) for item in payload.split(b"\0") if item}


def collect_changed_path_sources(repo: Path, base_ref: str) -> dict[Path, set[str]]:
    sources: dict[Path, set[str]] = {}
    committed = _nul_paths(
        _git(repo, "diff", "--name-only", "-z", f"{base_ref}...HEAD")
    )
    unstaged = _nul_paths(_git(repo, "diff", "--name-only", "-z"))
    untracked = _nul_paths(
        _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    )
    staged = _nul_paths(
        _git(repo, "diff", "--cached", "--name-only", "-z")
    )

    for path in committed:
        sources.setdefault(path, set()).add("head")
    for path in unstaged | untracked:
        sources.setdefault(path, set()).add("worktree")
    for path in staged:
        sources.setdefault(path, set()).add("index")
    return sources


def collect_changed_paths(repo: Path, base_ref: str) -> list[Path]:
    return sorted(
        collect_changed_path_sources(repo, base_ref),
        key=lambda path: path.as_posix(),
    )


def _audit_payload(
    relative: Path,
    payload: bytes | None,
    size: int,
    max_file_bytes: int,
) -> tuple[int, list[str]]:
    errors: list[str] = []

    parts = {part.lower() for part in relative.parts}
    if parts & FORBIDDEN_DIRS:
        errors.append(f"{relative}: forbidden artifact directory")
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"{relative}: forbidden artifact suffix")
    name = relative.name.lower()
    if (
        name in FORBIDDEN_NAMES
        or name.startswith("tokenizer.model")
        or (name.startswith("sentencepiece") and name.endswith(".model"))
    ):
        errors.append(f"{relative}: forbidden model/tokenizer artifact")

    if size > max_file_bytes:
        errors.append(f"{relative}: {size} bytes exceeds per-file limit {max_file_bytes}")
    # Oversized blobs are already rejected. Avoid materializing model-sized
    # index objects solely to run the remaining content checks.
    if payload is None:
        return size, errors
    if b"\0" in payload:
        errors.append(f"{relative}: binary content detected")
        return size, errors
    if LOCAL_HOME in payload:
        errors.append(f"{relative}: hard-coded local home path")
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(payload):
            errors.append(f"{relative}: possible {label}")
    return size, errors


def _index_entry(repo: Path, relative: Path) -> tuple[str, str] | None:
    listing = _git(
        repo,
        "ls-files",
        "--stage",
        "-z",
        "--",
        relative.as_posix(),
    )
    records = [record for record in listing.split(b"\0") if record]
    if not records:
        return None

    stage_zero: list[tuple[str, str]] = []
    for record in records:
        try:
            metadata, _ = record.split(b"\t", 1)
            mode, object_id, stage = metadata.split()
        except ValueError as exc:
            raise AuditError(f"malformed index entry for {relative}") from exc
        if stage == b"0":
            stage_zero.append((mode.decode("ascii"), object_id.decode("ascii")))
    if len(stage_zero) != 1:
        raise AuditError(f"{relative}: unresolved or ambiguous index entry")
    return stage_zero[0]


def _head_entry(repo: Path, relative: Path) -> tuple[str, str] | None:
    listing = _git(repo, "ls-tree", "-z", "HEAD", "--", relative.as_posix())
    records = [record for record in listing.split(b"\0") if record]
    if not records:
        return None
    if len(records) != 1:
        raise AuditError(f"{relative}: ambiguous HEAD entry")
    try:
        metadata, _ = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.split()
    except ValueError as exc:
        raise AuditError(f"malformed HEAD entry for {relative}") from exc
    if object_type != b"blob":
        return mode.decode("ascii"), object_id.decode("ascii")
    return mode.decode("ascii"), object_id.decode("ascii")


def _audit_git_blob(
    repo: Path,
    relative: Path,
    source: str,
    max_file_bytes: int,
) -> tuple[int, list[str]]:
    entry = (
        _index_entry(repo, relative)
        if source == "index"
        else _head_entry(repo, relative)
    )
    # A changed path without an entry in the selected tree is a deletion.
    if entry is None:
        return 0, []
    mode, object_id = entry
    if not mode.startswith("100"):
        return 0, [f"{relative}: delivery entries must be regular files"]

    raw_size = _git(repo, "cat-file", "-s", object_id).strip()
    try:
        size = int(raw_size)
    except ValueError as exc:
        raise AuditError(f"invalid blob size for {relative}: {raw_size!r}") from exc
    payload = None
    if size <= max_file_bytes:
        payload = _git(repo, "cat-file", "blob", object_id)
    return _audit_payload(relative, payload, size, max_file_bytes)


def audit_path(
    repo: Path,
    relative: Path,
    max_file_bytes: int,
    source: str = "worktree",
) -> tuple[int, list[str]]:
    if source in {"index", "head"}:
        return _audit_git_blob(repo, relative, source, max_file_bytes)
    if source != "worktree":
        raise AuditError(f"unsupported delivery source for {relative}: {source}")

    full_path = repo / relative
    if full_path.is_symlink():
        return 0, [f"{relative}: delivery entries must be regular files"]
    if not full_path.exists():
        return 0, []
    if not full_path.is_file():
        return 0, [f"{relative}: delivery entries must be regular files"]

    size = full_path.stat().st_size
    payload = full_path.read_bytes() if size <= max_file_bytes else None
    return _audit_payload(relative, payload, size, max_file_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument(
        "--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES
    )
    parser.add_argument(
        "--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    if args.max_total_bytes <= 0 or args.max_file_bytes <= 0:
        raise AuditError("byte limits must be positive")

    path_sources = collect_changed_path_sources(repo, args.base_ref)
    if not path_sources:
        raise AuditError(f"no delivery changes found relative to {args.base_ref}")
    paths = sorted(path_sources, key=lambda path: path.as_posix())

    total_bytes = 0
    errors: list[str] = []
    for relative in paths:
        source_sizes: list[int] = []
        for source in sorted(path_sources[relative]):
            size, path_errors = audit_path(
                repo,
                relative,
                args.max_file_bytes,
                source,
            )
            source_sizes.append(size)
            for error in path_errors:
                if error not in errors:
                    errors.append(error)
        # Candidate versions of one path are mutually exclusive in a commit.
        # Count the largest one while auditing every candidate's content.
        total_bytes += max(source_sizes, default=0)
    if total_bytes > args.max_total_bytes:
        errors.append(
            f"delivery total {total_bytes} bytes exceeds limit {args.max_total_bytes}"
        )

    print(f"base_ref={args.base_ref}")
    print(f"changed_files={len(paths)}")
    print(f"total_bytes={total_bytes}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("delivery_audit=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
