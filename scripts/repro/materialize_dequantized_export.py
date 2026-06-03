#!/usr/bin/env python3
"""Materialize a compressed-tensors export into a dense HF model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.common.io import ensure_dir
from algorithm.common.modeling import load_model_and_tokenizer
from scripts.repro.evaluate_real_quant_export_dequant import load_dequantized_export


INFERENCE_FILE_NAMES = {
    "chat_template.jinja",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_shard_size", default="5GB")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    model, tokenizer_bundle = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    export_stats = load_dequantized_export(model, Path(args.export_dir))
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer_bundle.save_pretrained(str(output_dir))

    copied = []
    source_dir = Path(args.model_path)
    if source_dir.is_dir():
        for name in sorted(INFERENCE_FILE_NAMES):
            src = source_dir / name
            dst = output_dir / name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
                copied.append(name)

    metadata = {
        "model_path": args.model_path,
        "export_dir": args.export_dir,
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "dequantized_export": export_stats,
        "copied_inference_files": copied,
        "format": "dense_hf_dequantized_from_compressed_tensors_export",
    }
    (output_dir / "dequantized_export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
