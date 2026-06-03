#!/usr/bin/env python3
"""Smoke test a vLLM compressed-tensors int4 export."""

from __future__ import annotations

import argparse
import json
import sys
import types


def _set_transformers_attr(transformers, name: str, value) -> None:
    setattr(transformers, name, value)
    if hasattr(transformers, "_objects"):
        transformers._objects[name] = value
    if hasattr(transformers, "__all__") and name not in transformers.__all__:
        transformers.__all__.append(name)


def _install_compat_shims() -> None:
    import transformers

    if not hasattr(transformers, "Gemma3Config"):
        _set_transformers_attr(transformers, "Gemma3Config", transformers.PretrainedConfig)
    if not hasattr(transformers, "AutoVideoProcessor"):
        class _AutoVideoProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise NotImplementedError("AutoVideoProcessor shim is not available in this transformers build.")

        _set_transformers_attr(transformers, "AutoVideoProcessor", _AutoVideoProcessor)

    if "transformers.video_processing_utils" not in sys.modules:
        video_processing_utils = types.ModuleType("transformers.video_processing_utils")

        class _BaseVideoProcessor:
            pass

        video_processing_utils.BaseVideoProcessor = _BaseVideoProcessor
        sys.modules["transformers.video_processing_utils"] = video_processing_utils

    try:
        import transformers.configuration_utils as configuration_utils

        if not hasattr(configuration_utils, "ALLOWED_LAYER_TYPES"):
            configuration_utils.ALLOWED_LAYER_TYPES = {}
        if not hasattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES"):
            configuration_utils.ALLOWED_ATTENTION_LAYER_TYPES = configuration_utils.ALLOWED_LAYER_TYPES
    except Exception:
        pass

    if "transformers.masking_utils" not in sys.modules:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except Exception:
            ALL_ATTENTION_FUNCTIONS = {}

        class _Registry(dict):
            def register(self, name, value):
                self[name] = value

        masking_utils = types.ModuleType("transformers.masking_utils")
        registry = _Registry()
        if isinstance(ALL_ATTENTION_FUNCTIONS, dict):
            for key, value in ALL_ATTENTION_FUNCTIONS.items():
                registry[key] = value
        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS = registry
        sys.modules["transformers.masking_utils"] = masking_utils

    if "vllm.transformers_utils.gguf_utils" not in sys.modules:
        gguf_utils = types.ModuleType("vllm.transformers_utils.gguf_utils")
        gguf_utils.check_gguf_file = lambda *args, **kwargs: False
        gguf_utils.is_gguf = lambda *args, **kwargs: False
        gguf_utils.is_remote_gguf = lambda *args, **kwargs: False
        gguf_utils.split_remote_gguf = lambda model: (model, "")
        gguf_utils.get_gguf_file_path_from_hf = lambda model, *args, **kwargs: model
        gguf_utils.maybe_patch_hf_config_from_gguf = lambda config, *args, **kwargs: config
        gguf_utils.detect_gguf_multimodal = lambda *args, **kwargs: False
        sys.modules["vllm.transformers_utils.gguf_utils"] = gguf_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--compat_shims", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compat_shims:
        _install_compat_shims()

    from vllm import LLM
    from vllm import SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    print(json.dumps({"prompt": args.prompt, "completion": text}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
