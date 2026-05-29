from types import SimpleNamespace

import pytest

from evaluation import vlm_eval


@pytest.mark.parametrize("model_type", sorted(vlm_eval.QWEN3_VL_MODEL_TYPES))
def test_build_wrapper_routes_qwen3_vl_variants(monkeypatch, model_type):
    expected_wrapper = object()
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    tokenizer_bundle = object()
    common_args = {"device": "cpu"}
    modules = {"BaseModel": object, "DATASET_TYPE": object()}

    def build_qwen3_wrapper(
        actual_model,
        actual_tokenizer_bundle,
        actual_common_args,
        actual_base_model_cls,
        actual_dataset_type_resolver,
    ):
        assert actual_model is model
        assert actual_tokenizer_bundle is tokenizer_bundle
        assert actual_common_args is common_args
        assert actual_base_model_cls is modules["BaseModel"]
        assert actual_dataset_type_resolver is modules["DATASET_TYPE"]
        return expected_wrapper

    monkeypatch.setattr(vlm_eval, "_build_qwen3_wrapper", build_qwen3_wrapper)

    assert (
        vlm_eval._build_wrapper(model, tokenizer_bundle, common_args, modules)
        is expected_wrapper
    )


@pytest.mark.parametrize("model_type", sorted(vlm_eval.QWEN3_5_MODEL_TYPES))
def test_qwen3_5_variants_disable_thinking(model_type):
    assert vlm_eval._qwen3_disable_thinking(model_type) is True


def test_qwen3_vl_keeps_thinking_enabled():
    assert vlm_eval._qwen3_disable_thinking("qwen3_vl") is False
