from __future__ import annotations

import torch
from transformers import MixtralConfig
from transformers import MixtralForCausalLM

from algorithm.common.modeling import capture_first_block_inputs
from algorithm.common.modeling import find_prunable_linear_layers
from algorithm.common.modeling import get_text_backbone
from algorithm.common.modeling import normalize_mixtral_expert_intermediate_size_for_hf_save


def _build_tiny_mixtral() -> MixtralForCausalLM:
    config = MixtralConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
    )
    model = MixtralForCausalLM(config)
    model.eval()
    return model


def test_mixtral_prunable_layers_are_discovered_and_router_is_ignored():
    model = _build_tiny_mixtral()
    backbone = get_text_backbone(model)

    assert backbone.prefix == "model"
    assert len(backbone.layers) == 1

    prunable = find_prunable_linear_layers(backbone.layers[0])
    expected = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "block_sparse_moe.experts.0.w1",
        "block_sparse_moe.experts.0.w2",
        "block_sparse_moe.experts.0.w3",
        "block_sparse_moe.experts.1.w1",
        "block_sparse_moe.experts.1.w2",
        "block_sparse_moe.experts.1.w3",
    }

    assert expected.issubset(prunable)
    assert "block_sparse_moe.gate" not in prunable


def test_mixtral_capture_and_save_normalization_smoke():
    model = _build_tiny_mixtral()
    backbone = get_text_backbone(model)
    calibration_batches = [
        (
            torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long),
            torch.full((1, 8), -100, dtype=torch.long),
        )
    ]

    inputs, layer_kwargs = capture_first_block_inputs(
        model=model,
        backbone=backbone,
        calibration_batches=calibration_batches,
        device="cpu",
    )

    assert inputs.shape == (1, 8, model.config.hidden_size)
    assert isinstance(layer_kwargs, dict)
    assert normalize_mixtral_expert_intermediate_size_for_hf_save(model) == 0
