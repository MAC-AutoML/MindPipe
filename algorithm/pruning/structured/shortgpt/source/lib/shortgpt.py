"""ShortGPT layer importance computation."""

from __future__ import annotations

from typing import List, Tuple

import torch
from tqdm import tqdm

from lib.metrics import block_influence


@torch.inference_mode()
def compute_layer_importances(
    model,
    calibration_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> List[float]:
    """Compute per-layer importance scores using Block Influence.

    Each calibration batch is a (input_ids, labels) tuple where input_ids
    has shape (1, sequence_length). We run a forward pass with
    ``output_hidden_states=True`` and accumulate BI scores between
    consecutive layer hidden states.

    Returns a list of importance scores, one per transformer layer.
    """
    model.eval()

    # Determine the number of layers via the first batch
    first_input = calibration_batches[0][0].to(device)
    dummy_out = model(first_input, output_hidden_states=True, use_cache=False)
    n_layers = len(dummy_out.hidden_states) - 1  # hidden_states[0] is embedding
    del dummy_out

    importances = [0.0] * n_layers

    for input_ids, _labels in tqdm(calibration_batches, desc="ShortGPT computing layer importances"):
        chunk = input_ids.to(device)

        outputs = model(chunk, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        for i in range(n_layers):
            in_h = hidden_states[i]
            out_h = hidden_states[i + 1]
            bi = block_influence(in_h, out_h)
            importances[i] += bi.sum().cpu().item()

        del outputs, hidden_states

    return importances
