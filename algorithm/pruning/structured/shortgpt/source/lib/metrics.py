"""Block Influence metric for ShortGPT layer importance scoring."""

import torch


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular: bool = False,
) -> torch.Tensor:
    """Compute the Block Influence score between consecutive layer hidden states.

    Higher BI means the layer transforms the representation more (more important).
    Lower BI means the layer is closer to identity (less important, candidate for pruning).
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return torch.arccos(sim) / torch.pi

    return 1 - sim
# Maintenance touch for repository metadata refresh.
