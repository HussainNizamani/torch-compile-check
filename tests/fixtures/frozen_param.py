"""A model with a frozen parameter, for the grad oracle's presence set.

PLAN.md "grad" makes the set of tensors that ended up with a ``.grad`` part of
the contract, and a frozen parameter is the case where that set is legitimately
smaller than the parameter list. ``named_parameters()`` still yields the frozen
layer's weight and bias; the backward pass leaves them alone, and both lanes must
agree about that. An oracle that read the presence set off the parameter list
rather than off the gradients would report this model as a divergence in every
lane.
"""

from __future__ import annotations

import torch
from torch import nn


class PartlyFrozen(nn.Module):
    """One layer that trains and one that does not."""

    def __init__(self, features: int = 4, hidden: int = 6, out_features: int = 3) -> None:
        super().__init__()
        self.frozen = nn.Linear(features, hidden)
        self.trainable = nn.Linear(hidden, out_features)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trainable(torch.relu(self.frozen(x)))


# Weights and inputs are seeded locally so importing this file twice is stable.
with torch.random.fork_rng():
    torch.manual_seed(1234)
    model = PartlyFrozen()

_generator = torch.Generator().manual_seed(1234)
inputs = (torch.randn(4, 4, generator=_generator),)
