"""A target that writes into its own input, for the isolation test.

The runner's contract (PLAN.md "Runner semantics") is that inputs are deep
cloned per backend, so a backend never sees a tensor another backend mutated.
This file is what makes that testable: after one backend has run, the input it
was handed is all zeros, and the next backend must still be handed the original.
"""

from __future__ import annotations

import torch
from torch import nn


class Mutator(nn.Module):
    """Zeroes its input in place, then reports the sum of what it was given."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        x.zero_()
        return total


model = Mutator()

# No requires_grad: an in-place write to a leaf that requires grad is an
# autograd error, and the point of this fixture is the mutation, not the grad.
inputs = (torch.full((3, 4), 2.0),)
