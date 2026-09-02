"""A model that legitimately aliases and mutates, for the alias oracle.

The oracle's negative test needs a target whose relation is not empty: a model
that returns fresh tensors proves only that an oracle stays quiet when there is
nothing to see. This one returns two views of its own input, writes into that
input in place, and returns a tensor built from one of the views, so a real run
has aliases, a mutation, and an output that is not related to anything.

None of that is a defect. Every lane must produce the same relation, and the
oracle must say nothing at all -- a checker that fires on ordinary aliasing
would be worse than no checker.

No ``requires_grad``: an in-place write to a leaf that requires grad is an
autograd error, and the point of this fixture is the aliasing, not the grad
(the same reason ``mutating.py`` says so).
"""

from __future__ import annotations

import torch
from torch import nn


class ViewsAndMutation(nn.Module):
    """Mutates its first input, and returns two views of it."""

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x.relu_()
        head = x[:, :2]
        return head, torch.cat([head, y], dim=1), x.t()


model = ViewsAndMutation()

# Seeded locally so importing this file twice gives the same tensors.
_generator = torch.Generator().manual_seed(1234)
inputs = (
    torch.randn(4, 6, generator=_generator),
    torch.randn(4, 3, generator=_generator),
)
