"""A tiny MLP following the discovery convention, the reference M1 target.

PLAN.md "Discovery convention": a module-level ``model`` and a module-level
``inputs``. ``get_inputs()`` is here as well so the tests can exercise both
halves of the convention against one file; the two return equal tensors, so a
test may use either without the values changing under it.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Linear, ReLU, Linear. Small enough that inductor compiles it in seconds."""

    def __init__(self, in_features: int = 8, hidden: int = 16, out_features: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_inputs() -> tuple[torch.Tensor]:
    """A fixed batch of 4. Seeded locally so importing this file twice is stable."""
    generator = torch.Generator().manual_seed(1234)
    return (torch.randn(4, 8, generator=generator),)


# Weights are seeded the same way, for the same reason.
with torch.random.fork_rng():
    torch.manual_seed(1234)
    model = MLP()

inputs = get_inputs()
