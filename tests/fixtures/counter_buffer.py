"""A model whose forward pass writes to its own buffer, for the isolation test.

PLAN.md "Runner semantics" clones the inputs per backend so that a mutation by
one lane is invisible to the next. Module state needs the same isolation and did
not have it before M2-2: this model multiplies by a counter it increments on
every call, so a second lane handed the same object computes with a different
number and the numerics oracle reports a divergence no backend caused. It stands
in for the real case, which is BatchNorm running statistics in train mode.

No ``requires_grad`` on the input, and the counter is a buffer rather than a
parameter: the point of this fixture is the state, not the gradient.
"""

from __future__ import annotations

import torch
from torch import nn


class Counter(nn.Module):
    """Multiplies its input by the number of times it has been called."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x * self.calls


model = Counter()

inputs = (torch.ones(3, 4),)
