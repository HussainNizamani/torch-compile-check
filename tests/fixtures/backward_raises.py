"""A target whose backward pass raises under the compiled lane only.

PLAN.md "grad" names backward-only divergence as this oracle's bug class, and
the sharpest form of it is a lane whose forward pass answers and whose backward
does not. Reaching that with a real op would mean depending on a bug in the
installed wheel; reaching it with a backend of our own works on every torch and
every architecture, the same trick ``compile_only_raises.py`` plays one pass
earlier.

The backend runs the traced graph unchanged and wraps every differentiable
output in an autograd function whose ``backward`` raises. So the compiled
forward is correct to the last bit -- the numerics oracle has nothing to say
about this target -- and only the backward diverges, which is exactly the shape
that would otherwise need a partitioner bug to produce.

Registration is guarded because this module is imported twice in a normal test
run, once by the test and once by the CLI's own discovery, and
``register_backend`` refuses a duplicate name.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

BACKEND = "compile_check_backward_raises"
"""The backend name to pass to ``--backends``."""

MESSAGE = "this backend's backward raises on purpose"


class _RaiseOnBackward(torch.autograd.Function):
    """Identity forwards, an exception backwards."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(MESSAGE)


def _poison(value: Any) -> Any:
    """Route one differentiable output through the raising function."""
    if isinstance(value, torch.Tensor) and value.requires_grad and value.is_floating_point():
        return _RaiseOnBackward.apply(value)
    return value


def _compile(gm: Any, example_inputs: Any) -> Any:
    """A torch.compile backend whose forward is the graph and whose backward is not."""
    del example_inputs

    def compiled(*args: Any, **kwargs: Any) -> Any:
        return torch.utils._pytree.tree_map(_poison, gm(*args, **kwargs))

    return compiled


if BACKEND not in torch._dynamo.list_backends(exclude_tags=()):
    torch._dynamo.register_backend(_compile, name=BACKEND)


class Tiny(nn.Module):
    """One linear layer, so the backward has parameters to reach."""

    def __init__(self, features: int = 4, out_features: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


with torch.random.fork_rng():
    torch.manual_seed(1234)
    model = Tiny()

inputs = (torch.ones(2, 4),)
