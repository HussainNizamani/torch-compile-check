"""A full training step -- forward, loss, and (once the grad oracle lands)
backward -- as a validation target.

PLAN.md "grad" describes the oracle this exercises: reduce the differentiable
outputs to a scalar, call ``backward()``, and compare both the values and the
set of tensors that received a gradient, between eager and each compiled
backend. That oracle is a stub today ("not yet" in the checks table until
M2-2), but the runner already runs the backward pass unconditionally
whenever an input or a parameter requires grad (``runner.py``'s
``_run_backward``, which every other target in this suite also triggers by
having trainable parameters), so this target is real work today, not a
placeholder: it forces the compiler through the backward graph for a
training-shaped computation, and the moment the oracle lands, this file
starts reporting grad findings with no change needed.

The whole training step -- linear layers, ReLU, and the cross-entropy loss
-- is the module's ``forward()``, deliberately, rather than a bare `fn` that
calls the loss function around a separate model: the module-level ``model``
name makes it an ``nn.Module``, and ``runner.py`` only records
per-parameter gradients (``result.param_grads``) when the target is one.
The input ``x`` also carries ``requires_grad=True``, so the input-gradient
half of the check (``result.input_grads``) is exercised too, which a
training loop's input normally does not need but the oracle's input/param
distinction (PLAN.md "grad") is written to catch either way.
"""

from __future__ import annotations

import torch
from torch import nn


class TrainStep(nn.Module):
    """Two linear layers and a cross-entropy loss, as one differentiable call."""

    def __init__(self, in_features: int = 8, hidden: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return nn.functional.cross_entropy(logits, y)


# Weights seeded the same way get_inputs() seeds its tensors, for the same
# reason cases/mlp.py does: importing this file twice must not change the
# values compared.
with torch.random.fork_rng():
    torch.manual_seed(7)
    model = TrainStep()


def get_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed batch of 8, with the input carrying ``requires_grad``."""
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(8, 8, generator=generator, requires_grad=True)
    y = torch.randint(0, 4, (8,), generator=generator)
    return (x, y)
