"""A target whose symbols are named nothing the convention looks for.

Only ``--entry`` and ``--inputs`` can reach these, which is what makes it the
override fixture.
"""

from __future__ import annotations

import torch
from torch import nn


class Net(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


net = Net()


def make_inputs() -> tuple[torch.Tensor]:
    return (torch.ones(2, 3),)


# A namespace, so the dotted attribute path in an override spec is covered too.
class bundle:  # noqa: N801 - a namespace, deliberately lowercase like a module
    net = net
    make_inputs = staticmethod(make_inputs)
