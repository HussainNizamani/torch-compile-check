"""A target whose weights are drawn while the module is being imported.

The shape the M2-3 housekeeping is about: PLAN.md's discovery convention asks
for a module-level ``model``, and the natural way to write one -- the way
``torchvision.models.resnet18(weights=None)`` is written -- constructs it at
module scope, so its parameters are initialised during ``load_target``. A seed
applied after that import cannot reach them, and two runs of the same command
would then compare two different models.

Deliberately tiny and deliberately not seeded here: the fixture must draw from
the global generator, because the global generator is exactly what ``--seed`` is
supposed to have set by the time this line runs. ``requires_grad`` on the input
so the run exercises the backward pass as well, which is the shape the grad
tolerance policy of the same slice was measured on.
"""

from __future__ import annotations

import torch

model = torch.nn.Linear(8, 4)

inputs = (torch.randn(3, 8, requires_grad=True),)
