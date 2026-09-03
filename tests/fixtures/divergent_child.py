"""A three-block model where exactly one block makes the compiled lane wrong.

The minimizer's delta-debugging pass (PLAN.md "Minimizer, v1", step 1) needs a
target where replacing a child changes the answer to "does the finding still
reproduce" -- and where it changes it for one child and not for the others.
Every real example of that is a torch bug on one wheel and one architecture,
which is not something a test suite can depend on, so the divergence here is a
backend of our own, registered by name, on the pattern of
``tests/fixtures/compile_only_raises.py``.

How it is arranged. ``middle`` is the guilty block: it is the only one that
calls ``torch.erf``, and the backend perturbs the compiled output by a constant
if and only if the graph it is handed contains that call. So replacing ``head``
or ``tail`` with ``torch.nn.Identity`` leaves the divergence in place and the
minimizer keeps the replacement, and replacing ``middle`` takes the ``erf`` out
of the graph, the perturbation with it, and the finding too -- so the minimizer
puts it back and records it as the block the finding lives in. What comes out is
"the bug needs this one child", which is the whole point of the pass.

The perturbation is a constant add rather than a scaling, so the gradients are
identical in both lanes and the grad oracle stays quiet: this fixture is about
one numerics finding, and a second finding from another oracle would make the
delta-debugging assertion ambiguous.

The batch is eight so the input-shrinking pass has something to halve, and the
divergence does not depend on the batch size, so it survives every halving down
to one.

Registration is guarded because this module is imported twice in a normal test
run -- once by the test and once by the CLI's own discovery -- and
``register_backend`` refuses a duplicate name.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

BACKEND = "torch_compile_check_perturbs"
"""The backend name to pass to ``--backends``."""

PERTURBATION = 0.5
"""Far outside any float32 tolerance, so the finding is never a rounding call."""


class Guilty(nn.Module):
    """The one block the perturbing backend keys on."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


class DivergentMLP(nn.Module):
    """Two shape-preserving linears around the block that matters."""

    def __init__(self, features: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(features, features)
        self.middle = Guilty()
        self.tail = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.middle(self.head(x)))


def _perturb_if_guilty(gm: Any, example_inputs: Any) -> Any:
    """Compile to the graph itself, plus a constant when ``erf`` is in it.

    Dynamo hands a backend the traced graph, so "does this graph still contain
    the guilty block" is a question about the nodes rather than about the module
    tree -- which is what makes the answer change when the minimizer stubs
    ``middle`` out and not when it stubs anything else.
    """
    del example_inputs  # the decision is about the graph, not the values
    guilty = any(node.op == "call_function" and node.target is torch.erf for node in gm.graph.nodes)
    if not guilty:
        return gm.forward

    def perturbed(*args: Any, **kwargs: Any) -> Any:
        return torch.utils._pytree.tree_map(
            lambda leaf: leaf + PERTURBATION if isinstance(leaf, torch.Tensor) else leaf,
            gm.forward(*args, **kwargs),
        )

    return perturbed


if BACKEND not in torch._dynamo.list_backends(exclude_tags=()):
    torch._dynamo.register_backend(_perturb_if_guilty, name=BACKEND)


# Seeded locally so importing this file twice gives the same weights and inputs.
with torch.random.fork_rng():
    torch.manual_seed(1234)
    model = DivergentMLP()

_generator = torch.Generator().manual_seed(1234)
inputs = (torch.randn(8, 4, generator=_generator),)
