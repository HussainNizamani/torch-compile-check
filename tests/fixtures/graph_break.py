"""A target that Dynamo cannot capture as one graph, on purpose.

The graph oracle's positive fixture. Two breaks, chosen because they are the two
shapes a user actually hits and because neither depends on a torch that has not
learned to trace it yet:

``print`` is a builtin Dynamo refuses to trace (gb0059 in torch's graph-break
catalogue), and a debug print left in a forward pass is the single most common
reason a real model does not compile into one graph.

``if x.sum() > 0`` branches on a tensor, which Dynamo cannot resolve without the
values (gb0170, "Data-dependent branching"). That is the same break
``cases/distributions_validation_branch.py`` reproduces inside
``torch.distributions``, at a size that compiles in a second.

The answers are unaffected -- the default ``fullgraph=False`` falls back to the
interpreter at each break and returns exactly what eager returns -- which is the
point PLAN.md "graph" makes: graph breaks are not bugs, they explain why a user
is not getting the speedup they expect. Under ``--fullgraph`` the same file
cannot be compiled at all, which is the oracle's ``fail`` rule.
"""

from __future__ import annotations

import torch


def fn(x: torch.Tensor) -> torch.Tensor:
    """Two graph breaks and no divergence."""
    y = x * 2
    print("compile-check graph_break fixture")
    if y.sum() > 0:
        y = y + 1
    return y - 0.5


def get_inputs() -> tuple[torch.Tensor]:
    """A fixed vector. Seeded locally so importing this file twice is stable."""
    generator = torch.Generator().manual_seed(1234)
    return (torch.randn(8, generator=generator),)


inputs = get_inputs()
