"""A dict-output target whose second leaf diverges in dtype under inductor.

Modeled on ``cases/dtype_promotion.py`` (191308: int8 batched matmul promoted
to int64 under ``torch.compile(backend="inductor")``). PLAN.md "metadata"
names that dtype divergence; this fixture wraps the same shape of bug in a
dict output, for the regression-test emitter's ``_leaf()``: the whole return
is not a bare tensor here, so indexing it by leaf position the way a tuple
output would be indexed -- ``actual[1]`` -- is a lookup for the dict key ``1``
and raises ``KeyError`` rather than comparing anything.
``torch.utils._pytree.tree_leaves`` sorts a dict by key, so ``"identity"``
comes before ``"product"`` and the divergent leaf lands at flattened index 1.

Version marker. Measured on torch ``2.14.0+cpu`` (git ``08187d9``, aarch64,
CPU-only, caches disabled): eager returns ``torch.int8`` for both entries and
the ``inductor`` lane promotes ``"product"`` to ``torch.int64``, so the case is
RED here, same as its twin. This fixture is not part of the version-tracked
regression corpus in ``cases/``: what it exists to exercise is the emitter's
leaf indexing, not the promotion bug itself, so it carries no marker in
``cases/markers.py`` and is not expected to stay RED forever.
"""

from __future__ import annotations

import torch


def fn(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    """One output untouched, one the same int8 matmul from 191308, under a dict."""
    return {"identity": a + b, "product": torch.matmul(a, b)}


# Shapes and values verbatim from cases/dtype_promotion.py: (1, 1, 2) @ (1, 2, 2),
# all ones, so every output element is 2 and a value comparison passes in both
# worlds -- only the dtype, and the emitter's leaf indexing, are under test.
inputs = (
    torch.ones((1, 1, 2), dtype=torch.int8),
    torch.ones((1, 2, 2), dtype=torch.int8),
)
