"""191308 as a compile-check target: int8 batched matmul, promoted to int64.

Issue: https://github.com/pytorch/pytorch/issues/191308 -- "[Inductor][CPU]
torch.compile changes int8 batched matmul output dtype to int64". PLAN.md
"metadata" names this as the bug the metadata oracle exists for: the values are
arguably defensible, the dtype is not, so an oracle that only compares numbers
reports clean.

Twin file, deliberately. ``cases/dtype_int8_matmul_promotion.py`` is the C-1
corpus entry for the same bug: a standalone RED/GREEN script that FINDINGS.md
keys on, driven by its own ``main()``. This file is the same reproducer written
to the discovery convention of PLAN.md, a module-level ``fn`` and ``inputs``, so
that ``compile-check cases/dtype_promotion.py`` runs it through the tool itself
-- which the corpus script cannot be, since it exposes ``build()`` rather than
the two module-level names discovery looks for.

Version marker. Measured on torch ``2.14.0+cpu`` (git ``08187d9``, aarch64,
CPU-only, caches disabled): eager returns ``torch.int8`` and the ``inductor``
lane returns ``torch.int64``, so the case is RED here and the metadata oracle
reports it. It may well pass on another torch -- the issue is open and unfixed
as of 2026-09-02, but the shape family that triggers it is narrow, and the
plain 2-D ``(4, 8) @ (8, 4)`` int8 matmul stays int8 on this same build. The
test that covers this file is written for both outcomes and asserts the oracle's
rule on a synthetic result, so a torch that fixes the bug turns the case green
rather than turning the suite red.
"""

from __future__ import annotations

import torch


def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The reproducer from the issue, unchanged: a 3-D int8 matmul."""
    return torch.matmul(a, b)


# Shapes and values verbatim from the issue: (1, 1, 2) @ (1, 2, 2), all ones,
# so every output element is 2 and a value comparison passes in both worlds.
inputs = (
    torch.ones((1, 1, 2), dtype=torch.int8),
    torch.ones((1, 2, 2), dtype=torch.int8),
)
