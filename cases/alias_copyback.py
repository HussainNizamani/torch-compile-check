"""195451 as a torch-compile-check target: a functional result that comes back as its input.

Issue: https://github.com/pytorch/pytorch/issues/195451 -- inductor's
``should_reinplace_scatter()`` treats a ``slice_scatter`` followed by a direct
copy-back as profitable, reinplaces the scatter, and returns the mutated input
itself. PLAN.md "alias" names this as the bug the alias oracle exists for: the
values are identical in both worlds, so an oracle that only compares numbers
reports clean, and the defect is that the returned tensor *is* the input, so a
caller that writes into the result corrupts it.

Twin file, deliberately, on the pattern of ``cases/dtype_promotion.py`` beside
``cases/dtype_int8_matmul_promotion.py``. ``cases/alias_slice_scatter_copyback.py``
is the corpus entry for the same bug: a standalone RED/GREEN script that
FINDINGS.md keys on, driven by its own ``main()``. This file is the same
reproducer written to the discovery convention of PLAN.md, a module-level ``fn``
and ``inputs``, so that ``torch-compile-check cases/alias_copyback.py`` runs it through
the tool itself -- which the corpus script cannot be, since it exposes
``build()`` rather than the two module-level names discovery looks for.

Version marker. Measured on torch ``2.14.0+cpu`` (git ``08187d9``, aarch64,
CPU-only, caches disabled): eager and ``aot_eager`` return an independent
tensor, and the ``inductor`` lane returns the input object itself, so the case is
RED here and the alias oracle reports it against ``inductor`` alone -- "first
diverges at inductor". The fix (PR 195484) was open and unmerged as of
2026-09-02; a torch that lands it turns this case green rather than turning the
suite red, which is how the test that covers this file is written.
"""

from __future__ import annotations

import torch


def fn(x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """The reproducer from the issue, unchanged.

    ``updated`` is a functional result: in eager it is a new tensor that happens
    to hold what ``x`` holds after the copy-back, and writing into it must not
    reach ``x``.
    """
    updated = torch.slice_scatter(x, src, 0, 0, 1)
    x.copy_(updated)
    return updated


# Values verbatim from the issue. Every element is equal in both worlds after
# the call, which is the point: only the aliasing differs.
inputs = (torch.tensor([1.0, 2.0]), torch.tensor([10.0]))
