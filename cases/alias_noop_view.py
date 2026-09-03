"""191449 as a torch-compile-check target: two outputs collapsed into one object.

Issue: https://github.com/pytorch/pytorch/issues/191449 -- PLAN.md "alias"
names this as the bug the alias oracle exists for: ``base = x + 1; return
base, base.view(-1)`` gives two distinct tensor objects in eager, sharing one
storage. AOTAutograd's metadata analysis misclassifies a no-grad no-op view
as ``non_alias`` instead of an alias needing regeneration, and Inductor's
``remove_noop_ops`` / ``pointless_view`` passes act on that misclassification
by returning one Python object for both outputs: ``out_base is out_alias``,
directly observable, and with a real consequence (see
``cases/alias_noop_view_identity.py`` for the downstream ``resize_()``
corruption this enables between two compiled graphs).

Twin file, deliberately, on the pattern of ``cases/dtype_promotion.py`` beside
``cases/dtype_int8_matmul_promotion.py`` and ``cases/alias_copyback.py``
beside ``cases/alias_slice_scatter_copyback.py``.
``cases/alias_noop_view_identity.py`` is the corpus entry for the same bug: a
standalone RED/GREEN script that FINDINGS.md keys on, driven by its own
``main()`` and exercising the fuller ``resize_()`` shape that shows the
consequence. This file is the plain identity-collapse reproducer written to
the discovery convention of PLAN.md, a module-level ``fn`` and ``inputs``, so
that ``torch-compile-check cases/alias_noop_view.py`` catches the bug by itself
through the alias oracle's object-identity check -- which is exactly the
``_identity_probe()`` helper embedded in the standalone script, promoted to
its own file because discovery needs a module-level ``fn``, not a nested one.

Version marker. Measured on torch ``2.14.0+cpu`` (git ``08187d9``, aarch64,
CPU-only, caches disabled): eager returns two distinct objects sharing a
storage (``output[0]~output[1] overlapping``), and the ``inductor`` lane
collapses them into one object (``output[0]~output[1] same object``), so the
case is RED here -- exit 1, one alias finding, field ``identity_added``,
message "inductor returned one object for output[0] and output[1] and eager
returned distinct objects that share a storage" -- and the stage verdict
names ``inductor`` as the first diverging backend. The fix (PR 191844,
merged 2026-09-02T03:45:57Z, commit ``a3586f0018``) lives in AOTAutograd, not
in Inductor, so a torch containing that commit is expected to turn this case
GREEN (exit 0) even though the divergence was only ever *observable* on the
``inductor`` backend -- ``cases/alias_noop_view_identity.py``'s docstring has
the fuller account of why the fix location and the first diverging backend
differ without contradiction. The test that covers this file
(``tests/test_corpus_twins.py``) is written for both outcomes.
"""

from __future__ import annotations

import torch


def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reproducer from the issue, unchanged: a base and its no-op view.

    In eager these are two distinct Python objects sharing one storage. The
    alias oracle compares that relation between eager and each compiled
    backend; a backend that hands back one object for both breaks it.
    """
    base = x + 1
    return base, base.view(-1)


inputs = (torch.zeros(1),)
