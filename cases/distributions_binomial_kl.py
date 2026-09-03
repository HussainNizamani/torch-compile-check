"""194593 as a torch-compile-check target: a `fullgraph=True` capturability break,
honest about the two different answers `torch-compile-check` gives depending on
`--fullgraph`.

Issue: https://github.com/pytorch/pytorch/issues/194593 (sibling:
https://github.com/pytorch/pytorch/issues/194596, same class,
Wishart.mean). PLAN.md "graph" names this as a fullgraph compile-support
gap, not a numerical mismatch: `_kl_binomial_binomial` branches on a tensor
condition (`if (p.total_count < q.total_count).any():`), which Dynamo
cannot capture under `fullgraph=True` ("Data-dependent branching", gb0170).

Twin file, deliberately, on the pattern of the other twins in this
directory. `cases/distributions_validation_branch.py` is the corpus entry
for the same bug: a standalone RED/GREEN script that FINDINGS.md keys on,
driven by its own `main()` and running both the default and `fullgraph=True`
modes itself for the context probe. This file is the same reproducer
written to the discovery convention of PLAN.md, a module-level `fn` and
`inputs`, so that `torch-compile-check cases/distributions_binomial_kl.py` runs it
through the tool itself -- but the tool only takes one `--fullgraph` value
per invocation, so this file's contract is stated for both.

Two expectations, both measured on torch `2.14.0+cpu` (git `08187d9`,
aarch64, CPU-only, caches disabled), and both are the point of this twin:

- `torch-compile-check cases/distributions_binomial_kl.py` (default,
  `fullgraph=False`): the branch graph-breaks and Dynamo falls back to
  eager for that piece, so every backend still returns the correct answer.
  Exit 0, clean, no findings -- this is the honest "no bug visible here"
  reading a caller gets by default.
- `torch-compile-check cases/distributions_binomial_kl.py --fullgraph`: Dynamo
  cannot capture the branch at all, so `aot_eager` and `inductor` both raise
  during compilation rather than returning a divergent value. A lane that
  raised while eager did not is exit 1 regardless of `--fail-on` (PLAN.md
  "CLI surface for v1" -- an exception belongs to no oracle category, and a
  lane that could not run is not a lane that passed). On this build the
  first diverging backend is `aot_eager` (checked before `inductor`, and
  both raise identically, since the failure is in Dynamo's tracing,
  upstream of backend codegen); `cases/distributions_validation_branch.py`
  ran only the default `inductor` backend and reports the class of failure
  without ranking the two.

As of 2026-09-02 no fix has merged upstream (see
`cases/distributions_validation_branch.py` for the maintainer thread), so
both outcomes above are expected to hold on any current torch; a torch that
lands a fix would turn the `--fullgraph` invocation exit 0 as well.
"""

from __future__ import annotations

import torch
from torch.distributions import Binomial, kl_divergence


def fn(
    p_count: torch.Tensor,
    p_prob: torch.Tensor,
    q_count: torch.Tensor,
    q_prob: torch.Tensor,
) -> torch.Tensor:
    """The reproducer from the issue, unchanged."""
    return kl_divergence(
        Binomial(total_count=p_count, probs=p_prob),
        Binomial(total_count=q_count, probs=q_prob),
    )


# Values verbatim from the issue.
inputs = (
    torch.tensor([5.0]),
    torch.tensor([0.2]),
    torch.tensor([3.0]),
    torch.tensor([0.4]),
)
