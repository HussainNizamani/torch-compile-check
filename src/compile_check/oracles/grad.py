"""Gradients oracle.

PLAN.md "Oracles": compares the ``.grad`` of every input and parameter after one
backward on a deterministic scalar reduction, and the set of tensors that
received a grad at all; grad values pass through the numerics rule and the grad
presence set must be identical. Its bug class is backward-only divergence and
partitioner bugs.

PLAN.md "grad": the oracle activates when any input or parameter has
``requires_grad``. Grads are zeroed before each backward so a leaked
accumulation cannot be mistaken for a divergence. Only one backward step is run;
multi-step training loop correctness is out of scope for v1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["check"]


def check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> list[dict[str, Any]]:
    """Compare the two runs' gradients and grad presence sets.

    Args:
        eager: the reference run record.
        compiled: the run record under test.
        rtol: relative tolerance override, passed to the numerics comparison.
        atol: absolute tolerance override, passed to the numerics comparison.

    Returns:
        One record per divergent gradient or presence mismatch; empty when the
        runs agree.
    """
    raise NotImplementedError("the grad oracle lands in M2")
