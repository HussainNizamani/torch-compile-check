"""The five oracles, one per clause of the compile contract.

PLAN.md "Definitions": the compile contract is that ``compile(f)(x)`` is
observationally equivalent to ``f(x)`` except for speed. Concretely: the same
numbers, the same aliasing and mutation behaviour, the same dtype, shape, and
stride, the same gradients, and no silent fallback. Each of the five oracles
checks one clause of that contract.
"""

from __future__ import annotations

__all__ = ["ORACLES"]

# Also the vocabulary of --fail-on, in the order PLAN.md "Oracles" lists them.
ORACLES: tuple[str, ...] = ("numerics", "alias", "metadata", "grad", "graph")
