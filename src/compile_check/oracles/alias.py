"""Alias and mutation oracle.

PLAN.md "Oracles": compares storage identity and overlap among all outputs and
inputs, the input mutation set, and Python object identity; passes when the
compiled relation equals the eager relation exactly. It is the oracle for the
195451 inductor reinplacing bug and for 191449 / PR 191844, AOTAutograd
aliased-output identity.

PLAN.md "alias": two tensors are related when they share an untyped storage and
their byte ranges overlap -- storage identity alone is not sufficient, since two
disjoint views of one buffer share a data pointer. Object identity is recorded
separately, because "output 0 is the same object as input 1" is a stronger
contract than "output 0 aliases input 1", and 191449 lives in that gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["check"]


def check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare the two runs' alias relations and mutation sets.

    Args:
        eager: the reference run record.
        compiled: the run record under test.

    Returns:
        One record per added alias, dropped alias, added mutation, or dropped
        mutation; empty when the relations match.
    """
    raise NotImplementedError("the alias oracle lands in M2")
