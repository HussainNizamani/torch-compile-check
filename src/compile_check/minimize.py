"""Submodule delta debug, input shrink, minifier handoff.

PLAN.md "Package layout": ``minimize.py`` -- submodule delta debug, input
shrink, minifier handoff.

PLAN.md "Minimizer, v1": v1 works at the module and input level -- delta
debugging over ``nn.Module`` children, plus input shrinking, with the FX graph
level handed off to torch's built-in accuracy minifier. It runs only after a
finding, and it is allowed to give up.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["minimize"]


def minimize(
    fn: Any,
    inputs: Any,
    reproduces: Callable[[Any, Any], bool],
) -> dict[str, Any]:
    """Shrink a reproducing case as far as it still reproduces.

    Args:
        fn: the callable the finding was produced from.
        inputs: the inputs the finding was produced with.
        reproduces: predicate returning whether a candidate pair still shows the
            finding.

    Returns:
        The minimization record: the stubbed model, the shrunk inputs, the
        subtrees that could not be replaced, and the minifier handoff result.
    """
    raise NotImplementedError("the minimizer lands in M3")
