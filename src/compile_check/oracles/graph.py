"""Graph health oracle.

PLAN.md "Oracles": compares graph break count and reasons, unique graph count
across a repeat call, and compile wall time; informational unless ``--fail-on
graph`` is set. Its bug class is recompile storms and silent fullgraph
regressions.

PLAN.md "graph": the counts come from ``torch._dynamo.explain`` and from
``torch._dynamo.utils.counters['stats']['unique_graphs']`` sampled around the
repeat call. Graph breaks are not bugs; they explain why a user is not getting
the speedup they expect. With ``--baseline FILE`` the oracle reports only new
breaks, which is the mode the GitHub Action uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["check"]


def check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare graph health against the reference run and an optional baseline.

    Args:
        eager: the reference run record.
        compiled: the run record under test.
        baseline: a stored graph-health baseline; when given, only breaks absent
            from it are reported.

    Returns:
        One record per new break, recompile, or fullgraph regression; empty when
        graph health is unchanged.
    """
    raise NotImplementedError("the graph oracle lands in M3")
