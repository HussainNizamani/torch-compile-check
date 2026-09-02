"""Terminal report.

PLAN.md "Reports": terminal output is plain ANSI with no third-party dependency.
One line per backend per oracle, findings expanded underneath with the first
divergent element index, the two values, and the tolerance that was in force.
Every finding names both the failing check and the implicated stage, since the
pair is what a reader acts on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["render"]


def render(result: Mapping[str, Any]) -> str:
    """Render a run result as the terminal report.

    Args:
        result: the run result, one record per backend per oracle.

    Returns:
        The report text, ready to print.
    """
    raise NotImplementedError("the terminal report lands in M1")
