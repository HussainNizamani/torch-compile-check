"""Inductor-suite regression test emitter.

PLAN.md "Package layout": ``report/pytest_case.py`` -- inductor-suite regression
test emitter.

PLAN.md "Regression test emission": alongside the repro, the tool emits the same
case as a drop-in regression test in the idiom the inductor suite already uses,
a ``common``-style eager versus compiled comparison written as a test method
body suitable for ``test/inductor/test_torchinductor.py``. The claim is not that
the test applies unmodified; it is that the test is half-written instead of
unwritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["emit"]


def emit(result: Mapping[str, Any]) -> str:
    """Emit the finding as an inductor-suite-style test method.

    Args:
        result: the run result the finding came from.

    Returns:
        The test method source.
    """
    raise NotImplementedError("the regression test emitter lands in M3")
