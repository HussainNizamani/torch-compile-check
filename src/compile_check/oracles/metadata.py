"""Metadata oracle.

PLAN.md "Oracles": compares dtype, shape, stride, ``requires_grad``, device, and
contiguity, per output; passes on exact equality on every field. It is the
oracle that catches 191308, int8 matmul silently promoted to int64.

PLAN.md "metadata": stride is compared but reported at a lower severity than
dtype and shape, because a layout change alone is usually a performance decision
rather than a correctness defect. It still appears in the report, since a stride
change combined with an alias change is how a reinplacing bug presents.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["check"]


def check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare the two runs' per-output metadata.

    Args:
        eager: the reference run record.
        compiled: the run record under test.

    Returns:
        One record per field that differs; empty when every field matches.
    """
    raise NotImplementedError("the metadata oracle lands in M1")
