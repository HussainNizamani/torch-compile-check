"""Numerics oracle.

PLAN.md "Oracles": compares output tensor values, per output; passes when
``assert_close`` holds within the per-dtype tolerance, plus NaN and inf position
parity. It is the oracle for the 190765 CPU inductor miscompile and for the
194593 and 194596 divergent validation branches.

PLAN.md "numerics": the comparison runs with ``check_dtype=False`` and
``check_stride=False``, because dtype and stride are the metadata oracle's job
and reporting one divergence twice hides which one is the real defect. A
tolerance-level difference is never called a bug; a NaN or inf that appears or
disappears is, because that is a category difference rather than a rounding one.
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
    """Compare the two runs' outputs and return the findings.

    Args:
        eager: the reference run record.
        compiled: the run record under test.
        rtol: relative tolerance override for every dtype.
        atol: absolute tolerance override for every dtype.

    Returns:
        One record per divergence; empty when the runs agree.
    """
    raise NotImplementedError("the numerics oracle lands in M1")
