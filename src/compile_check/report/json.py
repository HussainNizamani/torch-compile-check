"""JSON report.

PLAN.md "Reports": JSON is versioned with a top-level ``schema_version``
integer, bumped on any incompatible field change. It carries the environment
block (architecture always included, see cross-architecture parity), the run
configuration, and one record per backend per oracle with a machine-readable
finding list. This is the CI-consumable artifact, and it is the unit of
comparison for cross-architecture parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["SCHEMA_VERSION", "dump"]

SCHEMA_VERSION = 1


def dump(result: Mapping[str, Any], path: Path) -> None:
    """Write a run result to *path* as versioned JSON.

    Args:
        result: the run result, one record per backend per oracle.
        path: the file to write.
    """
    raise NotImplementedError("the JSON report lands in M3")
