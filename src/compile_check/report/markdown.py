"""Markdown report.

PLAN.md "Reports": Markdown is an issue draft formatted the way PyTorch issues
expect -- a short description, the minimal repro inline as a fenced Python
block, expected versus actual, the stage-localization verdict, the emitted
regression test, and an environment block with torch version and git hash,
Python version, OS, architecture, CPU or GPU model, and the backend
configuration that was in force. The tool drafts; the human reads it, edits it,
and files it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["render"]


def render(result: Mapping[str, Any]) -> str:
    """Render a run result as a Markdown issue draft.

    Args:
        result: the run result, one record per backend per oracle.

    Returns:
        The Markdown draft.
    """
    raise NotImplementedError("the Markdown report lands in M3")
