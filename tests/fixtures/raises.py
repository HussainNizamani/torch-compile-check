"""A target that always raises, for the exception-capture test.

A backend that raises is a result the stage localization reads, not a crash, so
the runner records it rather than letting it out.
"""

from __future__ import annotations

import torch


def fn(x: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("this target is broken on purpose")


inputs = (torch.ones(2, 2),)
