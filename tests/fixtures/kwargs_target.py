"""A target whose ``inputs`` is a mapping, so it is passed as keyword arguments."""

from __future__ import annotations

import torch


def fn(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale


inputs = {"x": torch.full((2, 3), 1.5), "scale": 2.0}
