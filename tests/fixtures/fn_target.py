"""A plain function target returning a dict, so the pytree path is exercised.

Discovery convention, second alternative on both lines: no ``model``, so ``fn``
is used; no ``inputs``, so ``get_inputs()`` is called.
"""

from __future__ import annotations

import torch


def fn(x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Two outputs of different dtypes, under a dict rather than a tuple."""
    z = torch.tanh(x @ y)
    return {"z": z, "positive": (z > 0).sum()}


def get_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(3, 5, generator=generator),
        torch.randn(5, 2, generator=generator),
    )
