"""Entry and input resolution.

PLAN.md "Package layout": ``discover.py`` -- entry and input resolution.

PLAN.md "Discovery convention": given a file path with no overrides, import the
module and look for a module-level ``model`` (an ``nn.Module``) or ``fn``, in
that order, then for a module-level ``inputs`` or a callable ``get_inputs()``,
in that order. An override flag wins over a discovered symbol. If neither
resolves, the tool exits 2 naming the two symbols it looked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["resolve"]


def resolve(
    path: Path,
    *,
    entry: str | None = None,
    inputs: str | None = None,
) -> tuple[Any, Any]:
    """Return the ``(callable, inputs)`` pair to test.

    Args:
        path: the python file to import.
        entry: ``module:callable`` override for the model or function.
        inputs: ``module:callable`` override for the input factory.

    Returns:
        The callable under test and the inputs to call it with.
    """
    raise NotImplementedError("discovery lands in M1")
