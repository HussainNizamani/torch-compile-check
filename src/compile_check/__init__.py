"""compile-check: differential testing for ``torch.compile``.

Bring your own model; compile-check tells you whether ``torch.compile`` changed
its answers, and if so hands you a minimal repro and a ready-to-file report.

Status: M0 scaffold. Only :mod:`compile_check.env` does real work; every other
module is a typed stub that raises :class:`NotImplementedError`.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Kept in step with the ``version`` field of pyproject.toml; tests/test_cli.py
# fails if the two drift apart.
__version__ = "0.0.1.dev0"
