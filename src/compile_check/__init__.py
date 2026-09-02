"""compile-check: differential testing for ``torch.compile``.

Bring your own model; compile-check tells you whether ``torch.compile`` changed
its answers, and if so hands you a minimal repro and a ready-to-file report.

Status: M1 in progress. :mod:`compile_check.env`, :mod:`compile_check.discover`,
:mod:`compile_check.runner`, and the numerics and metadata oracles are
implemented; the alias, grad, and graph oracles, the localizer, the minimizer,
and the reports are still typed stubs that raise :class:`NotImplementedError`.
The CLI's main path is not wired up until M1-3.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Kept in step with the ``version`` field of pyproject.toml; tests/test_cli.py
# fails if the two drift apart.
__version__ = "0.0.1.dev0"
