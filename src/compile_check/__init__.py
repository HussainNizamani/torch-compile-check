"""compile-check: differential testing for ``torch.compile``.

Bring your own model; compile-check tells you whether ``torch.compile`` changed
its answers, and if so hands you a minimal repro and a ready-to-file report.

Status: M3 in progress. The CLI's main path runs: :mod:`compile_check.discover`
resolves a target, :mod:`compile_check.runner` runs it under every backend, all
five oracles of PLAN.md "Oracles" compare the lanes, :mod:`compile_check.localize`
names the compilation stage a divergence first appears in, and
:mod:`compile_check.report.terminal` prints the report. The minimizer and the
JSON, Markdown, and pytest-case reports (M3-2, M3-3) are still typed stubs that
raise :class:`NotImplementedError`, and the terminal report says so rather than
letting an unwritten feature read as a check that passed.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Kept in step with the ``version`` field of pyproject.toml; tests/test_cli.py
# fails if the two drift apart.
__version__ = "0.0.1.dev0"
