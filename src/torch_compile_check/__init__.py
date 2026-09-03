"""torch-compile-check: differential testing for ``torch.compile``.

Bring your own model; torch-compile-check tells you whether ``torch.compile`` changed
its answers, and if so hands you a minimal repro and a ready-to-file report.

Status: the v1 surface of PLAN.md is complete and this is 0.1.0.
:mod:`torch_compile_check.discover` resolves a target, :mod:`torch_compile_check.runner`
runs it under every backend, all five oracles of PLAN.md "Oracles" compare the
lanes, :mod:`torch_compile_check.localize` names the compilation stage a divergence
first appears in, :mod:`torch_compile_check.minimize` shrinks a reproducing case, and
:mod:`torch_compile_check.report` writes the terminal report, the JSON artifact, the
Markdown issue draft, and a regression test. What is *not* in v1 is listed in
PLAN.md "Non-goals for v1" and "v0.2 outlook", and the terminal report says so
rather than letting an unwritten check read as a check that passed.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Kept in step with the ``version`` field of pyproject.toml; tests/test_cli.py
# fails if the two drift apart.
__version__ = "0.1.0"
