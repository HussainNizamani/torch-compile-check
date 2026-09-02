"""Test-session setup that has to happen before torch is imported.

``torch._inductor.config.force_disable_caches`` is read from the environment
when torch is imported, so setting the variable inside a test is too late: the
first test module (or fixture) that imports torch would have fixed the value
already. conftest.py is imported before any test module, which makes this the
one place it can be done.

The CLI does the same thing for a real run, in ``main()`` before its own torch
import (PLAN.md "Runner semantics": caches are disabled by default by setting
``TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`` before torch does any compiling).

The second variable is about disk rather than about correctness. Disabling the
caches stops inductor *reading* an earlier artifact; it does not stop it
*writing* the generated code, which lands under ``/tmp/torchinductor_<user>``
and is never cleaned up. On the development box that directory reached roughly
800 MB across local runs. The suite therefore points ``TORCHINDUCTOR_CACHE_DIR``
at a directory of its own and removes it when the session ends, so a test run
costs no permanent disk. An outer value is respected and left alone, so a
developer who has already pointed the variable somewhere keeps that choice.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

CACHE_DIR_VAR = "TORCHINDUCTOR_CACHE_DIR"

# None when the environment already named a directory: it is not ours, so it is
# not ours to delete.
_OWNED_CACHE_DIR: Path | None = None
if not os.environ.get(CACHE_DIR_VAR):
    _OWNED_CACHE_DIR = Path(tempfile.mkdtemp(prefix="compile-check-inductor-"))
    os.environ[CACHE_DIR_VAR] = str(_OWNED_CACHE_DIR)

assert "torch" not in sys.modules, "conftest.py must run before torch is imported"

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_sessionfinish(session, exitstatus) -> None:
    """Delete the codegen directory this session created, if it created one."""
    del session, exitstatus  # the hook's signature, not something this needs
    if _OWNED_CACHE_DIR is not None:
        shutil.rmtree(_OWNED_CACHE_DIR, ignore_errors=True)
