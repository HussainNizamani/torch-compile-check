"""Test-session setup that has to happen before torch is imported.

``torch._inductor.config.force_disable_caches`` is read from the environment
when torch is imported, so setting the variable inside a test is too late: the
first test module (or fixture) that imports torch would have fixed the value
already. conftest.py is imported before any test module, which makes this the
one place it can be done.

The CLI does the same thing for a real run, in ``main()`` before its own torch
import (PLAN.md "Runner semantics": caches are disabled by default by setting
``TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`` before torch does any compiling).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

assert "torch" not in sys.modules, "conftest.py must run before torch is imported"

FIXTURES = Path(__file__).resolve().parent / "fixtures"
