"""A target that runs under eager and blows up the moment it is compiled.

PLAN.md "CLI surface for v1", fixed in M1-3: a compiled lane that raises while
eager does not is a divergence and exits 1 regardless of ``--fail-on``, because
an exception belongs to no oracle category. Until this fixture that rule was
only reachable by monkeypatching the runner, which tests the exit-code branch
but not that a real backend failure ever reaches it.

The failure is a backend of our own, registered by name, rather than an op that
happens to break on some torch version: a backend that raises is the one way to
make the compiled path fail on every torch and every architecture without also
being a bug report about the installed wheel. ``fn`` itself is a multiply, so
the eager lane is healthy and the run has a reference world.

Registration is guarded because this module is imported twice in a normal test
run -- once by the test, so the name exists before the CLI validates
``--backends``, and once by the CLI's own discovery -- and
``register_backend`` refuses a duplicate name.
"""

from __future__ import annotations

from typing import Any

import torch

BACKEND = "torch_compile_check_raises"
"""The backend name to pass to ``--backends``."""

MESSAGE = "this backend raises on purpose"


def _raise_on_compile(gm: Any, example_inputs: Any) -> Any:
    """A torch.compile backend that never returns a compiled callable."""
    del gm, example_inputs
    raise RuntimeError(MESSAGE)


if BACKEND not in torch._dynamo.list_backends(exclude_tags=()):
    torch._dynamo.register_backend(_raise_on_compile, name=BACKEND)


def fn(x: torch.Tensor) -> torch.Tensor:
    return x * 2


inputs = (torch.ones(2, 3),)
