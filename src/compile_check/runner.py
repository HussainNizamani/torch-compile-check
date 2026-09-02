"""Seeding, cloning, reset, per-backend execution.

PLAN.md "Package layout": ``runner.py`` -- seeding, cloning, reset, per-backend
execution.

PLAN.md "Runner semantics": the runner establishes that any difference it
reports comes from the backend and not from the harness. The RNG is seeded
identically before every run, inputs are deep cloned per backend, each backend
runs after ``torch.compiler.reset()``, caches are disabled by default, the eager
run is the reference world, and each backend is called twice so the graph oracle
can see whether a recompile happened.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_backend"]


def run_backend(
    fn: Any,
    inputs: Any,
    backend: str,
    *,
    seed: int = 0,
    fullgraph: bool = False,
    dynamic: bool = False,
) -> dict[str, Any]:
    """Run ``fn`` under one backend and return everything the oracles compare.

    Args:
        fn: the callable under test.
        inputs: the inputs, deep cloned before use.
        backend: ``eager``, ``aot_eager``, ``aot_eager_decomp_partition``, or
            ``inductor``.
        seed: RNG seed, applied before the run.
        fullgraph: passed through to ``torch.compile``.
        dynamic: passed through to ``torch.compile``.

    Returns:
        The run record: outputs, input state before and after, and graph
        statistics.
    """
    raise NotImplementedError("the runner lands in M1")
