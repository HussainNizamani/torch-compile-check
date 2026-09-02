"""Backend ablation ladder, stage verdict.

PLAN.md "Package layout": ``localize.py`` -- backend ablation ladder, stage
verdict.

PLAN.md "Stage localization": running more than one backend is not redundancy,
it is the diagnosis. A divergence first seen at ``aot_eager`` implicates Dynamo
capture, AOTAutograd, functionalization, or decompositions; at
``aot_eager_decomp_partition`` but not ``aot_eager``, decomposition or the
partitioner; at ``inductor`` only, lowering, scheduling, or codegen.
"""

from __future__ import annotations

__all__ = ["implicated_stage"]


def implicated_stage(first_divergent_backend: str) -> str:
    """Return the compilation stage implicated by the first divergent backend.

    Args:
        first_divergent_backend: the earliest backend on the ablation ladder at
            which the finding reproduces.

    Returns:
        The stage verdict, as it appears in the report.
    """
    raise NotImplementedError("stage localization lands in M1")
