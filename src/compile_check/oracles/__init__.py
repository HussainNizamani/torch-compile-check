"""The five oracles, one per clause of the compile contract.

PLAN.md "Definitions": the compile contract is that ``compile(f)(x)`` is
observationally equivalent to ``f(x)`` except for speed. Concretely: the same
numbers, the same aliasing and mutation behaviour, the same dtype, shape, and
stride, the same gradients, and no silent fallback. Each of the five oracles
checks one clause of that contract.

Two names, deliberately not one. :data:`ORACLE_NAMES` is the ``--fail-on``
vocabulary from PLAN.md "CLI surface for v1", which is fixed at five for v1 and
is what a typo is checked against; :data:`ORACLES` is the registry of the
oracles that actually run today, so a build reports on what it implements
without the CLI having to hard-code a second list. The two converged in M3-1,
when the graph oracle -- the last one missing -- landed. They are still two
names: the vocabulary is a promise to the command line and the registry is a
statement about this build, and a v0.2 oracle would separate them again.
"""

from __future__ import annotations

from collections.abc import Sequence

from compile_check.oracles.alias import AliasOracle
from compile_check.oracles.base import (
    DEFAULT_GRAD_TOL_FACTOR,
    SEVERITIES,
    Baseline,
    BaselineEntry,
    Finding,
    Oracle,
    OracleConfig,
    Severity,
)
from compile_check.oracles.grad import GradOracle
from compile_check.oracles.graph import GraphOracle
from compile_check.oracles.metadata import MetadataOracle
from compile_check.oracles.numerics import NumericsOracle
from compile_check.results import BackendResult

__all__ = [
    "DEFAULT_GRAD_TOL_FACTOR",
    "ORACLES",
    "ORACLE_NAMES",
    "SEVERITIES",
    "Baseline",
    "BaselineEntry",
    "Finding",
    "Oracle",
    "OracleConfig",
    "Severity",
    "run_oracles",
]

# The --fail-on vocabulary, in the order PLAN.md "Oracles" lists them.
ORACLE_NAMES: tuple[str, ...] = ("numerics", "alias", "metadata", "grad", "graph")

# The oracles a run can actually use, keyed by their --fail-on name. All five
# since M3-1; the lookup stays keyed rather than positional so a build that ever
# ships fewer reports on what it implements instead of silently passing.
ORACLES: dict[str, Oracle] = {
    "numerics": NumericsOracle(),
    "alias": AliasOracle(),
    "metadata": MetadataOracle(),
    "grad": GradOracle(),
    "graph": GraphOracle(),
}


def run_oracles(
    eager: BackendResult,
    other: BackendResult,
    cfg: OracleConfig,
    names: Sequence[str] | None = None,
) -> list[Finding]:
    """Run the implemented oracles over one backend and collect their findings.

    Args:
        eager: the reference world.
        other: the lane under test.
        cfg: the run's tolerances and flags.
        names: restrict to these categories; unknown or not-yet-implemented
            names are skipped, since :func:`compile_check.cli.parse_fail_on` is
            where a typo is reported. ``None`` runs every implemented oracle.

    Returns:
        Every finding, grouped by oracle in registry order.
    """
    selected = ORACLES.values() if names is None else [ORACLES[n] for n in names if n in ORACLES]
    findings: list[Finding] = []
    for oracle in selected:
        findings.extend(oracle.compare(eager, other, cfg))
    return findings
