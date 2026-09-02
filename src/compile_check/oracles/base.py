"""What an oracle is: a finding, a run configuration, and the shared protocol.

PLAN.md "Oracles": five oracles run against every backend result, each with a
defined comparison, a pass rule, and a known bug it would have caught. They
differ in what they compare and agree on everything else, and this module is the
everything else: the record one divergence produces, the knobs a run passes
down, and the one method the report and the localizer call.

Two rules from :mod:`compile_check.results` hold here as well. Nothing imports
torch, so ``compile_check.cli`` can name the oracle registry without paying for
the torch import; tensors are therefore typed ``Any`` and every torch call lives
inside an oracle's own lazily imported function. And a :class:`Finding` is data:
its ``details`` carry strings, numbers, and lists rather than live tensors, so
that the same record survives the terminal report of M1-3 and the JSON report of
M3 without either having to re-derive it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from compile_check.results import BackendResult

__all__ = [
    "DEFAULT_GRAD_TOL_FACTOR",
    "SEVERITIES",
    "Finding",
    "Oracle",
    "OracleConfig",
    "Severity",
    "align_outputs",
]

log = logging.getLogger("compile_check")

# What the grad oracle multiplies the numerics tolerances by, and why the
# number is a flag rather than a constant.
#
# A gradient is a sum over every path that reaches a tensor, and compilation is
# free to fuse and reassociate that sum, so the float32 error in a gradient
# accumulates over the whole backward rather than over one output element. The
# M2-2 verification hit the boundary of that: a compiled resnet18 gradient
# about 1.24e-5 from eager's against a float32 atol of 1e-5, so the same run
# came back clean or failing depending on which way the last bit fell. Ten is
# one decimal order, which clears that case, and it is small enough that a real
# backward-only divergence -- orders of magnitude, not factors -- still fails.
#
# Ten is not enough for every model, and this is the honest record of that, per
# PLAN.md "Tolerance policy" ("start with the assert_close defaults, measure the
# false-positive rate on the validation set, and only move to per-op tolerances
# if the defaults prove too tight"). Measured on torchvision resnet18 at
# 2x3x64x64 on torch 2.14.0+cpu, aarch64, CPU, seed 0, and the training mode is
# half the measurement rather than an incidental detail:
#
#   model.eval()   inductor needs a factor of 1.0 to pass (worst gradient
#                  1.34e-5 against an atol of 1e-5), which is the borderline
#                  above, sitting on the wrong side of it by a last bit.
#   model.train()  inductor needs a factor of about 161 -- 61 of 63 gradients
#                  are past 1x and 21 are past 10x.
#
# Batch norm is the difference. In eval mode it is an affine map over frozen
# running statistics; in train mode the normalisation is over the batch, so
# every gradient reaching conv1 travels back through a mean and a variance
# computed from the activations, and the reassociation compilation is free to do
# has that much further to accumulate over. `validation/targets/tv_resnet18.py`
# is the eval-mode row, deliberately (its docstring says why); the train-mode
# row is the same architecture with `model.train()`, which is how anyone
# actually differentiating a resnet runs it.
#
# aot_eager reproduces eager's gradients bit for bit in both modes, so the
# spread is inductor's codegen and not the harness. It is not a miscompile
# either: against a float64 eager reference the two lanes sit at the same order
# of error in the train-mode row (worst per-tensor max|diff|/max|ref|: 3.4e-5
# for eager, 3.9e-5 for inductor), which is PLAN.md "The oracle blind spot"'s
# "both imprecise" rather than "compiled is wrong": the assert_close float32 defaults
# are simply below the noise floor of a deep backward, and no single constant
# is right for both a two-layer MLP (which needs 1x, bit-identical) and a
# resnet. So the default clears the measured borderline case, the flag is how a
# real model is run, and per-op or per-model tolerances are the documented next
# step rather than a bigger constant here.
#
# The tolerances for *outputs* stay exactly where they were: the reason for the
# looser rule is the backward's accumulation, and nothing else.
# `--grad-tol-factor 1` turns it off.
DEFAULT_GRAD_TOL_FACTOR = 10.0

Severity = Literal["fail", "warn", "info"]
"""``fail`` breaks the contract, ``warn`` is a legitimate choice worth seeing,
``info`` is context that is never a verdict (PLAN.md "Oracles")."""

# Ordered strongest first, which is the order the report groups by.
SEVERITIES: tuple[Severity, ...] = ("fail", "warn", "info")


@dataclass(frozen=True)
class Finding:
    """One divergence, in the form every report renders and the CLI counts."""

    oracle: str
    """The oracle that produced it, one of :data:`compile_check.oracles.ORACLE_NAMES`."""

    backend: str
    """The lane compared against eager, e.g. ``"inductor"``."""

    output_index: int | None
    """Index into the flattened outputs, or ``None`` for a whole-run finding.

    ``None`` is what a structural divergence gets: "eager returned 2 outputs and
    inductor returned 3" belongs to no single index.
    """

    severity: Severity
    message: str
    """One line, already written for a human; the report prints it as it is."""

    details: dict[str, Any] = field(default_factory=dict)
    """Machine-readable context: what was expected, what came back, and the
    tolerances or fields the decision was made with."""


@dataclass(frozen=True)
class OracleConfig:
    """Everything the oracles need from the command line, in one object.

    Passed to every :meth:`Oracle.compare` so that adding a knob does not change
    five signatures.
    """

    rtol: float | None = None
    """``--rtol``: overrides the per-dtype default for every dtype, or ``None``."""

    atol: float | None = None
    """``--atol``: the same for the absolute tolerance."""

    grad: bool = True
    """Whether the run ran a backward pass at all, i.e. not ``--no-grad``.

    The grad oracle needs to tell "no tensor in this run required a gradient"
    from "the user switched the check off", and the records alone cannot: both
    leave every lane with no gradients. Carried here so the second case can be
    said out loud in the report rather than passing as a clean grad row.
    """

    grad_tol_factor: float = DEFAULT_GRAD_TOL_FACTOR
    """``--grad-tol-factor``: what the grad oracle multiplies the tolerances by.

    Applies to gradient *values* only. The presence set is a set comparison with
    no tolerance to widen, and the output tolerances are deliberately untouched:
    the looser rule exists because of how a backward pass accumulates, and
    lending it to the forward comparison would weaken the oracle that catches
    190765. See :data:`DEFAULT_GRAD_TOL_FACTOR` for the measurement behind the
    default.
    """

    fp64: bool = False
    """``--fp64-oracle``: compare both worlds against an fp64 eager reference.

    PLAN.md "The oracle blind spot": the flag exists to separate "compiled is
    wrong" from "both are imprecise", and it costs an extra run, which is why it
    is a flag and not the default.
    """

    fp64_reference: BackendResult | None = None
    """The ``eager_fp64`` pseudo-backend result, when the runner produced one.

    Carried on the config rather than on the RunSet lane list because it is not
    a backend under test: it is a reference the numerics oracle reads, and the
    stage localizer of M1-3 must never treat it as a lane that diverged.
    """


@runtime_checkable
class Oracle(Protocol):
    """One clause of the compile contract, checked against the eager reference."""

    name: str
    """The ``--fail-on`` category this oracle's findings are counted under."""

    def compare(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Compare one backend against the eager reference.

        Args:
            eager: the reference world (PLAN.md "Runner semantics").
            other: the lane under test.
            cfg: the run's tolerances and flags.

        Returns:
            Every divergence found, in output order; empty when this oracle's
            clause of the contract holds.
        """
        ...


def align_outputs(
    eager: BackendResult,
    other: BackendResult,
    oracle: str,
) -> tuple[list[tuple[int, Any, Any]], list[Finding]]:
    """Pair the two runs' output leaves up, and report a structural divergence.

    Every oracle needs the same two guards before it can compare anything, and
    both guards are findings in their own right, so they live here rather than
    being written twice and worded differently.

    A lane that raised is not a divergence this function reports. The exception
    is already on the :class:`~compile_check.results.BackendResult`, PLAN.md
    "Stage localization" is what reads it, and turning it into a finding per
    oracle would report one failure three times.

    Args:
        eager: the reference run.
        other: the run under test.
        oracle: the name to stamp on any structural finding.

    Returns:
        The ``(index, eager_leaf, other_leaf)`` triples that can be compared,
        and the findings that describe why some could not be.
    """
    if not eager.ok or not other.ok:
        log.debug(
            "%s: nothing to compare, %s raised",
            oracle,
            "eager" if not eager.ok else other.backend,
        )
        return [], []

    findings: list[Finding] = []
    if (
        eager.output_spec is not None
        and other.output_spec is not None
        and eager.output_spec != other.output_spec
    ):
        findings.append(
            Finding(
                oracle=oracle,
                backend=other.backend,
                output_index=None,
                severity="fail",
                message=(
                    f"output structure differs: eager returned {eager.output_spec}, "
                    f"{other.backend} returned {other.output_spec}"
                ),
                details={
                    "field": "output_spec",
                    "expected": str(eager.output_spec),
                    "got": str(other.output_spec),
                },
            )
        )
    if len(eager.outputs) != len(other.outputs):
        findings.append(
            Finding(
                oracle=oracle,
                backend=other.backend,
                output_index=None,
                severity="fail",
                message=(
                    f"output count differs: eager returned {len(eager.outputs)} leaves, "
                    f"{other.backend} returned {len(other.outputs)}"
                ),
                details={
                    "field": "output_count",
                    "expected": len(eager.outputs),
                    "got": len(other.outputs),
                },
            )
        )

    # The leaves that both runs have are still worth comparing: "three of the
    # four outputs also have the wrong dtype" is more useful than the count
    # difference alone.
    pairs = [
        (index, expected, got)
        for index, (expected, got) in enumerate(zip(eager.outputs, other.outputs, strict=False))
    ]
    return pairs, findings
