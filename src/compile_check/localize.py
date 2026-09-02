"""Backend ablation ladder, stage verdict.

PLAN.md "Package layout": ``localize.py`` -- backend ablation ladder, stage
verdict.

PLAN.md "Stage localization": running more than one backend is not redundancy,
it is the diagnosis. A divergence first seen at ``aot_eager`` implicates Dynamo
capture, AOTAutograd, functionalization, or decompositions; at
``aot_eager_decomp_partition`` but not ``aot_eager``, decomposition or the
partitioner; at ``inductor`` only, lowering, scheduling, or codegen. This is the
ladder PyTorch maintainers walk by hand when triaging a compile bug, and doing
it automatically is a large part of what makes a generated report worth reading.

PLAN.md "Where divergence appears is not always where the fix belongs": the
verdict names the first backend whose output violates the contract, which is
where the divergence becomes *observable*, not necessarily where the defect
lives. The worked example is 191449, which collapses two outputs into one object
under ``inductor`` and not under ``aot_eager`` while the fix lives in
AOTAutograd's metadata analysis. Everything here is therefore worded "first
diverges at <backend>" and never "the bug is in <backend>"; :attr:`StageVerdict.note`
carries the caveat so that no report has to remember to add it.

Nothing here imports torch: a verdict is computed from the records in
:mod:`compile_check.results` and the findings the oracles already produced.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from compile_check.oracles.base import Finding
from compile_check.results import CapturedException, RunSet
from compile_check.runner import ABLATION_LADDER, FP64_BACKEND

__all__ = [
    "CLEAN",
    "MODEL",
    "NO_REFERENCE",
    "STAGES",
    "BackendSummary",
    "StageVerdict",
    "implicated_stage",
    "localize",
]

log = logging.getLogger("compile_check")

# The three verdicts that name no compilation stage, because no compilation
# stage is implicated.
CLEAN = "clean"
"""No backend diverged from eager."""

MODEL = "model"
"""Eager itself raised, so the model is broken before compile is involved.

PLAN.md "Runner semantics": the eager run is the reference world; if it raises,
the tool reports the model as broken and exits 2.
"""

NO_REFERENCE = "no reference"
"""No eager lane ran, so nothing was compared. Not the same as ``clean``."""

# PLAN.md "Stage localization", the table, one entry per rung of the ladder.
STAGES: dict[str, str] = {
    "aot_eager": "capture/AOTAutograd/decomposition",
    "aot_eager_decomp_partition": "decomposition/partitioner",
    "inductor": "inductor lowering/codegen",
}

# PLAN.md "Where divergence appears is not always where the fix belongs".
OBSERVABILITY_CAVEAT = (
    "that is where the divergence becomes observable, not necessarily where the fix belongs"
)


def implicated_stage(first_divergent_backend: str) -> str:
    """Return the compilation stage implicated by the first divergent backend.

    Args:
        first_divergent_backend: the earliest backend on the ablation ladder at
            which the finding reproduces.

    Returns:
        The stage verdict, as it appears in the report. A backend the ladder
        does not know -- a user is free to ask for any backend the installed
        torch registers -- is named rather than mapped, because guessing which
        stage an unknown backend implicates would be inventing a diagnosis.
    """
    stage = STAGES.get(first_divergent_backend)
    if stage is not None:
        return stage
    log.debug("no ladder entry for backend %s", first_divergent_backend)
    return f"the {first_divergent_backend} backend"


@dataclass(frozen=True)
class BackendSummary:
    """What one lane did, counted: the per-backend summary of a verdict."""

    backend: str

    fail: int = 0
    """Findings that break the contract. These, and only these, move the ladder."""

    warn: int = 0
    """Findings that are a legitimate choice worth seeing (a layout difference)."""

    info: int = 0
    """Context that is never a verdict (the fp64 reference distance)."""

    raised: CapturedException | None = None
    """Set when the lane's first call raised, so it produced nothing to compare."""

    raised_on_repeat: CapturedException | None = None
    """Set when the lane answered once and raised on the repeat call.

    Recorded and reported, deliberately not localized on. An answer that does
    not reproduce is graph health, which PLAN.md "Oracles" gives to the graph
    oracle, informational unless ``--fail-on graph`` is set; that oracle lands
    in M3. Letting it set the stage here would produce a correctness verdict out
    of a check that has not been written yet.
    """

    @property
    def diverged(self) -> bool:
        """Whether this lane is a divergence the ladder should stop at.

        The rule of the M1-3 brief: a lane diverges if it raised, or if any
        oracle produced a fail-severity finding against it. A ``warn`` does not
        count -- a contiguous-to-contiguous stride change is the metadata
        oracle's example of a difference that is not a defect -- and neither
        does an ``info``.
        """
        return self.raised is not None or self.fail > 0


@dataclass(frozen=True)
class StageVerdict:
    """The stage-localization verdict, in the form every report prints."""

    stage: str
    """:data:`CLEAN`, :data:`MODEL`, :data:`NO_REFERENCE`, or a stage from
    :data:`STAGES`."""

    first_divergent_backend: str | None
    """The first lane on the ladder that diverged, or ``None`` when none did.

    ``"eager"`` when the model itself raised: the ladder never got started.
    """

    summary: str
    """The verdict as one sentence, already worded for a human."""

    note: str | None
    """The sentence under it: the observability caveat, or why there is no verdict."""

    backends: tuple[BackendSummary, ...]
    """One entry per lane that ran, in ablation-ladder order."""

    eager_exception: CapturedException | None = None
    """The reference world's exception, when :attr:`stage` is :data:`MODEL`."""

    @property
    def clean(self) -> bool:
        """Whether every lane agreed with eager."""
        return self.stage == CLEAN

    @property
    def compared(self) -> bool:
        """Whether a comparison happened at all.

        ``False`` for :data:`MODEL` and :data:`NO_REFERENCE`, which are the two
        outcomes a caller must not read as "checked and clean".
        """
        return self.stage not in (MODEL, NO_REFERENCE)


def localize(runset: RunSet, findings: Sequence[Finding] = ()) -> StageVerdict:
    """Walk the ablation ladder and name the stage the divergence first appears in.

    The rules, in the order they are applied:

    1. no eager lane -- :data:`NO_REFERENCE`, nothing was compared;
    2. eager raised -- :data:`MODEL`, the model is broken before compile is
       involved, and the ladder stops there;
    3. otherwise the first lane in :data:`~compile_check.runner.ABLATION_LADDER`
       order that raised or drew a fail-severity finding names the stage, via
       :func:`implicated_stage`;
    4. no such lane -- :data:`CLEAN`.

    Args:
        runset: the run to read, from :func:`compile_check.runner.run_all`.
        findings: every finding the oracles produced for that run, in any order.

    Returns:
        The verdict, with a per-backend summary of the counts behind it.
    """
    summaries = _summarize(runset, findings)
    eager = runset.eager

    if eager is None:
        return StageVerdict(
            stage=NO_REFERENCE,
            first_divergent_backend=None,
            summary="nothing was compared: this run has no eager lane to be the reference",
            note=("PLAN.md makes eager the reference world; add it to --backends to get a verdict"),
            backends=summaries,
        )

    if not eager.ok:
        assert eager.exception is not None
        return StageVerdict(
            stage=MODEL,
            first_divergent_backend="eager",
            summary=(
                f"the model raised {eager.exception.type} under eager, "
                "so nothing was compiled and nothing was compared"
            ),
            note=_first_line(eager.exception.message),
            backends=summaries,
            eager_exception=eager.exception,
        )

    first = next((entry.backend for entry in summaries if entry.diverged), None)
    if first is None:
        return StageVerdict(
            stage=CLEAN,
            first_divergent_backend=None,
            summary=f"clean: no backend diverged from eager across {len(summaries) - 1} lanes",
            note=None,
            backends=summaries,
        )

    return StageVerdict(
        stage=implicated_stage(first),
        first_divergent_backend=first,
        summary=f"first diverges at {first}, which implicates {implicated_stage(first)}",
        note=OBSERVABILITY_CAVEAT,
        backends=summaries,
    )


def _summarize(runset: RunSet, findings: Sequence[Finding]) -> tuple[BackendSummary, ...]:
    """Count the findings per lane, with the lanes in ablation-ladder order.

    The eager lane is summarized too, so the report can show what the reference
    world did, but it is never a candidate for the divergence point: findings
    are stamped with the lane compared *against* eager, so eager's counts are
    zero by construction and :meth:`BackendSummary.diverged` on it would only be
    reachable through its own exception, which rule 2 has already handled.
    """
    counts: dict[str, dict[str, int]] = {
        name: {"fail": 0, "warn": 0, "info": 0} for name in runset.results
    }
    for finding in findings:
        bucket = counts.get(finding.backend)
        if bucket is None:
            # A finding against a lane that is not in this run. The fp64
            # reference is stamped with the lane under test, not with itself, so
            # this is a caller mixing two runs rather than an expected case.
            log.warning(
                "finding from oracle %s names backend %r, which is not in this run",
                finding.oracle,
                finding.backend,
            )
            continue
        bucket[finding.severity] += 1

    return tuple(
        BackendSummary(
            backend=name,
            fail=counts[name]["fail"],
            warn=counts[name]["warn"],
            info=counts[name]["info"],
            raised=runset.results[name].exception,
            raised_on_repeat=runset.results[name].second_call_exception,
        )
        for name in _ladder_order(runset)
    )


def _ladder_order(runset: RunSet) -> list[str]:
    """The lanes that ran, ordered by the ablation ladder rather than by run order.

    The ladder is the diagnosis, so the order it is walked in must not depend on
    the order the user typed ``--backends`` in. Eager comes first because the
    ladder starts there; a backend the ladder does not know keeps its run-order
    position at the end, because the ladder cannot place it and dropping it
    would hide a lane that diverged.
    """
    names = [name for name in runset.results if name != FP64_BACKEND]
    known = [name for name in ABLATION_LADDER if name in names]
    return known + [name for name in names if name not in ABLATION_LADDER]


def _first_line(message: str) -> str:
    """The first non-blank line of a torch message, which is usually many."""
    for line in message.splitlines():
        if line.strip():
            return line.strip()
    return message.strip()
