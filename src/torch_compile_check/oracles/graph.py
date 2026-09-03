"""Graph health oracle.

PLAN.md "Oracles": compares graph break count and reasons, unique graph count
across a repeat call, and compile wall time; informational unless ``--fail-on
graph`` is set. Its bug class is recompile storms and silent fullgraph
regressions.

PLAN.md "graph": the counts come from ``torch._dynamo.explain`` and from
``torch._dynamo.utils.counters['stats']['unique_graphs']`` sampled around the
repeat call. Graph breaks are not bugs; they explain why a user is not getting
the speedup they expect. With ``--baseline FILE`` the oracle reports only new
breaks, which is the mode the GitHub Action uses.

Nothing here runs a model. :func:`torch_compile_check.runner.run_backend` does the
``explain`` pass and leaves a :class:`~torch_compile_check.results.GraphHealth` on the
lane's result; this module is the rules that read it, and it imports no torch.

Four rules decide a severity, and they are the whole oracle:

``fail``  the lane answered once and raised on the repeat call; or
          ``--fullgraph`` was requested and the graph broke anyway; or a
          baseline was given and a break appeared that is not in it.
``warn``  the repeat call with identical inputs compiled a new graph; or a
          baseline was given and has no entry for this lane.
``info``  everything else a break has to say.

Severity is this oracle's own judgement and does not consult ``--fail-on``,
exactly as it does not for the other four: PLAN.md "CLI surface for v1" makes
``--fail-on`` the rule for turning a finding into exit code 1, not the rule for
what a finding is. So a fullgraph break is a ``fail`` row in the checks table on
every run, and it becomes exit code 1 when the run asked for ``--fail-on graph``
-- which is what "informational by default" means here.

The baseline file is this module's format, so reading it, writing it, and
comparing against it live here rather than in a module of their own that
PLAN.md "Package layout" does not name.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch_compile_check.oracles.base import (
    Baseline,
    BaselineEntry,
    Finding,
    OracleConfig,
    Severity,
)
from torch_compile_check.results import BackendResult, GraphBreak, GraphHealth, RunSet

__all__ = [
    "BASELINE_COUNT_KEY",
    "BASELINE_REASONS_KEY",
    "BaselineError",
    "GraphOracle",
    "baseline_entry",
    "read_baseline",
    "summarise_reason",
    "write_baseline",
]

log = logging.getLogger("torch_compile_check")

# The two keys of one baseline entry, from the M3-1 brief's
# ``{backend: {graph_break_count, break_reasons[]}}``. Named because the reader,
# the writer, and the error messages all have to spell them the same way.
BASELINE_COUNT_KEY = "graph_break_count"
BASELINE_REASONS_KEY = "break_reasons"

# How long a reason summary is allowed to be. Dynamo's first line is a sentence
# ("Failed to trace builtin operator"); the cap is there for a torch that one
# day writes a paragraph, so that a baseline file stays readable in a diff.
MAX_REASON_CHARS = 160

# torch's own stable identifier for a break class, as it appears in the
# documentation link Dynamo appends to every reason
# (".../compile-graph-break-site/gb/gb0059.html"). Worth pulling out: the prose
# around it is free to be reworded by a torch upgrade and the id is not, so a
# summary that carries it survives one.
_GB_ID = re.compile(r"\bgb(\d{4,})\b")


class BaselineError(ValueError):
    """A ``--baseline`` file that cannot be read as one.

    A subclass of :class:`ValueError` because that is what a malformed file is,
    and because :mod:`torch_compile_check.cli` turns it into the tool-error exit code
    with the message printed as one line: a baseline that does not parse is a
    problem with the file the user named, not a crash of the run.
    """


class GraphOracle:
    """Did compilation capture the model, and does it keep capturing it?"""

    name: str = "graph"

    def compare(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Report this lane's graph health against the run's expectations.

        ``eager`` is unused, and that is the difference between this oracle and
        the other four. They ask whether two worlds agree; this one asks what
        happened to one of them, because eager has no graphs to disagree with.
        The parameter stays because it is the
        :class:`~torch_compile_check.oracles.base.Oracle` protocol.

        A lane that raised is still reported on. Every other oracle stops there
        (:func:`~torch_compile_check.oracles.base.align_outputs` returns nothing to
        compare), and under ``--fullgraph`` that is exactly the case worth
        explaining: the lane raised *because* the graph broke.
        """
        del eager
        findings = self._repeat_call(other)

        health = other.graph_health
        if health is None:
            # Not compiled, or a hand-built record. Silence, rather than a row
            # of zeroes that would read as a lane which traced cleanly.
            log.debug("no graph health recorded for backend %s", other.backend)
            return findings
        if health.explain_error is not None:
            findings.extend(self._unmeasured(other, health))
            return findings

        findings.extend(self._breaks(other, health, cfg))
        findings.extend(self._recompile(other, health))
        return findings

    def _repeat_call(self, other: BackendResult) -> list[Finding]:
        """Repeat-call health, the M1-3 carry-over this oracle took over.

        A backend that answers once and then throws produced a result that is
        not reproducible. PLAN.md "CLI surface for v1" left it recorded but
        unowned until the graph oracle existed; it is a ``fail`` here, so
        ``--fail-on graph`` makes it exit code 1 and the default run reports it
        without changing the verdict.
        """
        captured = other.second_call_exception
        if captured is None:
            return []
        return [
            Finding(
                oracle=self.name,
                backend=other.backend,
                output_index=None,
                severity="fail",
                message=(
                    f"{other.backend} answered the first call and raised "
                    f"{captured.type} on the repeat call with the same inputs: "
                    f"{_first_line(captured.message)}"
                ),
                details={
                    "field": "second_call",
                    "expected": "the same answer as the first call",
                    "got": f"raised {captured.type}",
                },
            )
        ]

    def _unmeasured(self, other: BackendResult, health: GraphHealth) -> list[Finding]:
        """The ``explain`` pass itself raised, so there are no counts to report.

        ``warn`` rather than ``info``: a lane that answered its calls and could
        still not be traced is surprising, and a graph row saying "pass" here
        would be claiming a clean trace nobody performed. It is not a ``fail``
        because what failed is the tool's measurement rather than the compiler's
        contract.

        Nothing at all when the lane itself raised. The target raises, so of
        course it raised under ``explain`` too; the exception is already
        reported against the lane and the stage verdict is already built from
        it, and saying it a second time in the graph oracle's words would turn
        one broken model into two findings. The checks table still shows a dash
        rather than a pass, because no graph health was recorded either way.
        """
        assert health.explain_error is not None
        if not other.ok:
            log.debug(
                "%s raised, and so did its explain pass (%s); the lane's own exception is "
                "the finding",
                other.backend,
                health.explain_error.type,
            )
            return []
        return [
            Finding(
                oracle=self.name,
                backend=other.backend,
                output_index=None,
                severity="warn",
                message=(
                    f"graph health was not measured for {other.backend}: the "
                    f"torch._dynamo.explain pass raised {health.explain_error.type}: "
                    f"{_first_line(health.explain_error.message)}"
                ),
                details={"field": "explain", "got": f"raised {health.explain_error.type}"},
            )
        ]

    def _breaks(
        self,
        other: BackendResult,
        health: GraphHealth,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Every graph break, at the severity this run's flags give it."""
        reported = _reported_breaks(health)
        if not reported:
            return []

        if cfg.fullgraph:
            # The strongest rule, applied before the baseline is consulted: a
            # baseline records breaks a run accepts, and --fullgraph is the run
            # saying it accepts none.
            return [
                self._break_finding(
                    other,
                    health,
                    summary,
                    item,
                    severity="fail",
                    why=(
                        "--fullgraph was requested and the graph broke anyway, so this "
                        "lane could not be captured as one graph"
                    ),
                )
                for summary, item in reported
            ]

        if cfg.baseline is None:
            return [
                self._break_finding(other, health, summary, item, severity="info", why="")
                for summary, item in reported
            ]

        return self._against_baseline(other, health, cfg.baseline, reported)

    def _against_baseline(
        self,
        other: BackendResult,
        health: GraphHealth,
        baseline: Baseline,
        reported: Sequence[tuple[str, GraphBreak | None]],
    ) -> list[Finding]:
        """PLAN.md "GitHub Action": fail on new breaks, and only on new breaks.

        A break the baseline already lists produces nothing at all. That is the
        point of the file: a real model has graph breaks on day one, and a check
        that reported them every run is a check people turn off.
        """
        entry = baseline.entries.get(other.backend)
        if entry is None:
            # Not a failure. A baseline that does not know this lane is a stale
            # baseline, and failing the run for it would punish the wrong thing;
            # saying so loudly and falling back to the informational reading is
            # what lets the user regenerate it.
            findings = [
                Finding(
                    oracle=self.name,
                    backend=other.backend,
                    output_index=None,
                    severity="warn",
                    message=(
                        f"{baseline.path} has no baseline for {other.backend}, so its "
                        f"{health.break_count} graph break"
                        f"{'' if health.break_count == 1 else 's'} are reported rather "
                        "than compared; rerun with --write-baseline to record them"
                    ),
                    details={"field": "baseline", "expected": other.backend, "got": "no entry"},
                )
            ]
            findings.extend(
                self._break_finding(other, health, summary, item, severity="info", why="")
                for summary, item in reported
            )
            return findings

        known = set(entry.break_reasons)
        # ``item is None`` is a break torch counted but recorded no reason for.
        # It has no identity to be new against, so it is left to the count rule
        # below rather than compared as a reason that is missing from the file.
        new = [
            (summary, item)
            for summary, item in reported
            if item is not None and summary not in known
        ]
        findings = [
            self._break_finding(
                other,
                health,
                summary,
                item,
                severity="fail",
                why=f"this break is not in {baseline.path}",
            )
            for summary, item in new
        ]
        if not new and health.break_count > entry.graph_break_count:
            # No unfamiliar reason, but more breaks than the file accepts: the
            # same break now happens more often, which is still a new break.
            findings.append(
                Finding(
                    oracle=self.name,
                    backend=other.backend,
                    output_index=None,
                    severity="fail",
                    message=(
                        f"{other.backend} broke the graph {health.break_count} times "
                        f"against {entry.graph_break_count} in {baseline.path}, with no "
                        "reason the baseline does not already list"
                    ),
                    details={
                        "field": "graph_break_count",
                        "expected": entry.graph_break_count,
                        "got": health.break_count,
                    },
                )
            )
        elif health.break_count < entry.graph_break_count:
            findings.append(
                Finding(
                    oracle=self.name,
                    backend=other.backend,
                    output_index=None,
                    severity="info",
                    message=(
                        f"{other.backend} broke the graph {health.break_count} times "
                        f"against {entry.graph_break_count} in {baseline.path}; the "
                        "baseline is looser than this run and can be tightened with "
                        "--write-baseline"
                    ),
                    details={
                        "field": "graph_break_count",
                        "expected": entry.graph_break_count,
                        "got": health.break_count,
                    },
                )
            )
        return findings

    def _break_finding(
        self,
        other: BackendResult,
        health: GraphHealth,
        summary: str,
        item: GraphBreak | None,
        *,
        severity: Severity,
        why: str,
    ) -> Finding:
        """One graph break, worded for the reader and detailed for the report.

        Dynamo's full explanation is deliberately not in ``details``: it is a
        dozen lines of hints per break, it is already on the lane's
        :class:`~torch_compile_check.results.GraphHealth` for the JSON and Markdown
        reports of M3-2, and putting it here would bury five findings in one
        screen of the same three URLs.
        """
        where = f" at {item.user_frame}" if item is not None and item.user_frame else ""
        tail = f"; {why}" if why else ""
        return Finding(
            oracle=self.name,
            backend=other.backend,
            output_index=None,
            severity=severity,
            message=f"{other.backend} broke the graph{where}: {summary}{tail}",
            details={
                "field": "break_reasons",
                "reason": summary,
                "user_frame": item.user_frame if item is not None else None,
                "graph_count": health.graph_count,
                "break_count": health.break_count,
                "op_count": health.op_count,
                "compile_wall_s": _seconds(other.first_call_s),
            },
        )

    def _recompile(self, other: BackendResult, health: GraphHealth) -> list[Finding]:
        """The repeat call compiled something new, on inputs it had already seen.

        PLAN.md "graph" names recompile storms as this oracle's bug class, and
        the counter moving across a call with identical inputs is the signal.
        ``warn`` and not ``fail``: it costs compile time rather than
        correctness, and the M3-1 brief fixes the three conditions that fail.
        """
        if not health.recompiled:
            return []
        return [
            Finding(
                oracle=self.name,
                backend=other.backend,
                output_index=None,
                severity="warn",
                message=(
                    f"{other.backend} compiled {health.recompiles} more graph"
                    f"{'' if health.recompiles == 1 else 's'} on the repeat call with "
                    "the same inputs, so a guard did not hold and the model recompiles "
                    "on every call"
                ),
                details={
                    "field": "unique_graphs",
                    "expected": health.unique_graphs_before,
                    "got": health.unique_graphs_after,
                    "second_call_s": _seconds(other.second_call_s),
                },
            )
        ]


def summarise_reason(reason: str) -> str:
    """A graph break reason as the one line a baseline stores and compares.

    Dynamo's reason is a paragraph: a headline, an explanation, two or three
    hints, a developer debug context, and a link to the documentation page for
    the break class. The headline is what a human reads and the ``gbNNNN`` id in
    that link is what survives a rewording, so the summary is both -- the id
    when there is one, and the headline always.

    Args:
        reason: the reason text as ``torch._dynamo.explain`` reported it.

    Returns:
        One line, whitespace collapsed and capped at :data:`MAX_REASON_CHARS`.
    """
    lines = [line.strip() for line in reason.splitlines() if line.strip()]
    headline = " ".join(lines[0].split()) if lines else "unknown graph break"
    if len(headline) > MAX_REASON_CHARS:
        headline = headline[: MAX_REASON_CHARS - 1].rstrip() + "…"
    found = _GB_ID.search(reason)
    return f"gb{found.group(1)}: {headline}" if found else headline


def baseline_entry(result: BackendResult) -> BaselineEntry | None:
    """The baseline entry one lane's run would record, or ``None``.

    ``None`` for a lane with no measured graph health: an uncompiled lane, or
    one whose ``explain`` pass raised. Writing a zero-break entry for either
    would put a clean baseline on disk for a run that never established one,
    and every later run would then compare against a fiction.

    Only the breaks that came with a reason are written. A break torch counted
    without recording why has no identity to store, and a placeholder in the
    file would be a line nobody can act on and that every later run would read
    as a break of its own; the count keeps it visible.
    """
    health = result.graph_health
    if health is None or not health.measured:
        return None
    return BaselineEntry(
        graph_break_count=health.break_count,
        break_reasons=tuple(summarise_reason(item.reason) for item in health.breaks),
    )


def read_baseline(path: str | Path) -> Baseline:
    """Parse a ``--baseline`` file.

    The format is the M3-1 brief's, and deliberately small enough to review in a
    pull request::

        {"inductor": {"graph_break_count": 1, "break_reasons": ["gb0059: ..."]}}

    Unknown keys inside an entry are ignored, so a file written by a later
    version still reads here; a wrong *shape* is not, because a baseline that
    silently parsed as empty would turn every accepted break into a new one.
    Both named keys are required for the same reason: defaulting a missing one
    is how a file that lost half an entry becomes the strictest baseline there
    is without saying so.

    Args:
        path: the file named by ``--baseline``.

    Returns:
        The parsed baseline, with its path attached for the findings to name.

    Raises:
        BaselineError: the file is missing, is not JSON, or is not this shape --
            which :mod:`torch_compile_check.cli` turns into exit code 2 with the
            message printed as one line.
    """
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineError(
            f"cannot read the graph baseline {location}: {exc.strerror or exc}; "
            "write one first with --write-baseline"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"the graph baseline {location} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BaselineError(
            f"the graph baseline {location} must be an object keyed by backend name, "
            f"got a {type(raw).__name__}"
        )

    entries: dict[str, BaselineEntry] = {}
    for backend, value in raw.items():
        entries[str(backend)] = _entry(location, str(backend), value)
    return Baseline(path=str(location), entries=entries)


def _entry(location: Path, backend: str, value: Any) -> BaselineEntry:
    """One backend's entry, checked field by field.

    Both keys are required. They used to default to ``0`` and ``[]``, which
    turned a truncated or hand-edited entry into the strictest baseline there
    is -- zero accepted breaks, no accepted reason -- so every break the lane
    really has came back as a *new* break and failed the job for a reason the
    message did not mention (M3-1 verifier). A file that cannot be read as a
    baseline is named as one, with the missing field in the sentence.
    """
    if not isinstance(value, dict):
        raise BaselineError(
            f"the graph baseline {location} has a {type(value).__name__} for backend "
            f"{backend!r}, expected an object with {BASELINE_COUNT_KEY} and "
            f"{BASELINE_REASONS_KEY}"
        )
    missing = [key for key in (BASELINE_COUNT_KEY, BASELINE_REASONS_KEY) if key not in value]
    if missing:
        raise BaselineError(
            f"the graph baseline {location} is missing "
            f"{' and '.join(repr(key) for key in missing)} for backend {backend!r}; "
            f"every entry needs {BASELINE_COUNT_KEY} and {BASELINE_REASONS_KEY}, and "
            "--write-baseline writes both"
        )
    count = value[BASELINE_COUNT_KEY]
    reasons = value[BASELINE_REASONS_KEY]
    # bool is an int in Python and would sail through the check below; a
    # baseline that recorded `true` breaks is a typo, not a count of one.
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise BaselineError(
            f"the graph baseline {location} has {BASELINE_COUNT_KEY}={count!r} for "
            f"backend {backend!r}, expected a non-negative integer"
        )
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise BaselineError(
            f"the graph baseline {location} has a {BASELINE_REASONS_KEY} for backend "
            f"{backend!r} that is not a list of strings"
        )
    return BaselineEntry(graph_break_count=count, break_reasons=tuple(reasons))


def write_baseline(path: str | Path, runset: RunSet) -> list[str]:
    """Write the graph health of every measured lane as a baseline file.

    Args:
        path: the file named by ``--write-baseline``. Its parent directory is
            created, because a baseline conventionally lives in a directory of
            its own (``.torch-compile-check/baseline.json`` in the Action's docs) and
            failing on the first run for a missing directory helps nobody.
        runset: the run to record.

    Returns:
        The backends written, in run order, so the caller can say what it wrote
        and can tell an empty baseline from a full one.

    Raises:
        OSError: the file could not be written. Left to the caller, which turns
            it into the tool-error exit code.
    """
    measured = ((name, baseline_entry(result)) for name, result in runset.results.items())
    payload = {
        name: {
            BASELINE_COUNT_KEY: entry.graph_break_count,
            BASELINE_REASONS_KEY: list(entry.break_reasons),
        }
        for name, entry in measured
        if entry is not None
    }
    location = Path(path)
    if location.parent and not location.parent.exists():
        location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return list(payload)


def _reported_breaks(health: GraphHealth) -> list[tuple[str, GraphBreak | None]]:
    """Every break to report, as ``(summary, record)`` pairs.

    Usually one pair per reason ``explain`` returned. The odd case is a lane
    whose break *count* is higher than the number of reasons it recorded --
    torch derives the count from the graphs and the reasons from the tracer, so
    the two can disagree -- and those extra breaks get a pair with no record
    rather than disappearing. A count with no reasons at all is the same case at
    its limit, which is what a lane Dynamo captured nothing for looks like.
    """
    pairs: list[tuple[str, GraphBreak | None]] = [
        (summarise_reason(item.reason), item) for item in health.breaks
    ]
    for _ in range(max(0, health.break_count - len(pairs))):
        pairs.append(("no reason recorded", None))
    return pairs


def _first_line(message: str) -> str:
    """The first non-blank line of a torch message, which is usually many."""
    for line in message.splitlines():
        if line.strip():
            return line.strip()
    return message.strip()


def _seconds(value: float | None) -> float | None:
    """A wall time rounded for a report, or ``None`` when the call did not happen."""
    return None if value is None else round(value, 4)
