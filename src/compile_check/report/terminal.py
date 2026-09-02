"""Terminal report.

PLAN.md "Reports": terminal output is plain ANSI with no third-party dependency.
One line per backend per oracle, findings expanded underneath with the first
divergent element index, the two values, and the tolerance that was in force.
Every finding names both the failing check and the implicated stage, since the
pair is what a reader acts on.

The report is six blocks, in the order a reader needs them: what ran, where it
ran (PLAN.md "Cross-architecture parity is a feature" puts the architecture in
every environment block, because a run whose provenance is ambiguous is not
usable as parity evidence), what each lane did, an oracle-by-backend table, the
findings themselves, and the stage verdict.

:func:`render` returns a string and decides nothing: colour is a parameter
rather than a call to ``isatty`` in here, so the same report can be rendered for
a terminal, for a test, and for a log without the renderer having to guess which
it is. ``cli.py`` is where the TTY question is answered.

Nothing here imports torch. A report is rendered from the records the runner and
the oracles already produced.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from compile_check import __version__
from compile_check.localize import MODEL, NO_REFERENCE, StageVerdict
from compile_check.oracles import ORACLE_NAMES, ORACLES, Finding, Severity
from compile_check.results import BackendResult, RunSet
from compile_check.runner import FP64_BACKEND

__all__ = ["DEFAULT_MAX_FINDINGS", "render"]

Paint = Callable[..., str]
"""``paint(text, *styles) -> text``: the one styling primitive the blocks share."""

# Per oracle group, not per report: a run with fifty identical numerics findings
# and one metadata finding must still show the metadata one.
DEFAULT_MAX_FINDINGS = 10

# The width the prose wraps at. Chosen to match the project's 100-column line
# limit once the report's own indent is taken off.
_WIDTH = 96

_INDENT = "  "

# Plain ANSI, no dependency. Applied only when the caller asked for colour.
_RESET = "\033[0m"
_CODES: dict[str, str] = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}

_SEVERITY_COLOUR: dict[str, str] = {"fail": "red", "warn": "yellow", "info": "cyan"}

# Rendered first in a finding's detail line, in this order, because these are
# the fields PLAN.md "Reports" asks for: which field diverged, the two values,
# and the tolerance that was in force. Anything else follows, sorted.
_DETAIL_ORDER: tuple[str, ...] = (
    "field",
    "expected",
    "got",
    "expected_dtype",
    "got_dtype",
    "expected_type",
    "got_type",
    "rtol",
    "atol",
    "first_differing_index",
    "differing_elements",
    "expected_count",
    "got_count",
    "verdict",
)

# Already said in the finding's own message; repeating it would double the
# report's length for no information.
_DETAIL_SKIP: frozenset[str] = frozenset({"assert_close", "error"})


def render(
    runset: RunSet,
    findings: Sequence[Finding],
    verdict: StageVerdict,
    *,
    fail_on: Sequence[str] = (),
    max_findings: int = DEFAULT_MAX_FINDINGS,
    color: bool = False,
) -> str:
    """Render one run as the terminal report.

    Args:
        runset: the run, from :func:`compile_check.runner.run_all`.
        findings: every finding the oracles produced, in oracle order.
        verdict: the stage verdict, from :func:`compile_check.localize.localize`.
        fail_on: the ``--fail-on`` categories, so the checks table can show
            which of them would turn a finding into exit code 1. Every
            implemented oracle runs whatever this says; the flag decides the
            exit code, not which checks happen.
        max_findings: how many findings to print per oracle group. The rest are
            counted, never silently dropped.
        color: emit ANSI colour. ``cli.py`` decides this from ``--color`` and
            whether stdout is a TTY.

    Returns:
        The report text, ready to print. No trailing newline.
    """
    paint = _painter(color)
    blocks = [
        _header(runset, paint),
        _environment(runset, paint),
        _backends(runset, paint),
        _checks(runset, findings, fail_on, paint),
        _findings(findings, max_findings, verdict.compared, paint),
        _stage(verdict, paint),
        _next_steps(paint),
    ]
    return "\n\n".join(block for block in blocks if block)


def _painter(color: bool) -> Paint:
    """Return the function that applies (or drops) an ANSI style."""

    def paint(text: str, *styles: str) -> str:
        if not color or not styles:
            return text
        codes = "".join(_CODES[style] for style in styles if style in _CODES)
        return f"{codes}{text}{_RESET}" if codes else text

    return paint


def _header(runset: RunSet, paint: Paint) -> str:
    """One line: which tool, which version, which target."""
    return f"{paint('compile-check', 'bold')} {__version__}   target {runset.target_name}"


def _environment(runset: RunSet, paint: Paint) -> str:
    """The environment block that travels with every report.

    PLAN.md "Cross-architecture parity is a feature": the architecture is always
    carried, alongside the torch version and git hash, because parity in v1
    means running the tool on two machines and comparing the two reports.
    """
    env = runset.env
    torch_line = str(env.get("torch_version"))
    git_hash = env.get("torch_git_version")
    if git_hash:
        torch_line += f" (git {str(git_hash)[:12]})"

    machine = f"{env.get('machine')}"
    flags = env.get("cpu_flags")
    if flags:
        machine += f"   cpu flags {flags}"

    caches = env.get("inductor_force_disable_caches")
    if caches is None:
        cache_line = "unknown: torch did not report force_disable_caches"
    elif caches:
        cache_line = "disabled (force_disable_caches=True)"
    else:
        cache_line = "ENABLED (force_disable_caches=False, --allow-caches)"

    rows = [
        ("torch", torch_line),
        ("python", str(env.get("python_version"))),
        ("platform", str(env.get("platform"))),
        ("machine", machine),
        ("device", f"{runset.device}   cuda available {_yes_no(env.get('cuda_available'))}"),
        (
            "run",
            f"backends {', '.join(runset.backends)}   seed {runset.seed}   "
            f"fullgraph {_on_off(runset.fullgraph)}   dynamic {_on_off(runset.dynamic)}   "
            f"grad {_on_off(runset.grad)}",
        ),
        ("caches", cache_line),
    ]
    return _section("environment", [f"{name:<10}{value}" for name, value in rows], paint)


def _backends(runset: RunSet, paint: Paint) -> str:
    """One row per lane: how many outputs, how long, and whether it survived."""
    rows: list[tuple[str, BackendResult]] = list(runset.results.items())
    if runset.fp64 is not None:
        # Labelled as what it is. PLAN.md "The oracle blind spot" makes the fp64
        # pass a reference the numerics oracle reads, not a lane under test, and
        # the stage verdict must never name it.
        rows.append((f"{FP64_BACKEND} (reference)", runset.fp64))

    width = max((len(name) for name, _ in rows), default=len("backend"))
    width = max(width, len("backend"))
    lines = [f"{'backend':<{width}}{'outputs':>9}{'first call':>13}{'second call':>13}  status"]
    for name, result in rows:
        lines.append(
            f"{name:<{width}}{len(result.outputs):>9}"
            f"{_seconds(result.first_call_s):>13}{_seconds(result.second_call_s):>13}  "
            f"{_status(result, paint)}"
        )
    return _section("backends", lines, paint)


def _status(result: BackendResult, paint: Paint) -> str:
    """The status cell for one lane."""
    if result.exception is not None:
        return paint(f"raised {result.exception.type}", "red")
    if result.second_call_exception is not None:
        return paint(
            f"ok, then raised {result.second_call_exception.type} on the repeat call",
            "yellow",
        )
    if result.grad_error is not None:
        return paint(f"ok, backward raised {result.grad_error.type}", "yellow")
    return paint("ok", "green")


def _checks(
    runset: RunSet,
    findings: Sequence[Finding],
    fail_on: Sequence[str],
    paint: Paint,
) -> str:
    """The oracle-by-backend table.

    Columns are the lanes compared against eager, because eager is the reference
    world and comparing it with itself answers nothing. Rows are all five
    oracles of PLAN.md "Oracles", including the three that land in M2 and M3: an
    oracle that has not been written must not read as an oracle that found
    nothing.
    """
    lanes = [result.backend for result in runset.others]
    if not lanes:
        return _section(
            "checks",
            ["no lane to compare: this run has only the eager reference"],
            paint,
        )

    counted = _count(findings)
    raised = {name for name, result in runset.results.items() if not result.ok}
    reference_raised = runset.eager is None or not runset.eager.ok

    name_width = max(len(name) for name in ORACLE_NAMES)
    lane_widths = [max(len(lane), 16) for lane in lanes]
    header = f"{'oracle':<{name_width}}  fail-on  " + "  ".join(
        f"{lane:<{width}}" for lane, width in zip(lanes, lane_widths, strict=True)
    )
    lines = [header.rstrip()]
    for oracle in ORACLE_NAMES:
        marker = "yes" if oracle in fail_on else "no "
        cells = [
            _cell(counted, oracle, lane, lane in raised or reference_raised, paint)
            for lane in lanes
        ]
        padded = "  ".join(
            _pad(cell, width) for cell, width in zip(cells, lane_widths, strict=True)
        )
        lines.append(f"{oracle:<{name_width}}  {marker:<7}  {padded}".rstrip())

    legend = ["", paint("pass = no finding   not yet = the oracle lands in M2 or M3", "dim")]
    if raised or reference_raised:
        legend.append(paint("-    = the lane raised, so there was nothing to compare", "dim"))
    return _section("checks", lines + legend, paint)


def _cell(
    counted: Mapping[tuple[str, str], dict[str, int]],
    oracle: str,
    lane: str,
    lane_raised: bool,
    paint: Paint,
) -> str:
    """One cell of the checks table."""
    if oracle not in ORACLES:
        return paint("not yet", "dim")
    if lane_raised:
        return "-"
    counts = counted.get((oracle, lane))
    if not counts:
        return paint("pass", "green")
    parts = [
        paint(f"{counts[severity]} {severity}", _SEVERITY_COLOUR[severity])
        for severity in ("fail", "warn", "info")
        if counts.get(severity)
    ]
    return " ".join(parts)


def _count(findings: Iterable[Finding]) -> dict[tuple[str, str], dict[str, int]]:
    """Findings counted by ``(oracle, backend)`` and severity."""
    counted: dict[tuple[str, str], dict[str, int]] = {}
    for finding in findings:
        bucket = counted.setdefault(
            (finding.oracle, finding.backend), {"fail": 0, "warn": 0, "info": 0}
        )
        bucket[finding.severity] += 1
    return counted


def _findings(
    findings: Sequence[Finding],
    max_findings: int,
    compared: bool,
    paint: Paint,
) -> str:
    """The findings themselves, grouped by oracle, strongest severity first.

    ``compared`` is the difference between "checked and clean" and "not
    checked". A run whose eager lane raised, or which had no eager lane, has no
    findings for the same reason it has no verdict, and an empty findings block
    must not let those read alike.
    """
    if not findings and not compared:
        return _section(
            "findings",
            [paint("not checked: nothing was compared, see the stage block below", "yellow")],
            paint,
        )
    if not findings:
        return _section("findings", [paint("none", "green")], paint)

    lines: list[str] = []
    for oracle in ORACLE_NAMES:
        group = [finding for finding in findings if finding.oracle == oracle]
        if not group:
            continue
        group.sort(key=lambda finding: (_severity_rank(finding.severity), finding.backend))
        counts = ", ".join(
            f"{sum(1 for f in group if f.severity == severity)} {severity}"
            for severity in ("fail", "warn", "info")
            if any(f.severity == severity for f in group)
        )
        if lines:
            lines.append("")
        lines.append(paint(f"{oracle}  ({counts})", "bold"))
        for finding in group[:max_findings]:
            lines.extend(_finding_lines(finding, paint))
        hidden = len(group) - max_findings
        if hidden > 0:
            lines.append(
                paint(
                    f"  {hidden} more {oracle} finding{'s' if hidden > 1 else ''} "
                    f"not shown (--max-findings {max_findings})",
                    "dim",
                )
            )
    return _section("findings", lines, paint)


def _finding_lines(finding: Finding, paint: Paint) -> list[str]:
    """One finding: its heading, its message wrapped, and its details."""
    where = "run" if finding.output_index is None else f"output[{finding.output_index}]"
    heading = (
        f"  {paint(f'[{finding.severity}]', _SEVERITY_COLOUR[finding.severity])} "
        f"{finding.backend} {where}"
    )
    lines = [heading]
    lines += textwrap.wrap(
        finding.message,
        width=_WIDTH,
        initial_indent="      ",
        subsequent_indent="      ",
    ) or ["      (no message)"]
    detail = _detail_line(finding.details)
    if detail:
        lines += textwrap.wrap(
            detail, width=_WIDTH, initial_indent="      ", subsequent_indent="      "
        )
    return lines


def _detail_line(details: Mapping[str, Any]) -> str:
    """The machine-readable context of a finding, as one human-readable line."""
    keys = [key for key in _DETAIL_ORDER if key in details]
    keys += sorted(key for key in details if key not in _DETAIL_ORDER and key not in _DETAIL_SKIP)
    if not keys:
        return ""
    return "   ".join(f"{key} {_show(details[key])}" for key in keys)


def _stage(verdict: StageVerdict, paint: Paint) -> str:
    """The stage-localization verdict, and the caveat that goes with it."""
    style = "green" if verdict.clean else "red" if verdict.stage == MODEL else "yellow"
    if verdict.stage == NO_REFERENCE:
        style = "yellow"
    lines = [paint(verdict.summary, style)]
    if verdict.note:
        lines += textwrap.wrap(verdict.note, width=_WIDTH)
    if verdict.eager_exception is not None:
        lines.append("")
        lines.append("eager traceback (first lines):")
        lines += [f"  {line}" for line in verdict.eager_exception.traceback]

    repeat = [entry for entry in verdict.backends if entry.raised_on_repeat is not None]
    for entry in repeat:
        assert entry.raised_on_repeat is not None
        lines.append("")
        lines += textwrap.wrap(
            f"{entry.backend} answered once and raised "
            f"{entry.raised_on_repeat.type} on the repeat call. That is graph health, "
            "which the graph oracle owns; it lands in M3 and does not change this verdict.",
            width=_WIDTH,
        )
    return _section("stage", lines, paint)


def _next_steps(paint: Paint) -> str:
    """What to run next. Honest about what does not exist yet."""
    return _section(
        "next",
        [
            paint(
                "run with --json to save the result, --md for an issue draft (both land in M3)",
                "dim",
            )
        ],
        paint,
    )


def _section(title: str, lines: Sequence[str], paint: Paint) -> str:
    """A titled block, every line indented under it."""
    body = [f"{_INDENT}{line}" if line else "" for line in lines]
    return "\n".join([paint(title, "bold"), *body])


def _pad(cell: str, width: int) -> str:
    """Left-justify a cell to ``width``, counting the visible characters only."""
    visible = _visible_length(cell)
    return cell + " " * max(0, width - visible)


def _visible_length(text: str) -> int:
    """The printed width of a string that may carry ANSI escapes."""
    length = 0
    in_escape = False
    for char in text:
        if in_escape:
            in_escape = char != "m"
            continue
        if char == "\033":
            in_escape = True
            continue
        length += 1
    return length


def _severity_rank(severity: Severity) -> int:
    """Sort key: fail before warn before info."""
    return {"fail": 0, "warn": 1, "info": 2}.get(severity, 3)


def _show(value: Any) -> str:
    """Render a details value the way a terminal report wants to read it."""
    if isinstance(value, list):
        return f"({', '.join(str(item) for item in value)})"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _seconds(value: float | None) -> str:
    """Format a wall time, or a dash when the call did not happen."""
    return "-" if value is None else f"{value:.4f}s"


def _yes_no(value: bool | None) -> str:
    """``yes``/``no``, or ``unknown`` when the fact could not be established."""
    return "unknown" if value is None else ("yes" if value else "no")


def _on_off(value: bool) -> str:
    """``on``/``off``, which reads better than ``True``/``False`` in a report."""
    return "on" if value else "off"
