"""Run the corpus and print what it said, as a Markdown table.

``python -m cases.summary`` from the repository root runs each standalone
RED/GREEN script in ``cases/`` and prints one table of observed against expected
verdicts. CI appends it to ``$GITHUB_STEP_SUMMARY`` on every matrix cell, so the
question "which of the five bugs does *this* torch still have" is answered on
the job page rather than by reading a log. ``python cases/summary.py`` works
too, because every other file in this directory does and a module that broke the
habit would be a papercut; see the bootstrap below the imports.

Output goes to stdout and the workflow redirects it. That is deliberate: a
module that wrote to ``$GITHUB_STEP_SUMMARY`` itself would be untestable
anywhere else and silent when the variable is unset, and a redirect is the one
line of workflow that makes both cases obvious.

:func:`observe` is also what the two corpus test modules use, so "run the
standalone script and read its verdict" is written once. It is cached per
process: the scripts are slow (each compiles), and within one process the answer
cannot change, since the torch under them cannot.

Nothing here decides that a disagreement is a failure. PLAN.md "Regression
corpus" makes the marker a recorded expectation and the script the measurement;
when they disagree it is the marker that is out of date, or the torch that moved,
and either way what is wanted is a visible note rather than a red build. The
exit code is 0 unless this module could not run at all.
"""

from __future__ import annotations

import functools
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent

# Run as a script, Python puts `cases/` on the path and not the repository root,
# so `cases.markers` does not resolve; run with `-m` from the root, it already
# does and this is a no-op. The alternative was a module that is the one file in
# this directory you cannot just run, which is worse than four lines and a noqa.
if __package__ in {None, ""}:  # pragma: no cover - only true under `python cases/summary.py`
    sys.path.insert(0, str(CASES_DIR.parent))

from cases.markers import CASES, MARKERS, Verdict, expected_verdict  # noqa: E402

__all__ = ["Observation", "main", "observe", "render_table"]

# The exit codes the standalone scripts agree on (cases/README.md "Adding a
# case"): 1 is RED, 0 is GREEN, and 2 is a crash, which establishes neither.
_EXIT_VERDICT: dict[int, Verdict] = {0: "GREEN", 1: "RED"}

_ISSUE_URL = "https://github.com/pytorch/pytorch/issues/{issue}"


@dataclass(frozen=True)
class Observation:
    """What one standalone script actually did on this torch."""

    case: str
    verdict: Verdict
    """``RED``, ``GREEN``, or ``UNKNOWN`` when the script crashed or was killed.

    A crash is ``UNKNOWN`` and not ``GREEN``: a case that could not run
    established nothing, and reading it as clean would be the one mistake a
    regression corpus must not make.
    """

    exit_code: int
    line: str
    """The script's own one-line verdict, which carries the torch build it
    measured and what it saw. Empty when it printed nothing."""


@functools.cache
def observe(case: str, timeout: float = 900.0) -> Observation:
    """Run one standalone script in a subprocess and read its verdict.

    A subprocess rather than an import, for the reason ``tests/test_corpus_twins.py``
    gives: the scripts are not import-safe as modules, each defines its own
    ``main()`` and calls ``sys.exit``. Running them is also the point -- the
    script's live verdict on the installed torch is the ground truth the marker
    is checked against, and a hardcoded version window would go stale.

    Args:
        case: the script's stem, a key of :data:`~cases.markers.MARKERS`.
        timeout: seconds before the run is abandoned as ``UNKNOWN``. Generous,
            because each script compiles.

    Returns:
        The observation. Never raises for a case that failed: a script that
        crashed, timed out, or printed nothing is ``UNKNOWN`` with whatever it
        did say.
    """
    script = CASES_DIR / f"{case}.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Observation(
            case=case, verdict="UNKNOWN", exit_code=-1, line=f"timed out ({timeout}s)"
        )
    return Observation(
        case=case,
        verdict=_EXIT_VERDICT.get(completed.returncode, "UNKNOWN"),
        exit_code=completed.returncode,
        line=_verdict_line(completed.stdout, completed.stderr),
    )


def _verdict_line(stdout: str, stderr: str) -> str:
    """The script's ``RED``/``GREEN``/``CRASH`` line, or the best line there is."""
    for line in stdout.splitlines():
        if line.startswith(("RED ", "GREEN ", "CRASH ")):
            return line.strip()
    tail = [line for line in (stdout + stderr).splitlines() if line.strip()]
    return tail[-1].strip() if tail else ""


def render_table(observations: list[Observation], torch_version: str, git_version: str) -> str:
    """One Markdown table of observed against expected, plus a one-line count."""
    heading = (
        f"### compile-check regression corpus -- torch {torch_version}"
        f"{f' (git {git_version[:12]})' if git_version else ''}, "
        f"python {platform.python_version()}, {platform.machine()}"
    )
    rows = [
        "| Case | Issue | Oracle | Observed | Expected | Agrees |",
        "|---|---|---|---|---|---|",
    ]
    agree = differ = unplaced = 0
    for observation in observations:
        marker = MARKERS[observation.case]
        expected = expected_verdict(observation.case, torch_version, git_version)
        if expected == "UNKNOWN" or observation.verdict == "UNKNOWN":
            note, unplaced = "not placed", unplaced + 1
        elif expected == observation.verdict:
            note, agree = "yes", agree + 1
        else:
            note, differ = "**no**", differ + 1
        issue = f"[#{marker.issue}]({_ISSUE_URL.format(issue=marker.issue)})"
        rows.append(
            f"| `{observation.case}` | {issue} | {marker.oracle} | "
            f"{observation.verdict} | {expected} | {note} |"
        )
    total = len(observations)
    tally = (
        f"{total} case{'' if total == 1 else 's'}: {agree} agree with the marker, "
        f"{differ} disagree, {unplaced} could not be placed. A disagreement is a "
        "note, not a failure: it means the marker is out of date or this torch "
        "moved, and `cases/markers.py` is where it is recorded."
    )
    return "\n".join([heading, "", *rows, "", tally])


def main(argv: list[str] | None = None) -> int:
    """Run every corpus case and print the table. Always exit 0 on a real run."""
    del argv  # no flags: the module does one thing, and CI redirects its output
    import torch

    observations = [observe(case) for case in CASES]
    print(
        render_table(
            observations, str(torch.__version__), getattr(torch.version, "git_version", "")
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
