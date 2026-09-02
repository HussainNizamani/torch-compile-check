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
standalone script and read its verdict" is written once. It is cached twice
over, because the scripts are slow -- each one compiles -- and within one
environment the answer cannot change.

Per process, with ``functools.cache``: the torch under the scripts cannot move
while the interpreter is running.

Across processes, in a JSON file under the system temporary directory: CI runs
the whole corpus through ``pytest`` and then runs it again in the job-summary
step, which used to cost a second full set of compiles per matrix cell (M2-3
note). An entry is reused only when the *fingerprint* matches -- torch version
and build hash, Python version, machine, interpreter path -- and only when the
script's own source is byte-for-byte what produced it, so a torch upgrade or an
edited case re-runs rather than reports a stale verdict. Nothing depends on the
file: it is written best-effort, every read failure falls back to running the
script, and the table says how many rows came from it.
``COMPILE_CHECK_OBSERVATIONS`` points the file somewhere else; set to an empty
value it switches that half of the caching off entirely.

Nothing here decides that a disagreement is a failure. PLAN.md "Regression
corpus" makes the marker a recorded expectation and the script the measurement;
when they disagree it is the marker that is out of date, or the torch that moved,
and either way what is wanted is a visible note rather than a red build. The
exit code is 0 unless this module could not run at all.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).resolve().parent

# Run as a script, Python puts `cases/` on the path and not the repository root,
# so `cases.markers` does not resolve; run with `-m` from the root, it already
# does and this is a no-op. The alternative was a module that is the one file in
# this directory you cannot just run, which is worse than four lines and a noqa.
if __package__ in {None, ""}:  # pragma: no cover - only true under `python cases/summary.py`
    sys.path.insert(0, str(CASES_DIR.parent))

from cases.markers import CASES, MARKERS, Verdict, expected_verdict  # noqa: E402

__all__ = [
    "CACHE_ENV_VAR",
    "Observation",
    "cache_file",
    "main",
    "observe",
    "render_table",
]

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

    cached: bool = False
    """Whether this came out of the cross-process cache rather than a run.

    Reported rather than hidden: the table is a measurement, and "this row was
    produced twenty minutes ago by the pytest step" is part of what it is.
    """


CACHE_ENV_VAR = "COMPILE_CHECK_OBSERVATIONS"
"""Where to keep the cross-process observation cache. Empty value: nowhere."""


# What the cached verdicts were measured under. An entry is reused only when
# every part of this still matches, because every part of it can change what a
# case does: the torch build most obviously, the interpreter because a second
# virtual environment on the same machine has a different one, and the machine
# because /tmp is shared on a build host.
def _fingerprint() -> str:
    import torch

    return "|".join(
        [
            str(torch.__version__),
            getattr(torch.version, "git_version", "") or "",
            platform.python_version(),
            platform.machine(),
            sys.executable,
        ]
    )


def cache_file() -> Path | None:
    """The observation cache's path, or ``None`` when caching is switched off.

    Keyed by the corpus directory, so two checkouts on one machine do not read
    each other's verdicts, and under the system temporary directory rather than
    in the repository, because it is a measurement of this machine and not
    something to commit.
    """
    override = os.environ.get(CACHE_ENV_VAR)
    if override is not None:
        return Path(override) if override.strip() else None
    digest = hashlib.sha256(str(CASES_DIR).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"compile-check-observations-{digest}.json"


def _read_cache() -> dict[str, Any]:
    """Every cached observation that was measured under this fingerprint."""
    path = cache_file()
    if path is None:
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A cache is a speed-up. Anything unreadable -- absent, truncated by a
        # killed run, written by another user -- means running the script.
        return {}
    if not isinstance(document, dict) or document.get("fingerprint") != _fingerprint():
        return {}
    entries = document.get("observations")
    return entries if isinstance(entries, dict) else {}


def _cached(case: str, source: str) -> Observation | None:
    """One reusable observation, or ``None`` when there is nothing to reuse."""
    entry = _read_cache().get(case)
    if not isinstance(entry, dict) or entry.get("source") != source:
        return None
    verdict = entry.get("verdict")
    if verdict not in ("RED", "GREEN", "UNKNOWN") or not isinstance(entry.get("exit_code"), int):
        return None
    return Observation(
        case=case,
        verdict=verdict,
        exit_code=entry["exit_code"],
        line=str(entry.get("line", "")),
        cached=True,
    )


def _store(case: str, observation: Observation, source: str) -> None:
    """Record one observation for the next process, best effort.

    Read-modify-write rather than a whole-file rewrite from memory, because the
    pytest run that fills this file and a ``python -m cases.summary`` beside it
    are two processes with two different sets of cases in hand. The rename is
    what makes a half-written file impossible for the reader.
    """
    path = cache_file()
    if path is None:
        return
    document: dict[str, Any] = {"fingerprint": _fingerprint(), "observations": {}}
    existing = _read_cache()
    if existing:
        document["observations"] = dict(existing)
    document["observations"][case] = {
        "verdict": observation.verdict,
        "exit_code": observation.exit_code,
        "line": observation.line,
        "source": source,
    }
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # Same rule as the read: never fail a run over the cache.
        temporary.unlink(missing_ok=True)


def _source_digest(script: Path) -> str | None:
    """The script's content hash, or ``None`` when it cannot be read.

    The observation is a measurement of *this* file; an edited case that reused
    the verdict of the one before it would be the corpus lying about itself.
    """
    try:
        return hashlib.sha256(script.read_bytes()).hexdigest()
    except OSError:
        return None


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
        did say. ``Observation.cached`` says whether it was measured now or
        read back from the file another process left; see the module docstring
        for when an entry is allowed to be reused.
    """
    script = CASES_DIR / f"{case}.py"
    source = _source_digest(script)
    if source is not None:
        reused = _cached(case, source)
        if reused is not None:
            return reused

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Not cached, deliberately: a timeout says more about the machine that
        # was busy than about the case, and a stored one would keep saying it.
        return Observation(
            case=case, verdict="UNKNOWN", exit_code=-1, line=f"timed out ({timeout}s)"
        )
    observation = Observation(
        case=case,
        verdict=_EXIT_VERDICT.get(completed.returncode, "UNKNOWN"),
        exit_code=completed.returncode,
        line=_verdict_line(completed.stdout, completed.stderr),
    )
    if source is not None:
        _store(case, observation, source)
    return observation


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
    reused = sum(1 for observation in observations if observation.cached)
    if reused:
        # Said out loud, because a reader is entitled to know which rows are a
        # fresh measurement and which came from the pytest run that preceded
        # this one. The fingerprint the cache is keyed by is in the heading
        # above: same torch, same interpreter, same machine, same case source.
        tally += (
            f" {reused} of the {total} {'was' if reused == 1 else 'were'} reused from the "
            f"observation cache ({cache_file()}) rather than re-run."
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
