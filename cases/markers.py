"""Known-bad version markers for the regression corpus.

PLAN.md "Regression corpus": several of these bugs are fixed on current torch,
so a case cannot simply assert "this fails". Each case carries a known-bad
version marker recording the torch versions where the bug reproduces and the
version or commit where it was fixed, and the test suite reads the marker and
the running torch version to decide whether the case is expected to produce a
finding or expected to be clean.

This module is that marker table, as data rather than as prose in five
docstrings. The docstrings stay, because a case file has to explain itself to
someone reading it alone; what lives here is the part a test and a CI summary
have to be able to *compute* with. :func:`expected_verdict` is the one place the
version arithmetic happens, and it is deliberately the only thing in this file
that has any logic in it.

Nothing here imports torch. The running version is passed in as a string, so the
table can be read, tested, and printed without a torch install -- and so the
same function can answer for a version that is not the one this process is
running, which is what makes it testable at all.

What a verdict means, since the three words are load-bearing:

* ``RED`` -- this torch is expected to reproduce the bug. The standalone script
  exits 1, and ``compile-check`` on the twin reports the finding.
* ``GREEN`` -- this torch is expected to be clean, because it carries the fix.
  A case whose bug is fixed is not deleted; it becomes a regression test that
  the fix is still there.
* ``UNKNOWN`` -- the version could not be placed against what is recorded. Not
  the same as either of the other two, and never treated as one: a marker that
  guessed would be worse than a marker that says it does not know.

Precedence, in the order :func:`expected_verdict` applies it:

1. the build's own git commit, when it is the fix commit or one of the commits a
   RED was measured on. A hash is exact where a version string is an inference.
2. the fix point, when one is recorded and the build can be placed against it.
   A landed commit is the strongest general statement available, so it beats the
   measured-version list below: a build at or after the fix that still comes
   back RED is a disagreement worth a warning, not a marker to be overridden.
3. :attr:`CaseMarker.known_bad`, the versions a RED was actually measured on.
4. RED, when no fix is recorded at all. The issue is open, so every build has
   the bug until one of the fields above says otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CASES",
    "MARKERS",
    "CaseMarker",
    "Verdict",
    "expected_verdict",
    "parse_torch_version",
]

Verdict = Literal["RED", "GREEN", "UNKNOWN"]

RED: Verdict = "RED"
GREEN: Verdict = "GREEN"
UNKNOWN: Verdict = "UNKNOWN"

# torch version strings, in the three shapes this table has to place:
#   2.14.0+cpu                     a release
#   2.15.0.dev20260901+cpu         a nightly, dated
#   2.16.0a0+git279f79e            a build from a checkout, undated
# The local part after "+" is dropped: it names the wheel flavour (cpu, cu130)
# or the checkout, never the point in history the fix is measured against.
_VERSION = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)"
    r"(?:\.dev(?P<dev>\d+)|(?P<pre>a\d+|b\d+|rc\d+))?"
    r"(?:\+.*)?$"
)


@dataclass(frozen=True)
class _Version:
    """A torch version, split into the parts a fix point is compared against."""

    release: tuple[int, ...]
    """``2.15.0`` as ``(2, 15, 0)``. Compared element by element."""

    nightly: str | None
    """The ``dev20260901`` date, as ``"20260901"``, or ``None``.

    A string rather than an int because it is compared as a date and never
    arithmetically, and because a lexicographic comparison of ``YYYYMMDD`` is
    the date order.
    """

    prerelease: bool
    """Whether this build comes before its own release: a ``dev`` nightly or an
    ``a0``/``rc`` build. It matters because such a build carries the version it
    is heading *towards*: ``2.15.0.dev20260901`` is not 2.15.0, it is some point
    on the way to it, and a fix that landed during that window may or may not be
    in it."""


@dataclass(frozen=True)
class CaseMarker:
    """What is known about when one corpus case is RED and when it is GREEN.

    Every field is a fact with a provenance, which is why there are this many of
    them: the difference between "we measured RED on this build" and "the fix
    landed in this release, so anything later must be GREEN" is the difference
    between evidence and inference, and a table that flattened the two would let
    an inference go stale without anyone noticing.
    """

    case: str
    """The standalone script's stem, e.g. ``alias_slice_scatter_copyback``."""

    issue: int
    """The pytorch/pytorch issue number the case encodes."""

    oracle: str
    """The ``--fail-on`` category that reports this case's twin when it is RED.

    The oracle named in the case header. Note it is the oracle for the *twin's*
    shape: ``alias_noop_view_identity`` is an alias bug, but its own reproducer
    goes on to ``resize_()`` through the collapsed view, so what reaches the
    report from that shape is a numerics and metadata divergence. The twin
    ``alias_noop_view.py`` returns the base and the view together, which is the
    shape the alias oracle sees.
    """

    manifests_as: Literal["finding", "raised_lane"]
    """How a RED reaches the report on this build.

    ``finding`` is the ordinary case: an oracle compares the two lanes and
    reports. ``raised_lane`` is 194593, where the compiled lane does not return
    a divergent answer at all -- Dynamo cannot trace the branch under
    ``fullgraph=True``, so the lane raises, which is exit 1 regardless of
    ``--fail-on`` and belongs to no oracle category. The distinction is here
    because a test that demanded a finding from :attr:`oracle` would fail on a
    case that is working exactly as documented.
    """

    signal: str
    """FINDINGS.md's signal type: what kind of divergence this is."""

    known_bad: tuple[str, ...] = ()
    """Torch version prefixes a RED was *measured* on, not inferred.

    Matched with ``startswith``, so ``"2.14.0"`` covers ``2.14.0+cpu`` and
    ``2.14.0+cu130``. Written precisely enough not to swallow a fixed build:
    a nightly prefix carries its whole date.
    """

    known_bad_git: tuple[str, ...] = ()
    """Torch build commits a RED was measured on, as full or prefix hashes.

    ``torch.version.git_version``, which is the commit the wheel was built from.
    Exact where a version string is an inference, and the only handle on a build
    whose version string cannot be placed at all.
    """

    fix_pr: int | None = None
    """The pytorch/pytorch PR that fixes it, when one has landed."""

    fix_commit: str | None = None
    """The commit that PR landed as. A build at exactly this commit is GREEN."""

    fixed_in_release: str | None = None
    """The first torch release that contains the fix, e.g. ``"2.15.0"``.

    A release at or after this is GREEN. A *pre*-release of this same version
    (a nightly or an ``a0`` build heading towards it) is not decided by this
    field alone, because it may predate the fix inside the release window; that
    is what :attr:`fixed_after_nightly` is for.
    """

    fixed_after_nightly: str | None = None
    """The first nightly date containing the fix, as ``"YYYYMMDD"``.

    A nightly dated at or after this carries the fix; one dated before it does
    not. Nightlies are cut from main, so the date is a point in history and the
    comparison is sound.
    """

    note: str = ""
    """One sentence a human needs that the fields above do not carry."""


# The five C-1 corpus cases. Every RED below was measured, not assumed; the
# provenance is in FINDINGS.md and in each case's own docstring.
#
# The measurements this table was written from: torch 2.14.0+cpu, git
# 08187d9e0fba026dc8217405802ab5381dc88d90, aarch64, CPU-only, caches disabled,
# 2026-09-02. Four of the five scripts exit 1 there and
# numerics_cpu_inductor_miscompile exits 0. The 2.15.0.dev20260901+cpu column
# comes from the C-1 clean-venv run recorded in FINDINGS.md.
MARKERS: dict[str, CaseMarker] = {
    "alias_slice_scatter_copyback": CaseMarker(
        case="alias_slice_scatter_copyback",
        issue=195451,
        oracle="alias",
        manifests_as="finding",
        signal="inductor-only miscompile",
        known_bad=("2.14.0", "2.15.0.dev20260824", "2.15.0.dev20260831", "2.15.0.dev20260901"),
        known_bad_git=("08187d9e0fba026dc8217405802ab5381dc88d90",),
        # PR 195484 was open and unmerged as of 2026-09-02, so there is no fix
        # point to compare against and every build is expected RED. FINDINGS.md
        # records one GREEN, in a venv that had an earlier local iteration of
        # that PR hand-applied to reinplace.py; that is the fix being present,
        # not the bug being absent, and it is exactly the disagreement this
        # marker is supposed to surface as a warning rather than silently
        # absorb.
        note="PR 195484 open and unmerged as of 2026-09-02, so RED everywhere",
    ),
    "alias_noop_view_identity": CaseMarker(
        case="alias_noop_view_identity",
        issue=191449,
        oracle="alias",
        manifests_as="finding",
        signal="inductor-only miscompile",
        known_bad=("2.14.0", "2.15.0.dev20260901"),
        known_bad_git=("08187d9e0fba026dc8217405802ab5381dc88d90",),
        fix_pr=191844,
        fix_commit="a3586f00181a395b066cdf3a8933c2b47b7a6890",
        # Merged into main at 2026-09-02T03:45:57Z, while 2.15 was still the
        # open development line (the nightlies of that week are 2.15.0.dev), so
        # the 2.15.0 release will carry it and the nightlies from the 0902 cut
        # onwards do.
        fixed_in_release="2.15.0",
        fixed_after_nightly="20260902",
        note="fix lives in AOTAutograd; the divergence is only observable under inductor",
    ),
    "dtype_int8_matmul_promotion": CaseMarker(
        case="dtype_int8_matmul_promotion",
        issue=191308,
        oracle="metadata",
        manifests_as="finding",
        signal="inductor-only miscompile",
        known_bad=("2.14.0", "2.15.0.dev20260901"),
        known_bad_git=("08187d9e0fba026dc8217405802ab5381dc88d90",),
        note="no fix PR found as of 2026-09-02; the triggering shape family is narrow",
    ),
    "distributions_validation_branch": CaseMarker(
        case="distributions_validation_branch",
        issue=194593,
        oracle="graph",
        # The compiled lane raises rather than answering differently, so this
        # RED is exit 1 by the raised-lane rule and not by an oracle finding.
        # The graph oracle lands in M3 and would report the break as graph
        # health; it would not change this verdict.
        manifests_as="raised_lane",
        signal="fullgraph capturability break, backend-independent",
        known_bad=("2.14.0", "2.15.0.dev20260901"),
        known_bad_git=("08187d9e0fba026dc8217405802ab5381dc88d90",),
        note="RED only under --fullgraph; the default mode graph-breaks and stays correct",
    ),
    "numerics_cpu_inductor_miscompile": CaseMarker(
        case="numerics_cpu_inductor_miscompile",
        issue=190765,
        oracle="numerics",
        manifests_as="finding",
        signal="none observed (fixed upstream)",
        # Deliberately empty. FINDINGS.md expects RED on torch <= 2.13.x from
        # the issue's own environment, and nobody here has run a 2.13 build, so
        # recording it as measured would be a claim this repo cannot support.
        # The fix point below carries it instead: anything before 2.14 is RED by
        # inference, which is what it is.
        known_bad=(),
        fix_pr=190966,
        fixed_in_release="2.14.0",
        note="fixed upstream by #190966 (ModularIndexing negativity guard); GREEN from 2.14 on",
    ),
}

CASES: tuple[str, ...] = tuple(MARKERS)
"""Every corpus case with a marker, in the order the table declares them."""


def parse_torch_version(version: str) -> _Version | None:
    """Split a torch version string, or ``None`` when it is not one.

    Args:
        version: ``torch.__version__``, e.g. ``"2.15.0.dev20260901+cpu"``.

    Returns:
        The release tuple, the nightly date if there is one, and whether the
        build precedes its own release. ``None`` for anything unparseable, which
        the caller turns into ``UNKNOWN`` rather than into a guess.
    """
    match = _VERSION.match(version.strip())
    if match is None:
        return None
    return _Version(
        release=tuple(int(part) for part in match.group("release").split(".")),
        nightly=match.group("dev"),
        prerelease=bool(match.group("dev") or match.group("pre")),
    )


def expected_verdict(case: str, torch_version: str, git_version: str | None = None) -> Verdict:
    """Whether ``case`` is expected to be RED or GREEN on this torch.

    Args:
        case: the standalone script's stem, a key of :data:`MARKERS`.
        torch_version: ``torch.__version__``.
        git_version: ``torch.version.git_version``, the commit the wheel was
            built from, when it is known. Optional, and worth passing: it is
            the only field that can place a build whose version string cannot
            be, and the only exact answer for a build sitting on the fix commit
            itself.

    Returns:
        ``"RED"``, ``"GREEN"``, or ``"UNKNOWN"``. See the module docstring for
        what the three mean and the order the rules are applied in.

    Raises:
        KeyError: ``case`` is not in the table. A typo in a case name is a bug
            in the caller, not an unknown verdict, so it is not quietly folded
            into ``"UNKNOWN"``.
    """
    if case not in MARKERS:
        raise KeyError(f"no marker for {case!r}; the corpus cases are {', '.join(CASES)}")
    marker = MARKERS[case]

    if git_version:
        if marker.fix_commit and _same_commit(git_version, marker.fix_commit):
            return GREEN
        if any(_same_commit(git_version, commit) for commit in marker.known_bad_git):
            return RED

    parsed = parse_torch_version(torch_version)
    if parsed is None:
        return UNKNOWN

    against_fix = _placed_against_the_fix(parsed, marker)
    if against_fix is not None:
        return GREEN if against_fix else RED

    if any(torch_version.startswith(prefix) for prefix in marker.known_bad):
        return RED
    if marker.fixed_in_release or marker.fixed_after_nightly:
        # A fix is known and this build could not be placed against it: a
        # prerelease of the very release the fix landed in, with no date to
        # compare. It may or may not carry the fix, and saying either would be
        # inventing the answer.
        return UNKNOWN
    return RED


def _placed_against_the_fix(parsed: _Version, marker: CaseMarker) -> bool | None:
    """Is this build at or after the fix? ``None`` when it cannot be placed.

    Two comparisons, and the split between them is the whole subtlety of the
    file. A release compares by version number. A prerelease -- a nightly, an
    ``a0`` build -- carries the version it is heading towards, so it compares by
    version number only when the lines differ, and by nightly date when the
    build is a prerelease of the very release the fix landed in.
    """
    if marker.fixed_in_release is None and marker.fixed_after_nightly is None:
        return None

    fixed_release = (
        tuple(int(part) for part in marker.fixed_in_release.split("."))
        if marker.fixed_in_release
        else None
    )
    if fixed_release is not None:
        line, fixed_line = _line(parsed.release), _line(fixed_release)
        if not parsed.prerelease:
            return parsed.release >= fixed_release
        if line != fixed_line:
            # A prerelease of a different release line is placed by the line
            # alone: every 2.16 build comes after the 2.15 release, and every
            # 2.13 build comes before it.
            return line > fixed_line

    # Either a prerelease of the release the fix landed in, or a marker that
    # only knows the nightly date. Both need the date.
    if marker.fixed_after_nightly and parsed.nightly:
        return parsed.nightly >= marker.fixed_after_nightly
    return None


def _line(release: tuple[int, ...]) -> tuple[int, ...]:
    """The ``(major, minor)`` release line a version belongs to."""
    return release[:2]


def _same_commit(left: str, right: str) -> bool:
    """Whether two git hashes name the same commit, allowing for abbreviation.

    Torch reports a full 40-character hash in ``torch.version.git_version`` and
    an issue or a changelog usually quotes ten. The shorter of the two decides
    the comparison, with a floor so that a stray short string cannot match
    everything.
    """
    left, right = left.strip().lower(), right.strip().lower()
    width = min(len(left), len(right))
    if width < 7:
        return False
    return left[:width] == right[:width]
