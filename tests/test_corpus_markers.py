"""The corpus markers, checked against what the corpus actually does here.

PLAN.md "Regression corpus": each case carries a known-bad version marker, and
"the test suite reads the marker and the running torch version and decides
whether the case is expected to produce a finding or expected to be clean".
This module is that reading, plus the two rules that make it useful rather than
brittle.

A disagreement is a **warning, not a failure**. The marker is a record of what
was measured and inferred at a moment; the script is the measurement, live, on
whatever torch this cell of the matrix installed. When they part company the
interesting news is which way, and turning it into a red build would mean a
nightly that fixes a bug upstream breaks this repository -- which is the exact
failure mode PLAN.md's "a case cannot simply assert 'this fails'" is about. The
suite still fails on the things that are the tool's fault, and
``tests/test_corpus_oracles.py`` is where those assertions live.

Every case also prints one machine-readable line, ``CORPUS <case>
observed=... expected=... torch=...``, so a CI log can be grepped even when the
job summary is not to hand. Visible under ``pytest -s``; the same table in a
nicer shape is ``python -m cases.summary``.

The marker arithmetic itself -- which version is after which fix -- is tested
here too, against version strings rather than against this machine's torch. It
is the half of the file that would otherwise only ever be exercised on one
build, and the half most likely to be wrong.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

from cases.markers import CASES, MARKERS, expected_verdict, parse_torch_version
from cases.summary import CASES_DIR, observe, render_table
from compile_check.oracles import ORACLE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSION = str(torch.__version__)
GIT_VERSION = getattr(torch.version, "git_version", "") or ""


@pytest.mark.parametrize("case", CASES)
def test_the_marker_agrees_with_what_the_case_does_on_this_torch(case):
    observation = observe(case)
    expected = expected_verdict(case, TORCH_VERSION, GIT_VERSION)

    # The machine-readable line, printed for every case whatever the outcome:
    # the agreements are the evidence that the disagreements mean something.
    print(f"CORPUS {case} observed={observation.verdict} expected={expected} torch={TORCH_VERSION}")

    if observation.verdict == "UNKNOWN":
        pytest.skip(
            f"{case} exited {observation.exit_code} and established neither RED nor "
            f"GREEN on this torch: {observation.line or '(no output)'}"
        )
    if expected == "UNKNOWN":
        pytest.skip(
            f"cases/markers.py cannot place torch {TORCH_VERSION} for {case}; "
            f"it observed {observation.verdict}"
        )
    if observation.verdict != expected:
        warnings.warn(
            f"corpus marker out of date: {case} is {observation.verdict} on torch "
            f"{TORCH_VERSION} (git {GIT_VERSION[:12]}) and cases/markers.py expects "
            f"{expected}. The script said: {observation.line!r}. Update the marker "
            f"for issue {MARKERS[case].issue} in cases/markers.py.",
            stacklevel=1,
        )


def test_the_summary_table_covers_every_case_and_counts_the_agreements():
    observations = [observe(case) for case in CASES]
    table = render_table(observations, TORCH_VERSION, GIT_VERSION)

    assert TORCH_VERSION in table
    for case in CASES:
        assert f"`{case}`" in table
        assert f"[#{MARKERS[case].issue}]" in table
    # One row per case plus the two header rows, so nothing is silently dropped.
    assert table.count("\n| ") == len(CASES) + 1
    assert f"{len(CASES)} cases:" in table


def test_the_summary_module_resolves_the_way_ci_invokes_it():
    # CI's job-summary step is `python -m cases.summary >> $GITHUB_STEP_SUMMARY`
    # from the repository root, and the way that breaks is a path one: `cases`
    # has no __init__.py and resolves only as a namespace package under the
    # root. Importing it in a fresh interpreter from that directory is the whole
    # check, and unlike running the module it costs no compiles.
    completed = subprocess.run(
        [sys.executable, "-c", "import cases.summary as s; print(s.CASES_DIR)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == CASES_DIR


# --------------------------------------------------------------------------
# the version arithmetic, against strings rather than against this machine
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "release", "nightly", "prerelease"),
    [
        ("2.14.0+cpu", (2, 14, 0), None, False),
        ("2.15.0.dev20260901+cpu", (2, 15, 0), "20260901", True),
        ("2.16.0a0+git279f79e", (2, 16, 0), None, True),
        ("2.14.1", (2, 14, 1), None, False),
    ],
)
def test_a_torch_version_string_splits_into_the_parts_a_fix_is_placed_against(
    version, release, nightly, prerelease
):
    parsed = parse_torch_version(version)
    assert parsed is not None
    assert (parsed.release, parsed.nightly, parsed.prerelease) == (release, nightly, prerelease)


@pytest.mark.parametrize("version", ["", "nightly", "not-a-version", "2.x.0"])
def test_a_version_string_that_cannot_be_parsed_is_never_guessed_at(version):
    assert parse_torch_version(version) is None
    for case in CASES:
        assert expected_verdict(case, version) == "UNKNOWN"


def test_an_open_issue_is_expected_red_on_every_build_that_can_be_read():
    # 195451: PR 195484 was open and unmerged as of 2026-09-02, so there is no
    # fix point and no build gets a GREEN out of this table.
    for version in ("2.13.0+cpu", "2.14.0+cpu", "2.15.0.dev20260902+cpu", "2.99.0"):
        assert expected_verdict("alias_slice_scatter_copyback", version) == "RED"


def test_a_fix_that_landed_in_a_release_turns_the_case_green_from_there_on():
    # 190765, fixed by #190966 and present from 2.14 onwards.
    case = "numerics_cpu_inductor_miscompile"
    assert expected_verdict(case, "2.13.1+cpu") == "RED"
    assert expected_verdict(case, "2.14.0+cpu") == "GREEN"
    assert expected_verdict(case, "2.15.0.dev20260901+cpu") == "GREEN"


def test_a_fix_that_landed_mid_nightly_splits_the_release_line_by_date():
    # 191449, merged 2026-09-02T03:45:57Z while 2.15 was still the open
    # development line: the nightlies of that week are all 2.15.0.dev, and only
    # the date separates the ones that carry the fix from the ones that do not.
    case = "alias_noop_view_identity"
    assert expected_verdict(case, "2.15.0.dev20260901+cpu") == "RED"
    assert expected_verdict(case, "2.15.0.dev20260902+cpu") == "GREEN"
    assert expected_verdict(case, "2.15.0.dev20261001+cpu") == "GREEN"
    # A release, and an earlier line, both decided by version alone.
    assert expected_verdict(case, "2.15.0") == "GREEN"
    assert expected_verdict(case, "2.14.0+cpu") == "RED"
    # A prerelease of the very release the fix landed in, with no date on it:
    # it may be from either side of the merge, and the table says so.
    assert expected_verdict(case, "2.15.0a0+gitdeadbeef") == "UNKNOWN"


def test_the_build_commit_answers_where_the_version_string_cannot():
    case = "alias_noop_view_identity"
    fix = MARKERS[case].fix_commit
    assert fix is not None
    # An undated prerelease of the fix's own release line is UNKNOWN by version
    # and GREEN when the commit is the fix itself. Abbreviated too, since that
    # is how a changelog quotes one.
    assert expected_verdict(case, "2.15.0a0+gitdeadbeef", fix) == "GREEN"
    assert expected_verdict(case, "2.15.0a0+gitdeadbeef", fix[:10]) == "GREEN"
    assert expected_verdict(case, "2.15.0a0+gitdeadbeef", "deadbeefcafe") == "UNKNOWN"
    # Too short to identify anything: not a match, rather than a match on four
    # hex characters.
    assert expected_verdict(case, "2.15.0a0+gitdeadbeef", fix[:4]) == "UNKNOWN"


def test_an_unknown_case_name_is_an_error_and_not_a_verdict():
    with pytest.raises(KeyError, match="no marker for 'nope'"):
        expected_verdict("nope", TORCH_VERSION)


def test_every_marker_names_a_script_and_an_oracle_that_exist():
    for case, marker in MARKERS.items():
        assert marker.case == case, "the key and the record must not drift apart"
        assert (CASES_DIR / f"{case}.py").is_file(), f"no standalone script for {case}"
        assert marker.oracle in ORACLE_NAMES, marker.oracle
        assert marker.manifests_as in ("finding", "raised_lane")
