"""The corpus observation cache in ``cases/summary.py``.

M2-3 left a note: CI runs the five corpus scripts through ``pytest`` and then
runs them again in the job-summary step, roughly a minute of compiles per matrix
cell for an answer it already had. :func:`cases.summary.observe` now writes each
verdict to a JSON file under the system temporary directory and reads it back in
the next process.

The whole risk of a cache like this is that it answers with something that is no
longer true, so every test below is about *when it must not be used*: a
different torch or interpreter, an edited case, a file that cannot be read, a
run that timed out, a run that crashed. The corpus is the thing that says
which bugs this torch still has, and a stale row in it is worse than a slow
one -- and an ``UNKNOWN`` row is the stalest kind there is, because it is not
a fact about the case at all, only about whatever was broken on the machine
that measured it.

None of these run the real corpus. ``CASES_DIR`` is pointed at a temporary
directory holding a script that costs nothing, which is also what lets the tests
count how many times it actually ran.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import cases.summary as summary
from cases.markers import CASES, CaseMarker
from cases.summary import Observation, cache_file, observe, render_table

REPO_ROOT = Path(__file__).resolve().parents[1]

# A stand-in for a corpus script: it appends to a file every time it runs, so a
# test can tell "reused the cache" from "ran it again" by counting, and it
# follows the RED/GREEN protocol of cases/README.md so `observe` reads it the
# way it reads a real one.
PROBE = """\
import pathlib
import sys

pathlib.Path({runs!r}).open("a").write("run\\n")
print({line!r})
sys.exit({code})
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A one-script corpus in a temporary directory, with its own cache file."""
    directory = tmp_path / "cases"
    directory.mkdir()
    monkeypatch.setattr(summary, "CASES_DIR", directory)
    monkeypatch.setenv(summary.CACHE_ENV_VAR, str(tmp_path / "observations.json"))
    observe.cache_clear()
    yield directory
    observe.cache_clear()


def write_probe(directory: Path, *, line: str = "RED probe torch=x", code: int = 1) -> Path:
    """Write the stand-in script, and hand back the file that counts its runs."""
    runs = directory / "runs.txt"
    (directory / "probe.py").write_text(PROBE.format(runs=str(runs), line=line, code=code))
    return runs


def run_count(runs: Path) -> int:
    return len(runs.read_text().splitlines()) if runs.exists() else 0


def test_a_second_process_reuses_the_verdict_the_first_one_measured(corpus):
    runs = write_probe(corpus)

    first = observe("probe")
    observe.cache_clear()  # what a new process starts with
    second = observe("probe")

    assert run_count(runs) == 1, "the script ran a second time instead of being reused"
    assert first.cached is False
    assert second.cached is True
    assert (second.verdict, second.exit_code, second.line) == (
        first.verdict,
        first.exit_code,
        first.line,
    )
    assert second.verdict == "RED"


def test_an_edited_case_is_measured_again_rather_than_answered_from_the_cache(corpus):
    runs = write_probe(corpus, line="RED probe torch=x", code=1)
    assert observe("probe").verdict == "RED"
    observe.cache_clear()

    # The bug is fixed upstream and the case now passes: same name, same cache
    # file, different file.
    write_probe(corpus, line="GREEN probe torch=x", code=0)
    again = observe("probe")

    assert run_count(runs) == 2
    assert again.cached is False
    assert again.verdict == "GREEN"


def test_a_cache_written_under_another_torch_is_ignored(corpus, tmp_path):
    runs = write_probe(corpus)
    observe("probe")
    observe.cache_clear()

    document = json.loads((tmp_path / "observations.json").read_text())
    assert document["fingerprint"]  # what it was measured under
    document["fingerprint"] = "torch 1.0|deadbeef|3.10.0|x86_64|/usr/bin/python"
    (tmp_path / "observations.json").write_text(json.dumps(document))

    assert observe("probe").cached is False
    assert run_count(runs) == 2


def test_a_cache_file_that_is_not_readable_json_costs_a_re_run_and_nothing_else(corpus, tmp_path):
    runs = write_probe(corpus)
    (tmp_path / "observations.json").write_text("half a file, killed mid-w")

    observation = observe("probe")

    assert observation.verdict == "RED"
    assert observation.cached is False
    assert run_count(runs) == 1
    # And the unreadable file is replaced rather than left to fail every run.
    assert json.loads((tmp_path / "observations.json").read_text())["observations"]["probe"]


def test_a_timed_out_run_is_never_cached(corpus, tmp_path):
    # A timeout says the machine was busy, not what the case does. Storing one
    # would keep saying UNKNOWN about a case that reproduces perfectly well.
    (corpus / "slow.py").write_text("import time\ntime.sleep(30)\n")

    observation = observe("slow", timeout=0.2)

    assert observation.verdict == "UNKNOWN"
    assert "timed out" in observation.line
    written = tmp_path / "observations.json"
    document = json.loads(written.read_text()) if written.exists() else {}
    assert "slow" not in document.get("observations", {})


def test_a_crash_is_never_cached(corpus, tmp_path):
    # A crash is environment state, not a verdict: a missing compiler or absent
    # Python headers makes every case in the corpus exit 2, and that must not
    # be readable as "this case was measured and found UNKNOWN" once the
    # environment is fixed.
    runs = write_probe(corpus, line="", code=2)

    observation = observe("probe")

    assert observation.verdict == "UNKNOWN"
    assert observation.exit_code == 2
    assert run_count(runs) == 1
    written = tmp_path / "observations.json"
    document = json.loads(written.read_text()) if written.exists() else {}
    assert "probe" not in document.get("observations", {})


def test_a_stale_unknown_entry_is_ignored_and_a_red_rerun_replaces_it(corpus, tmp_path):
    # An older cache file may hold an UNKNOWN entry written before this rule
    # existed (or one from a version of this file that could still write one).
    # It must be read back as a miss, exactly like no entry at all, and a real
    # verdict measured now must overwrite it.
    runs = write_probe(corpus, line="RED probe torch=x", code=1)
    source = summary._source_digest(corpus / "probe.py")
    written = tmp_path / "observations.json"
    written.write_text(
        json.dumps(
            {
                "fingerprint": summary._fingerprint(),
                "observations": {
                    "probe": {
                        "verdict": "UNKNOWN",
                        "exit_code": 2,
                        "line": "CRASH probe: c++ compile failed",
                        "source": source,
                    }
                },
            }
        )
    )

    observation = observe("probe")

    assert observation.cached is False, "an UNKNOWN entry must not be reused"
    assert observation.verdict == "RED"
    assert run_count(runs) == 1, "ignoring the stale entry must still run the script"

    document = json.loads(written.read_text())
    assert document["observations"]["probe"]["verdict"] == "RED", (
        "a real verdict must replace the stale UNKNOWN one"
    )


def test_setting_the_variable_empty_switches_the_cache_off(corpus, monkeypatch):
    monkeypatch.setenv(summary.CACHE_ENV_VAR, "")
    runs = write_probe(corpus)

    assert cache_file() is None
    observe("probe")
    observe.cache_clear()
    observe("probe")

    assert run_count(runs) == 2


def test_the_no_cache_flag_reads_and_writes_nothing(corpus, tmp_path, monkeypatch, capsys):
    # `--no-cache` is equivalent to TORCH_COMPILE_CHECK_OBSERVATIONS= (empty),
    # exercised end to end through main() rather than the env var directly, so
    # the flag itself -- not just what it sets -- is under test.
    runs = write_probe(corpus, line="RED probe torch=x", code=1)
    monkeypatch.setattr(summary, "CASES", ["probe"])
    # `summary.MARKERS` is the same dict object as `cases.markers.MARKERS` (one
    # `from cases.markers import MARKERS`), so a single `setitem` is visible to
    # both `render_table` here and `expected_verdict`'s own lookup, and
    # monkeypatch removes the key again after the test.
    monkeypatch.setitem(
        summary.MARKERS,
        "probe",
        CaseMarker(
            case="probe", issue=1, oracle="numerics", manifests_as="finding", signal="dtype"
        ),
    )
    # A cache file already sitting there, primed with a verdict that would be
    # reused if --no-cache did not stop the cache from being read at all.
    written = tmp_path / "observations.json"
    poisoned = {
        "fingerprint": "irrelevant -- never read",
        "observations": {
            "probe": {
                "verdict": "GREEN",
                "exit_code": 0,
                "line": "GREEN poisoned",
                "source": "deadbeef",
            }
        },
    }
    written.write_text(json.dumps(poisoned))

    exit_code = summary.main(["--no-cache"])

    assert exit_code == 0
    assert run_count(runs) == 1, "the probe must actually run, not be answered from the cache"
    output = capsys.readouterr().out
    assert "RED" in output
    assert "0 of the 1 were reused: the observation cache is off." in output
    # And the poisoned file is untouched -- --no-cache neither reads it (the
    # verdict above proves that) nor writes it.
    assert json.loads(written.read_text()) == poisoned


def test_the_default_cache_lives_outside_the_repository_and_is_keyed_by_checkout(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(summary.CACHE_ENV_VAR, raising=False)
    here = cache_file()
    assert here is not None
    assert REPO_ROOT not in here.parents, "a machine-local measurement must not be committable"

    monkeypatch.setattr(summary, "CASES_DIR", tmp_path / "another-checkout" / "cases")
    assert cache_file() != here


def test_the_table_says_which_rows_were_reused(corpus):
    # The reader is entitled to know how many rows were produced by an earlier
    # step rather than measured now -- and, just as importantly, when none
    # were: that is what a re-run after a crashed environment must say.
    reused = [Observation(case=CASES[0], verdict="RED", exit_code=1, line="RED", cached=True)]
    fresh = [Observation(case=CASES[0], verdict="RED", exit_code=1, line="RED")]

    assert "1 of the 1 was reused from the observation cache" in render_table(
        reused, "2.14.0+cpu", ""
    )
    assert "0 of the 1 were reused from the observation cache" in render_table(
        fresh, "2.14.0+cpu", ""
    )


def test_the_cache_is_shared_between_two_interpreters(tmp_path):
    # The point of the whole exercise, and the one thing an in-process
    # functools.cache cannot do: CI's job-summary step is a different process
    # from the pytest run that measured the corpus.
    directory = tmp_path / "cases"
    directory.mkdir()
    runs = write_probe(directory)
    program = (
        "import cases.summary as s;"
        f" s.CASES_DIR = __import__('pathlib').Path({str(directory)!r});"
        " print(s.observe('probe').cached)"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        summary.CACHE_ENV_VAR: str(tmp_path / "observations.json"),
    }

    first = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    second = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() == "False"
    assert second.stdout.strip() == "True"
    assert run_count(runs) == 1
