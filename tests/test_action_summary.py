"""Tests for ``action/summary.sh``, the composite action's job-summary renderer.

The script is what ``action/action.yml``'s "Run torch-compile-check" step calls once
per target, so these tests run *that file* rather than a copy of its logic, and
they run it against JSON reports the CLI writes here, in the test, rather than
against a stored document. A stored document would drift from the schema the
moment ``report/json.py`` changed and the test would keep passing; a report
written by the run under test cannot.

That is also why the reports are produced by ``main()`` and not hand-built: the
renderer's whole job is to read fields another module owns
(``backends[].graph.break_count`` and the top-level ``minimized`` object of
schema version 2), and a hand-built document would be this test agreeing with
itself about their names.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from torch_compile_check.cli import EXIT_FINDING, EXIT_OK, main

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SUMMARY_SH = REPO_ROOT / "action" / "summary.sh"

# The lanes are kept to the cheapest pair that still answers the question. The
# graph oracle reads torch._dynamo.explain, which does not depend on the
# backend, and the perturbing fixture backend is what makes a finding without
# needing inductor's codegen -- so no test here pays for a real compile.
GRAPH_BREAK = FIXTURES / "graph_break.py"
DIVERGENT = FIXTURES / "divergent_child.py"
PERTURBS = "torch_compile_check_perturbs"


def render(*args: str) -> str:
    """Run ``summary.sh`` and return its stdout, failing loudly on a usage error."""
    completed = subprocess.run(
        ["bash", str(SUMMARY_SH), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.fixture(scope="module")
def graph_break_report(tmp_path_factory) -> Path:
    """A clean run of the two-graph-break fixture, with no minimizer."""
    path = tmp_path_factory.mktemp("reports") / "graph_break.json"
    code = main(
        [
            str(GRAPH_BREAK),
            "--backends",
            "eager,aot_eager",
            "--fail-on",
            "numerics,alias,metadata,grad,graph",
            "--json",
            str(path),
        ]
    )
    assert code == EXIT_OK
    return path


@pytest.fixture(scope="module")
def minimized_report(tmp_path_factory) -> Path:
    """A run with a finding, shrunk: the report that carries a full ``minimized``."""
    path = tmp_path_factory.mktemp("reports") / "minimized.json"
    code = main(
        [str(DIVERGENT), "--backends", f"eager,{PERTURBS}", "--minimize", "--json", str(path)]
    )
    assert code == EXIT_FINDING
    return path


@pytest.fixture(scope="module")
def exhausted_report(tmp_path_factory) -> Path:
    """The same finding under a zero budget, so the minimizer reports partial."""
    path = tmp_path_factory.mktemp("reports") / "partial.json"
    code = main(
        [
            str(DIVERGENT),
            "--backends",
            f"eager,{PERTURBS}",
            "--minimize",
            "--budget",
            "0",
            "--json",
            str(path),
        ]
    )
    assert code == EXIT_FINDING
    return path


def test_the_row_carries_the_graph_break_count(graph_break_report):
    # The fixture breaks the graph twice, on purpose, and that is the number a
    # reader of the summary is looking for: PLAN.md "graph" keeps breaks
    # informational, so nothing else in the row would mention them.
    assert json.loads(graph_break_report.read_text())["backends"][1]["graph"]["break_count"] == 2

    row = render(
        "row",
        "tests/fixtures/graph_break.py",
        "0",
        "exit 0",
        "clean: nothing",
        str(graph_break_report),
    )

    assert row == ("| `tests/fixtures/graph_break.py` | 0 | exit 0 | 2 | clean: nothing |\n")


def test_the_row_reports_a_run_with_no_breaks_as_zero(minimized_report):
    row = render("row", "t.py", "1", "exit 1", "first diverges", str(minimized_report))

    assert row.split("|")[4].strip() == "0"


def test_the_row_degrades_to_a_placeholder_without_a_report(tmp_path):
    # A tool error can end a run before any JSON is written. The summary is a
    # convenience beside the verdict and must not fail with it.
    row = render("row", "gone.py", "2", "exit 2", "", str(tmp_path / "never-written.json"))

    assert row == "| `gone.py` | 2 | exit 2 | - | - |\n"


def test_the_row_degrades_on_a_document_that_is_not_a_report(tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text("this is not JSON at all")

    assert render("row", "t.py", "0", "exit 0", "-", str(junk)).split("|")[4].strip() == "-"


def test_the_row_names_the_lanes_when_they_disagree(graph_break_report, tmp_path):
    # Two lanes with different break counts is a real state -- a graph break
    # that only Dynamo's fullgraph path hits, or a lane that raised before it
    # could be measured -- and one number could only be wrong about it.
    document = json.loads(graph_break_report.read_text())
    document["backends"][1]["graph"]["break_count"] = 5
    document["backends"].append(
        {**document["backends"][1], "backend": "inductor", "graph": {"measured": False}}
    )
    edited = tmp_path / "disagreeing.json"
    edited.write_text(json.dumps(document))

    assert render("row", "t.py", "0", "exit 0", "-", str(edited)).split("|")[4].strip() == (
        "aot_eager 5, inductor n/a"
    )


def test_no_minimized_section_when_the_run_did_not_minimize(graph_break_report):
    # `minimized` is null rather than absent, which is the schema-2 way of
    # saying "not run" (report/json.py). Nothing at all is the right output.
    assert json.loads(graph_break_report.read_text())["minimized"] is None

    assert render("minimized", "tests/fixtures/graph_break.py", str(graph_break_report)) == ""


def test_no_minimized_section_for_a_schema_1_document(graph_break_report, tmp_path):
    document = json.loads(graph_break_report.read_text())
    del document["minimized"]
    document["schema_version"] = 1
    old = tmp_path / "v1.json"
    old.write_text(json.dumps(document))

    assert render("minimized", "t.py", str(old)) == ""


def test_no_minimized_section_without_a_report(tmp_path):
    assert render("minimized", "t.py", str(tmp_path / "never-written.json")) == ""


def test_the_minimized_section_reports_what_the_pass_removed(minimized_report):
    document = json.loads(minimized_report.read_text())["minimized"]
    # What the renderer is asked to describe, asserted here so a change in the
    # minimizer shows up as this test's own failure rather than as a silently
    # emptier summary.
    assert [stub["path"] for stub in document["stubs"]] == ["head", "tail"]
    assert [kept["path"] for kept in document["kept"]] == ["middle"]
    assert document["shrinks"] == [{"index": 0, "before": [8, 4], "after": [1, 4]}]

    section = render("minimized", "tests/fixtures/divergent_child.py", str(minimized_report))

    assert section.startswith("<details><summary><code>tests/fixtures/divergent_child.py</code>")
    assert section.rstrip().endswith("</details>")
    assert f"- finding: `numerics` on `{PERTURBS}`, output 0 (fail)" in section
    assert "- input: leaf 0 (8, 4) -> (1, 4)" in section
    assert "- stubbed: `head` (Linear) -> `torch.nn.Identity()`" in section
    assert "- stubbed: `tail` (Linear) -> `torch.nn.Identity()`" in section
    assert "- kept: `middle` (Guilty)" in section
    assert f"- cost: {document['steps']} candidate re-runs in {document['seconds']}s" in section
    # The handoff paragraph is the same constant advice for every target and is
    # already in the step log; repeating it per target would bury the rows.
    assert "TORCHDYNAMO_REPRO_AFTER" not in section


def test_the_minimized_section_says_when_the_budget_ran_out(exhausted_report):
    document = json.loads(exhausted_report.read_text())["minimized"]
    assert document["partial"] is True

    section = render("minimized", "t.py", str(exhausted_report))

    assert f"- **partial**: {document['partial_reason']}" in section


def test_a_bad_invocation_is_a_usage_error(tmp_path):
    # The call sites are in action.yml, so a wrong argument count is a bug in
    # the action rather than something to paper over with a placeholder row.
    completed = subprocess.run(
        ["bash", str(SUMMARY_SH), "row", "only-one-argument"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "usage: summary.sh" in completed.stderr
