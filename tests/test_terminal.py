"""Tests for the terminal report.

The report is rendered from synthetic records, not from a compiled run: it reads
:class:`~compile_check.results.RunSet` and
:class:`~compile_check.oracles.Finding` and nothing else, so it can be pinned
exactly without paying for a compile. The main test is a snapshot with the ANSI
stripped, which is the only way to catch a layout change that a substring
assertion would sail past.
"""

from __future__ import annotations

import re

import pytest

from compile_check import __version__
from compile_check.localize import localize
from compile_check.oracles import Finding
from compile_check.report.terminal import DEFAULT_MAX_FINDINGS, render
from compile_check.results import BackendResult, CapturedException, GraphBreak, GraphHealth, RunSet

ANSI = re.compile(r"\033\[[0-9;]*m")

ENV = {
    "torch_version": "2.14.0+cpu",
    "torch_git_version": "08187d9e0fba1234567890",
    "python_version": "3.10.12",
    "platform": "Linux-6.8.0-1057-oracle-aarch64-with-glibc2.35",
    "machine": "aarch64",
    "cpu_flags": "asimd asimdhp asimddp",
    "cuda_available": False,
    "inductor_force_disable_caches": True,
}

LANES = (("eager", 0.0005, 0.0001), ("aot_eager", 1.25, 0.0003), ("inductor", 3.5, 0.0004))

CAPTURE = CapturedException(
    type="RuntimeError",
    message="this target is broken on purpose",
    traceback=("Traceback (most recent call last):", "RuntimeError: broken on purpose"),
)


def strip(text: str) -> str:
    """The report as a terminal without colour support would show it."""
    return ANSI.sub("", text)


@pytest.fixture
def runset() -> RunSet:
    """Three clean lanes with fixed timings, so a snapshot can pin them."""
    built = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        env=dict(ENV),
    )
    for name, first, second in LANES:
        built.results[name] = BackendResult(
            backend=name,
            outputs=[1],
            first_call_s=first,
            second_call_s=second,
            # One graph, no breaks, and a counter that did not move: what a
            # compiled lane looks like when nothing is wrong. `None` for eager,
            # which is never compiled and so never has graphs -- and which the
            # checks table does not have a column for anyway.
            graph_health=None
            if name == "eager"
            else GraphHealth(
                graph_count=1, op_count=3, unique_graphs_before=1, unique_graphs_after=1
            ),
        )
    return built


@pytest.fixture
def findings() -> list[Finding]:
    """One divergent lane, three findings, two oracles, three severities."""
    return [
        Finding(
            oracle="metadata",
            backend="inductor",
            output_index=0,
            severity="fail",
            message="dtype differs: eager torch.int8, inductor torch.int64",
            details={"field": "dtype", "expected": "torch.int8", "got": "torch.int64"},
        ),
        Finding(
            oracle="metadata",
            backend="inductor",
            output_index=0,
            severity="warn",
            message=(
                "stride differs: eager (4, 1), inductor (1, 4), but both tensors are contiguous"
            ),
            details={"field": "stride", "expected": [4, 1], "got": [1, 4]},
        ),
        Finding(
            oracle="numerics",
            backend="inductor",
            output_index=0,
            severity="fail",
            message=(
                "Tensor-likes are not close!; Mismatched elements: 3 / 16 (18.8%); "
                "Greatest absolute difference: 0.5 at index (0, 2)"
            ),
            details={
                "rtol": 1.3e-6,
                "atol": 1e-5,
                "expected_dtype": "torch.float32",
                "got_dtype": "torch.float32",
                # Already in the message; the detail line must not repeat it.
                "assert_close": "the whole multi-line torch message",
            },
        ),
    ]


CLEAN_REPORT = f"""\
compile-check {__version__}   target m:model

environment
  torch     2.14.0+cpu (git 08187d9e0fba)
  python    3.10.12
  platform  Linux-6.8.0-1057-oracle-aarch64-with-glibc2.35
  machine   aarch64   cpu flags asimd asimdhp asimddp
  device    cpu   cuda available no
  run       backends eager, aot_eager, inductor   seed 0   fullgraph off   dynamic off   grad on
  module    deep copied per lane
  gradients compared at the numerics tolerances x10 (--grad-tol-factor 10)
  caches    disabled (force_disable_caches=True)

backends
  backend    outputs   first call  second call  status
  eager            1      0.0005s      0.0001s  ok
  aot_eager        1      1.2500s      0.0003s  ok
  inductor         1      3.5000s      0.0004s  ok

checks
  oracle    fail-on  aot_eager         inductor
  numerics  yes      pass              pass
  alias     yes      pass              pass
  metadata  yes      pass              pass
  grad      yes      pass              pass
  graph     no       pass              pass

  pass = no finding

findings
  none

stage
  clean: no backend diverged from eager across 2 lanes

next
  run with --json to save the result, --md for an issue draft, and --emit-test for a regression
  test (the minimizer lands in M3-3)"""


DIVERGENT_REPORT = f"""\
compile-check {__version__}   target m:model

environment
  torch     2.14.0+cpu (git 08187d9e0fba)
  python    3.10.12
  platform  Linux-6.8.0-1057-oracle-aarch64-with-glibc2.35
  machine   aarch64   cpu flags asimd asimdhp asimddp
  device    cpu   cuda available no
  run       backends eager, aot_eager, inductor   seed 0   fullgraph off   dynamic off   grad on
  module    deep copied per lane
  gradients compared at the numerics tolerances x10 (--grad-tol-factor 10)
  caches    disabled (force_disable_caches=True)

backends
  backend    outputs   first call  second call  status
  eager            1      0.0005s      0.0001s  ok
  aot_eager        1      1.2500s      0.0003s  ok
  inductor         1      3.5000s      0.0004s  ok

checks
  oracle    fail-on  aot_eager         inductor
  numerics  yes      pass              1 fail
  alias     no       pass              pass
  metadata  yes      pass              1 fail 1 warn
  grad      no       pass              pass
  graph     no       pass              pass

  pass = no finding

findings
  numerics  (1 fail)
    [fail] inductor output[0]
        Tensor-likes are not close!; Mismatched elements: 3 / 16 (18.8%); Greatest absolute
        difference: 0.5 at index (0, 2)
        expected_dtype torch.float32   got_dtype torch.float32   rtol 1.3e-06   atol 1e-05

  metadata  (1 fail, 1 warn)
    [fail] inductor output[0]
        dtype differs: eager torch.int8, inductor torch.int64
        field dtype   expected torch.int8   got torch.int64
    1 more metadata finding not shown (--max-findings 1)

stage
  first diverges at inductor, which implicates inductor lowering/codegen
  that is where the divergence becomes observable, not necessarily where the fix belongs

next
  run with --json to save the result, --md for an issue draft, and --emit-test for a regression
  test (the minimizer lands in M3-3)"""


def test_a_clean_run_renders_exactly_this(runset):
    report = render(
        runset,
        [],
        localize(runset, []),
        fail_on=["numerics", "alias", "metadata", "grad"],
    )
    assert report == CLEAN_REPORT


def test_a_divergent_run_renders_exactly_this(runset, findings):
    # Rendered with colour and stripped, which pins two things at once: the
    # layout, and that colour adds escapes and changes nothing else.
    report = render(
        runset,
        findings,
        localize(runset, findings),
        fail_on=["numerics", "metadata"],
        max_findings=1,
        color=True,
    )
    assert strip(report) == DIVERGENT_REPORT


def test_colour_is_off_unless_it_is_asked_for(runset, findings):
    plain = render(runset, findings, localize(runset, findings))
    painted = render(runset, findings, localize(runset, findings), color=True)

    assert "\033[" not in plain
    assert "\033[" in painted
    assert strip(painted) == plain


def test_every_finding_is_shown_by_default(runset, findings):
    report = render(runset, findings, localize(runset, findings), max_findings=DEFAULT_MAX_FINDINGS)

    assert "not shown" not in report
    assert report.count("[fail]") == 2
    assert report.count("[warn]") == 1


def test_the_hidden_findings_are_counted_not_dropped(runset, findings):
    report = render(runset, findings, localize(runset, findings), max_findings=1)

    assert "1 more metadata finding not shown (--max-findings 1)" in report
    # The cap is per oracle, so the single numerics finding is still printed.
    assert "Tensor-likes are not close" in report


def test_the_environment_block_always_carries_the_architecture(runset):
    # PLAN.md "Cross-architecture parity is a feature": a run whose provenance
    # is ambiguous is not usable as parity evidence.
    report = render(runset, [], localize(runset, []))

    assert "machine   aarch64" in report
    assert "2.14.0+cpu" in report
    assert "git 08187d9e0fba" in report


def test_a_raising_eager_lane_says_not_checked_rather_than_none(runset):
    for result in runset.results.values():
        result.exception = CAPTURE
        result.outputs = []
    verdict = localize(runset, [])
    report = render(runset, [], verdict)

    assert "not checked: nothing was compared" in report
    assert "findings\n  none" not in report
    assert "raised RuntimeError" in report
    assert "eager traceback (first lines):" in report
    # Every check cell is a dash: there was no reference to compare against.
    assert "-    = the lane raised, so there was nothing to compare" in report


def test_the_fp64_reference_is_labelled_and_never_a_lane(runset):
    # PLAN.md "The oracle blind spot": a reference the numerics oracle reads,
    # not a lane under test.
    runset.fp64 = BackendResult(
        backend="eager_fp64", outputs=[1], first_call_s=0.002, second_call_s=0.001
    )
    report = render(runset, [], localize(runset, []))

    assert "eager_fp64 (reference)" in report
    # Once, in the backends table. Never a column of the checks table and
    # never the backend a stage verdict names.
    assert report.count("eager_fp64") == 1
    assert "clean: no backend diverged from eager across 2 lanes" in report


def test_a_failing_repeat_call_is_surfaced_without_becoming_the_verdict(runset):
    runset.results["inductor"].second_call_exception = CAPTURE
    verdict = localize(runset, [])
    report = render(runset, [], verdict)

    assert "ok, then raised RuntimeError on the repeat call" in report
    assert "inductor answered once and raised RuntimeError on the repeat call" in report
    assert "clean: no backend diverged" in report


def test_a_run_with_only_the_eager_lane_has_nothing_to_compare():
    solo = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        env=dict(ENV),
        results={"eager": BackendResult(backend="eager", outputs=[1], first_call_s=0.1)},
    )
    report = render(solo, [], localize(solo, []))

    assert "no lane to compare: this run has only the eager reference" in report


def test_a_run_with_no_eager_lane_is_reported_as_having_no_reference():
    headless = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        env=dict(ENV),
        results={"inductor": BackendResult(backend="inductor", outputs=[1], first_call_s=0.1)},
    )
    report = render(headless, [], localize(headless, []))

    assert "no eager lane" in report
    assert "not checked: nothing was compared" in report


def test_allowed_caches_are_shouted_about(runset):
    runset.env["inductor_force_disable_caches"] = False
    report = render(runset, [], localize(runset, []))

    assert "ENABLED (force_disable_caches=False, --allow-caches)" in report


def test_the_module_row_says_what_actually_happened_to_the_module(runset):
    # The M3 brief's carry-over from the M2-2 review: a module the runner could
    # not deep copy left every lane sharing one object while this row still said
    # "deep copied per lane". A run whose lanes may have leaked state into each
    # other has to say so where the evidence is read.
    assert "module    deep copied per lane" in render(runset, [], localize(runset, []))

    runset.module_copy_error = "TypeError: cannot pickle 'module' object"
    report = render(runset, [], localize(runset, []))
    assert (
        "module    shared across every lane (deep copy failed: TypeError: cannot pickle "
        "'module' object)" in report
    )

    runset.share_module = True
    assert "module    shared across every lane (--share-module)" in render(
        runset, [], localize(runset, [])
    )


def test_a_plain_callable_is_not_claimed_to_have_been_copied(runset):
    runset.target_is_module = False
    report = render(runset, [], localize(runset, []))

    assert "module    not copied: the target is a plain callable" in report


def test_the_graph_row_is_a_real_check_and_not_a_placeholder(runset):
    report = render(runset, [], localize(runset, []), fail_on=["graph"])

    assert "graph     yes      pass              pass" in report
    assert "not yet" not in report


def test_a_lane_with_no_graph_health_gets_a_dash_and_says_why(runset):
    runset.results["inductor"].graph_health = None
    report = render(runset, [], localize(runset, []))

    assert "graph     no       pass              -" in report
    assert "-    = no graph health was recorded for that lane" in report


def test_a_graph_fail_is_reported_without_naming_a_compilation_stage(runset):
    # A graph break is the same answer reached with a slower plan, so the
    # ablation ladder has nothing to place. The stage block says that out loud
    # rather than leaving a "clean" verdict beside a red row unexplained.
    runset.results["inductor"].graph_health = GraphHealth(
        graph_count=2,
        break_count=1,
        breaks=(GraphBreak(reason="Data-dependent branching", user_frame="m.py:7 in forward"),),
    )
    finding = Finding(
        oracle="graph",
        backend="inductor",
        output_index=None,
        severity="fail",
        message="inductor broke the graph at m.py:7 in forward: Data-dependent branching",
        details={"field": "break_reasons"},
    )
    verdict = localize(runset, [finding])
    report = render(runset, [finding], verdict, fail_on=["graph"])

    assert "graph     yes      pass              1 fail" in report
    assert "clean: no backend diverged from eager across 2 lanes" in report
    # Wrapped prose, so compared with the line breaks collapsed.
    flat = " ".join(report.split())
    assert "inductor has 1 fail-severity graph finding." in flat
    assert "--fail-on graph is what turns it into exit code 1." in flat


def test_the_baseline_row_appears_only_when_there_is_a_baseline(runset):
    assert "baseline" not in render(runset, [], localize(runset, []))

    report = render(runset, [], localize(runset, []), baseline=".compile-check/baseline.json")
    assert (
        "baseline  .compile-check/baseline.json   (the graph oracle reports new breaks only)"
        in report
    )


def test_a_negative_cap_never_reports_more_hidden_than_exist(runset, findings):
    # A caller that clamps nothing must still get an honest count: the CLI
    # rejects a negative --max-findings, and the renderer does not trust it to.
    report = render(runset, findings, localize(runset, findings), max_findings=-5)

    assert "[fail]" not in report
    assert "2 more metadata findings not shown" in report
    assert "1 more numerics finding not shown" in report
