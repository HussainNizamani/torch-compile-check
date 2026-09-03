"""Every corpus case, through the real runner and the real oracles, once.

PLAN.md "Regression corpus" states the contract this module checks: import a
case, ``build()`` it, run it through :mod:`torch_compile_check.runner` and every
oracle, and ask the case's own ``check()`` what it thinks. If ``check()`` says
RED the tool must report at least one fail finding from the oracle the case
belongs to; if it says GREEN the tool must report none. The case is the ground
truth and this module is the tool being graded against it, which is the only
arrangement that survives a torch upgrade: a nightly that fixes a bug upstream
turns the case green and the assertion green with it, rather than turning the
suite red.

M2-1 and M2-2 grew one hand-written fixture and one hand-written test per case,
four of them, each repeating the same six lines of setup and each covering only
the case it was written for -- ``distributions_validation_branch`` and
``numerics_cpu_inductor_miscompile`` had no integration test at all. This module
is the parametrized version, and adding a case to :data:`CORPUS` is now the
whole cost of covering it.

Three neighbours, and what is deliberately not repeated from them:

* ``tests/test_corpus_twins.py`` runs each standalone script beside its
  discovery-convention twin through ``torch-compile-check``'s own ``main()`` and
  asserts the exit code and the stage line agree. That is the CLI's contract and
  this module does not restate it; what it takes from there is
  :data:`~tests.test_corpus_twins.TWINS`, so the flags a case needs and the
  backend its verdict lands on are written once.
* ``tests/test_corpus_markers.py`` asks whether the RED/GREEN each script
  reports still matches the version marker. That is a question about
  ``cases/markers.py`` being current, and its answer is a warning. This module's
  question is whether the *tool* saw what the script saw, and its answer is a
  failure.
* ``tests/test_oracles.py`` keeps the version-independent halves: a synthetic
  result carrying the shape of each bug, which pins the oracle's rule down on
  any torch, including one where the bug is long fixed.

The runs are module-scoped fixtures because each compiles three lanes, and the
whole file is a few minutes of inductor on a cold cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from cases.markers import MARKERS
from test_corpus_twins import TWINS
from torch_compile_check.discover import Target, import_target_module, load_target
from torch_compile_check.localize import localize
from torch_compile_check.oracles import Finding, OracleConfig, run_oracles
from torch_compile_check.results import BackendResult, CapturedException
from torch_compile_check.runner import run_all

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "cases"

# The ablation ladder of PLAN.md "Stage localization", which is what makes
# "inductor-only miscompile" a statement this module can check rather than a
# claim in FINDINGS.md: aot_eager runs the same graph without inductor's
# codegen, so a case that is clean there and fails here has been placed.
BACKENDS = ["eager", "aot_eager", "inductor"]

CFG = OracleConfig()


def _cfg(runset: Any) -> OracleConfig:
    """The oracle config for one corpus run.

    ``fullgraph`` is the only knob a case moves, and the graph oracle reads it:
    a graph break is informational when nobody asked for one graph and a broken
    promise when ``--fullgraph`` did. A config that did not carry it would grade
    the fullgraph cases under the wrong rule.
    """
    return OracleConfig(fullgraph=runset.fullgraph)


# tests/test_corpus_twins.py carries the flags each case needs as the CLI
# spells them, because that is what it passes to main(). This module calls
# run_all directly, so they are translated once, here.
_FLAG_TO_RUN_KWARG = {"--fullgraph": "fullgraph", "--dynamic": "dynamic"}

CheckArgs = Callable[[BackendResult, BackendResult], tuple[Any, Any, Any]]


def _outputs(eager: BackendResult, lane: BackendResult) -> tuple[Any, Any, Any]:
    """The first output of each lane: what most ``check()`` functions compare."""
    return eager.outputs[0], lane.outputs[0], None


def _live_input_and_output(eager: BackendResult, lane: BackendResult) -> tuple[Any, Any, Any]:
    """The (input, output) pair 195451's ``check()`` reads data pointers off.

    The live references, not the clones: the question is whether the returned
    tensor *is* the input's storage, and a clone has a storage of its own.
    """
    return (
        (eager.input_refs[0], eager.output_refs[0]),
        (lane.input_refs[0], lane.output_refs[0]),
        None,
    )


def _live_input_and_second_output(
    eager: BackendResult, lane: BackendResult
) -> tuple[Any, Any, Any]:
    """The (input, output) pair alias_view_slice_scatter_copyback's ``check()``
    reads data pointers off.

    Same shape as :func:`_live_input_and_output`, at ``output_refs[1]`` rather
    than ``[0]``: that case's ``fn`` returns ``(before, updated)``, and it is
    ``updated`` -- the second output -- the alias oracle finds aliasing `x`.
    Returning ``before`` alongside it is what makes Inductor's reinplace pass
    treat the view as eliminable in the first place; dropping it stops the
    case from reproducing (see the case's own docstring).
    """
    return (
        (eager.input_refs[0], eager.output_refs[1]),
        (lane.input_refs[0], lane.output_refs[1]),
        None,
    )


def _raised_or_returned(eager: BackendResult, lane: BackendResult) -> tuple[Any, Any, Any]:
    """The ``(raised, payload)`` pair 194593's ``check()`` reads.

    That case's RED is "the compiled lane raised where eager succeeded", so its
    ``check()`` takes the attempt rather than a value. The runner records an
    exception as a :class:`~torch_compile_check.results.CapturedException` -- data,
    with no traceback object and no live exception -- and ``check()`` reads
    ``type(payload).__name__`` and ``str(payload)`` off an exception, so the
    recorded type and message are rebuilt into one. A reconstruction, and said
    out loud as one: it is the case's message string that consumes it, never a
    verdict.
    """
    payload: Any = lane.outputs[0] if lane.ok else _rebuild(lane.exception)
    return eager.outputs[0], (not lane.ok, payload), None


def _repeated_outputs(eager: BackendResult, lane: BackendResult) -> tuple[Any, Any, Any]:
    """190765's ``check()`` takes a list of outputs from repeated compiled calls.

    One element, because the runner keeps one set of outputs per lane: it calls
    each backend twice and times both, but the second call's values are not
    recorded (``BackendResult.second_call_s`` is a duration, not a result). The
    value comparison is therefore exercised in full and the determinism half of
    that ``check()`` is vacuous here -- the standalone script is where the four
    repeated calls happen, and ``tests/test_corpus_markers.py`` runs it.
    """
    return eager.outputs[0], [lane.outputs[0]], None


def _rebuild(captured: CapturedException | None) -> Any:
    """The recorded exception as an object of a class with its name."""
    assert captured is not None
    return type(captured.type, (Exception,), {})(captured.message)


# One row per C-1 corpus case. `reports_as` is the set of oracles that carry the
# fail findings when the case is RED -- the "oracle named in the case header" of
# the M2-3 brief, resolved for the shape `build()` actually returns, which is
# not always the shape the twin returns:
#
#   alias_noop_view_identity's build() goes on to resize_() through the
#   collapsed view and returns `base + 1`, so what reaches the report is a
#   shape and a value divergence; the aliasing underneath it is only visible
#   when the base and the view come back together, which is the twin
#   cases/alias_noop_view.py and the second test in this file.
#
# `reports_as = None` means the RED reaches no *correctness* oracle: 194593's
# compiled lane raises under fullgraph rather than answering differently, which
# is exit 1 by the raised-lane rule and belongs to none of the four categories
# that compare two lanes. The graph oracle does reach it -- it measures the lane
# alone and names the break the fullgraph request could not survive -- which is
# why the assertion below allows graph fails on this row and nothing else.
CORPUS = (
    pytest.param(
        "alias_slice_scatter_copyback",
        _live_input_and_output,
        frozenset({"alias"}),
        id="alias_slice_scatter_copyback",
    ),
    pytest.param(
        "alias_noop_view_identity",
        _outputs,
        frozenset({"numerics", "metadata"}),
        id="alias_noop_view_identity",
    ),
    pytest.param(
        "dtype_int8_matmul_promotion",
        _outputs,
        frozenset({"metadata"}),
        id="dtype_int8_matmul_promotion",
    ),
    pytest.param(
        "distributions_validation_branch",
        _raised_or_returned,
        None,
        id="distributions_validation_branch",
    ),
    pytest.param(
        "numerics_cpu_inductor_miscompile",
        _repeated_outputs,
        frozenset({"numerics"}),
        id="numerics_cpu_inductor_miscompile",
    ),
    # Reviewer-reported siblings of alias_slice_scatter_copyback (2026-09-03),
    # not part of the original C-1 slice; see cases/README.md and
    # cases/markers.py's note on each.
    pytest.param(
        "alias_view_slice_scatter_copyback",
        _live_input_and_second_output,
        frozenset({"alias"}),
        id="alias_view_slice_scatter_copyback",
    ),
    pytest.param(
        "alias_diagonal_scatter_index_put_chain",
        _live_input_and_output,
        frozenset({"alias"}),
        id="alias_diagonal_scatter_index_put_chain",
    ),
)

# The twin table, keyed by the standalone script it belongs to, so this module
# and tests/test_corpus_twins.py cannot disagree about which flags a case needs
# or which backend its verdict lands on.
_TWINS_BY_CASE = {param.id: param.values for param in TWINS}


def _run_kwargs(case: str) -> dict[str, bool]:
    """``run_all`` keyword arguments for the flags the twin table records."""
    _standalone, _twin, extra_args, _backend = _TWINS_BY_CASE[case]
    return {_FLAG_TO_RUN_KWARG[flag]: True for flag in extra_args}


def _red_stage_backend(case: str) -> str:
    """The backend the stage verdict names when this case is RED."""
    return _TWINS_BY_CASE[case][3]


@pytest.fixture(scope="module")
def corpus_runs():
    """Every case run once, lazily, and cached for the whole module.

    A dict of runs rather than a parametrized fixture because the second test
    needs one of them by name, and because a case that is never asked for is
    never compiled.
    """
    runs: dict[str, tuple[Any, Any]] = {}

    def run(case: str):
        if case not in runs:
            module = import_target_module(str(CASES / f"{case}.py"))
            fn, example_inputs = module.build()
            target = Target(fn=fn, example_inputs=example_inputs, name=f"{case}:fn")
            runs[case] = (
                module,
                run_all(target, BACKENDS, seed=0, **_run_kwargs(case)),
            )
        return runs[case]

    return run


@pytest.mark.parametrize(("case", "check_args", "reports_as"), CORPUS)
def test_a_corpus_case_reports_exactly_when_its_own_check_says_red(
    corpus_runs, case: str, check_args: CheckArgs, reports_as: frozenset[str] | None
):
    module, runset = corpus_runs(case)
    eager = runset.results["eager"]
    # Eager raising is not a verdict about compilation, it is a broken case, and
    # it is the one outcome that must fail loudly rather than skip.
    assert eager.ok, eager.exception

    cfg = _cfg(runset)
    findings: dict[str, list[Finding]] = {
        lane.backend: run_oracles(eager, lane, cfg) for lane in runset.others
    }
    # check() is asked after every oracle has run, because 195451's RED probe
    # mutates the output it is handed -- writing into the compiled result to
    # prove the alias is load-bearing -- and an oracle reading that tensor
    # afterwards would be reading the probe rather than the run.
    verdicts = {lane.backend: module.check(*check_args(eager, lane)) for lane in runset.others}

    red_lanes = [backend for backend, (is_red, _) in verdicts.items() if is_red]
    for lane in runset.others:
        is_red, message = verdicts[lane.backend]
        fails = [finding for finding in findings[lane.backend] if finding.severity == "fail"]

        if not is_red:
            assert fails == [], (
                f"{case}'s own check() calls {lane.backend} clean ({message}) and "
                f"torch-compile-check reported {[f.message for f in fails]}"
            )
        elif reports_as is None:
            # The raised-lane RED: nothing to compare, so no *correctness*
            # oracle can say anything. What makes it exit 1 is the lane not
            # running at all.
            assert not lane.ok, (
                f"{case} is RED on {lane.backend} because the lane raises, and it "
                f"returned instead: {message}"
            )
            # The graph oracle is the exception, and since M3-1 it is the one
            # that explains this RED: 194593 raises under --fullgraph *because*
            # the graph broke, and graph health is measured from the lane alone
            # rather than by comparing it with eager. See `reports_as` above.
            assert {f.oracle for f in fails} <= {"graph"}, [f.message for f in fails]
        else:
            assert {finding.oracle for finding in fails} == reports_as, (
                f"{case} is RED on {lane.backend} ({message}) and torch-compile-check "
                f"reported {[(f.oracle, f.message) for f in fails]}"
            )
            assert {finding.backend for finding in fails} == {lane.backend}

    if not red_lanes:
        said = {backend: message for backend, (_, message) in verdicts.items()}
        pytest.skip(f"this torch does not reproduce {MARKERS[case].issue}: {said}")

    # RED somewhere, so the stage verdict must name the first lane that broke,
    # and the twin table's expectation is what it is measured against. The
    # ladder is what turns "inductor miscompiles this" from a claim into a
    # placement: three of these cases are clean under aot_eager.
    every_finding = [finding for lane in findings.values() for finding in lane]
    verdict = localize(runset, every_finding)
    assert verdict.first_divergent_backend == red_lanes[0]
    assert verdict.first_divergent_backend == _red_stage_backend(case)


def test_the_191449_identity_collapse_is_an_alias_fail_on_the_twin(corpus_runs):
    """The alias oracle's view of 191449, which the standalone shape hides.

    ``cases/alias_noop_view_identity.py``'s ``build()`` returns ``base + 1``
    after a ``resize_()`` through the collapsed view, so the divergence reaches
    the report as numerics and metadata and the aliasing itself never gets in
    front of the alias oracle. ``cases/alias_noop_view.py`` is the twin that
    returns the base and the view together, which is the shape the collapse is
    directly observable in -- and it is not covered by
    ``tests/test_corpus_twins.py``, which asserts the CLI's exit code and stage
    line rather than which oracle fired or what it said.
    """
    target = load_target(str(CASES / "alias_noop_view.py"))
    runset = run_all(target, BACKENDS, seed=0)
    eager = runset.results["eager"]
    inductor = runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception
    assert eager.output_refs[0] is not eager.output_refs[1], "eager itself collapsed the two"

    findings = [finding for lane in runset.others for finding in run_oracles(eager, lane, CFG)]
    alias_fails = [f for f in findings if f.oracle == "alias" and f.severity == "fail"]
    if inductor.output_refs[0] is not inductor.output_refs[1]:  # pragma: no cover - torch build
        assert alias_fails == []
        pytest.skip("this torch keeps the base and the view apart, so 191449 is fixed here")

    assert [f.details["field"] for f in alias_fails] == ["identity_added"]
    assert [f.backend for f in alias_fails] == ["inductor"]
    assert "one object for output[0] and output[1]" in alias_fails[0].message
    # PLAN.md "Where divergence appears is not always where the fix belongs":
    # the fix landed in AOTAutograd and the divergence still shows at inductor.
    assert localize(runset, findings).first_divergent_backend == "inductor"


def test_the_194593_fullgraph_break_is_a_graph_fail_naming_the_branch(corpus_runs):
    """The graph oracle's view of 194593, which no other oracle can reach.

    Under ``--fullgraph`` the compiled lane raises, so the four oracles that
    compare two lanes have nothing to compare and the report would otherwise say
    only "raised Unsupported". The graph oracle traces the same callable without
    the fullgraph demand and names the break that killed it, which is the
    diagnosis the issue is about: a data-dependent branch inside
    ``_kl_binomial_binomial``, not a numerical divergence.
    """
    _module, runset = corpus_runs("distributions_validation_branch")
    inductor = runset.results["inductor"]
    if inductor.ok:  # pragma: no cover - torch build
        pytest.skip("this torch captures the branch, so 194593 is fixed here")

    findings = run_oracles(runset.results["eager"], inductor, _cfg(runset), ["graph"])
    fails = [finding for finding in findings if finding.severity == "fail"]

    assert [f.details["reason"] for f in fails] == ["gb0170: Data-dependent branching"]
    assert "torch/distributions/kl.py" in str(fails[0].details["user_frame"])
    assert "_kl_binomial_binomial" in str(fails[0].details["user_frame"])
    assert "--fullgraph was requested and the graph broke anyway" in fails[0].message
    # And it still does not name a compilation stage: the ladder places
    # divergences, and the lane raising is what it placed here.
    assert localize(runset, findings).first_divergent_backend == "aot_eager"


def test_every_corpus_case_has_a_twin_and_a_marker():
    # The three tables that describe the corpus -- CORPUS here, TWINS in
    # tests/test_corpus_twins.py, MARKERS in cases/markers.py -- are written
    # separately because they answer different questions, and a case added to
    # one and forgotten in the others would silently lose its coverage.
    cases = {param.id for param in CORPUS}
    assert cases == set(_TWINS_BY_CASE)
    assert cases == set(MARKERS)
    for param in CORPUS:
        case, _check_args, reports_as = param.values
        if reports_as is None:
            assert MARKERS[case].manifests_as == "raised_lane"
        else:
            assert MARKERS[case].manifests_as == "finding"
            assert (CASES / f"{case}.py").is_file()


def test_the_corpus_scripts_all_expose_the_convention_they_are_read_through():
    # cases/README.md "Adding a case": build() returning (fn, example_inputs)
    # and check() comparing eager against compiled. This module reads both by
    # name, so the convention is worth one assertion rather than five obscure
    # AttributeErrors inside a three-lane run.
    for param in CORPUS:
        case = param.values[0]
        module = import_target_module(str(CASES / f"{case}.py"))
        fn, example_inputs = module.build()
        assert callable(fn), case
        assert isinstance(example_inputs, tuple), case
        assert all(isinstance(x, torch.Tensor) for x in example_inputs), case
        assert callable(module.check), case
