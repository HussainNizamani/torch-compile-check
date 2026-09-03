"""Tests for the stage-localization rules table.

Every case here is built from synthetic records rather than from a real run.
The ladder is a decision procedure over what the runner and the oracles already
produced, so it can be exercised exhaustively without compiling anything, and a
rules table is exactly the thing that should be tested one row at a time.
"""

from __future__ import annotations

import logging

import pytest

from torch_compile_check.localize import (
    CLEAN,
    MODEL,
    NO_REFERENCE,
    STAGES,
    BackendSummary,
    StageVerdict,
    implicated_stage,
    localize,
)
from torch_compile_check.oracles import Finding
from torch_compile_check.results import BackendResult, CapturedException, RunSet
from torch_compile_check.runner import ABLATION_LADDER

CAPTURE = CapturedException(
    type="RuntimeError",
    message="the lane blew up\nwith a second line",
    traceback=("Traceback (most recent call last):", "RuntimeError: the lane blew up"),
)


def make_runset(*backends: str, raised: dict[str, CapturedException] | None = None) -> RunSet:
    """A RunSet with one empty result per named backend, in the order given."""
    raised = raised or {}
    return RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        results={
            name: BackendResult(backend=name, outputs=[1], exception=raised.get(name))
            for name in backends
        },
    )


def finding(backend: str, severity: str = "fail", oracle: str = "numerics") -> Finding:
    """One finding against ``backend``, with no content the ladder reads."""
    return Finding(
        oracle=oracle,
        backend=backend,
        output_index=0,
        severity=severity,  # type: ignore[arg-type]
        message="synthetic",
        details={},
    )


# --- the rules table, one test per row -------------------------------------


def test_a_clean_run_names_no_stage():
    verdict = localize(make_runset("eager", "aot_eager", "inductor"), [])

    assert verdict.stage == CLEAN
    assert verdict.clean is True
    assert verdict.compared is True
    assert verdict.first_divergent_backend is None
    assert verdict.summary == "clean: no backend diverged from eager across 2 lanes"
    assert verdict.note is None


def test_eager_raising_is_the_model_not_a_stage():
    runset = make_runset("eager", "aot_eager", raised={"eager": CAPTURE, "aot_eager": CAPTURE})
    verdict = localize(runset, [])

    assert verdict.stage == MODEL
    assert verdict.first_divergent_backend == "eager"
    assert verdict.compared is False
    assert verdict.clean is False
    assert "raised RuntimeError under eager" in verdict.summary
    # The ladder stops: aot_eager raised too, and is not what is reported.
    assert "aot_eager" not in verdict.summary
    assert verdict.note == "the lane blew up"
    assert verdict.eager_exception is CAPTURE


def test_no_eager_lane_is_not_a_clean_run():
    verdict = localize(make_runset("aot_eager", "inductor"), [])

    assert verdict.stage == NO_REFERENCE
    assert verdict.clean is False
    assert verdict.compared is False
    assert "no eager lane" in verdict.summary


def test_a_fail_at_aot_eager_implicates_capture():
    runset = make_runset("eager", "aot_eager", "inductor")
    verdict = localize(runset, [finding("aot_eager"), finding("inductor")])

    assert verdict.first_divergent_backend == "aot_eager"
    assert verdict.stage == "capture/AOTAutograd/decomposition"
    assert verdict.summary == (
        "first diverges at aot_eager, which implicates capture/AOTAutograd/decomposition"
    )


def test_a_fail_only_at_inductor_implicates_lowering():
    runset = make_runset("eager", "aot_eager", "inductor")
    verdict = localize(runset, [finding("inductor")])

    assert verdict.first_divergent_backend == "inductor"
    assert verdict.stage == "inductor lowering/codegen"
    assert verdict.summary == (
        "first diverges at inductor, which implicates inductor lowering/codegen"
    )


def test_the_optional_fourth_lane_splits_the_capture_row():
    # PLAN.md "Stage localization": aot_eager_decomp_partition diverging while
    # aot_eager does not is the decomposition or the partitioner, not capture.
    runset = make_runset("eager", "aot_eager", "aot_eager_decomp_partition", "inductor")
    verdict = localize(runset, [finding("aot_eager_decomp_partition"), finding("inductor")])

    assert verdict.first_divergent_backend == "aot_eager_decomp_partition"
    assert verdict.stage == "decomposition/partitioner"


def test_a_lane_that_raised_diverges_even_with_no_findings():
    runset = make_runset("eager", "aot_eager", "inductor", raised={"inductor": CAPTURE})
    verdict = localize(runset, [])

    assert verdict.first_divergent_backend == "inductor"
    assert verdict.stage == "inductor lowering/codegen"


def test_the_verdict_never_says_where_the_bug_is():
    # PLAN.md "Where divergence appears is not always where the fix belongs".
    verdict = localize(make_runset("eager", "inductor"), [finding("inductor")])

    assert "first diverges at" in verdict.summary
    assert "the bug is in" not in verdict.summary
    assert verdict.note is not None
    assert "not necessarily where the fix belongs" in verdict.note


# --- what does and does not move the ladder --------------------------------


@pytest.mark.parametrize("severity", ["warn", "info"])
def test_only_fail_severity_moves_the_ladder(severity):
    runset = make_runset("eager", "aot_eager", "inductor")
    verdict = localize(runset, [finding("aot_eager", severity), finding("inductor", severity)])

    assert verdict.stage == CLEAN
    assert verdict.first_divergent_backend is None
    # Counted, though: a warn is reported even when it is not a verdict.
    counts = {entry.backend: entry for entry in verdict.backends}
    assert getattr(counts["aot_eager"], severity) == 1
    assert counts["aot_eager"].fail == 0


def test_a_failing_repeat_call_is_reported_but_does_not_set_the_stage():
    runset = make_runset("eager", "inductor")
    runset.results["inductor"].second_call_exception = CAPTURE
    verdict = localize(runset, [])

    assert verdict.stage == CLEAN
    counts = {entry.backend: entry for entry in verdict.backends}
    assert counts["inductor"].raised_on_repeat is CAPTURE
    assert counts["inductor"].diverged is False


def test_a_graph_fail_is_counted_but_never_names_a_compilation_stage():
    # M3-1: the ladder places a divergence, and a graph break is not one. It is
    # the same answer reached with a slower plan, so a stage verdict built from
    # it would read "first diverges at aot_eager" for a model whose numbers are
    # exactly right.
    runset = make_runset("eager", "aot_eager", "inductor")
    verdict = localize(runset, [finding("aot_eager", "fail", oracle="graph")])

    assert verdict.stage == CLEAN
    assert verdict.first_divergent_backend is None
    counts = {entry.backend: entry for entry in verdict.backends}
    assert counts["aot_eager"].fail == 1
    assert counts["aot_eager"].graph_fail == 1
    assert counts["aot_eager"].diverged is False


def test_a_correctness_fail_beside_a_graph_fail_still_sets_the_stage():
    runset = make_runset("eager", "aot_eager", "inductor")
    verdict = localize(
        runset,
        [finding("inductor", "fail", oracle="graph"), finding("inductor", "fail")],
    )

    assert verdict.first_divergent_backend == "inductor"
    counts = {entry.backend: entry for entry in verdict.backends}
    assert (counts["inductor"].fail, counts["inductor"].graph_fail) == (2, 1)


# --- ordering and counts ---------------------------------------------------


def test_the_ladder_order_beats_the_order_the_user_typed():
    # --backends inductor,aot_eager,eager runs in that order; the diagnosis
    # walks eager, aot_eager, inductor whatever the run order was.
    runset = make_runset("inductor", "aot_eager", "eager")
    verdict = localize(runset, [finding("aot_eager"), finding("inductor")])

    assert [entry.backend for entry in verdict.backends] == ["eager", "aot_eager", "inductor"]
    assert verdict.first_divergent_backend == "aot_eager"


def test_a_backend_off_the_ladder_keeps_its_place_at_the_end():
    runset = make_runset("eager", "cudagraphs", "inductor")
    verdict = localize(runset, [finding("cudagraphs")])

    assert [entry.backend for entry in verdict.backends] == ["eager", "inductor", "cudagraphs"]
    assert verdict.first_divergent_backend == "cudagraphs"
    # Not mapped to a stage, because guessing one would be inventing a diagnosis.
    assert verdict.stage == "the cudagraphs backend"


def test_the_summary_counts_every_severity_per_backend():
    runset = make_runset("eager", "aot_eager", "inductor")
    findings = [
        finding("inductor", "fail", "numerics"),
        finding("inductor", "fail", "metadata"),
        finding("inductor", "warn", "metadata"),
        finding("inductor", "info", "numerics"),
        finding("aot_eager", "warn", "metadata"),
    ]
    verdict = localize(runset, findings)
    counts = {entry.backend: entry for entry in verdict.backends}

    assert (counts["inductor"].fail, counts["inductor"].warn, counts["inductor"].info) == (2, 1, 1)
    assert (counts["aot_eager"].fail, counts["aot_eager"].warn) == (0, 1)
    assert (counts["eager"].fail, counts["eager"].warn, counts["eager"].info) == (0, 0, 0)


def test_a_finding_against_a_lane_that_did_not_run_is_reported_not_counted(caplog):
    runset = make_runset("eager", "inductor")
    with caplog.at_level(logging.WARNING, logger="torch_compile_check"):
        verdict = localize(runset, [finding("aot_eager")])

    assert verdict.stage == CLEAN
    assert "which is not in this run" in caplog.text


# --- the stage table itself ------------------------------------------------


def test_implicated_stage_covers_every_compiled_rung_of_the_ladder():
    assert set(STAGES) == set(ABLATION_LADDER) - {"eager"}
    for backend, stage in STAGES.items():
        assert implicated_stage(backend) == stage


def test_implicated_stage_names_an_unknown_backend_rather_than_guessing():
    assert implicated_stage("onnxrt") == "the onnxrt backend"


def test_a_verdict_is_immutable():
    verdict = localize(make_runset("eager"), [])
    assert isinstance(verdict, StageVerdict)
    assert isinstance(verdict.backends[0], BackendSummary)
    with pytest.raises(AttributeError):
        verdict.stage = "something else"  # type: ignore[misc]


def test_the_clean_summary_counts_lanes_in_the_singular_when_there_is_one():
    assert localize(make_runset("eager", "inductor"), []).summary == (
        "clean: no backend diverged from eager across 1 lane"
    )
