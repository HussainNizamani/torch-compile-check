"""Tests for the minimizer: the two passes, the budget, and the handoff note.

Two layers, deliberately. The shrink loop and the delta-debugging loop take
their predicate as a parameter, so most of what they do is tested against a
synthetic one and costs no compile at all; the end-to-end tests then run the
real predicate against ``tests/fixtures/divergent_child.py``, whose divergence
comes from a backend of our own rather than from a bug in the installed wheel.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_compile_check.discover import load_target
from torch_compile_check.minimize import (
    Budget,
    Minimization,
    Outcome,
    finding_key,
    handoff_note,
    minimize,
    reproducer,
    shrink_inputs,
    stub_children,
)
from torch_compile_check.oracles import Finding, OracleConfig, run_oracles
from torch_compile_check.report.pytest_case import select
from torch_compile_check.runner import run_all

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DIVERGENT = FIXTURES / "divergent_child.py"
PERTURBS = "torch_compile_check_perturbs"


def always(fn, args, kwargs):
    """A predicate that says every candidate still reproduces."""
    del fn, args, kwargs
    return Outcome(reproduced=True)


def never(fn, args, kwargs):
    """A predicate that says no candidate reproduces."""
    del fn, args, kwargs
    return Outcome(reproduced=False)


def while_leading_at_least(size: int):
    """Reproduces while every tensor leaf still has at least ``size`` rows."""

    def check(fn, args, kwargs):
        del fn
        leaves = torch.utils._pytree.tree_leaves((args, kwargs))
        return Outcome(
            reproduced=all(
                leaf.shape[0] >= size
                for leaf in leaves
                if isinstance(leaf, torch.Tensor) and leaf.dim()
            )
        )

    return check


def runs(fn, args, kwargs):
    """Reproduces exactly when the candidate model runs at all.

    Not a stand-in for a real oracle -- it is the honest predicate for "does a
    passthrough fit here", which is the branch of :func:`stub_children` that
    records a child it could not replace.
    """
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        return Outcome(reproduced=False, error=type(exc).__name__)
    return Outcome(reproduced=True)


def finding(**overrides) -> Finding:
    """A fail-severity finding, with whatever the test wants changed."""
    fields = {
        "oracle": "numerics",
        "backend": "inductor",
        "output_index": 0,
        "severity": "fail",
        "message": "Tensor-likes are not close!",
        "details": {},
    }
    fields.update(overrides)
    return Finding(**fields)


# --- the finding identity ---------------------------------------------------


def test_the_key_is_the_oracle_the_lane_the_index_and_the_field():
    assert finding_key(finding(details={"field": "dtype"})) == (
        "numerics",
        "inductor",
        0,
        "dtype",
    )


def test_the_key_ignores_the_message_and_the_values():
    # A shrunk input changes the first differing element and both numbers
    # around it, so a key that read them would call every successful halving a
    # different finding.
    before = finding(message="Greatest absolute difference: 0.5 at index (7, 2)")
    after = finding(
        message="Greatest absolute difference: 0.5 at index (0, 2)",
        details={"rtol": 1.3e-6, "atol": 1e-5},
    )
    assert finding_key(before) == finding_key(after)


def test_the_key_separates_two_lanes():
    assert finding_key(finding()) != finding_key(finding(backend="aot_eager"))


# --- input shrinking --------------------------------------------------------


def test_halving_stops_at_the_smallest_reproducing_batch():
    args, kwargs, shrinks, notes = shrink_inputs(
        None, (torch.zeros(16, 3),), {}, while_leading_at_least(4), budget=Budget()
    )
    assert args[0].shape == (4, 3)
    assert kwargs == {}
    assert [(s.index, s.before, s.after) for s in shrinks] == [(0, (16, 3), (4, 3))]
    assert notes == []


def test_halving_never_goes_below_one():
    args, _, shrinks, _ = shrink_inputs(None, (torch.zeros(5, 2),), {}, always, budget=Budget())
    assert args[0].shape == (1, 2)
    assert shrinks[0].after == (1, 2)


def test_leaves_that_share_a_leading_dimension_are_halved_together():
    # A batched model is handed a batch of activations and a batch of masks,
    # and halving one of them alone makes it raise rather than reproduce. The
    # group step is what shrinks such a target at all.
    def both_or_nothing(fn, args, kwargs):
        del fn, kwargs
        return Outcome(reproduced=args[0].shape[0] == args[1].shape[0])

    args, _, shrinks, _ = shrink_inputs(
        None, (torch.zeros(8, 3), torch.zeros(8)), {}, both_or_nothing, budget=Budget()
    )
    assert [tuple(leaf.shape) for leaf in args] == [(1, 3), (1,)]
    assert {s.index for s in shrinks} == {0, 1}


def test_a_leaf_of_its_own_is_still_offered_a_halving_after_the_group():
    # The group stops at 4 because leaf 1 cannot go below it; leaf 0 can, and
    # the per-leaf pass is what finds that out.
    def check(fn, args, kwargs):
        del fn, kwargs
        return Outcome(reproduced=args[1].shape[0] >= 4)

    args, _, _, _ = shrink_inputs(
        None, (torch.zeros(8, 3), torch.zeros(8)), {}, check, budget=Budget()
    )
    assert [tuple(leaf.shape) for leaf in args] == [(1, 3), (4,)]


def test_keyword_inputs_are_shrunk_too_and_stay_keyword():
    args, kwargs, shrinks, _ = shrink_inputs(
        None, (), {"x": torch.zeros(4, 2)}, always, budget=Budget()
    )
    assert args == ()
    assert kwargs["x"].shape == (1, 2)
    assert shrinks[0].index == 0


def test_a_target_without_a_batch_dimension_is_left_unchanged_with_a_note():
    args, _, shrinks, notes = shrink_inputs(
        None, (torch.zeros(1, 4), torch.zeros(())), {}, always, budget=Budget()
    )
    assert [tuple(leaf.shape) for leaf in args] == [(1, 4), ()]
    assert shrinks == []
    assert notes == [
        "no input has a leading dimension above 1, so there was nothing to halve "
        '(v1 shrinks the leading dimension only, see PLAN.md "Minimizer, v1")'
    ]


def test_a_target_with_no_tensor_input_says_so():
    _, _, shrinks, notes = shrink_inputs(None, (3, "x"), {}, always, budget=Budget())
    assert shrinks == []
    assert notes == ["no input is a tensor, so there was no dimension to shrink"]


def test_a_load_bearing_batch_is_reported_rather_than_silently_left_alone():
    args, _, shrinks, notes = shrink_inputs(None, (torch.zeros(8, 2),), {}, never, budget=Budget())
    assert args[0].shape == (8, 2)
    assert shrinks == []
    assert notes == [
        "every input's leading dimension is load-bearing: halving it stopped reproducing "
        "the finding, so the inputs are unchanged"
    ]


def test_a_non_tensor_leaf_is_carried_through_untouched():
    args, _, _, _ = shrink_inputs(None, (torch.zeros(4), "flag", 7), {}, always, budget=Budget())
    assert args[0].shape == (1,)
    assert args[1:] == ("flag", 7)


def test_a_shrunk_leaf_is_a_leaf_that_still_requires_grad():
    # PLAN.md "grad" compares `.grad`, and only a leaf ever gets one; a plain
    # slice of a tensor that requires grad is not a leaf.
    original = torch.zeros(4, 2, requires_grad=True)
    args, _, _, _ = shrink_inputs(None, (original,), {}, always, budget=Budget())
    assert args[0].requires_grad
    assert args[0].is_leaf
    assert args[0].shape == (1, 2)


def test_the_original_inputs_are_not_touched():
    original = torch.arange(8.0).reshape(4, 2)
    shrink_inputs(None, (original,), {}, always, budget=Budget())
    assert original.shape == (4, 2)


# --- submodule delta debugging ----------------------------------------------


class Head(nn.Module):
    """A shape-preserving block a passthrough can stand in for."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def test_a_plain_callable_has_no_children_and_says_so():
    _, stubs, kept, notes = stub_children(lambda x: x, (1,), {}, always, budget=Budget())
    assert (stubs, kept) == ([], [])
    assert notes == ["the target is a plain callable, so it has no children to replace"]


def test_a_module_with_no_children_says_so():
    _, stubs, kept, notes = stub_children(Head(), (torch.zeros(2),), {}, always, budget=Budget())
    assert (stubs, kept) == ([], [])
    assert notes == ["the target module has no children to replace"]


def test_a_replacement_that_keeps_the_finding_is_kept():
    model = nn.Sequential(Head(), Head())
    _, stubs, kept, _ = stub_children(model, (torch.zeros(2),), {}, always, budget=Budget())
    assert [(s.path, s.module) for s in stubs] == [("0", "Head"), ("1", "Head")]
    assert kept == []


def test_a_replacement_that_loses_the_finding_is_reverted_and_recorded():
    model = nn.Sequential(Head(), Head())
    _, stubs, kept, _ = stub_children(model, (torch.zeros(2),), {}, never, budget=Budget())
    assert stubs == []
    assert [(k.path, k.reason) for k in kept] == [
        ("0", "the finding did not survive the replacement, so it lives in here"),
        ("1", "the finding did not survive the replacement, so it lives in here"),
    ]


def test_a_child_a_passthrough_does_not_fit_says_which_it_was():
    # Replacing the first Linear leaves a (2, 4) activation flowing into a
    # Linear(8, 4), which raises; the second one is shape-preserving from the
    # rest of the model's point of view and is replaced.
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    _, stubs, kept, _ = stub_children(model, (torch.zeros(2, 4),), {}, runs, budget=Budget())
    assert [s.path for s in stubs] == ["1"]
    assert [(k.path, k.reason) for k in kept] == [
        ("0", "replacing it raised RuntimeError, so a passthrough does not fit there")
    ]


def test_a_stubbed_subtree_is_not_walked_into():
    model = nn.Sequential(nn.Sequential(Head(), Head()), Head())
    _, stubs, _, _ = stub_children(model, (torch.zeros(2),), {}, always, budget=Budget())
    # The whole block, not its two children: one candidate rather than three,
    # which is what makes the pass affordable on a deep model.
    assert [s.path for s in stubs] == ["0", "1"]


def test_an_identity_child_is_not_reported_as_a_reduction():
    model = nn.Sequential(nn.Identity(), Head())
    _, stubs, _, _ = stub_children(model, (torch.zeros(2),), {}, always, budget=Budget())
    assert [s.path for s in stubs] == ["1"]


def test_the_callers_module_is_never_edited():
    model = nn.Sequential(Head(), Head())
    before = copy.deepcopy(model)
    work, stubs, _, _ = stub_children(model, (torch.zeros(2),), {}, always, budget=Budget())
    assert stubs
    assert type(model[0]) is type(before[0]) is Head
    assert isinstance(work[0], nn.Identity)


def test_a_module_that_cannot_be_deep_copied_is_reported_rather_than_edited():
    class Uncopyable(nn.Module):
        def __deepcopy__(self, memo):
            raise TypeError("no")

    model = Uncopyable()
    model.child = Head()
    returned, stubs, kept, notes = stub_children(
        model, (torch.zeros(2),), {}, always, budget=Budget()
    )
    assert returned is model
    assert (stubs, kept) == ([], [])
    assert notes[0].startswith("the module could not be deep copied (TypeError: no)")


# --- the budget -------------------------------------------------------------


def test_a_zero_budget_stops_before_the_first_candidate():
    budget = Budget(seconds=0)
    assert budget.spend() is False
    assert budget.spent == 0
    assert "--budget of 0s ran out after 0 candidate re-runs" in budget.stopped


def test_the_step_ceiling_stops_and_says_which_ceiling_it_was():
    budget = Budget(steps=2)
    assert [budget.spend() for _ in range(3)] == [True, True, False]
    assert "ceiling of 2 candidate re-runs" in budget.stopped


def test_a_stopped_budget_stays_stopped():
    budget = Budget(steps=0)
    budget.spend()
    first = budget.stopped
    budget.spend()
    assert budget.stopped == first


def test_an_exhausted_budget_leaves_the_inputs_where_it_found_them():
    args, _, shrinks, notes = shrink_inputs(
        None, (torch.zeros(8, 2),), {}, always, budget=Budget(seconds=0)
    )
    assert args[0].shape == (8, 2)
    assert shrinks == []
    # And says which of the two it was. The predicate here is `always`, so
    # halving would have worked; the note used to claim the leading dimension
    # was load-bearing, which the pass had not checked and could not (M3-3
    # verifier).
    assert notes == [
        "the inputs were not shrunk: the --budget of 0s ran out after 0 candidate re-runs"
    ]


def test_an_exhausted_step_ceiling_is_reported_as_the_ceiling_and_not_as_a_measurement():
    _, _, shrinks, notes = shrink_inputs(
        None, (torch.zeros(8, 2),), {}, always, budget=Budget(steps=0)
    )
    assert shrinks == []
    assert notes == [
        "the inputs were not shrunk: the ceiling of 0 candidate re-runs was reached "
        "(--budget SECONDS is the other way to bound this)"
    ]


def test_a_record_a_ceiling_stopped_does_not_call_the_case_irreducible():
    # The same sentence one level up: `summary` is what the Action's job
    # summary prints, so this is the line most readers see.
    stopped = Minimization(
        finding=finding(),
        reproduced=True,
        partial=True,
        partial_reason="the --budget of 0s ran out after 0 candidate re-runs",
    )
    assert stopped.changed is False
    assert stopped.summary == (
        "nothing was reduced: the --budget of 0s ran out after 0 candidate re-runs"
    )
    assert "load-bearing" not in stopped.summary


# --- the minifier handoff ---------------------------------------------------


def test_the_handoff_names_both_environment_variables():
    note = handoff_note(finding())
    assert "TORCHDYNAMO_REPRO_AFTER=aot" in note
    assert "TORCHDYNAMO_REPRO_LEVEL=4" in note


def test_the_handoff_says_the_minifier_is_numerics_only_for_another_oracle():
    note = handoff_note(finding(oracle="alias"))
    assert "TORCHDYNAMO_REPRO_AFTER=aot" in note
    assert "compares numbers only" in note
    assert "would not isolate this alias finding" in note


def test_the_handoff_is_never_executed(monkeypatch):
    # PLAN.md "Minimizer, v1": the handoff is a note, not a run. Reading the
    # note must not change the process's own repro configuration.
    config = pytest.importorskip("torch._dynamo.config")
    before = (config.repro_after, config.repro_level)
    handoff_note(finding())
    assert (config.repro_after, config.repro_level) == before


# --- the record ------------------------------------------------------------


def test_a_run_with_nothing_to_minimize_carries_the_reason():
    record = Minimization.not_attempted("this run has no fail-severity finding")
    assert record.attempted is False
    assert record.changed is False
    assert record.summary == "nothing to minimize: this run has no fail-severity finding"


def test_a_record_that_reduced_nothing_says_so_rather_than_nothing():
    record = Minimization(finding=finding(), reproduced=True)
    assert record.attempted is True
    assert record.changed is False
    assert "every input and every child is load-bearing" in record.summary


# --- end to end, against the real predicate ---------------------------------


@pytest.fixture(scope="module")
def divergent():
    """The fixture run under eager and the perturbing backend, plus its finding."""
    target = load_target(str(DIVERGENT))
    runset = run_all(target, ["eager", PERTURBS])
    cfg = OracleConfig()
    findings = [item for lane in runset.others for item in run_oracles(runset.eager, lane, cfg)]
    top = select(findings)
    assert top is not None, "the fixture must diverge for the minimizer to have work"
    return target, runset, top, cfg


def test_the_minimizer_finds_the_one_child_the_finding_lives_in(divergent):
    target, runset, top, cfg = divergent
    record = minimize(target, runset, top, cfg)

    assert record.reproduced is True
    assert [(s.path, s.module) for s in record.stubs] == [("head", "Linear"), ("tail", "Linear")]
    assert [(k.path, k.module) for k in record.kept] == [("middle", "Guilty")]
    assert record.kept[0].reason.endswith("so it lives in here")
    assert record.partial is False


def test_the_minimizer_shrinks_the_batch_to_one_on_the_same_run(divergent):
    target, runset, top, cfg = divergent
    record = minimize(target, runset, top, cfg)

    assert [(s.index, s.before, s.after) for s in record.shrinks] == [(0, (8, 4), (1, 4))]
    assert record.changed is True
    assert record.summary == "2 child modules replaced with torch.nn.Identity() and 1 input shrunk"


def test_a_budget_that_expires_leaves_a_partial_result_marked_as_such(divergent):
    target, runset, top, cfg = divergent
    record = minimize(target, runset, top, cfg, budget=0)

    # The control re-run is outside the budget on purpose, so the finding is
    # still known to reproduce; everything after it is what did not happen.
    assert record.reproduced is True
    assert record.partial is True
    assert record.steps == 0
    assert (record.shrinks, record.stubs) == ((), ())
    assert "--budget of 0s ran out" in record.partial_reason


def test_the_step_ceiling_also_marks_the_result_partial(divergent):
    target, runset, top, cfg = divergent
    record = minimize(target, runset, top, cfg, steps=1)

    assert record.partial is True
    assert record.steps == 1
    assert "ceiling of 1 candidate re-runs" in record.partial_reason


def test_a_finding_that_does_not_reproduce_stops_before_either_pass(divergent):
    # Same run, but the key is one nothing will match: the control re-run comes
    # back negative and the record says nothing was minimized rather than
    # showing an empty reduction that a reader would take for a resistant case.
    target, runset, top, cfg = divergent
    record = minimize(target, runset, finding(oracle=top.oracle, backend="eager"), cfg)

    assert record.reproduced is False
    assert (record.shrinks, record.stubs, record.kept) == ((), (), ())
    assert record.notes[0].startswith("the finding did not reproduce on a re-run")
    assert "TORCHDYNAMO_REPRO_AFTER=aot" in record.handoff


def test_the_minimized_module_still_reproduces_when_it_is_run(divergent):
    # The claim the record makes, checked against real torch rather than
    # against the record's own bookkeeping: the stubbed model on the shrunk
    # input still diverges between the two lanes.
    target, runset, top, cfg = divergent
    record = minimize(target, runset, top, cfg)

    model = copy.deepcopy(target.fn)
    for stub in record.stubs:
        setattr(model, stub.path, nn.Identity())
    inputs = (target.example_inputs[0][: record.shrinks[0].after[0]],)

    check = reproducer(runset, top, cfg)
    assert check(model, inputs, {}).reproduced is True
