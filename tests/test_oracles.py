"""Tests for the four oracles: numerics, alias, metadata, and grad.

Most of these are hand-built tensor pairs rather than runs: an oracle's rules
are exactly what has to be pinned down, and a pair built here says what it is
testing in one line, where a compiled model saying the same thing costs seconds
and depends on what this torch happens to do. The integration tests in the
middle of the file -- the MLP, the aliasing fixture, the stateful module -- are
the ones that check the hand-built rules still describe real runs.

The regression corpus is not run here. ``tests/test_corpus_oracles.py`` puts
every case through the runner and every oracle in one parametrized test, graded
against the case's own ``check()``; what stays in this file is the shape of each
bug as a synthetic result, which holds on a torch where the bug is fixed.

conftest.py has already set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1, before torch
was imported, which is the only moment at which it can be set.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from compile_check.discover import import_target_module, load_target
from compile_check.oracles import (
    DEFAULT_GRAD_TOL_FACTOR,
    ORACLE_NAMES,
    ORACLES,
    Baseline,
    BaselineEntry,
    Finding,
    Oracle,
    OracleConfig,
    run_oracles,
)
from compile_check.oracles.alias import AliasOracle, relation
from compile_check.oracles.grad import GradOracle
from compile_check.oracles.graph import (
    MAX_REASON_CHARS,
    BaselineError,
    GraphOracle,
    baseline_entry,
    read_baseline,
    summarise_reason,
    write_baseline,
)
from compile_check.oracles.metadata import MetadataOracle
from compile_check.oracles.numerics import FALLBACK_TOLERANCES, NumericsOracle, resolve_tolerances
from compile_check.results import BackendResult, CapturedException, GraphBreak, GraphHealth, RunSet
from compile_check.runner import FP64_BACKEND, run_all, run_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES = REPO_ROOT / "cases"

NUMERICS = NumericsOracle()
ALIAS = AliasOracle()
METADATA = MetadataOracle()
GRAD = GradOracle()
GRAPH = GraphOracle()
CFG = OracleConfig()


def lane(backend: str, outputs: list[Any], **kwargs: Any) -> BackendResult:
    """A BackendResult carrying nothing but what the oracles read."""
    return BackendResult(backend=backend, outputs=list(outputs), **kwargs)


def pair(expected: Any, got: Any) -> tuple[BackendResult, BackendResult]:
    """A one-output eager lane and a one-output inductor lane."""
    return lane("eager", [expected]), lane("inductor", [got])


def fields(findings: list[Finding]) -> set[str | None]:
    """The ``details["field"]`` of every finding, for a set comparison."""
    return {finding.details.get("field") for finding in findings}


# --------------------------------------------------------------------------
# numerics: values
# --------------------------------------------------------------------------


def test_equal_tensors_produce_nothing():
    values = torch.randn(4, 3)
    assert NUMERICS.compare(*pair(values, values.clone()), CFG) == []
    assert METADATA.compare(*pair(values, values.clone()), CFG) == []


def test_a_difference_within_the_dtype_tolerance_is_not_a_finding():
    # PLAN.md "numerics": inductor fuses, fusion changes rounding, and a
    # bitwise check would fire on correct compilations. 5e-7 is about four
    # float32 ulps at 1.0, so the tensors really do differ; the default
    # tolerance (rtol 1.3e-6, atol 1e-5) is what makes it not a finding.
    values = torch.ones(8)
    got = values + 5e-7

    assert not torch.equal(values, got)
    assert NUMERICS.compare(*pair(values, got), CFG) == []


def test_a_difference_beyond_the_dtype_tolerance_is_a_fail():
    values = torch.ones(8)
    findings = NUMERICS.compare(*pair(values, values + 0.5), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    finding = findings[0]
    assert finding.oracle == "numerics"
    assert finding.backend == "inductor"
    assert finding.output_index == 0
    # The message is assert_close's own, which is why it carries the counts.
    assert "Mismatched elements: 8 / 8" in finding.message
    assert "\n" not in finding.message
    assert finding.details["rtol"] == pytest.approx(1.3e-6)
    assert finding.details["atol"] == pytest.approx(1e-5)
    assert "Greatest absolute difference" in finding.details["assert_close"]


def test_explicit_tolerances_override_the_dtype_defaults():
    values = torch.ones(8)
    got = values + 0.5

    assert NUMERICS.compare(*pair(values, got), OracleConfig(atol=1.0)) == []
    assert NUMERICS.compare(*pair(values, got), OracleConfig(rtol=1.0)) == []
    # And in the other direction: a difference the default swallows fails
    # under a tighter override.
    tight = OracleConfig(rtol=0.0, atol=0.0)
    findings = NUMERICS.compare(*pair(values, values + 5e-7), tight)
    assert [finding.severity for finding in findings] == ["fail"]
    assert findings[0].details["atol"] == 0.0


def test_integer_outputs_are_compared_exactly():
    expected = torch.tensor([1, 2, 3], dtype=torch.int64)
    findings = NUMERICS.compare(*pair(expected, torch.tensor([1, 2, 4])), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    assert findings[0].details["rtol"] == 0.0
    assert findings[0].details["atol"] == 0.0


def test_a_dtype_difference_alone_is_invisible_to_numerics():
    # PLAN.md "numerics": check_dtype=False, because the dtype is the metadata
    # oracle's job and reporting it twice hides which one is the real defect.
    expected = torch.tensor([1, 2], dtype=torch.int8)
    assert NUMERICS.compare(*pair(expected, expected.to(torch.int64)), CFG) == []


def test_a_stride_difference_alone_is_invisible_to_numerics():
    expected = torch.arange(12.0).reshape(3, 4)
    transposed = expected.t().contiguous().t()

    assert transposed.stride() != expected.stride()
    assert NUMERICS.compare(*pair(expected, transposed), CFG) == []


# --------------------------------------------------------------------------
# numerics: NaN and inf parity
# --------------------------------------------------------------------------


def test_a_nan_that_appears_is_its_own_finding():
    expected = torch.tensor([1.0, 2.0, 3.0])
    got = torch.tensor([1.0, float("nan"), 3.0])
    findings = NUMERICS.compare(*pair(expected, got), CFG)

    nan_findings = [f for f in findings if f.details.get("field") == "nan_mask"]
    assert len(nan_findings) == 1
    finding = nan_findings[0]
    assert finding.severity == "fail"
    assert "NaN positions differ in 1 of 3 elements" in finding.message
    assert finding.details["expected_count"] == 0
    assert finding.details["got_count"] == 1
    assert finding.details["first_differing_index"] == [1]


def test_a_nan_in_the_same_place_is_agreement():
    values = torch.tensor([1.0, float("nan"), 3.0])
    # equal_nan=True in the value comparison, and the masks match.
    assert NUMERICS.compare(*pair(values, values.clone()), CFG) == []


def test_an_inf_that_disappears_is_its_own_finding():
    expected = torch.tensor([1.0, float("inf")])
    got = torch.tensor([1.0, 3.4e38])
    findings = NUMERICS.compare(*pair(expected, got), CFG)

    inf_findings = [f for f in findings if f.details.get("field") == "inf_mask"]
    assert len(inf_findings) == 1
    assert inf_findings[0].severity == "fail"
    assert "inf positions differ" in inf_findings[0].message
    assert inf_findings[0].details["expected_count"] == 1
    assert inf_findings[0].details["got_count"] == 0


def test_integer_outputs_get_no_mask_findings():
    expected = torch.tensor([1, 2], dtype=torch.int32)
    findings = NUMERICS.compare(*pair(expected, torch.tensor([1, 5], dtype=torch.int32)), CFG)

    assert fields(findings) == {None}  # the value finding carries no field


# --------------------------------------------------------------------------
# numerics: non-tensor leaves and structure
# --------------------------------------------------------------------------


def test_equal_non_tensor_leaves_are_silent():
    assert NUMERICS.compare(*pair(3, 3), CFG) == []
    assert NUMERICS.compare(*pair("annotation", "annotation"), CFG) == []
    assert NUMERICS.compare(*pair(None, None), CFG) == []


def test_a_differing_non_tensor_leaf_is_a_fail():
    findings = NUMERICS.compare(*pair(3, 4), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    assert "non-tensor output differs" in findings[0].message
    assert findings[0].details == {"expected": "3", "got": "4"}


def test_a_tensor_against_a_non_tensor_is_a_fail_in_both_oracles():
    numeric = NUMERICS.compare(*pair(torch.ones(2), 3), CFG)
    meta = METADATA.compare(*pair(torch.ones(2), 3), CFG)

    assert [finding.severity for finding in numeric] == ["fail"]
    assert "values cannot be compared" in numeric[0].message
    assert [finding.severity for finding in meta] == ["fail"]
    assert meta[0].details == {"field": "type", "expected": "Tensor", "got": "int"}


def test_an_output_count_difference_is_a_fail_in_both_oracles():
    eager = lane("eager", [torch.ones(2), torch.ones(3)])
    other = lane("inductor", [torch.ones(2)])

    for oracle in (NUMERICS, METADATA):
        findings = oracle.compare(eager, other, CFG)
        counts = [f for f in findings if f.details.get("field") == "output_count"]
        assert len(counts) == 1
        assert counts[0].severity == "fail"
        assert counts[0].output_index is None
        assert counts[0].details["expected"] == 2
        assert counts[0].details["got"] == 1
        # The leaf both runs do have is still compared.
        assert len(findings) == 1


def test_an_output_structure_difference_is_a_fail():
    values = torch.ones(2)
    _, tuple_spec = torch.utils._pytree.tree_flatten((values, values))
    _, dict_spec = torch.utils._pytree.tree_flatten({"a": values, "b": values})
    eager = lane("eager", [values, values], output_spec=tuple_spec)
    other = lane("inductor", [values, values], output_spec=dict_spec)

    findings = NUMERICS.compare(eager, other, CFG)
    assert [finding.details["field"] for finding in findings] == ["output_spec"]
    assert findings[0].severity == "fail"
    assert findings[0].output_index is None


def test_a_lane_that_raised_is_not_compared():
    # PLAN.md "Stage localization" reads the exception; an oracle reporting it
    # again would say the same thing three times.
    raised = CapturedException(type="RuntimeError", message="boom", traceback=())
    eager = lane("eager", [torch.ones(2)])
    other = lane("inductor", [], exception=raised)

    assert NUMERICS.compare(eager, other, CFG) == []
    assert METADATA.compare(eager, other, CFG) == []
    assert NUMERICS.compare(other, eager, CFG) == []


def test_a_shape_difference_reaches_both_oracles():
    numeric = NUMERICS.compare(*pair(torch.ones(2, 3), torch.ones(3, 2)), CFG)
    meta = METADATA.compare(*pair(torch.ones(2, 3), torch.ones(3, 2)), CFG)

    assert [finding.severity for finding in numeric] == ["fail"]
    assert "shape" in numeric[0].message.lower()
    assert "shape" in fields(meta)


# --------------------------------------------------------------------------
# numerics: tolerance resolution
# --------------------------------------------------------------------------


def test_the_fallback_table_matches_the_torch_it_falls_back_from():
    # The table is only used when the private default_tolerances is missing, so
    # this is the test that notices the day the two drift apart.
    comparison = pytest.importorskip("torch.testing._comparison")
    for name, (rtol, atol) in FALLBACK_TOLERANCES.items():
        dtype = getattr(torch, name.removeprefix("torch."))
        assert comparison.default_tolerances(dtype) == (
            pytest.approx(rtol),
            pytest.approx(atol),
        ), name


def test_tolerances_are_the_loosest_of_the_dtypes_involved():
    # assert_close promotes, and check_dtype=False means both dtypes are in
    # play, so the float16 tolerance has to win over the float32 one.
    rtol, atol = resolve_tolerances(torch, (torch.float32, torch.float16), CFG)
    assert (rtol, atol) == (pytest.approx(1e-3), pytest.approx(1e-5))


def test_one_tolerance_override_leaves_the_other_at_its_default():
    rtol, atol = resolve_tolerances(torch, (torch.float32,), OracleConfig(atol=0.5))
    assert rtol == pytest.approx(1.3e-6)
    assert atol == pytest.approx(0.5)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def test_a_dtype_difference_is_a_metadata_fail():
    expected = torch.tensor([1, 2], dtype=torch.int8)
    findings = METADATA.compare(*pair(expected, expected.to(torch.int64)), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    finding = findings[0]
    assert finding.oracle == "metadata"
    assert finding.output_index == 0
    assert finding.details == {
        "field": "dtype",
        "expected": "torch.int8",
        "got": "torch.int64",
    }
    assert finding.message == "dtype differs: eager torch.int8, inductor torch.int64"


def test_a_shape_difference_is_a_metadata_fail():
    findings = METADATA.compare(*pair(torch.ones(2, 3), torch.ones(6)), CFG)

    shapes = [f for f in findings if f.details["field"] == "shape"]
    assert len(shapes) == 1
    assert shapes[0].severity == "fail"
    assert shapes[0].details["expected"] == [2, 3]
    assert shapes[0].details["got"] == [6]


def test_a_stride_difference_between_two_contiguous_tensors_is_a_warn():
    # A size-1 dimension has no meaningful stride, so two contiguous tensors of
    # the same shape can legitimately report different ones. Documented in
    # oracles/metadata.py.
    expected = torch.zeros(1, 4)
    got = torch.empty_strided((1, 4), (1, 1)).copy_(expected)

    assert expected.is_contiguous()
    assert got.is_contiguous()
    assert expected.stride() != got.stride()
    findings = METADATA.compare(*pair(expected, got), CFG)

    assert [finding.severity for finding in findings] == ["warn"]
    assert findings[0].details["field"] == "stride"
    assert "both tensors are contiguous" in findings[0].message
    assert NUMERICS.compare(*pair(expected, got), CFG) == []


def test_a_stride_difference_with_a_non_contiguous_side_is_a_fail():
    expected = torch.zeros(4, 4)
    got = torch.zeros(4, 4).t()

    findings = METADATA.compare(*pair(expected, got), CFG)
    assert fields(findings) == {"stride", "is_contiguous"}
    assert {finding.severity for finding in findings} == {"fail"}


def test_a_requires_grad_difference_is_a_fail():
    expected = torch.ones(3)
    findings = METADATA.compare(*pair(expected, expected.clone().requires_grad_(True)), CFG)

    assert [finding.details["field"] for finding in findings] == ["requires_grad"]
    assert findings[0].severity == "fail"
    assert findings[0].details == {"field": "requires_grad", "expected": False, "got": True}


def test_the_requires_grad_of_a_real_run_is_read_from_the_runners_record():
    # The field was vacuous on every real run before M2-2: what the oracle
    # compares are the runner's output clones, and a clone is detached, so both
    # sides always answered False. The finding below can only come from the
    # record, because neither clone in it says anything but False.
    target = load_target(str(FIXTURES / "mlp.py"))
    eager = run_all(target, ["eager"], seed=0).results["eager"]
    detached = lane("inductor", eager.outputs, output_requires_grad=[False])

    assert [tensor.requires_grad for tensor in eager.outputs] == [False]
    assert METADATA.compare(eager, eager, CFG) == []

    findings = METADATA.compare(eager, detached, CFG)
    assert [finding.details["field"] for finding in findings] == ["requires_grad"]
    assert findings[0].details == {"field": "requires_grad", "expected": True, "got": False}


def test_a_device_difference_is_a_fail():
    # The meta device stands in for a cuda/cpu divergence on a CPU-only box:
    # what is compared is the device type, not the index.
    findings = METADATA.compare(*pair(torch.zeros(2), torch.zeros(2, device="meta")), CFG)

    assert [finding.details["field"] for finding in findings] == ["device"]
    assert findings[0].severity == "fail"
    assert findings[0].details == {"field": "device", "expected": "cpu", "got": "meta"}


def test_a_layout_difference_is_a_fail():
    dense = torch.zeros(2, 2)
    findings = METADATA.compare(*pair(dense, dense.to_sparse()), CFG)

    layout = [f for f in findings if f.details["field"] == "layout"]
    assert len(layout) == 1
    assert layout[0].severity == "fail"
    assert layout[0].details["expected"] == "torch.strided"
    assert layout[0].details["got"] == "torch.sparse_coo"


def test_two_equal_non_tensor_leaves_are_not_the_metadata_oracles_business():
    assert METADATA.compare(*pair(3, 4), CFG) == []
    assert METADATA.compare(*pair(3, "three"), CFG)[0].details["field"] == "type"


# --------------------------------------------------------------------------
# the fp64 reference
# --------------------------------------------------------------------------


def fp64_lanes(eager_value: float, compiled_value: float, exact_value: float) -> Any:
    """Three one-element lanes: eager, compiled, and the fp64 reference."""
    eager = lane("eager", [torch.tensor([eager_value], dtype=torch.float32)])
    other = lane("inductor", [torch.tensor([compiled_value], dtype=torch.float32)])
    reference = lane(FP64_BACKEND, [torch.tensor([exact_value], dtype=torch.float64)])
    return eager, other, reference


def test_the_fp64_reference_says_nothing_when_eager_tracks_it():
    eager, other, reference = fp64_lanes(1.0, 1.0, 1.0)
    cfg = OracleConfig(fp64=True, fp64_reference=reference)

    assert NUMERICS.compare(eager, other, cfg) == []


def test_eager_drifting_from_fp64_is_an_info_finding():
    # Both lanes are 0.5 away from the true value: the compiled lane is not
    # wrong, both are imprecise, which is the case PLAN.md's blind spot section
    # says the two-way comparison cannot distinguish.
    eager, other, reference = fp64_lanes(1.5, 1.5, 1.0)
    cfg = OracleConfig(fp64=True, fp64_reference=reference)
    findings = NUMERICS.compare(eager, other, cfg)

    assert [finding.severity for finding in findings] == ["info"]
    finding = findings[0]
    assert finding.details["field"] == "fp64_reference"
    assert finding.details["verdict"] == "both imprecise"
    assert finding.details["eager_max_abs_diff"] == pytest.approx(0.5)
    assert finding.details["got_max_abs_diff"] == pytest.approx(0.5)
    assert "eager itself deviates from the fp64 reference" in finding.message


def test_a_compiled_lane_further_from_fp64_says_so():
    eager, other, reference = fp64_lanes(1.5, 2.5, 1.0)
    cfg = OracleConfig(fp64=True, fp64_reference=reference)
    findings = NUMERICS.compare(eager, other, cfg)

    info = [finding for finding in findings if finding.severity == "info"]
    assert info[0].details["verdict"] == "inductor is further from fp64 than eager is"
    # The ordinary value comparison still fails: the two worlds disagree.
    assert any(finding.severity == "fail" for finding in findings)


def test_the_fp64_reference_is_only_read_behind_the_flag():
    eager, other, reference = fp64_lanes(1.5, 1.5, 1.0)

    assert NUMERICS.compare(eager, other, OracleConfig(fp64_reference=reference)) == []
    # And a reference that raised is no reference at all.
    broken = lane(
        FP64_BACKEND,
        [],
        exception=CapturedException(type="RuntimeError", message="no double kernel", traceback=()),
    )
    assert NUMERICS.compare(eager, other, OracleConfig(fp64=True, fp64_reference=broken)) == []


# --------------------------------------------------------------------------
# alias: the relation between outputs and inputs
# --------------------------------------------------------------------------


def lanes(eager_fn: Any, compiled_fn: Any, inputs: tuple[Any, ...]) -> Any:
    """Two lanes, from two plain callables, without compiling anything.

    Both are run through the real :func:`run_backend`, so the records under test
    are the records a run really produces -- live output objects, live input
    objects, and the snapshots taken around the call -- rather than a hand-built
    approximation of them. The second is relabelled ``inductor`` because what an
    oracle compares is one lane against the reference, and running the compiler
    to obtain a divergence that can be written in one line would cost seconds
    and depend on what this torch happens to do.
    """
    expected = run_backend(eager_fn, inputs, "eager", grad=False)
    got = run_backend(compiled_fn, inputs, "eager", grad=False)
    got.backend = "inductor"
    return expected, got


def independent(x: torch.Tensor) -> torch.Tensor:
    """A result that shares nothing with the input, which is the usual case."""
    return x.clone()


def a_view_of_the_input(x: torch.Tensor) -> torch.Tensor:
    """A distinct object over the input's storage: an alias, not an identity."""
    return x[:]


def base_and_view(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 191449 shape in eager: two objects, one storage."""
    base = x + 1
    return base, base.view(-1)


def one_object_twice(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 191449 shape after the collapse: one object returned twice."""
    base = x + 1
    return base, base


def test_an_alias_the_compiled_lane_added_is_a_fail():
    # PLAN.md "alias", the 195451 class: the values are identical in both
    # worlds and the compiled result is a view of the input, so a caller that
    # writes into it corrupts the input.
    findings = ALIAS.compare(*lanes(independent, a_view_of_the_input, (torch.ones(4),)), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    finding = findings[0]
    assert finding.oracle == "alias"
    assert finding.backend == "inductor"
    assert finding.output_index == 0
    assert finding.details["field"] == "alias_added"
    assert "inductor output[0] aliases input[0]" in finding.message


def test_an_alias_the_compiled_lane_dropped_is_a_fail():
    # The other direction is a contract break too: a caller writing through the
    # view eager gave it expects the write to land.
    findings = ALIAS.compare(*lanes(a_view_of_the_input, independent, (torch.ones(4),)), CFG)

    assert [finding.details["field"] for finding in findings] == ["alias_dropped"]
    assert "eager output[0] aliases input[0]" in findings[0].message


def test_the_input_returned_as_the_output_is_a_fail():
    findings = ALIAS.compare(*lanes(independent, lambda x: x, (torch.ones(4),)), CFG)

    assert [finding.details["field"] for finding in findings] == ["identity_added"]
    assert findings[0].severity == "fail"
    assert "inductor returned input[0] itself as output[0]" in findings[0].message


def test_two_outputs_collapsed_into_one_object_is_a_fail():
    # PLAN.md "Where divergence appears is not always where the fix belongs":
    # this is 191449, whose fix lives in AOTAutograd and whose symptom is here.
    findings = ALIAS.compare(*lanes(base_and_view, one_object_twice, (torch.zeros(2),)), CFG)

    assert [finding.details["field"] for finding in findings] == ["identity_added"]
    assert findings[0].severity == "fail"
    assert "one object for output[0] and output[1]" in findings[0].message
    assert "distinct objects that share a storage" in findings[0].message


def test_one_object_split_into_two_is_a_fail():
    findings = ALIAS.compare(*lanes(one_object_twice, base_and_view, (torch.zeros(2),)), CFG)

    assert [finding.details["field"] for finding in findings] == ["identity_dropped"]
    assert "eager returned one object for output[0] and output[1]" in findings[0].message


def test_the_same_relation_in_both_lanes_is_silent():
    # The negative test, which is the one that keeps the oracle honest: an alias
    # is not a finding, a *difference* in aliasing is.
    assert (
        ALIAS.compare(*lanes(a_view_of_the_input, a_view_of_the_input, (torch.ones(4),)), CFG) == []
    )
    assert ALIAS.compare(*lanes(base_and_view, base_and_view, (torch.zeros(2),)), CFG) == []
    assert ALIAS.compare(*lanes(independent, independent, (torch.ones(4),)), CFG) == []


def test_a_mutation_only_the_compiled_lane_made_is_a_fail():
    def mutates(x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        x.zero_()
        return total

    findings = ALIAS.compare(*lanes(torch.Tensor.sum, mutates, (torch.ones(3),)), CFG)

    assert [finding.details["field"] for finding in findings] == ["mutation_added"]
    finding = findings[0]
    assert finding.severity == "fail"
    # A mutated input belongs to the run, not to an output index.
    assert finding.output_index is None
    assert finding.message == "inductor mutated input[0] in place and eager did not"


def test_a_mutation_the_compiled_lane_skipped_is_a_fail():
    def mutates(x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        x.zero_()
        return total

    findings = ALIAS.compare(*lanes(mutates, torch.Tensor.sum, (torch.ones(3),)), CFG)

    assert [finding.details["field"] for finding in findings] == ["mutation_dropped"]
    assert findings[0].message == "eager mutated input[0] in place and inductor did not"


def test_a_write_back_of_the_same_values_is_not_a_mutation():
    # PLAN.md "alias": the mutation set is over the bytes, so a call that writes
    # back what was already there is deliberately not counted.
    def writes_the_same_values(x: torch.Tensor) -> torch.Tensor:
        x.copy_(x.clone())
        return x.sum()

    assert (
        ALIAS.compare(*lanes(torch.Tensor.sum, writes_the_same_values, (torch.ones(3),)), CFG) == []
    )


def test_an_in_place_resize_is_a_metadata_mutation():
    def resizes(x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        x.resize_(2, 6)
        return total

    findings = ALIAS.compare(*lanes(torch.Tensor.sum, resizes, (torch.ones(3, 4),)), CFG)

    # Both, and in this order: the bytes the tensor covers are not the bytes it
    # covered, and the layout finding beside it says which of the two happened.
    assert [finding.details["field"] for finding in findings] == [
        "mutation_added",
        "metadata_mutation_added",
    ]
    layout = findings[1]
    assert layout.severity == "fail"
    assert "inductor changed the shape, stride of input[0] in place" in layout.message
    assert "float32(3, 4) stride (4, 1) offset 0 -> float32(2, 6) stride (6, 1) offset 0" in (
        layout.message
    )


def test_two_disjoint_views_of_one_buffer_are_not_an_alias():
    # PLAN.md "alias": storage identity alone is not sufficient, since two
    # disjoint views of one buffer share a data pointer. Recorded, never a
    # verdict.
    def two_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        buffer = x + 1
        return buffer[:2], buffer[2:]

    def two_tensors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        buffer = x + 1
        return buffer[:2].clone(), buffer[2:].clone()

    findings = ALIAS.compare(*lanes(two_views, two_tensors, (torch.ones(4),)), CFG)

    assert [finding.severity for finding in findings] == ["info"]
    assert findings[0].details["field"] == "buffer_sharing"
    assert "allocation choice rather than an alias" in findings[0].message
    # And the same relation on both sides says nothing at all.
    assert ALIAS.compare(*lanes(two_views, two_views, (torch.ones(4),)), CFG) == []


def test_an_output_that_overlaps_itself_is_recorded_as_context():
    # PLAN.md "alias": torch._debug_has_internal_overlap per tensor, as context,
    # since a self-overlapping output changes what a downstream write means.
    def expanded(x: torch.Tensor) -> torch.Tensor:
        return (x + 1).expand(3, 4)

    def materialised(x: torch.Tensor) -> torch.Tensor:
        return (x + 1).expand(3, 4).contiguous()

    findings = ALIAS.compare(*lanes(materialised, expanded, (torch.ones(1, 4),)), CFG)

    assert [finding.severity for finding in findings] == ["info"]
    assert findings[0].details == {
        "field": "internal_overlap",
        "expected": 0,
        "got": 1,
        "eager_relation": ["no aliases, no mutations"],
        "compiled_relation": ["output[0] self-overlapping"],
    }


def test_every_finding_carries_both_relations():
    findings = ALIAS.compare(*lanes(independent, a_view_of_the_input, (torch.ones(4),)), CFG)

    details = findings[0].details
    assert details["eager_relation"] == ["no aliases, no mutations"]
    assert details["compiled_relation"] == ["output[0]~input[0] overlapping"]
    assert details["expected"] == "output[0]~input[0] unrelated"
    assert details["got"] == "output[0]~input[0] overlapping"


def test_a_hand_built_pair_of_results_is_read_the_same_way():
    # The one pair assembled field by field rather than run: it pins that the
    # oracle reads output_refs and input_refs, and would notice a runner that
    # stopped filling them.
    shared = torch.ones(4)
    eager = BackendResult(
        backend="eager",
        outputs=[shared.clone()],
        output_refs=[shared.clone()],
        inputs_before=[shared.clone()],
        inputs_after=[shared.clone()],
        input_refs=[shared],
    )
    other = BackendResult(
        backend="inductor",
        outputs=[shared.clone()],
        output_refs=[shared],
        inputs_before=[shared.clone()],
        inputs_after=[shared.clone()],
        input_refs=[shared],
    )
    findings = ALIAS.compare(eager, other, CFG)

    assert [finding.details["field"] for finding in findings] == ["identity_added"]


def test_an_input_with_no_after_snapshot_is_unknown_rather_than_mutated():
    # A missing snapshot is not evidence of anything, and the one thing it must
    # not do is read as "mutated": comparing a tensor against a missing entry
    # would report every input of that lane as written to.
    eager, other = lanes(independent, independent, (torch.ones(4),))
    other.inputs_after = []
    findings = ALIAS.compare(eager, other, CFG)

    assert [finding.severity for finding in findings] == ["info"]
    assert findings[0].details["field"] == "mutation_unknown"
    assert findings[0].details["got"] == "input[0] mutation unknown"


def test_the_alias_oracle_says_nothing_about_a_lane_that_raised():
    eager, other = lanes(independent, independent, (torch.ones(4),))
    other.exception = CapturedException(type="RuntimeError", message="boom", traceback=())
    other.outputs, other.output_refs = [], []

    assert ALIAS.compare(eager, other, CFG) == []


def test_an_output_count_difference_is_reported_and_the_rest_still_compared():
    def one_output(x: torch.Tensor) -> torch.Tensor:
        return x[:]

    def two_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x.clone(), x.clone()

    findings = ALIAS.compare(*lanes(one_output, two_outputs, (torch.ones(4),)), CFG)

    fields = [finding.details["field"] for finding in findings]
    # One output against a pair of them differs in structure and in count, and
    # both of those are align_outputs' findings, worded once for every oracle.
    assert fields == ["output_spec", "output_count", "alias_dropped"]
    # The structural differences belong to no output; the alias one to output 0,
    # which both lanes have.
    assert [finding.output_index for finding in findings] == [None, None, 0]


def test_a_non_tensor_leaf_is_related_to_nothing():
    def with_a_label(x: torch.Tensor) -> tuple[torch.Tensor, str]:
        return x[:], "annotation"

    assert ALIAS.compare(*lanes(with_a_label, with_a_label, (torch.ones(4),)), CFG) == []


def test_a_relation_where_everything_is_related_is_compared_in_under_a_second():
    # The regression this pins (M2-2 housekeeping): every pair of 200 views of
    # one buffer is a link, so the relation holds 20 100 of them, and answering
    # 20 100 questions by scanning that tuple took 23.7 s on the development box
    # against 0.3 s once both relations are indexed. The bound is deliberately
    # loose -- what is being caught is a quadratic, not a slow machine.
    views = 200

    def many_views(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(x[index:] for index in range(views))

    eager, other = lanes(many_views, many_views, (torch.ones(views + 1),))
    assert len(relation(torch, eager).links) > views * views // 4

    started = time.perf_counter()
    findings = ALIAS.compare(eager, other, CFG)
    elapsed = time.perf_counter() - started

    assert findings == []
    assert elapsed < 1.0, f"the alias comparison took {elapsed:.2f}s for {views} views"


# --------------------------------------------------------------------------
# grad: the presence set, the values, and the backward that raised
# --------------------------------------------------------------------------


def grad_lane(backend: str, grads: dict[str, Any], **kwargs: Any) -> BackendResult:
    """A lane carrying one parameter gradient per name, and nothing else.

    Enough for the grad oracle: it reads the two gradient records and the
    backward's exception, and the forward's outputs only through ``ok``.
    """
    kwargs.setdefault("grad_ran", True)
    return BackendResult(backend=backend, param_grads=dict(grads), **kwargs)


def grad_pair(expected: dict[str, Any], got: dict[str, Any]) -> tuple[BackendResult, BackendResult]:
    """An eager lane and an inductor lane, each with its own gradients."""
    return grad_lane("eager", expected), grad_lane("inductor", got)


def test_the_same_gradients_in_both_lanes_are_silent():
    grads = {"w": torch.ones(2, 3), "b": torch.zeros(3)}
    assert GRAD.compare(*grad_pair(grads, {k: v.clone() for k, v in grads.items()}), CFG) == []


def test_a_gradient_only_eager_produced_is_a_fail_naming_the_parameter():
    expected = {"w": torch.ones(3), "b": torch.ones(2)}
    findings = GRAD.compare(*grad_pair(expected, {"w": torch.ones(3)}), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    finding = findings[0]
    assert finding.oracle == "grad"
    assert finding.backend == "inductor"
    # A gradient belongs to a parameter, not to an output leaf.
    assert finding.output_index is None
    assert finding.details["field"] == "grad_missing"
    assert finding.details["tensor"] == "parameter b"
    assert "eager produced a gradient for parameter b and inductor did not" in finding.message


def test_a_gradient_only_the_compiled_lane_produced_is_a_fail():
    got = {"w": torch.ones(3), "b": torch.ones(2)}
    findings = GRAD.compare(*grad_pair({"w": torch.ones(3)}, got), CFG)

    assert [finding.details["field"] for finding in findings] == ["grad_extra"]
    assert "inductor produced a gradient for parameter b and eager did not" in findings[0].message


def test_a_perturbed_gradient_is_a_fail_naming_the_parameter():
    expected = {"w": torch.ones(4), "b": torch.ones(2)}
    got = {"w": torch.ones(4), "b": torch.ones(2) + 0.5}
    findings = GRAD.compare(*grad_pair(expected, got), CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    finding = findings[0]
    assert finding.details["field"] == "grad_values"
    assert finding.details["tensor"] == "parameter b"
    assert finding.message.startswith("the gradient of parameter b differs: ")
    # The message is assert_close's, because the rule is the numerics rule.
    assert "Mismatched elements: 2 / 2" in finding.message
    # The numerics float32 rtol of 1.3e-6, times the grad factor, and the factor
    # itself alongside it: the tolerance a reader sees has to be the tolerance
    # the decision was made with, and at 10x it is not the one the numerics rows
    # of the same report were compared under.
    assert finding.details["rtol"] == pytest.approx(1.3e-6 * DEFAULT_GRAD_TOL_FACTOR)
    assert finding.details["tol_factor"] == DEFAULT_GRAD_TOL_FACTOR


def test_gradients_go_through_the_numerics_tolerances():
    # PLAN.md "grad": the values go through the numerics comparison, which means
    # the per-dtype tolerance and the --rtol/--atol overrides both apply here.
    expected = {"w": torch.ones(8)}
    within = {"w": torch.ones(8) + 5e-7}
    assert GRAD.compare(*grad_pair(expected, within), CFG) == []

    beyond = {"w": torch.ones(8) + 0.5}
    assert len(GRAD.compare(*grad_pair(expected, beyond), CFG)) == 1
    assert GRAD.compare(*grad_pair(expected, beyond), OracleConfig(atol=1.0)) == []


def test_the_grad_tolerance_factor_is_what_decides_a_borderline_gradient():
    # M2-3 housekeeping (b), as the measurement that produced it: the M2-2
    # verifier saw a compiled resnet18 gradient about 1.24e-5 from eager's
    # against a float32 atol of 1e-5, so the same run flipped between clean and
    # failing. 1.24e-5 is the number, not a round one, and the point of the test
    # is that the factor is the only thing that decides it.
    expected = {"w": torch.ones(8)}
    borderline = {"w": torch.ones(8) + 1.24e-5}

    exact = GRAD.compare(*grad_pair(expected, borderline), OracleConfig(grad_tol_factor=1.0))
    assert [finding.details["field"] for finding in exact] == ["grad_values"]
    assert "tol_factor" not in exact[0].details, "a factor of 1 is not worth a detail line"
    assert exact[0].details["atol"] == pytest.approx(1e-5)

    assert GRAD.compare(*grad_pair(expected, borderline), CFG) == []


def test_the_grad_tolerance_factor_does_not_reach_the_outputs():
    # The half that keeps the policy honest: the looser rule exists because of
    # how a backward accumulates, so it must not widen the comparison that
    # catches 190765. Same distance, same config, two different verdicts.
    borderline = torch.ones(8) + 1.24e-5
    output_findings = NUMERICS.compare(*pair(torch.ones(8), borderline), CFG)

    assert [finding.severity for finding in output_findings] == ["fail"]
    assert output_findings[0].details["atol"] == pytest.approx(1e-5)
    assert GRAD.compare(*grad_pair({"w": torch.ones(8)}, {"w": borderline}), CFG) == []


def test_the_grad_tolerance_factor_multiplies_the_rtol_and_atol_overrides():
    # Applied last, after --rtol and --atol, so the policy is one rule rather
    # than two that interact: --atol 1e-6 compares outputs at 1e-6 and gradients
    # at ten times that.
    cfg = OracleConfig(atol=1e-6, rtol=0.0, grad_tol_factor=10.0)
    expected = {"w": torch.zeros(4)}

    assert GRAD.compare(*grad_pair(expected, {"w": torch.full((4,), 9e-6)}), cfg) == []
    findings = GRAD.compare(*grad_pair(expected, {"w": torch.full((4,), 2e-5)}), cfg)
    assert [finding.details["field"] for finding in findings] == ["grad_values"]
    assert findings[0].details["atol"] == pytest.approx(1e-5)
    assert findings[0].details["rtol"] == pytest.approx(0.0)


def test_a_backward_that_raised_in_one_lane_only_is_a_fail():
    boom = CapturedException(type="RuntimeError", message="no backward here", traceback=("a", "b"))
    eager = grad_lane("eager", {"w": torch.ones(3)})
    other = grad_lane("inductor", {}, grad_ran=False, grad_error=boom)

    findings = GRAD.compare(eager, other, CFG)

    assert [finding.details["field"] for finding in findings] == ["grad_error_added"]
    assert findings[0].severity == "fail"
    assert "raised RuntimeError under inductor and completed under eager" in findings[0].message
    assert findings[0].details["expected"] == "completed"
    assert findings[0].details["got"] == "RuntimeError"
    # And nothing else: a lane whose backward did not finish has no gradients
    # for a reason already stated, and listing every missing one would bury it.
    assert len(findings) == 1


def test_a_backward_that_raised_in_eager_only_is_a_fail_the_other_way():
    boom = CapturedException(type="ValueError", message="not differentiable", traceback=())
    eager = grad_lane("eager", {}, grad_ran=False, grad_error=boom)
    other = grad_lane("inductor", {"w": torch.ones(3)})

    findings = GRAD.compare(eager, other, CFG)

    assert [finding.details["field"] for finding in findings] == ["grad_error_dropped"]
    assert "raised ValueError under eager and completed under inductor" in findings[0].message


def test_both_backwards_raising_is_not_a_divergence():
    boom = CapturedException(type="RuntimeError", message="no backward here", traceback=())
    eager = grad_lane("eager", {}, grad_ran=False, grad_error=boom)
    other = grad_lane("inductor", {}, grad_ran=False, grad_error=boom)

    assert GRAD.compare(eager, other, CFG) == []


def test_no_grad_reports_an_info_line_rather_than_a_clean_row():
    grads = {"w": torch.ones(3)}
    findings = GRAD.compare(*grad_pair(grads, grads), OracleConfig(grad=False))

    assert [finding.severity for finding in findings] == ["info"]
    assert findings[0].details["field"] == "grad_disabled"
    assert "--no-grad switched the backward pass off" in findings[0].message


def test_the_grad_oracle_says_nothing_about_a_lane_that_raised():
    eager = grad_lane("eager", {"w": torch.ones(3)})
    other = grad_lane("inductor", {})
    other.exception = CapturedException(type="RuntimeError", message="boom", traceback=())

    assert GRAD.compare(eager, other, CFG) == []


def test_the_grad_oracle_leaves_the_output_requires_grad_flag_to_metadata():
    # PLAN.md "grad" and "metadata" divide this deliberately: reporting one
    # divergence from two oracles hides which one is the real defect.
    values = torch.ones(3)
    eager = lane("eager", [values], output_requires_grad=[True])
    other = lane("inductor", [values.clone()], output_requires_grad=[False])

    assert GRAD.compare(eager, other, CFG) == []
    assert [finding.details["field"] for finding in METADATA.compare(eager, other, CFG)] == [
        "requires_grad"
    ]


# --------------------------------------------------------------------------
# graph: breaks, baselines, recompiles, and the repeat call
#
# Hand-built GraphHealth records rather than compiled runs, for the reason the
# module docstring gives: the rules are what has to be pinned down, and which
# reasons a particular torch reports for a particular fixture is not a rule.
# The two integration tests further down run tests/fixtures/graph_break.py for
# real and check that these rules still describe one.
# --------------------------------------------------------------------------

# The shape of one entry of ExplainOutput.break_reasons, as torch 2.14 fills it
# in: a multi-line explanation ending in a link whose gbNNNN id names the break
# class, and a user stack whose last frame is the line that broke.
PRINT_BREAK = """\
Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print`
  Hint: Avoid calling builtin `print`.

 For more details about this graph break, please visit: \
https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0059.html"""

BRANCH_BREAK = "Data-dependent branching\n  Explanation: ... gb/gb0170.html"

PRINT_SUMMARY = "gb0059: Failed to trace builtin operator"
BRANCH_SUMMARY = "gb0170: Data-dependent branching"


def graph_lane(
    *reasons: str,
    backend: str = "inductor",
    break_count: int | None = None,
    **kwargs: Any,
) -> BackendResult:
    """An inductor lane whose only content is its graph health."""
    breaks = tuple(
        GraphBreak(reason=reason, user_frame=f"m.py:{index + 1} in forward")
        for index, reason in enumerate(reasons)
    )
    return BackendResult(
        backend=backend,
        first_call_s=1.5,
        second_call_s=0.001,
        graph_health=GraphHealth(
            graph_count=len(breaks) + 1,
            break_count=len(breaks) if break_count is None else break_count,
            breaks=breaks,
            op_count=7,
            unique_graphs_before=3,
            unique_graphs_after=3,
            **kwargs,
        ),
    )


def baseline(*reasons: str, backend: str = "inductor", count: int | None = None) -> Baseline:
    """A one-backend baseline holding exactly these reason summaries."""
    return Baseline(
        path="b.json",
        entries={
            backend: BaselineEntry(
                graph_break_count=len(reasons) if count is None else count,
                break_reasons=tuple(reasons),
            )
        },
    )


def test_a_lane_that_captured_one_graph_says_nothing():
    assert GRAPH.compare(lane("eager", []), graph_lane(), CFG) == []


def test_a_lane_with_no_graph_health_is_not_a_lane_with_no_graph_breaks():
    # A hand-built record, or an uncompiled lane. Silence rather than a clean
    # report: the checks table renders it as a dash, not as "pass".
    assert GRAPH.compare(lane("eager", []), lane("inductor", []), CFG) == []


def test_a_graph_break_is_an_info_finding_carrying_the_reason():
    findings = GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, BRANCH_BREAK), CFG)

    assert [f.severity for f in findings] == ["info", "info"]
    assert [f.details["reason"] for f in findings] == [PRINT_SUMMARY, BRANCH_SUMMARY]
    assert "Failed to trace builtin operator" in findings[0].message
    assert "m.py:1 in forward" in findings[0].message
    assert findings[0].details["break_count"] == 2
    assert findings[0].details["compile_wall_s"] == 1.5
    # PLAN.md "Oracles": graph health is informational, so nothing here is a
    # divergence and the run stays clean without --fail-on graph.
    assert findings[0].oracle == "graph"
    assert findings[0].output_index is None


def test_fullgraph_turns_every_break_into_a_fail():
    findings = GRAPH.compare(
        lane("eager", []), graph_lane(PRINT_BREAK), OracleConfig(fullgraph=True)
    )

    assert [f.severity for f in findings] == ["fail"]
    assert "--fullgraph was requested and the graph broke anyway" in findings[0].message


def test_fullgraph_on_a_lane_that_captured_one_graph_is_still_silent():
    assert GRAPH.compare(lane("eager", []), graph_lane(), OracleConfig(fullgraph=True)) == []


def test_a_baseline_that_lists_every_break_produces_nothing():
    cfg = OracleConfig(baseline=baseline(PRINT_SUMMARY, BRANCH_SUMMARY))
    assert GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, BRANCH_BREAK), cfg) == []


def test_a_baseline_with_fewer_breaks_fails_naming_the_new_reason():
    cfg = OracleConfig(baseline=baseline(PRINT_SUMMARY))
    findings = GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, BRANCH_BREAK), cfg)

    assert [f.severity for f in findings] == ["fail"]
    assert findings[0].details["reason"] == BRANCH_SUMMARY
    assert "this break is not in b.json" in findings[0].message
    # And only the new one: PLAN.md "GitHub Action" fails on new breaks only,
    # because a check that reported the accepted ones every run gets turned off.
    assert PRINT_SUMMARY not in findings[0].message


def test_the_same_reason_breaking_more_often_than_the_baseline_is_a_fail():
    # No unfamiliar reason, but two of it where the file accepts one: an
    # additional break is a new break, whatever it is called.
    cfg = OracleConfig(baseline=baseline(PRINT_SUMMARY, count=1))
    findings = GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, PRINT_BREAK), cfg)

    assert [f.severity for f in findings] == ["fail"]
    assert findings[0].details == {
        "field": "graph_break_count",
        "expected": 1,
        "got": 2,
    }
    assert "with no reason the baseline does not already list" in findings[0].message


def test_a_break_with_no_recorded_reason_is_left_to_the_count_rule():
    # A break torch counted without saying why has no identity to be new
    # against, so it neither fails a matching baseline as an unknown reason...
    cfg = OracleConfig(baseline=baseline(PRINT_SUMMARY, count=3))
    assert GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, break_count=3), cfg) == []
    # ...nor gets written into one as a line nobody can act on.
    entry = baseline_entry(graph_lane(PRINT_BREAK, break_count=3))
    assert entry == BaselineEntry(graph_break_count=3, break_reasons=(PRINT_SUMMARY,))


def test_a_baseline_looser_than_the_run_is_an_info_not_a_fail():
    cfg = OracleConfig(baseline=baseline(PRINT_SUMMARY, BRANCH_SUMMARY, count=5))
    findings = GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK), cfg)

    assert [f.severity for f in findings] == ["info"]
    assert "the baseline is looser than this run" in findings[0].message


def test_a_baseline_with_no_entry_for_this_lane_is_a_warn_and_not_a_fail():
    cfg = OracleConfig(baseline=baseline(PRINT_SUMMARY, backend="aot_eager"))
    findings = GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK), cfg)

    assert [f.severity for f in findings] == ["warn", "info"]
    assert "b.json has no baseline for inductor" in findings[0].message


def test_a_break_count_higher_than_the_recorded_reasons_is_still_reported():
    findings = GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, break_count=3), CFG)

    assert [f.details["reason"] for f in findings] == [
        PRINT_SUMMARY,
        "no reason recorded",
        "no reason recorded",
    ]


def test_a_second_call_that_raised_is_a_graph_fail():
    other = graph_lane()
    other.second_call_exception = CapturedException(
        type="RuntimeError", message="guard failed\nsecond line", traceback=()
    )
    findings = GRAPH.compare(lane("eager", []), other, CFG)

    assert [f.severity for f in findings] == ["fail"]
    assert findings[0].message == (
        "inductor answered the first call and raised RuntimeError on the repeat "
        "call with the same inputs: guard failed"
    )
    assert findings[0].details["field"] == "second_call"


def test_a_recompile_on_the_repeat_call_is_a_warn():
    other = graph_lane()
    assert other.graph_health is not None
    other.graph_health = replace(other.graph_health, unique_graphs_after=5)
    findings = GRAPH.compare(lane("eager", []), other, CFG)

    assert [f.severity for f in findings] == ["warn"]
    assert "compiled 2 more graphs on the repeat call" in findings[0].message
    assert findings[0].details["expected"] == 3
    assert findings[0].details["got"] == 5


def test_an_unreadable_counter_is_not_reported_as_a_recompile():
    other = graph_lane()
    assert other.graph_health is not None
    other.graph_health = replace(other.graph_health, unique_graphs_after=None)

    assert other.graph_health.recompiled is False
    assert GRAPH.compare(lane("eager", []), other, CFG) == []


def test_an_explain_pass_that_raised_is_a_warn_rather_than_a_pass():
    other = BackendResult(
        backend="inductor",
        graph_health=GraphHealth(
            explain_error=CapturedException(
                type="TypeError", message="cannot trace this", traceback=()
            )
        ),
    )
    findings = GRAPH.compare(lane("eager", []), other, CFG)

    assert [f.severity for f in findings] == ["warn"]
    assert "graph health was not measured for inductor" in findings[0].message


def test_a_lane_that_raised_and_could_not_be_traced_is_not_reported_twice():
    # The target raises, so of course it raised under explain too. That
    # exception is already reported against the lane and is what the stage
    # verdict is built from; saying it again here would turn one broken model
    # into two findings.
    other = BackendResult(
        backend="inductor",
        exception=CapturedException(type="RuntimeError", message="broken", traceback=()),
        graph_health=GraphHealth(
            explain_error=CapturedException(type="RuntimeError", message="broken", traceback=())
        ),
    )
    assert GRAPH.compare(lane("eager", []), other, CFG) == []


def test_a_negative_break_count_is_floored_before_it_reaches_a_baseline():
    # torch computes graph_break_count as graph_count - 1, so a callable it
    # captured nothing for reports -1. GraphHealth floors it in the runner; this
    # is the rule stated where a baseline would otherwise read it as progress.
    empty = BackendResult(backend="inductor", graph_health=GraphHealth(break_count=0))
    entry = baseline_entry(empty)

    assert entry == BaselineEntry(graph_break_count=0, break_reasons=())


# --- the reason summary, which is what a baseline compares on ---------------


def test_a_reason_summary_is_the_headline_and_the_stable_break_id():
    assert summarise_reason(PRINT_BREAK) == PRINT_SUMMARY
    assert summarise_reason(BRANCH_BREAK) == BRANCH_SUMMARY


def test_a_reason_with_no_break_id_keeps_its_headline():
    assert summarise_reason("generic_jump TensorVariable()") == "generic_jump TensorVariable()"


def test_a_reason_summary_is_capped_and_never_empty():
    assert summarise_reason("") == "unknown graph break"
    assert summarise_reason("   \n\n") == "unknown graph break"
    assert len(summarise_reason("word " * 200)) == MAX_REASON_CHARS
    assert summarise_reason("word " * 200).endswith("…")


# --- the baseline file ------------------------------------------------------


def test_a_baseline_written_from_a_run_reads_back_identically(tmp_path):
    runset = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        results={
            "eager": lane("eager", []),
            "inductor": graph_lane(PRINT_BREAK, BRANCH_BREAK),
        },
    )
    path = tmp_path / "nested" / "baseline.json"
    assert write_baseline(path, runset) == ["inductor"]

    parsed = read_baseline(path)
    assert set(parsed.entries) == {"inductor"}
    assert parsed.entries["inductor"] == BaselineEntry(
        graph_break_count=2, break_reasons=(PRINT_SUMMARY, BRANCH_SUMMARY)
    )
    # Round trip: the file it just wrote silences exactly the breaks it recorded.
    cfg = OracleConfig(baseline=parsed)
    assert GRAPH.compare(lane("eager", []), graph_lane(PRINT_BREAK, BRANCH_BREAK), cfg) == []


def test_a_lane_whose_graph_health_was_not_measured_is_left_out_of_a_baseline(tmp_path):
    # A zero-break entry for a lane that never established one would be a
    # fiction every later run compared against.
    unmeasured = BackendResult(
        backend="inductor",
        graph_health=GraphHealth(
            explain_error=CapturedException(type="TypeError", message="no", traceback=())
        ),
    )
    runset = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        results={"eager": lane("eager", []), "inductor": unmeasured},
    )
    path = tmp_path / "baseline.json"

    assert write_baseline(path, runset) == []
    assert json.loads(path.read_text()) == {}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("not json at all", "is not valid JSON"),
        ("[1, 2]", "must be an object keyed by backend name"),
        ('{"inductor": 3}', "expected an object with graph_break_count"),
        ('{"inductor": {"graph_break_count": -1}}', "expected a non-negative integer"),
        ('{"inductor": {"graph_break_count": true}}', "expected a non-negative integer"),
        ('{"inductor": {"break_reasons": [1]}}', "not a list of strings"),
    ],
)
def test_a_baseline_that_is_not_this_shape_is_refused(tmp_path, content, expected):
    path = tmp_path / "baseline.json"
    path.write_text(content)

    with pytest.raises(BaselineError) as excinfo:
        read_baseline(path)
    assert expected in str(excinfo.value)


def test_a_missing_baseline_names_the_flag_that_writes_one(tmp_path):
    with pytest.raises(BaselineError) as excinfo:
        read_baseline(tmp_path / "absent.json")
    assert "write one first with --write-baseline" in str(excinfo.value)


def test_a_baseline_entry_ignores_keys_it_does_not_know(tmp_path):
    # A file written by a later version still reads here; a wrong shape does not.
    path = tmp_path / "baseline.json"
    path.write_text('{"inductor": {"graph_break_count": 1, "break_reasons": ["x"], "torch": "2"}}')

    assert read_baseline(path).entries["inductor"].break_reasons == ("x",)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def test_the_registry_is_a_subset_of_the_fail_on_vocabulary():
    assert set(ORACLES) <= set(ORACLE_NAMES)
    # All five since M3-1, in the order PLAN.md "Oracles" lists them.
    assert list(ORACLES) == list(ORACLE_NAMES)
    for name, oracle in ORACLES.items():
        assert isinstance(oracle, Oracle)
        assert oracle.name == name


def test_run_oracles_selects_by_name():
    expected = torch.tensor([1, 2], dtype=torch.int8)
    eager, other = pair(expected, expected.to(torch.int64) + 1)

    everything = run_oracles(eager, other, CFG)
    assert {finding.oracle for finding in everything} == {"numerics", "metadata"}
    assert {f.oracle for f in run_oracles(eager, other, CFG, ["metadata"])} == {"metadata"}
    # An oracle with nothing to say contributes nothing, rather than raising:
    # cli.parse_fail_on is where an unknown name is reported. These two lanes
    # carry no gradients and no graph health, so both are silent.
    assert run_oracles(eager, other, CFG, ["grad", "graph"]) == []


# --------------------------------------------------------------------------
# integration: real runs
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_runset():
    """One eager-plus-inductor run of the MLP, shared: inductor costs seconds."""
    target = load_target(str(FIXTURES / "mlp.py"))
    return run_all(target, ["eager", "inductor"], seed=0, fp64=True)


def test_a_clean_model_produces_no_fail_findings(mlp_runset):
    findings = run_oracles(
        mlp_runset.results["eager"],
        mlp_runset.results["inductor"],
        OracleConfig(fp64=True, fp64_reference=mlp_runset.fp64),
    )
    assert [finding for finding in findings if finding.severity == "fail"] == []


def test_the_graph_oracle_is_silent_on_a_model_that_captures_in_one_graph(mlp_runset):
    inductor = mlp_runset.results["inductor"]
    assert GRAPH.compare(mlp_runset.results["eager"], inductor, CFG) == []
    # Not vacuous: the explain pass really ran and really found one graph.
    assert inductor.graph_health is not None
    assert inductor.graph_health.measured is True
    assert (inductor.graph_health.graph_count, inductor.graph_health.break_count) == (1, 0)
    assert inductor.graph_health.op_count > 0
    # And the eager lane has no graph health at all: it was never compiled.
    assert mlp_runset.results["eager"].graph_health is None


@pytest.fixture(scope="module")
def graph_break_runset():
    """The deliberate-break fixture, run under inductor once."""
    target = load_target(str(FIXTURES / "graph_break.py"))
    return run_all(target, ["eager", "inductor"], seed=0)


def test_a_real_graph_break_reaches_the_oracle_with_its_reason_and_its_line(graph_break_runset):
    eager, inductor = graph_break_runset.results["eager"], graph_break_runset.results["inductor"]
    assert inductor.graph_health is not None
    findings = GRAPH.compare(eager, inductor, CFG)

    assert {f.severity for f in findings} == {"info"}
    reasons = " ".join(f.details["reason"] for f in findings)
    assert "Failed to trace builtin operator" in reasons
    assert all("graph_break.py" in str(f.details["user_frame"]) for f in findings)
    # PLAN.md "graph": a break is not a bug. The answers still match, so no
    # other oracle has anything to say about this lane.
    assert [f for f in run_oracles(eager, inductor, CFG) if f.severity == "fail"] == []


def test_a_real_break_under_fullgraph_is_a_fail_on_a_lane_that_raised():
    # The shape of cases/distributions_validation_branch.py, at fixture size:
    # fullgraph=True makes the break a hard error, so the lane produces nothing
    # for the other four oracles and the graph oracle is the only one that can
    # say why. That is why it does not stop at a lane that raised.
    target = load_target(str(FIXTURES / "graph_break.py"))
    runset = run_all(target, ["eager", "inductor"], seed=0, fullgraph=True)
    eager, inductor = runset.results["eager"], runset.results["inductor"]
    assert not inductor.ok

    findings = GRAPH.compare(eager, inductor, OracleConfig(fullgraph=True))
    assert {f.severity for f in findings} == {"fail"}
    assert "--fullgraph was requested and the graph broke anyway" in findings[0].message
    # The correctness oracles have nothing to compare, and say nothing.
    assert [f for f in run_oracles(eager, inductor, CFG) if f.oracle != "graph"] == []


def test_a_baseline_written_from_a_real_run_silences_that_run(tmp_path, graph_break_runset):
    path = tmp_path / "baseline.json"
    assert write_baseline(path, graph_break_runset) == ["inductor"]
    entry = read_baseline(path).entries["inductor"]
    assert entry.graph_break_count == 2

    cfg = OracleConfig(baseline=read_baseline(path))
    eager, inductor = graph_break_runset.results["eager"], graph_break_runset.results["inductor"]
    assert GRAPH.compare(eager, inductor, cfg) == []


def test_the_grad_oracle_is_silent_on_a_clean_model(mlp_runset):
    eager, inductor = mlp_runset.results["eager"], mlp_runset.results["inductor"]
    assert GRAD.compare(eager, inductor, CFG) == []
    # And the silence is not vacuous: four parameters really were differentiated
    # in both lanes, and the oracle really compared their gradients.
    assert eager.grad_ran is True
    assert inductor.grad_ran is True
    assert len(eager.grad_present) == 4
    assert eager.grad_present == inductor.grad_present


def test_a_perturbed_parameter_gradient_from_a_real_run_names_the_parameter(mlp_runset):
    # The brief's synthetic half of the grad oracle's positive coverage: the
    # gradients come from a real compiled run, one of them is moved, and the
    # oracle must point at that parameter by name and at nothing else.
    eager = mlp_runset.results["eager"]
    moved = dict(eager.param_grads)
    moved["net.0.weight"] = moved["net.0.weight"] + 0.5
    compiled = lane(
        "inductor",
        eager.outputs,
        input_grads=list(eager.input_grads),
        param_grads=moved,
        grad_ran=True,
        output_requires_grad=list(eager.output_requires_grad),
    )

    findings = GRAD.compare(eager, compiled, CFG)

    assert [finding.details["field"] for finding in findings] == ["grad_values"]
    assert findings[0].severity == "fail"
    assert findings[0].details["tensor"] == "parameter net.0.weight"
    assert "the gradient of parameter net.0.weight differs" in findings[0].message
    # The other three parameters agree, and the oracle says nothing about them.
    assert "net.2.weight" not in findings[0].message


@pytest.fixture(scope="module")
def frozen_param_runset():
    """A model with a frozen layer, through every lane; inductor costs seconds."""
    target = load_target(str(FIXTURES / "frozen_param.py"))
    return run_all(target, ["eager", "aot_eager", "inductor"], seed=0)


def test_a_frozen_parameter_is_absent_from_both_presence_sets(frozen_param_runset):
    # The presence set is the set of tensors that got a gradient, not the set of
    # parameters: a frozen layer is in every lane's parameter list and in no
    # lane's gradients, and an oracle that confused the two would report this
    # model as a divergence in every lane.
    eager = frozen_param_runset.results["eager"]
    assert eager.ok, eager.exception
    assert eager.grad_present == ("parameter trainable.weight", "parameter trainable.bias")

    for other in frozen_param_runset.others:
        assert other.ok, other.exception
        assert other.grad_present == eager.grad_present
        assert GRAD.compare(eager, other, CFG) == []
    # And the frozen layer really is still declared as a parameter.
    frozen = load_target(str(FIXTURES / "frozen_param.py")).fn.frozen
    assert [parameter.requires_grad for parameter in frozen.parameters()] == [False, False]


def test_a_backward_that_only_the_compiled_lane_cannot_run_is_a_fail():
    # The live half of the grad_error rule: a backend whose forward is the
    # traced graph unchanged and whose backward raises, so the divergence is in
    # the backward pass and nowhere else.
    fixture = FIXTURES / "backward_raises.py"
    backend = import_target_module(str(fixture)).BACKEND
    runset = run_all(load_target(str(fixture)), ["eager", backend], seed=0)

    eager, other = runset.results["eager"], runset.results[backend]
    assert eager.ok, eager.exception
    assert other.ok, other.exception  # the forward answered in both lanes
    assert eager.grad_ran is True
    assert other.grad_ran is False

    findings = GRAD.compare(eager, other, CFG)

    assert [finding.details["field"] for finding in findings] == ["grad_error_added"]
    assert findings[0].severity == "fail"
    assert f"raised RuntimeError under {backend} and completed under eager" in findings[0].message
    # The forward is bit-identical, so no other oracle has anything to say: the
    # divergence really is backward-only.
    assert NUMERICS.compare(eager, other, CFG) == []


def test_the_alias_oracle_is_silent_on_a_clean_model(mlp_runset):
    # The negative test that matters most for this oracle: a linear-relu-linear
    # stack allocates plenty of buffers, and none of that is an alias
    # divergence. A checker that fires here would be worse than no checker.
    eager, inductor = mlp_runset.results["eager"], mlp_runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception
    assert ALIAS.compare(eager, inductor, CFG) == []
    # And the relation it compared is not empty of information by accident: the
    # run really did have outputs and inputs to relate.
    assert relation(torch, inductor).outputs == 1
    assert relation(torch, inductor).inputs == 1


def test_the_fp64_reference_runs_beside_the_backends(mlp_runset):
    reference = mlp_runset.fp64
    assert reference is not None
    assert reference.backend == FP64_BACKEND
    assert reference.ok, reference.exception
    assert [tensor.dtype for tensor in reference.outputs] == [torch.float64]
    # It is a reference, not a lane: nothing that walks the backends sees it.
    assert mlp_runset.backends == ["eager", "inductor"]
    assert [result.backend for result in mlp_runset.others] == ["inductor"]
    # The float32 weights the backends ran with are untouched by the widening.
    assert mlp_runset.results["eager"].outputs[0].dtype == torch.float32


@pytest.fixture(scope="module")
def aliasing_runset():
    """A model that really aliases and really mutates, through every lane."""
    target = load_target(str(FIXTURES / "aliasing.py"))
    return run_all(target, ["eager", "aot_eager", "inductor"], seed=0)


def test_a_model_that_legitimately_aliases_and_mutates_is_not_a_finding(aliasing_runset):
    eager = aliasing_runset.results["eager"]
    assert eager.ok, eager.exception
    for lane in aliasing_runset.others:
        assert lane.ok, lane.exception
        assert ALIAS.compare(eager, lane, CFG) == []

    # And the silence is not vacuous: the relation every lane agreed on has two
    # views of the input in it, an unrelated output, and a mutated input.
    described = relation(torch, eager).describe()
    assert "output[0]~input[0] overlapping" in described
    assert "output[2]~input[0] overlapping" in described
    assert "output[1]~input[0] overlapping" not in described
    assert "input[0] mutated (values)" in described
    assert "input[1] mutated (values)" not in described


def test_a_stateful_module_is_a_numerics_finding_only_when_the_lanes_share_it():
    # Why the runner deep copies the module per lane (M2-2 housekeeping): the
    # counter this fixture multiplies by is module state, and one object shared
    # across the lanes carries it from the first into the second. The oracle is
    # right both times -- the outputs really do differ -- which is exactly why
    # the divergence must not be the harness's to cause.
    target = load_target(str(FIXTURES / "counter_buffer.py"))
    shared = run_all(target, ["eager", "aot_eager"], seed=0, share_module=True)
    findings = run_oracles(shared.results["eager"], shared.results["aot_eager"], CFG)
    fails = [finding for finding in findings if finding.severity == "fail"]
    assert [finding.oracle for finding in fails] == ["numerics"]

    isolated = run_all(target, ["eager", "aot_eager"], seed=0)
    assert run_oracles(isolated.results["eager"], isolated.results["aot_eager"], CFG) == []


# --------------------------------------------------------------------------
# the regression corpus, as shapes rather than as runs
#
# Whether each corpus case still reproduces on the installed torch, and whether
# compile-check reports it when it does, is tests/test_corpus_oracles.py: one
# parametrized run of every case through the runner and every oracle, graded
# against the case's own check(). What is left here is the half of each case
# that does not depend on the torch build -- the shape of the bug, handed to an
# oracle as a synthetic result. It pins the rule down on a torch where the bug
# is long fixed, which is the point of writing it separately, and it costs one
# eager run instead of three lanes.
# --------------------------------------------------------------------------


def test_the_191308_shape_is_flagged_from_a_synthetic_result():
    # The version-independent half of 191308: whatever this torch does,
    # a lane that came back int64 where eager returned int8 is a metadata fail.
    target = load_target(str(CASES / "dtype_promotion.py"))
    eager = run_all(target, ["eager"], seed=0).results["eager"]
    assert eager.outputs[0].dtype == torch.int8

    promoted = lane("inductor", [eager.outputs[0].to(torch.int64)])
    findings = METADATA.compare(eager, promoted, CFG)

    assert [finding.severity for finding in findings] == ["fail"]
    assert findings[0].details == {
        "field": "dtype",
        "expected": "torch.int8",
        "got": "torch.int64",
    }


def test_the_195451_shape_is_flagged_from_a_synthetic_result():
    # The other version-independent shape: whatever this torch does, a lane that
    # handed back the input object as its output is an alias fail. Built on top
    # of a real eager run rather than out of thin air, because the relation the
    # oracle compares is read off live tensors and their storages -- but one
    # lane rather than three, since the compiled world here is the synthetic one.
    target = load_target(str(CASES / "alias_copyback.py"))
    eager = run_all(target, ["eager"], seed=0).results["eager"]
    reinplaced = BackendResult(
        backend="inductor",
        outputs=list(eager.outputs),
        output_refs=[eager.input_refs[0]],
        inputs_before=list(eager.inputs_before),
        inputs_after=list(eager.inputs_after),
        input_refs=list(eager.input_refs),
        input_meta_before=list(eager.input_meta_before),
        input_meta_after=list(eager.input_meta_after),
    )
    findings = ALIAS.compare(eager, reinplaced, CFG)

    assert [f.details["field"] for f in findings] == ["identity_added"]
    assert findings[0].message == (
        "inductor returned input[0] itself as output[0] and eager returned a distinct object"
    )
