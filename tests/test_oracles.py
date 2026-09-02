"""Tests for the numerics and metadata oracles.

Most of these are hand-built tensor pairs rather than runs: an oracle's rules
are exactly what has to be pinned down, and a pair built here says what it is
testing in one line, where a compiled model saying the same thing costs seconds
and depends on what this torch happens to do. The two integration tests at the
end are the ones that check the hand-built rules still describe real runs.

conftest.py has already set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1, before torch
was imported, which is the only moment at which it can be set.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch

from compile_check.discover import Target, import_target_module, load_target
from compile_check.localize import localize
from compile_check.oracles import (
    ORACLE_NAMES,
    ORACLES,
    Finding,
    Oracle,
    OracleConfig,
    run_oracles,
)
from compile_check.oracles.alias import AliasOracle, relation
from compile_check.oracles.metadata import MetadataOracle
from compile_check.oracles.numerics import FALLBACK_TOLERANCES, NumericsOracle, resolve_tolerances
from compile_check.results import BackendResult, CapturedException
from compile_check.runner import FP64_BACKEND, run_all, run_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES = REPO_ROOT / "cases"

NUMERICS = NumericsOracle()
ALIAS = AliasOracle()
METADATA = MetadataOracle()
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
# the registry
# --------------------------------------------------------------------------


def test_the_registry_is_a_subset_of_the_fail_on_vocabulary():
    assert set(ORACLES) <= set(ORACLE_NAMES)
    assert list(ORACLES) == ["numerics", "alias", "metadata"]
    for name, oracle in ORACLES.items():
        assert isinstance(oracle, Oracle)
        assert oracle.name == name


def test_run_oracles_selects_by_name():
    expected = torch.tensor([1, 2], dtype=torch.int8)
    eager, other = pair(expected, expected.to(torch.int64) + 1)

    everything = run_oracles(eager, other, CFG)
    assert {finding.oracle for finding in everything} == {"numerics", "metadata"}
    assert {f.oracle for f in run_oracles(eager, other, CFG, ["metadata"])} == {"metadata"}
    # A category no oracle implements yet contributes nothing, rather than
    # raising: cli.parse_fail_on is where an unknown name is reported.
    assert run_oracles(eager, other, CFG, ["grad"]) == []


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


@pytest.fixture(scope="module")
def dtype_promotion_runset():
    """The 191308 case through the real backends; inductor costs seconds."""
    target = load_target(str(CASES / "dtype_promotion.py"))
    return run_all(target, ["eager", "inductor"], seed=0)


def test_the_191308_case_is_reported_when_this_torch_still_has_the_bug(dtype_promotion_runset):
    eager = dtype_promotion_runset.results["eager"]
    inductor = dtype_promotion_runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception

    findings = run_oracles(eager, inductor, CFG)
    dtypes = (eager.outputs[0].dtype, inductor.outputs[0].dtype)
    if dtypes[0] == dtypes[1]:  # pragma: no cover - depends on the torch build
        # The issue is open, but the shape family is narrow; a torch that fixes
        # it must turn this case green, not turn the suite red.
        assert findings == []
        pytest.skip(f"this torch keeps the dtype ({dtypes[0]}), so there is nothing to report")

    fails = [finding for finding in findings if finding.severity == "fail"]
    assert [finding.details["field"] for finding in fails] == ["dtype"]
    assert fails[0].oracle == "metadata"
    # PLAN.md "metadata": the values were arguably defensible, the dtype was
    # not, which is exactly what makes this the metadata oracle's bug.
    assert [finding for finding in findings if finding.oracle == "numerics"] == []
    assert torch.equal(
        eager.outputs[0].to(torch.int64),
        inductor.outputs[0].to(torch.int64),
    )


def test_the_191308_shape_is_flagged_from_a_synthetic_result():
    # The version-independent half of the case above: whatever this torch does,
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


# --------------------------------------------------------------------------
# integration: the alias cases from the regression corpus
# --------------------------------------------------------------------------


def corpus_case(name: str) -> Any:
    """Import a corpus script and hand back the module.

    The case scripts are the source of truth for whether a bug reproduces on the
    installed torch: each carries its own ``build()`` and its own ``check()``,
    and the tests below ask *them* rather than hard-coding a version window that
    would go stale.
    """
    return import_target_module(str(CASES / f"{name}.py"))


@pytest.fixture(scope="module")
def slice_scatter_run():
    """The 195451 case through the real runner; inductor costs seconds."""
    case = corpus_case("alias_slice_scatter_copyback")
    fn, example_inputs = case.build()
    target = Target(fn=fn, example_inputs=example_inputs, name="alias_slice_scatter_copyback:fn")
    return case, run_all(target, ["eager", "aot_eager", "inductor"], seed=0)


def test_the_195451_case_is_reported_when_this_torch_still_has_the_bug(slice_scatter_run):
    case, runset = slice_scatter_run
    eager = runset.results["eager"]
    inductor = runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception

    findings = [f for lane in runset.others for f in run_oracles(eager, lane, CFG)]
    alias_fails = [f for f in findings if f.oracle == "alias" and f.severity == "fail"]

    # The case script's own verdict, asked of the objects this run produced, and
    # asked after the oracles have run: its RED probe mutates the output it is
    # given, to prove the alias is load-bearing.
    is_red, message = case.check(
        (eager.input_refs[0], eager.output_refs[0]),
        (inductor.input_refs[0], inductor.output_refs[0]),
        None,
    )
    if not is_red:  # pragma: no cover - depends on the torch build
        assert alias_fails == []
        pytest.skip(f"this torch does not reproduce 195451: {message}")

    assert [f.details["field"] for f in alias_fails] == ["identity_added"]
    assert [f.backend for f in alias_fails] == ["inductor"]
    # The numbers agree in both worlds; only the aliasing differs, which is what
    # makes this the alias oracle's bug and not the numerics oracle's.
    assert [f for f in findings if f.oracle == "numerics" and f.severity == "fail"] == []
    verdict = localize(runset, findings)
    assert verdict.first_divergent_backend == "inductor"
    assert "first diverges at inductor" in verdict.summary


def test_the_195451_shape_is_flagged_from_a_synthetic_result(slice_scatter_run):
    # The version-independent half of the case above: whatever this torch does,
    # a lane that handed back the input object as its output is an alias fail.
    _case, runset = slice_scatter_run
    eager = runset.results["eager"]
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


@pytest.fixture(scope="module")
def noop_view_run():
    """The 191449 case through the real runner; inductor costs seconds."""
    case = corpus_case("alias_noop_view_identity")
    fn, example_inputs = case.build()
    target = Target(fn=fn, example_inputs=example_inputs, name="alias_noop_view_identity:fn")
    return case, run_all(target, ["eager", "aot_eager", "inductor"], seed=0)


def test_the_191449_case_is_reported_when_this_torch_still_has_the_bug(noop_view_run):
    case, runset = noop_view_run
    eager = runset.results["eager"]
    inductor = runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception

    findings = [f for lane in runset.others for f in run_oracles(eager, lane, CFG)]
    fails = [f for f in findings if f.severity == "fail"]
    is_red, message = case.check(eager.outputs[0], inductor.outputs[0], None)
    if not is_red:  # pragma: no cover - depends on the torch build
        assert fails == []
        pytest.skip(f"this torch does not reproduce 191449: {message}")

    # RED, so the tool must report it -- but not from this oracle. What this
    # case *returns* is `base + 1` after a resize_ through a no-op view, so the
    # divergence reaches the report as a shape and a value difference; the
    # aliasing underneath it is only visible when the base and the view are
    # returned together, which is the run below.
    assert fails != []
    assert {f.oracle for f in fails} == {"numerics", "metadata"}
    assert localize(runset, findings).first_divergent_backend == "inductor"


@pytest.fixture(scope="module")
def base_and_view_run():
    """The alias-visible form of 191449: a base and its no-op view, together.

    The corpus script carries this shape as its own identity probe; running it
    through the real runner is what puts the collapse in front of the oracle.
    """

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = x + 1
        return base, base.view(-1)

    target = Target(fn=fn, example_inputs=(torch.zeros(1),), name="inline:base_and_view")
    return run_all(target, ["eager", "aot_eager", "inductor"], seed=0)


def test_the_191449_identity_collapse_is_an_alias_fail(base_and_view_run):
    runset = base_and_view_run
    eager = runset.results["eager"]
    inductor = runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception
    assert eager.output_refs[0] is not eager.output_refs[1], "eager itself collapsed the two"

    findings = [f for lane in runset.others for f in run_oracles(eager, lane, CFG)]
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
