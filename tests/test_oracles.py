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

from pathlib import Path
from typing import Any

import pytest
import torch

from compile_check.discover import load_target
from compile_check.oracles import (
    ORACLE_NAMES,
    ORACLES,
    Finding,
    Oracle,
    OracleConfig,
    run_oracles,
)
from compile_check.oracles.metadata import MetadataOracle
from compile_check.oracles.numerics import FALLBACK_TOLERANCES, NumericsOracle, resolve_tolerances
from compile_check.results import BackendResult, CapturedException
from compile_check.runner import FP64_BACKEND, run_all

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES = REPO_ROOT / "cases"

NUMERICS = NumericsOracle()
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
# the registry
# --------------------------------------------------------------------------


def test_the_registry_is_a_subset_of_the_fail_on_vocabulary():
    assert set(ORACLES) <= set(ORACLE_NAMES)
    assert list(ORACLES) == ["numerics", "metadata"]
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
