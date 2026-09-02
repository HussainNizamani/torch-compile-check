"""Tests for the regression-test emitter.

Two things are checked of every emitted file, because both are ways a generated
test can be worse than none: that it is valid Python (``compile()``), and that
the assertion in it is the one the oracle actually failed on. What the emitted
file does when it *runs* is exercised end to end in tests/test_cli.py, where a
real divergence is available to emit from.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from compile_check.localize import localize
from compile_check.oracles import Finding
from compile_check.report.pytest_case import emit, select
from compile_check.results import BackendResult, CapturedException, RunSet, TargetSource

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "cases"

ENV = {
    "torch_version": "2.14.0+cpu",
    "torch_git_version": "08187d9e0fba",
    "python_version": "3.10.12",
    "platform": "Linux-aarch64",
    "machine": "aarch64",
    "cpu_flags": "asimd",
    "cuda_available": False,
    "inductor_force_disable_caches": True,
}

MODULE = """\
import torch


def fn(x):
    return x + 1


inputs = (torch.ones(4),)
"""

MODEL_MODULE = """\
import torch
from torch import nn


class Tiny(nn.Module):
    def forward(self, x):
        return self.net(x)


model = Tiny()
inputs = (torch.ones(4, requires_grad=True),)
"""


def make_runset(
    text: str = MODULE,
    *,
    entry: str = "fn",
    name: str = "sample:fn",
    file: str = "sample.py",
    is_module: bool = False,
) -> RunSet:
    """A two-lane run around a literal target file."""
    runset = RunSet(
        target_name=name,
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        target_is_module=is_module,
        target_source=TargetSource(file=file, text=text, entry=entry, inputs="inputs"),
        env=dict(ENV),
    )
    runset.results["eager"] = BackendResult(backend="eager", outputs=[1])
    runset.results["inductor"] = BackendResult(backend="inductor", outputs=[1])
    return runset


def emitted(runset: RunSet, findings: list[Finding], **kwargs) -> str:
    """Emit, and fail the test rather than returning ``None`` unnoticed."""
    case = emit(runset, findings, localize(runset, findings), **kwargs)
    assert case is not None
    compile(case, "<emitted>", "exec")
    return case


def finding(oracle: str, **kwargs) -> Finding:
    """One fail-severity finding, with the fields a test cares about."""
    return Finding(
        oracle=oracle,
        backend=kwargs.pop("backend", "inductor"),
        output_index=kwargs.pop("output_index", 0),
        severity=kwargs.pop("severity", "fail"),
        message=kwargs.pop("message", "something differs"),
        details=kwargs.pop("details", {}),
    )


def test_a_dtype_finding_becomes_a_dtype_assertion():
    case = emitted(
        make_runset(),
        [
            finding(
                "metadata",
                message="dtype differs: eager torch.int8, inductor torch.int64",
                details={"field": "dtype", "expected": "torch.int8", "got": "torch.int64"},
            )
        ],
    )

    assert "self.assertEqual(actual.dtype, expected.dtype)" in case
    assert 'actual = torch.compile(fn, backend="inductor")(*make_inputs())' in case
    assert "expected = fn(*make_inputs())" in case
    # The message goes in as a comment, so the test says what it is defending.
    assert "# dtype differs: eager torch.int8, inductor torch.int64" in case


def test_a_finding_past_the_first_leaf_is_indexed_through_tree_leaves():
    # A dict (or any other non-sequence pytree) output does not support
    # actual[N] by leaf position -- actual[1] on a two-key dict is a lookup for
    # the key 1, and raises KeyError rather than comparing anything.
    # tree_leaves(actual)[N] is the same flattening the runner indexed the
    # finding against, so it names the leaf whatever shape actual is.
    runset = make_runset()
    runset.results["eager"] = BackendResult(backend="eager", outputs=[1, 2])
    case = emitted(
        runset,
        [
            finding(
                "metadata",
                output_index=1,
                message="dtype differs: eager torch.int8, inductor torch.int64",
                details={"field": "dtype", "expected": "torch.int8", "got": "torch.int64"},
            )
        ],
    )

    assert (
        "self.assertEqual(torch.utils._pytree.tree_leaves(actual)[1].dtype, "
        "torch.utils._pytree.tree_leaves(expected)[1].dtype)" in case
    )
    assert "actual[1]" not in case
    assert "expected[1]" not in case


@pytest.mark.parametrize(
    ("field", "assertion"),
    [
        ("shape", "self.assertEqual(actual.shape, expected.shape)"),
        ("stride", "self.assertEqual(actual.stride(), expected.stride())"),
        ("device", "self.assertEqual(actual.device.type, expected.device.type)"),
        ("requires_grad", "self.assertEqual(actual.requires_grad, expected.requires_grad)"),
        ("layout", "self.assertEqual(actual.layout, expected.layout)"),
        ("type", "self.assertIs(type(actual), type(expected))"),
    ],
)
def test_every_metadata_field_has_an_assertion_of_its_own(field, assertion):
    case = emitted(make_runset(), [finding("metadata", details={"field": field})])

    assert assertion in case


def test_a_numerics_finding_becomes_assert_close_at_the_tolerances_it_failed_at():
    case = emitted(
        make_runset(),
        [
            finding(
                "numerics",
                details={"rtol": 1.3e-06, "atol": 1e-05, "expected_dtype": "torch.float32"},
            )
        ],
    )

    assert "torch.testing.assert_close(actual, expected, rtol=1.3e-06, atol=1e-05)" in case


def test_an_identity_finding_becomes_an_identity_and_a_data_ptr_check():
    # The 195451 shape: the compiled lane handed back the input object itself.
    case = emitted(
        make_runset(),
        [
            finding(
                "alias",
                message="inductor returned input[0] itself as output[0] and eager returned a "
                "distinct object",
                details={"field": "identity_added", "left": "output[0]", "right": "input[0]"},
            )
        ],
    )

    assert "self.assertIsNot(actual, compiled_inputs[0])" in case
    assert "self.assertNotEqual(actual.data_ptr(), compiled_inputs[0].data_ptr())" in case
    # Both input sets are built, because the relation is about the inputs too.
    assert "eager_inputs = make_inputs()" in case
    assert "compiled_inputs = make_inputs()" in case


def test_an_identity_that_was_dropped_asserts_the_identity_holds():
    case = emitted(
        make_runset(),
        [
            finding(
                "alias",
                details={"field": "identity_dropped", "left": "output[0]", "right": "output[1]"},
            )
        ],
    )

    assert "self.assertIs(actual, actual)" not in case
    assert "self.assertIs(" in case


def test_an_alias_finding_becomes_a_data_ptr_comparison():
    case = emitted(
        make_runset(),
        [
            finding(
                "alias", details={"field": "alias_added", "left": "output[0]", "right": "input[1]"}
            )
        ],
    )

    assert "self.assertNotEqual(actual.data_ptr(), compiled_inputs[1].data_ptr())" in case


def test_a_mutation_finding_compares_the_input_the_lanes_disagree_about():
    case = emitted(
        make_runset(),
        [
            finding(
                "alias", output_index=None, details={"field": "mutation_added", "left": "input[0]"}
            )
        ],
    )

    assert "torch.testing.assert_close(compiled_inputs[0], eager_inputs[0])" in case


def test_an_input_gradient_finding_runs_a_backward_in_both_lanes():
    case = emitted(
        make_runset(),
        [
            finding(
                "grad",
                output_index=None,
                message="the gradient of input[0] differs",
                details={
                    "field": "grad_values",
                    "tensor": "input[0]",
                    "rtol": 1e-05,
                    "atol": 1e-05,
                },
            )
        ],
    )

    assert "expected.sum().backward()" in case
    assert "actual.sum().backward()" in case
    assert (
        "torch.testing.assert_close(compiled_inputs[0].grad, eager_inputs[0].grad, "
        "rtol=1e-05, atol=1e-05)" in case
    )


def test_a_parameter_gradient_finding_runs_the_lanes_one_after_the_other():
    # torch.compile(model) wraps the module rather than copying it, so both
    # backwards would otherwise accumulate into one .grad.
    case = emitted(
        make_runset(MODEL_MODULE, entry="model", name="sample:model", is_module=True),
        [
            finding(
                "grad",
                output_index=None,
                message="the gradient of parameter net.0.weight differs",
                details={"field": "grad_values", "tensor": "parameter net.0.weight"},
            )
        ],
    )

    assert case.count("model.zero_grad(set_to_none=True)") == 2
    assert 'expected_grad = model.get_parameter("net.0.weight").grad.clone()' in case
    assert 'actual_grad = model.get_parameter("net.0.weight").grad' in case
    assert "torch.testing.assert_close(actual_grad, expected_grad)" in case


def test_a_graph_finding_asserts_on_both_the_break_reasons_and_the_count():
    # torch's own graph_break_count is unreliable (runner.py floors it with
    # len(break_reasons)): a case can come back graph_break_count 0 with a
    # break reason recorded, which would make an assertion on the count alone
    # pass on the very case it was written for.
    case = emitted(
        make_runset(),
        [finding("graph", output_index=None, details={"field": "break_reasons"})],
    )

    assert "explained = torch._dynamo.explain(fn)(*make_inputs())" in case
    assert "self.assertEqual(list(explained.break_reasons), [])" in case
    assert (
        "self.assertEqual(max(explained.graph_break_count, len(explained.break_reasons)), 0)"
        in case
    )


def test_a_repeat_call_finding_calls_the_compiled_function_twice():
    case = emitted(
        make_runset(),
        [
            finding(
                "graph",
                output_index=None,
                message="inductor answered once and raised RuntimeError on the repeat call",
                details={"field": "second_call"},
            )
        ],
    )

    assert case.count("compiled(*make_inputs())") == 2


def test_a_structural_finding_falls_back_to_comparing_the_whole_return():
    case = emitted(
        make_runset(),
        [finding("numerics", output_index=None, details={"field": "output_count"})],
    )

    assert "torch.testing.assert_close(actual, expected)" in case


def test_a_lane_that_raised_with_no_finding_still_gets_a_test():
    runset = make_runset()
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="backend compiler failed\nsecond line", traceback=()
    )
    case = emitted(runset, [])

    assert "# inductor raised where eager did not: RuntimeError: backend compiler failed" in case
    # The compiled call is the assertion: if it raises, the test fails.
    assert 'actual = torch.compile(fn, backend="inductor")(*make_inputs())' in case


def test_nothing_is_emitted_when_the_eager_reference_itself_raised():
    # A compiled lane that also raised did not diverge from a working eager
    # run -- there is no eager behaviour left to assert against, whatever the
    # compiled lane did, so "raised where eager did not" would be false here.
    runset = make_runset()
    runset.results["eager"].exception = CapturedException(
        type="RuntimeError", message="the model is broken", traceback=()
    )
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="backend compiler failed", traceback=()
    )

    assert emit(runset, [], localize(runset, [])) is None


def test_nothing_is_emitted_when_no_eager_lane_ran_at_all():
    # The NO_REFERENCE sibling of the MODEL-stage case above: no eager result
    # at all, so a compiled lane that raised still did not diverge from
    # anything -- there is no reference run to compare it against, let alone
    # one it disagreed with.
    runset = make_runset()
    del runset.results["eager"]
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="backend compiler failed", traceback=()
    )

    assert emit(runset, [], localize(runset, [])) is None


def test_a_clean_run_emits_nothing():
    runset = make_runset()

    assert emit(runset, [], localize(runset, [])) is None


@pytest.mark.parametrize("severity", ["warn", "info"])
def test_a_warning_or_a_note_is_not_worth_a_regression_test(severity):
    # A contiguous-to-contiguous stride change is a legitimate layout choice and
    # the fp64 distance is context; a test asserting either would fail the day a
    # backend makes a different legal choice.
    runset = make_runset()
    findings = [finding("metadata", severity=severity, details={"field": "stride"})]

    assert select(findings) is None
    assert emit(runset, findings, localize(runset, findings)) is None


def test_a_run_with_no_target_source_emits_nothing():
    runset = make_runset()
    runset.target_source = None
    findings = [finding("metadata", details={"field": "dtype"})]

    assert emit(runset, findings, localize(runset, findings)) is None


def test_the_strongest_finding_in_oracle_order_is_the_one_written():
    chosen = select(
        [
            finding("graph", severity="fail"),
            finding("metadata", severity="fail"),
            finding("numerics", severity="fail"),
            finding("numerics", severity="warn"),
        ]
    )

    assert chosen is not None
    assert chosen.oracle == "numerics"
    assert chosen.severity == "fail"


def test_the_emitted_file_imports_torch_and_the_stdlib_and_nothing_else():
    case = emitted(make_runset(), [finding("metadata", details={"field": "dtype"})])
    tree = ast.parse(case)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.partition(".")[0])
    # PLAN.md "Engineering decisions" keeps torch the only dependency, and a
    # test the user cannot run is not a test: unittest is the standard library.
    assert {"torch", "unittest"} <= imported
    assert imported <= {"torch", "unittest", "__future__"}


def test_the_emitted_file_carries_the_issue_the_target_names():
    text = (CASES / "dtype_promotion.py").read_text(encoding="utf-8")
    runset = make_runset(text, name="dtype_promotion:fn", file="cases/dtype_promotion.py")
    case = emitted(runset, [finding("metadata", details={"field": "dtype"})])

    assert "Issue: https://github.com/pytorch/pytorch/issues/191308" in case
    assert "# https://github.com/pytorch/pytorch/issues/191308" in case


def test_the_target_is_inlined_once_with_the_inputs_as_a_factory():
    case = emitted(make_runset(), [finding("metadata", details={"field": "dtype"})])

    assert "def fn(x):" in case
    # The module-level assignment is dropped in favour of the factory, so the
    # same tuple is not in the file twice under a name nothing calls.
    assert "inputs = (torch.ones(4),)" not in case
    assert "return (torch.ones(4),)" in case


def test_a_whole_file_fallback_says_so_in_the_header():
    text = MODULE.replace("def fn(x):", "if True:\n    pass\n\n\ndef fn(x):")
    runset = make_runset(text)
    runset.target_source = TargetSource(
        file="sample.py", text=text, entry="missing_name", inputs="inputs"
    )
    case = emit(runset, [finding("metadata")], localize(runset, [finding("metadata")]))

    assert case is None or "whole file is inlined below" in case


def test_the_non_standalone_form_is_the_class_alone():
    findings = [finding("metadata", details={"field": "dtype"})]
    runset = make_runset()
    case = emit(runset, findings, localize(runset, findings), standalone=False)

    assert case is not None
    assert case.startswith("def make_inputs():")
    assert "import unittest" not in case
    assert "def fn(x):" not in case
    assert "class TestSample(unittest.TestCase):" in case
