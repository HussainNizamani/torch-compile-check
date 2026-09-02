"""Inductor-suite regression test emitter.

PLAN.md "Package layout": ``report/pytest_case.py`` -- inductor-suite regression
test emitter.

PLAN.md "Regression test emission": alongside the repro, the tool emits the same
case as a drop-in regression test in the idiom the inductor suite already uses,
a ``common``-style eager versus compiled comparison written as a test method
body suitable for ``test/inductor/test_torchinductor.py``. The claim is not that
the test applies unmodified; it is that the test is half-written instead of
unwritten.

Three rules decide what comes out, and each is a way the emitter could otherwise
produce something worse than nothing.

A test is written for a ``fail`` finding, never for a ``warn`` or an ``info``. A
contiguous-to-contiguous stride change is PLAN.md "metadata"'s example of a
legitimate layout choice and the fp64 distance is context; a regression test
asserting either would be a test that fails the day a backend makes a different
legal choice. A clean run gets no file at all and is told so.

The assertion is the one the oracle actually failed on, not a general
comparison. A dtype divergence becomes an assertion about ``dtype``, an alias
divergence becomes an identity and ``data_ptr`` check, a gradient divergence
becomes a comparison of gradients. A test that asserted everything would fail
for reasons the report never claimed.

The file it writes runs on its own. ``test/inductor/test_torchinductor.py`` gets
its tensor-aware ``assertEqual`` from ``torch.testing._internal``, which needs
test-only dependencies the tool does not have and a user may not have either, so
tensor comparisons here go through the public ``torch.testing.assert_close``
instead and the class wrapper is plain ``unittest``. What is pasted into the
suite is the method body, which is identical either way.

Nothing here imports torch. It writes Python; it does not run it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from compile_check import __version__
from compile_check.localize import StageVerdict
from compile_check.oracles import ORACLE_NAMES, Finding
from compile_check.report import repro as repro_source
from compile_check.results import RunSet

__all__ = ["emit", "select"]

log = logging.getLogger("compile_check")

_INDENT = " " * 8

# Fields whose divergence has an assertion of its own, keyed by the oracle that
# reports them. Anything not here falls back to comparing the whole return with
# assert_close, which is the honest general answer: it walks structures, tensors,
# and scalars alike and says which of them differed.
_METADATA_ASSERTIONS: dict[str, str] = {
    "dtype": "self.assertEqual({actual}.dtype, {expected}.dtype)",
    "shape": "self.assertEqual({actual}.shape, {expected}.shape)",
    "stride": "self.assertEqual({actual}.stride(), {expected}.stride())",
    "requires_grad": "self.assertEqual({actual}.requires_grad, {expected}.requires_grad)",
    "device": "self.assertEqual({actual}.device.type, {expected}.device.type)",
    "is_contiguous": "self.assertEqual({actual}.is_contiguous(), {expected}.is_contiguous())",
    "layout": "self.assertEqual({actual}.layout, {expected}.layout)",
    "type": "self.assertIs(type({actual}), type({expected}))",
}


def select(findings: Sequence[Finding]) -> Finding | None:
    """The finding a regression test would be written from, or ``None``.

    ``fail`` severity only, then oracle order, then the order the lanes ran --
    which is ablation-ladder order, so the earliest rung that diverged wins. A
    test written against the earliest failing lane is the more useful test:
    PLAN.md "Stage localization" makes that lane the diagnosis, and a test that
    only exercises ``inductor`` says nothing about whether the same divergence
    is already there at ``aot_eager``.
    """
    fails = [finding for finding in findings if finding.severity == "fail"]
    if not fails:
        return None
    return min(fails, key=lambda finding: _oracle_rank(finding.oracle))


def emit(
    runset: RunSet,
    findings: Sequence[Finding],
    verdict: StageVerdict,
    *,
    standalone: bool = True,
) -> str | None:
    """Emit the top finding as a regression test.

    Args:
        runset: the run, from :func:`compile_check.runner.run_all`.
        findings: every finding the oracles produced.
        verdict: the stage verdict, for the case where a lane raised and left no
            finding behind to write a test from.
        standalone: write a file that runs on its own -- header, imports, and the
            target inlined. ``False`` emits the input factory and the test class
            alone, for a report that has already quoted the target above it,
            which is what the Markdown draft wants: the same code twice in one
            issue is one copy a reader has to diff against the other.

    Returns:
        The test's source, or ``None`` when there is nothing to test: a clean
        run, or a run whose target source was not available to inline. ``None``
        is a result, not a failure -- the caller says so and writes no file,
        because a regression test that asserts nothing is worse than no file.
    """
    finding = select(findings)
    raised = _raised_lane(verdict)
    if finding is None and raised is None:
        log.debug("nothing to emit: no fail-severity finding and no lane raised")
        return None

    repro = repro_source.extract(runset.target_source)
    if repro is None or repro.entry is None:
        log.debug("nothing to emit: the target's source is not available to inline")
        return None

    if finding is not None:
        lane = finding.backend
        body = _body(runset, repro, finding, lane)
    else:
        assert raised is not None  # the pair above is exhaustive
        lane, why = raised
        body = _raised_body(repro, lane, why)
    return _file(runset, repro, finding, lane, body, standalone)


def _file(
    runset: RunSet,
    repro: repro_source.Repro,
    finding: Finding | None,
    lane: str,
    body: list[str],
    standalone: bool,
) -> str:
    """The whole file: docstring, imports, the target, and the test class.

    Spaced the way isort and PEP 8 space a module -- one blank line between the
    import groups, two before every top-level definition -- because the first
    thing a maintainer does with this file is run their formatter over it, and a
    diff that is all whitespace hides the part that is not.
    """
    definitions: list[str] = []
    if standalone:
        definitions.append(repro.body_without_inputs if repro.inputs_expr else repro.body)
    if repro.inputs_expr:
        definitions.append(_factory(repro))
    definitions.append(_test_class(runset, repro, finding, lane, body))

    if not standalone:
        return "\n\n\n".join(definitions) + "\n"

    # `from __future__` is only legal directly under the docstring, and the two
    # import groups after it are stdlib then third party, which is where the
    # target's own imports belong.
    third_party = list(repro.imports)
    if not _imports_torch(repro):
        # Every emitted test calls torch.compile, so torch has to be imported
        # even when the target file did not import it under that name.
        third_party.insert(0, "import torch")
    groups = [
        _docstring(runset, repro, finding, lane),
        "\n".join(repro.future_imports),
        "import unittest",
        "\n".join(third_party),
    ]
    header = "\n\n".join(group for group in groups if group)
    footer = 'if __name__ == "__main__":\n    unittest.main()'
    return "\n\n\n".join([header, *definitions, footer]) + "\n"


def _docstring(
    runset: RunSet,
    repro: repro_source.Repro,
    finding: Finding | None,
    lane: str,
) -> str:
    """The file's own header: what it came from, and what it is not."""
    env = runset.env
    lines = [
        f'"""Regression test drafted by compile-check {__version__}.',
        "",
        f"Target: {repro.file or runset.target_name} ({runset.target_name})",
    ]
    if repro.issue is not None:
        lines.append(f"Issue: {repro.issue}")
    if finding is not None:
        where = "the run" if finding.output_index is None else f"output[{finding.output_index}]"
        lines.append(f"Finding: [{finding.severity}] {finding.oracle} {lane} {where}")
        lines.append(f"  {_one_line(finding.message)}")
    else:
        lines.append(f"Finding: {lane} raised where eager did not")
    lines.append(
        f"Observed on torch {env.get('torch_version')} ({env.get('machine')}), "
        f"python {env.get('python_version')}"
    )
    lines += [
        "",
        "The method below is the idiom test/inductor/test_torchinductor.py uses -- eager",
        "against compiled, asserting the one property that diverged -- so it can be lifted",
        "into that file. The imports and the class around it are what make this file also",
        "run on its own with pytest; the suite's own TestCase would give assertEqual its",
        "tensor semantics, and torch.testing.assert_close is the public equivalent.",
    ]
    if not repro.complete:
        lines += [
            "",
            "The target's whole file is inlined below: its entry point or its inputs are not",
            "bound by a plain top-level statement, so the pieces could not be isolated.",
        ]
    lines += [
        "",
        "It is half-written rather than ready. Check that the assertion says what you mean",
        "before you file it.",
        '"""',
    ]
    return "\n".join(lines)


def _factory(repro: repro_source.Repro) -> str:
    """The input factory, so that each lane gets inputs of its own.

    PLAN.md "Runner semantics" clones the inputs per backend, and a test that
    did not would compare a compiled call against an eager call that had already
    mutated what it was given -- which is exactly the class of bug the alias
    oracle exists for.
    """
    return "\n".join(
        [
            "def make_inputs():",
            '    """A fresh set per lane, as the runner gives each backend its own clone."""',
            f"    return {_reindent(repro.inputs_expr or '()')}",
        ]
    )


def _reindent(expression: str) -> str:
    """Indent an expression's continuation lines into a function body.

    The expression is the user's own text, so its continuation lines are
    indented for where it was, not for where it is going. A multi-line tuple
    lifted into a ``return`` reads wrong at the file's left margin, and adding
    four spaces per line fixes it -- except inside a triple-quoted string, where
    leading whitespace is part of the value, so that case is left alone.
    """
    lines = expression.splitlines()
    if len(lines) == 1 or '"""' in expression or "'''" in expression:
        return expression
    return "\n".join([lines[0], *(f"    {line}" for line in lines[1:])])


def _test_class(
    runset: RunSet,
    repro: repro_source.Repro,
    finding: Finding | None,
    lane: str,
    body: list[str],
) -> str:
    """The class wrapper and the one test method inside it."""
    oracle = finding.oracle if finding is not None else "raises"
    name = _method_name(runset, oracle, lane)
    lines = [f"class {_class_name(runset)}(unittest.TestCase):", f"    def {name}(self):"]
    if repro.issue is not None:
        lines.append(f"{_INDENT}# {repro.issue}")
    lines += [f"{_INDENT}{line}" if line else "" for line in body]
    return "\n".join(lines)


def _body(
    runset: RunSet,
    repro: repro_source.Repro,
    finding: Finding,
    lane: str,
) -> list[str]:
    """The test method's body: run both lanes, then assert what diverged.

    Four shapes, because the four oracles ask for four different runs. Numerics
    and metadata compare two returns and can pass their inputs inline; alias is
    a statement about the inputs too, so both sets are kept; grad needs a
    backward pass, and a *parameter* gradient needs the two lanes run one after
    the other, since a compiled module shares its parameters with the module it
    was compiled from and both backwards would otherwise land in one ``.grad``.
    """
    if finding.oracle == "graph":
        return _graph_body(repro, finding, lane)
    if finding.oracle == "grad":
        return _grad_body(repro, finding, lane)
    if finding.oracle == "alias":
        return _alias_body(runset, repro, finding, lane)

    leaves = _leaf_count(runset)
    lines = _two_lanes(repro, lane, leaves)
    lines.append(f"# {_one_line(finding.message)}")
    lines += _assertions(
        finding,
        _leaf("expected", finding.output_index, leaves),
        _leaf("actual", finding.output_index, leaves),
    )
    return lines


def _two_lanes(repro: repro_source.Repro, lane: str, leaves: int) -> list[str]:
    """Run the target under eager and under one compiled lane."""
    call = _call(repro)
    lines = [
        f"expected = {repro.entry}{call}",
        f'actual = torch.compile({repro.entry}, backend="{lane}"){call}',
    ]
    if leaves > 1:
        lines.append(f"# the run had {leaves} output leaves; the report indexes them flattened")
    return lines


def _assertions(finding: Finding, expected: str, actual: str) -> list[str]:
    """The assertion the oracle failed on, in the oracle's own terms."""
    field = finding.details.get("field")
    if finding.oracle == "metadata" and isinstance(field, str) and field in _METADATA_ASSERTIONS:
        return [_METADATA_ASSERTIONS[field].format(expected=expected, actual=actual)]
    if finding.oracle == "numerics":
        return [f"torch.testing.assert_close({actual}, {expected}{_tolerances(finding)})"]
    # Structural findings (an output count or a pytree spec that differs) and
    # anything a later oracle adds: assert_close walks the whole return and
    # names what differed, which is the general form of every check above.
    return [f"torch.testing.assert_close(actual, expected{_tolerances(finding)})"]


def _alias_body(
    runset: RunSet,
    repro: repro_source.Repro,
    finding: Finding,
    lane: str,
) -> list[str]:
    """Both input sets kept, so the relation between them can be asserted on."""
    return [
        *_kept_inputs(repro, lane),
        f"# {_one_line(finding.message)}",
        *_alias_assertions(finding, _leaf_count(runset)),
    ]


def _kept_inputs(repro: repro_source.Repro, lane: str) -> list[str]:
    """The two lanes, with their input sets bound rather than passed inline."""
    star = "**" if repro.keyword_inputs else "*"
    build = _inputs_source(repro)
    lines = []
    if build is None:
        # No expression that builds a second independent set. Saying so is the
        # only honest option: a test that ran both lanes against one set would
        # compare the compiled call against inputs the eager call had already
        # mutated, which is precisely what the alias oracle reports on.
        lines.append("# the tool could not build a second set of inputs; both lanes share these")
        lines += ["eager_inputs = inputs", "compiled_inputs = inputs"]
    else:
        lines += [f"eager_inputs = {build}", f"compiled_inputs = {build}"]
    lines += [
        f"expected = {repro.entry}({star}eager_inputs)",
        f'actual = torch.compile({repro.entry}, backend="{lane}")({star}compiled_inputs)',
    ]
    return lines


def _alias_assertions(finding: Finding, leaves: int) -> list[str]:
    """PLAN.md "alias": identity, then storage, then the mutation set.

    Written against the compiled lane alone. The eager relation is what the
    assertion encodes -- "these two must not be the same object" -- so the test
    states the contract rather than re-deriving it from a second run.
    """
    field = finding.details.get("field")
    left = _entity(finding.details.get("left"), leaves)
    right = _entity(finding.details.get("right"), leaves)
    if left is None:
        return ["torch.testing.assert_close(actual, expected)"]

    if field in {"identity_added", "identity_dropped"} and right is not None:
        keyword = "assertIsNot" if field == "identity_added" else "assertIs"
        lines = [f"self.{keyword}({left}, {right})"]
        if field == "identity_added":
            # The stronger half of the same statement: two names for one object
            # share a first byte, and a data_ptr check is what survives a lane
            # that returns a fresh view of the same storage.
            lines.append(f"self.assertNotEqual({left}.data_ptr(), {right}.data_ptr())")
        return lines
    if field in {"alias_added", "alias_dropped"} and right is not None:
        keyword = "assertNotEqual" if field == "alias_added" else "assertEqual"
        return [f"self.{keyword}({left}.data_ptr(), {right}.data_ptr())"]
    if field in {"mutation_added", "mutation_dropped"}:
        # The input the two lanes disagree about, compared after both calls: the
        # lane that mutated it in place has left different bytes behind.
        eager = left.replace("compiled_inputs", "eager_inputs")
        return [f"torch.testing.assert_close({left}, {eager})"]
    return ["torch.testing.assert_close(actual, expected)"]


def _grad_body(
    repro: repro_source.Repro,
    finding: Finding,
    lane: str,
) -> list[str]:
    """PLAN.md "grad": one backward per lane, then the gradient that differed."""
    tensor = finding.details.get("tensor")
    message = f"# {_one_line(finding.message)}"
    if isinstance(tensor, str) and tensor.startswith("parameter "):
        return _parameter_grad_body(repro, finding, lane, tensor[len("parameter ") :], message)

    lines = [
        *_kept_inputs(repro, lane),
        # The runner's own reduction (PLAN.md "Runner semantics"): a fixed
        # scalar, so the backward is deterministic and the two lanes are
        # differentiating the same function of the output.
        "expected.sum().backward()",
        "actual.sum().backward()",
        message,
    ]
    if isinstance(tensor, str) and tensor.startswith("input["):
        index = tensor[len("input[") : -1]
        lines.append(
            "torch.testing.assert_close("
            f"compiled_inputs[{index}].grad, eager_inputs[{index}].grad{_tolerances(finding)})"
        )
        return lines
    lines.append(f"torch.testing.assert_close(actual, expected{_tolerances(finding)})")
    return lines


def _parameter_grad_body(
    repro: repro_source.Repro,
    finding: Finding,
    lane: str,
    parameter: str,
    message: str,
) -> list[str]:
    """A parameter gradient: one lane at a time, with the grads cleared between.

    ``torch.compile(model)`` wraps the module rather than copying it, so the two
    lanes hold the same parameter objects and a backward in each would add the
    two gradients into one ``.grad``. The eager gradient is therefore taken and
    cloned first, and the buffer is cleared before the compiled lane runs.
    ``get_parameter`` takes the same dotted name the runner recorded the
    gradient under, so the test names the parameter the report named.
    """
    call = _call(repro)
    return [
        f"{repro.entry}.zero_grad(set_to_none=True)",
        f"expected = {repro.entry}{call}",
        "expected.sum().backward()",
        f'expected_grad = {repro.entry}.get_parameter("{parameter}").grad.clone()',
        f"{repro.entry}.zero_grad(set_to_none=True)",
        f'actual = torch.compile({repro.entry}, backend="{lane}"){call}',
        "actual.sum().backward()",
        f'actual_grad = {repro.entry}.get_parameter("{parameter}").grad',
        message,
        f"torch.testing.assert_close(actual_grad, expected_grad{_tolerances(finding)})",
    ]


def _graph_body(repro: repro_source.Repro, finding: Finding, lane: str) -> list[str]:
    """PLAN.md "graph": a break count, or a repeat call that did not survive."""
    call = _call(repro)
    entry = repro.entry
    if finding.details.get("field") == "second_call":
        return [
            f"# {_one_line(finding.message)}",
            f'compiled = torch.compile({entry}, backend="{lane}")',
            f"compiled{call}",
            f"compiled{call}",
        ]
    return [
        f"# {_one_line(finding.message)}",
        f"explained = torch._dynamo.explain({entry}){call}",
        "self.assertEqual(explained.graph_break_count, 0)",
    ]


def _raised_body(repro: repro_source.Repro, lane: str, message: str) -> list[str]:
    """The lane raised and produced no finding: calling it is the whole test."""
    call = _call(repro)
    entry = repro.entry
    return [
        f"# {lane} raised where eager did not: {message}",
        f"expected = {entry}{call}",
        f'actual = torch.compile({entry}, backend="{lane}"){call}',
        "torch.testing.assert_close(actual, expected)",
    ]


def _call(repro: repro_source.Repro) -> str:
    """The argument list, built from the factory when there is one."""
    build = _inputs_source(repro)
    if build is None:
        return (
            f"({'**' if repro.keyword_inputs else '*'}{repro.inputs_ref})"
            if repro.inputs_ref
            else "()"
        )
    return f"({'**' if repro.keyword_inputs else '*'}{build})"


def _inputs_source(repro: repro_source.Repro) -> str | None:
    """The expression the emitted file builds one fresh set of inputs with.

    ``make_inputs()`` when the file carries the factory this module writes, and
    ``None`` when there was no expression to write one from -- which is the
    caller's cue to say so rather than to pretend the two lanes were given
    independent inputs.
    """
    return "make_inputs()" if repro.inputs_expr else None


def _leaf(name: str, index: int | None, leaves: int) -> str:
    """One output leaf, indexed only when the run had more than one.

    A single-output target returns the leaf itself, and writing ``actual[0]``
    for it would index into the tensor. Only the first leaf gets that treatment:
    a finding naming a later index against a one-leaf run is a record that
    disagrees with itself, and collapsing the two to the same name would turn it
    into an assertion comparing something with itself.
    """
    if index is None or (index == 0 and leaves <= 1):
        return name
    return f"{name}[{index}]"


def _entity(label: Any, leaves: int) -> str | None:
    """One alias-relation label as an expression in the emitted test."""
    if not isinstance(label, str):
        return None
    found = re.fullmatch(r"(output|input)\[(\d+)]", label)
    if found is None:
        return None
    kind, index = found.group(1), int(found.group(2))
    if kind == "input":
        return f"compiled_inputs[{index}]"
    return _leaf("actual", index, leaves)


def _tolerances(finding: Finding) -> str:
    """The tolerances the comparison was made under, as keyword arguments."""
    rtol, atol = finding.details.get("rtol"), finding.details.get("atol")
    if not isinstance(rtol, int | float) or not isinstance(atol, int | float):
        return ""
    return f", rtol={rtol:g}, atol={atol:g}"


def _raised_lane(verdict: StageVerdict) -> tuple[str, str] | None:
    """The first compiled lane that raised, and the first line of why."""
    for entry in verdict.backends:
        if entry.backend != "eager" and entry.raised is not None:
            return entry.backend, f"{entry.raised.type}: {_one_line(entry.raised.message)}"
    return None


def _leaf_count(runset: RunSet) -> int:
    """How many output leaves the reference lane produced."""
    eager = runset.eager
    return len(eager.outputs) if eager is not None else 1


def _method_name(runset: RunSet, oracle: str, lane: str) -> str:
    """``test_<target>_<oracle>_<lane>``, as a valid Python identifier."""
    stem = _identifier(runset.target_name.replace(":", "_"))
    return f"test_{stem}_{oracle}_{_identifier(lane)}".lower()


def _class_name(runset: RunSet) -> str:
    """A class name derived from the target, for a file that runs on its own."""
    stem = _identifier(runset.target_name.partition(":")[0])
    parts = [part.capitalize() for part in stem.split("_") if part]
    return f"Test{''.join(parts) or 'CompileCheck'}"


def _identifier(text: str) -> str:
    """``text`` reduced to something that can appear in a Python name."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"target_{cleaned}"
    return cleaned


def _imports_torch(repro: repro_source.Repro) -> bool:
    """Whether the inlined source already imports torch under that name."""
    return bool(re.search(r"^import torch$", repro.source, flags=re.MULTILINE))


def _oracle_rank(oracle: str) -> int:
    """Where an oracle sits in PLAN.md's order, unknown names last."""
    return ORACLE_NAMES.index(oracle) if oracle in ORACLE_NAMES else len(ORACLE_NAMES)


def _one_line(message: str) -> str:
    """The first line of a message, which for a torch error is one of many."""
    for line in message.splitlines():
        if line.strip():
            return " ".join(line.split())
    return message.strip()
