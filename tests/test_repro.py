"""Tests for the repro reduction the Markdown draft and the emitter share.

Everything here is a read of source text, so the tests are source text: no run,
no torch, and the corpus cases used as fixtures are the real files the tool is
pointed at in practice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from torch_compile_check.report.repro import extract
from torch_compile_check.results import TargetSource

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "cases"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def source(text: str, entry: str = "fn", inputs: str | None = "inputs") -> TargetSource:
    """A target record around a literal file body."""
    return TargetSource(file="target.py", text=text, entry=entry, inputs=inputs)


def from_file(path: Path, entry: str, inputs: str) -> TargetSource:
    """A target record around one of the repository's own files."""
    return TargetSource(
        file=str(path),
        text=path.read_text(encoding="utf-8"),
        entry=entry,
        inputs=inputs,
    )


def test_a_corpus_case_reduces_to_its_import_its_function_and_its_inputs():
    repro = extract(from_file(CASES / "dtype_promotion.py", "fn", "inputs"))

    assert repro is not None
    assert repro.complete is True
    assert repro.imports == ("import torch",)
    assert repro.future_imports == ("from __future__ import annotations",)
    assert repro.source.startswith("import torch\n\n\ndef fn(")
    assert repro.source.rstrip().endswith(")")
    # The module docstring is prose about the bug, not part of the reproducer.
    assert "Twin file, deliberately" not in repro.source


def test_the_inputs_expression_builds_a_fresh_set_and_the_reference_names_it():
    repro = extract(from_file(CASES / "dtype_promotion.py", "fn", "inputs"))

    assert repro is not None
    # The right-hand side, so that evaluating it twice makes two independent
    # sets; the name, for a block that already carries the assignment.
    assert repro.inputs_expr is not None
    assert repro.inputs_expr.startswith("(\n    torch.ones((1, 1, 2)")
    assert repro.inputs_ref == "inputs"
    assert "inputs = (" in repro.source
    assert "inputs = (" not in repro.body_without_inputs


def test_the_issue_the_docstring_names_is_carried():
    repro = extract(from_file(CASES / "alias_copyback.py", "fn", "inputs"))

    assert repro is not None
    assert repro.issue == "https://github.com/pytorch/pytorch/issues/195451"


def test_a_docstring_with_no_issue_link_carries_none():
    repro = extract(source('"""No link here."""\nfn = len\ninputs = (1,)\n'))

    assert repro is not None
    assert repro.issue is None


def test_a_helper_the_entry_point_calls_is_kept_and_the_rest_is_dropped():
    repro = extract(
        source(
            "import torch\n"
            "\n"
            "UNUSED = 3\n"
            "\n"
            "def helper(x):\n"
            "    return x + 1\n"
            "\n"
            "def fn(x):\n"
            "    return helper(x)\n"
            "\n"
            "inputs = (torch.ones(2),)\n"
        )
    )

    assert repro is not None
    assert "def helper" in repro.source
    assert "UNUSED" not in repro.source


def test_a_class_the_entry_point_is_built_from_is_kept():
    repro = extract(
        source(
            "import torch\n"
            "from torch import nn\n"
            "\n"
            "class Tiny(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
            "\n"
            "model = Tiny()\n"
            "inputs = (torch.ones(2),)\n",
            entry="model",
        )
    )

    assert repro is not None
    assert repro.complete is True
    assert "class Tiny" in repro.source
    assert "model = Tiny()" in repro.source


def test_a_factory_is_kept_and_called_rather_than_copied():
    repro = extract(
        source(
            "import torch\n"
            "\n"
            "def fn(x):\n"
            "    return x\n"
            "\n"
            "def get_inputs():\n"
            "    return (torch.ones(2),)\n",
            inputs="get_inputs",
        )
    )

    assert repro is not None
    assert repro.inputs_expr == "get_inputs()"
    assert repro.inputs_ref == "get_inputs()"
    # Nothing to drop: the factory is a definition the test still calls.
    assert repro.body_without_inputs == repro.body


def test_an_entry_bound_inside_a_block_falls_back_to_the_whole_file():
    # The reference fixture is exactly this shape: `model` is built inside a
    # `with torch.random.fork_rng()`, so the binding depends on control flow the
    # reduction would have to reproduce.
    repro = extract(from_file(FIXTURES / "mlp.py", "model", "inputs"))

    assert repro is not None
    assert repro.complete is False
    assert repro.source.startswith('"""A tiny MLP')
    assert "with torch.random.fork_rng():" in repro.source
    # Still answered, because it is a fact about the file rather than about the
    # reduction: a whole-file repro wants a factory too.
    assert repro.inputs_expr == "get_inputs()"


def test_the_whole_file_fallback_still_hands_its_future_imports_out():
    # M3-3 defect fix. `from __future__` is only legal directly under a module
    # docstring, and the fallback used to leave it in the middle of the block --
    # so report/pytest_case.py, which pastes that block into a file of its own,
    # emitted a test that did not parse at all. Proven by parsing an emitted-
    # shaped file rather than by looking at the fields.
    text = (
        '"""A target whose entry point is bound inside a block."""\n'
        "from __future__ import annotations\n\n"
        "import torch\n\n"
        "with torch.no_grad():\n"
        "    model = torch.nn.Identity()\n\n"
        "inputs = (torch.ones(2),)\n"
    )
    repro = extract(source(text, entry="model"))

    assert repro is not None
    assert repro.complete is False
    assert repro.future_imports == ("from __future__ import annotations",)
    assert "from __future__" not in repro.source
    assert "with torch.no_grad():" in repro.source
    # The shape the emitter builds: docstring, futures, imports, then the block.
    compile(
        '"""Emitted."""\n\n'
        + "\n".join(repro.future_imports)
        + "\n\nimport unittest\n\n\n"
        + repro.source
        + "\n",
        "<emitted>",
        "exec",
    )


def test_a_file_that_no_longer_parses_falls_back_to_its_text():
    repro = extract(source("def fn(:\n"))

    assert repro is not None
    assert repro.complete is False
    assert repro.source == "def fn(:\n"


def test_a_run_with_no_source_reduces_to_nothing():
    assert extract(None) is None
    assert extract(TargetSource(file=None, text=None, entry="fn", inputs="inputs")) is None


@pytest.mark.parametrize("names", [("fn", None), (None, "inputs"), (None, None)])
def test_an_unresolvable_entry_or_inputs_name_falls_back(names):
    entry, inputs = names
    text = "import torch\n\ndef fn(x):\n    return x\n\ninputs = (torch.ones(2),)\n"
    repro = extract(TargetSource(file="t.py", text=text, entry=entry, inputs=inputs))

    assert repro is not None
    if entry is None and inputs is None:
        # Nothing to reduce towards: the whole file is the repro.
        assert repro.complete is False
    else:
        assert repro.complete is True
