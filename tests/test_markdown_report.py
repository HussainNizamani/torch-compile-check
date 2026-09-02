"""Tests for the Markdown issue draft.

The main test is a snapshot of the stable parts -- everything but the wall times
the draft deliberately does not carry -- because a substring assertion would
sail past a section that quietly stopped being rendered. The environment values
are fixed by the fixture for the same reason.
"""

from __future__ import annotations

import pytest

from compile_check import __version__
from compile_check.localize import localize
from compile_check.oracles import Finding
from compile_check.report.markdown import render, title
from compile_check.results import BackendResult, CapturedException, RunSet, TargetSource

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

TARGET = '''\
"""A case with an issue link.

Issue: https://github.com/pytorch/pytorch/issues/191308 -- the int8 matmul.
"""

from __future__ import annotations

import torch


def fn(a: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, a)


inputs = (torch.ones((2, 2), dtype=torch.int8),)
'''

FINDING = Finding(
    oracle="metadata",
    backend="inductor",
    output_index=0,
    severity="fail",
    message="dtype differs: eager torch.int8, inductor torch.int64",
    details={"field": "dtype", "expected": "torch.int8", "got": "torch.int64"},
)


@pytest.fixture
def runset() -> RunSet:
    """Two lanes over a target whose source the draft can quote."""
    built = RunSet(
        target_name="dtype_promotion:fn",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        target_is_module=False,
        target_source=TargetSource(
            file="cases/dtype_promotion.py", text=TARGET, entry="fn", inputs="inputs"
        ),
        env=dict(ENV),
    )
    built.results["eager"] = BackendResult(backend="eager", outputs=[1])
    built.results["inductor"] = BackendResult(backend="inductor", outputs=[1])
    return built


def draft(runset: RunSet, findings=(FINDING,), **kwargs) -> str:
    """The draft for one run, with the verdict derived as the CLI derives it."""
    return render(runset, list(findings), localize(runset, list(findings)), **kwargs)


EXPECTED = f"""\
# [inductor] torch.compile changes the output dtype of cases/dtype_promotion.py

> Drafted by [compile-check](https://github.com/HussainNizamani/compile-check) {__version__}. \
The line above is the issue title; everything below is the body. Read it, check it, and edit it \
before filing -- the tool drafts, a person files.

`dtype_promotion:fn` was run under eager and `inductor` on the environment at the bottom of this \
report. The oracles reported 1 finding, all of them fail-severity. First diverges at inductor, \
which implicates inductor lowering/codegen.

## Repro

The target's own source, reduced to the statements the entry point and the inputs need. It is not \
a *minimized* repro: the minimizer lands in M3-3.

```python
from __future__ import annotations

import torch


def fn(a: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, a)


inputs = (torch.ones((2, 2), dtype=torch.int8),)

# compile-check ran it like this. It gives each lane its own clone of the inputs,
# so rebuild them between the two calls if the target mutates what it is given.
expected = fn(*inputs)
actual = torch.compile(fn, backend="inductor")(*inputs)
```

## Expected versus got

- **[fail] metadata · inductor · output[0]** — dtype differs: eager torch.int8, inductor torch.int64
  - expected `torch.int8`, got `torch.int64`, field `dtype`

## Stage

First diverges at inductor, which implicates inductor lowering/codegen.

That is where the divergence becomes observable, not necessarily where the fix belongs.

## Regression test

A starting point in the idiom `test/inductor/test_torchinductor.py` uses, against the target \
above. It is half-written rather than ready: check the assertion says what you mean before you \
paste it in.

```python
def make_inputs():
    \"\"\"A fresh set per lane, as the runner gives each backend its own clone.\"\"\"
    return (torch.ones((2, 2), dtype=torch.int8),)


class TestDtypePromotion(unittest.TestCase):
    def test_dtype_promotion_fn_metadata_inductor(self):
        # https://github.com/pytorch/pytorch/issues/191308
        expected = fn(*make_inputs())
        actual = torch.compile(fn, backend="inductor")(*make_inputs())
        # dtype differs: eager torch.int8, inductor torch.int64
        self.assertEqual(actual.dtype, expected.dtype)
```

## Environment

- **torch**: 2.14.0+cpu (git `08187d9e0fba`)
- **python**: 3.10.12
- **os**: Linux-6.8.0-1057-oracle-aarch64-with-glibc2.35
- **architecture**: aarch64, cpu flags `asimd asimdhp asimddp`
- **device**: cpu (cuda available: no)
- **backends**: `eager`, `inductor`
- **compile flags**: fullgraph=False, dynamic=False, backward=on, seed=0
- **module**: not copied: the target is a plain callable, with no state to isolate
- **gradient tolerance**: the numerics tolerances x10
- **inductor caches**: disabled (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`)

## How this was produced

```console
$ compile-check cases/dtype_promotion.py --backends eager,inductor --fail-on metadata
```
"""


def test_a_divergent_run_drafts_exactly_this(runset):
    assert draft(runset, fail_on=["metadata"]) == EXPECTED


def test_the_draft_never_writes_a_disclosure_line(runset):
    # Whether and how a person discloses tooling on an issue they file under
    # their own name is theirs to decide, so the tool does not decide it.
    text = draft(runset)

    assert "AI assisted" not in text
    assert "Assisted by" not in text


def test_the_title_names_the_lane_the_field_and_the_target(runset):
    assert title(runset, [FINDING], localize(runset, [FINDING])) == (
        "[inductor] torch.compile changes the output dtype of cases/dtype_promotion.py"
    )


@pytest.mark.parametrize(
    ("oracle", "expected"),
    [
        ("numerics", "the output values"),
        ("alias", "the aliasing of the outputs"),
        ("grad", "the gradients"),
        ("graph", "the graph health"),
    ],
)
def test_every_oracle_has_a_phrase_for_the_title(runset, oracle, expected):
    finding = Finding(
        oracle=oracle, backend="inductor", output_index=0, severity="fail", message="differs"
    )

    assert expected in title(runset, [finding], localize(runset, [finding]))


def test_a_clean_run_drafts_a_record_rather_than_a_bug_report(runset):
    text = draft(runset, findings=())

    assert text.startswith("# compile-check found no divergence on cases/dtype_promotion.py")
    assert "a record of a clean run rather than a bug report" in text
    # Nothing to assert, so no test section; the environment block stays,
    # because a clean run on a named machine is still evidence.
    assert "## Regression test" not in text
    assert "## Environment" in text


def test_a_lane_that_raised_is_named_in_the_title(runset):
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="backend compiler failed", traceback=()
    )
    verdict = localize(runset, [])

    assert title(runset, [], verdict) == (
        "[inductor] torch.compile raises RuntimeError on cases/dtype_promotion.py"
    )


def test_a_run_with_no_reference_says_there_is_nothing_to_file(runset):
    del runset.results["eager"]
    text = draft(runset, findings=())

    assert "was not compared: the run had no eager lane" in text
    assert "There is nothing to file here" in text


def test_a_model_that_raised_in_eager_says_so(runset):
    runset.results["eager"].exception = CapturedException(
        type="ValueError", message="the model is broken", traceback=()
    )
    text = draft(runset, findings=())

    assert "the model raised under eager, so nothing was compiled" in text


def test_a_model_that_raised_in_eager_names_it_in_the_title_not_the_lane(runset):
    # A compiled lane that also raised did not diverge from a working eager
    # run, so the title must not name it the way test_a_lane_that_raised_is_
    # named_in_the_title below names a real divergence.
    runset.results["eager"].exception = CapturedException(
        type="ValueError", message="the model is broken", traceback=()
    )
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="backend compiler failed", traceback=()
    )
    verdict = localize(runset, [])

    assert title(runset, [], verdict) == (
        "compile-check could not compare cases/dtype_promotion.py: "
        "the eager reference raised ValueError"
    )
    assert "torch.compile raises" not in title(runset, [], verdict)


def test_a_model_that_raised_in_eager_carries_no_regression_test_section(runset):
    runset.results["eager"].exception = CapturedException(
        type="ValueError", message="the model is broken", traceback=()
    )
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="backend compiler failed", traceback=()
    )
    text = draft(runset, findings=())

    assert "## Regression test" not in text
    assert "raised where eager did not" not in text


def test_the_findings_are_capped_and_the_rest_counted(runset):
    findings = [FINDING] * 4
    text = draft(runset, findings=findings, max_findings=2)

    assert text.count("- **[fail] metadata") == 2
    assert "2 further findings are not listed here." in text


def test_a_target_with_no_source_says_the_repro_is_missing(runset):
    runset.target_source = None
    text = draft(runset)

    assert "there is nothing to inline here" in text
    assert "Attach the target file before filing." in text


def test_the_baseline_and_the_grad_tolerance_reach_the_environment_block(runset):
    text = draft(runset, baseline="baseline.json", grad_tol_factor=1.0)

    assert "- **graph baseline**: `baseline.json` (new breaks only)" in text
    assert "- **gradient tolerance**: the numerics tolerances x1" in text
    assert "--baseline baseline.json" in text


def test_the_command_block_carries_the_flags_that_change_the_run(runset):
    runset.fullgraph = True
    runset.dynamic = True
    runset.grad = False
    runset.share_module = True
    runset.seed = 7
    text = draft(runset)

    assert (
        "$ compile-check cases/dtype_promotion.py --backends eager,inductor --fullgraph "
        "--dynamic --no-grad --share-module --seed 7" in text
    )
