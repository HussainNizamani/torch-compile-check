# Reports

One run, three optional artifacts, off the same result — PLAN.md "Reports".
None is written unless its flag is passed; see [usage.md](usage.md#artifacts)
for the flags. All three below are from the same real run:

```console
$ compile-check cases/dtype_promotion.py --json out.json --md draft.md --emit-test test_case.py
compile-check: wrote out.json (--json)
compile-check: wrote draft.md (--md)
compile-check: wrote test_case.py (--emit-test)
```

## JSON — `--json OUT.JSON`

The CI-consumable artifact, and the unit of cross-architecture comparison
(PLAN.md "Cross-architecture parity is a feature", [cross-arch.md](cross-arch.md)).
`schema_version` is a top-level integer, bumped on any incompatible field
change; the authoritative schema is the docstring at the top of
[`src/compile_check/report/json.py`](../src/compile_check/report/json.py),
which `build()` implements and `validate()` enforces by hand (torch stays
the package's only runtime dependency, so the schema does not earn
`jsonschema`). `dump()` refuses to write a document that fails its own
`validate()`.

The current version is **2**. Version 1 (M3-2) was the first artifact:
`schema_version`, `tool`, `target`, `environment`, `run`, `backends`,
`findings`, `verdict`, `counts`, `exit_code`. Version 2 (M3-3) added exactly
one field, the top-level `minimized` object — `null` means `--minimize` was
not asked for, and a record whose `changed` is `false` means it ran and
nothing could be reduced; a v1 document has no such key at all, which is why
the version moved rather than the field being added silently. There is no
timestamp in the document: parity in v1 is running the tool on two machines
and diffing the two JSON files, and a field that differs on every run for no
reason makes that diff worse.

The top level, from a real run of the command above (values elided for
length; the field names and shapes are exactly what shipped):

```json
{
  "schema_version": 2,
  "tool": {"name": "compile-check", "version": "0.0.1.dev0"},
  "target": {"name": "dtype_promotion:fn", "file": ".../cases/dtype_promotion.py", "entry": "fn", "inputs": "inputs"},
  "environment": {"torch_version": "2.14.0+cpu", "machine": "aarch64", "cuda_available": false, "...": "..."},
  "run": {"device": "cpu", "seed": 0, "backends": ["eager", "aot_eager", "inductor"], "...": "..."},
  "backends": [{"backend": "eager", "reference": false, "ok": true, "...": "..."}, "..."],
  "findings": [
    {
      "oracle": "metadata",
      "backend": "inductor",
      "output_index": 0,
      "severity": "fail",
      "message": "dtype differs: eager torch.int8, inductor torch.int64",
      "details": {"field": "dtype", "expected": "torch.int8", "got": "torch.int64"}
    }
  ],
  "verdict": {"stage": "...", "first_divergent_backend": "inductor", "clean": false, "...": "..."},
  "minimized": null,
  "counts": {"fail": 1, "warn": 0, "info": 0},
  "exit_code": 1
}
```

`environment` always carries the architecture (`machine`), `cuda_available`,
and the torch version and git hash — the fields
[cross-arch.md](cross-arch.md) diffs between two runs, alongside
`schema_version` and `findings`. `minimized` is `null` here because
`--minimize` was not passed on this run; see
[usage.md's minimizer section](usage.md#the-minimizer) for a run where it
is not.

## Markdown — `--md REPORT.MD`

An issue draft in the shape the PyTorch tracker expects: a title naming the
lane and what changed, the minimal repro inline as a fenced Python block
(the target's own source, reduced to the statements the entry point and the
inputs need — a *shorter* file, not a *smaller case*: that is what
`--minimize` is for), expected versus got per finding, the stage
verdict with PLAN.md's observability caveat carried verbatim, the emitted
regression test when `--emit-test` also ran, and an environment block. The
tool drafts; a person reads it, edits it, and files it. From the same run:

```markdown
# [inductor] torch.compile changes the output dtype of cases/dtype_promotion.py

> Drafted by [compile-check](https://github.com/HussainNizamani/compile-check) 0.0.1.dev0. The line above is the issue title; everything below is the body. Read it, check it, and edit it before filing -- the tool drafts, a person files.

`dtype_promotion:fn` was run under eager and `aot_eager`, `inductor` on the environment at the bottom of this report. The oracles reported 1 finding, all of them fail-severity. First diverges at inductor, which implicates inductor lowering/codegen.

## Repro
...
## Expected versus got

- **[fail] metadata · inductor · output[0]** — dtype differs: eager torch.int8, inductor torch.int64
  - expected `torch.int8`, got `torch.int64`, field `dtype`

## Stage

First diverges at inductor, which implicates inductor lowering/codegen.

That is where the divergence becomes observable, not necessarily where the fix belongs.
...
## How this was produced

$ compile-check cases/dtype_promotion.py --backends eager,aot_eager,inductor --fail-on numerics,alias,metadata,grad
```

Three things the draft deliberately does not do, from
[`report/markdown.py`](../src/compile_check/report/markdown.py)'s own
docstring: it adds no AI-disclosure line (whether and how a person discloses
tooling on an issue they file under their own name is theirs to decide, not
the tool's); it never says "the bug is in `<stage>`", only "first diverges
at", because a maintainer reading the stronger claim from a tool that ran
three backends would be right to stop reading (see PLAN.md "Where divergence
appears is not always where the fix belongs"); and it says explicitly
whether `--minimize` ran, rather than letting a reader assume a repro is
minimal because a tool produced it.

## Regression test — `--emit-test TEST.PY`

The top **fail**-severity finding, written as a drop-in regression test in
the idiom `test/inductor/test_torchinductor.py` already uses — a `common`-style
eager-versus-compiled comparison as a test method body. The claim is not
that it applies unmodified; it is that it is half-written instead of
unwritten. A clean run, or one whose only findings are `warn`/`info`
severity (a legal contiguous-to-contiguous stride change, for instance),
writes no file and says which of the two it was — asserting nothing would
be a worse artifact than no artifact.

```python
class TestDtypePromotion(unittest.TestCase):
    def test_dtype_promotion_fn_metadata_inductor(self):
        # https://github.com/pytorch/pytorch/issues/191308
        expected = fn(*make_inputs())
        actual = torch.compile(fn, backend="inductor")(*make_inputs())
        # dtype differs: eager torch.int8, inductor torch.int64
        self.assertEqual(actual.dtype, expected.dtype)
```

The assertion is the one the oracle actually failed on, never a general
comparison: a dtype divergence becomes `assertEqual` on `.dtype`, an alias
divergence becomes an identity and `data_ptr` check, a gradient divergence
compares gradients, and anything else falls back to
`torch.testing.assert_close`. Tensor comparisons go through the public
`torch.testing.assert_close` rather than the inductor suite's own
`assertEqual`, which needs test-only internals a user may not have; the
method body pastes into the suite unchanged either way. This file **runs on
its own**, which is the EXECUTE-ARTIFACTS property that matters for an
emitted artifact — running it is not the same as reading it:

```console
$ python test_case.py -v
test_dtype_promotion_fn_metadata_inductor (__main__.TestDtypePromotion) ... FAIL

======================================================================
FAIL: test_dtype_promotion_fn_metadata_inductor (__main__.TestDtypePromotion)
----------------------------------------------------------------------
Traceback (most recent call last):
  ...
AssertionError: torch.int64 != torch.int8

----------------------------------------------------------------------
Ran 1 test in 3.260s

FAILED (failures=1)
```

`FAIL` is the correct outcome here, not a broken example: issue
[#191308](https://github.com/pytorch/pytorch/issues/191308) is still open
upstream on this torch build, so a test asserting the contract holds is
supposed to fail until the fix lands, exactly like a regression test filed
against the bug would. Once a fix merges, the same file — pasted into
`test/inductor/test_torchinductor.py`, or run standalone — turns green and
starts guarding the fix.
