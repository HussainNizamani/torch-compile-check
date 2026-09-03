# The regression corpus

`cases/` holds one tiny model per known `torch.compile` bug class: five bug
classes, one row per bug class in [PLAN.md "Why this project, and why
us"](../PLAN.md), plus two more cases that are reviewer-reported sibling
shapes of one of them ([#195451](https://github.com/pytorch/pytorch/issues/195451),
`cases/README.md`), seven cases in total as of this table. The detailed
reference, covering how the file shapes in it relate, what `markers.py` and
`summary.py` do, and the walkthrough for adding a new case, is
[`cases/README.md`](../cases/README.md); this page is the short version plus
what running it looks like today.

## Two files per case

- A **standalone RED/GREEN script** (`dtype_int8_matmul_promotion.py`,
  `alias_slice_scatter_copyback.py`, ...) that reproduces the bug as
  literally as the issue describes it, prints one `RED`/`GREEN` line with
  the torch version and build hash it measured, and exits non-zero when the
  bug reproduces. `FINDINGS.md` is the ground-truth table these scripts fill
  in.
- A **discovery-convention twin** (`dtype_promotion.py`,
  `alias_copyback.py`, ...): the same reproducer trimmed to a module-level
  `model`/`fn` plus `inputs`/`get_inputs()`, so `torch-compile-check cases/<twin>.py`
  exercises the bug through the real tool. `tests/test_corpus_twins.py` runs
  every standalone script and its twin together on every test run and
  asserts they agree on exit code and stage line, so the two cannot drift
  apart silently.

Two cases, `alias_view_slice_scatter_copyback.py` and
`alias_diagonal_scatter_index_put_chain.py` -- reviewer-reported sibling
shapes of `alias_slice_scatter_copyback.py`'s bug, added 2026-09-03 outside
the original slice -- combine both shapes into one file instead: the same
file is the standalone script and, by also exposing module-level `fn` and
`inputs`, its own twin. See `cases/README.md` for why.

## The known-bad marker

Several of these bugs are fixed on current torch, so a case cannot simply
assert "this fails". `cases/markers.py` records, per case, the torch
versions a `RED` was measured on and the version or commit that fixed it;
`expected_verdict(case, torch_version, git_version)` answers `RED`, `GREEN`,
or `UNKNOWN` for the torch actually running. `tests/test_corpus_markers.py`
compares the marker against the live verdict and warns rather than fails on
a disagreement — a nightly that fixes a bug upstream must not turn this
repository red.

## Running the corpus

```console
$ python -m cases.summary
### torch-compile-check regression corpus -- torch 2.14.0+cpu (git 08187d9e0fba), python 3.10.12, aarch64

| Case | Issue | Oracle | Observed | Expected | Agrees |
|---|---|---|---|---|---|
| `alias_slice_scatter_copyback` | #195451 | alias | RED | RED | yes |
| `alias_noop_view_identity` | #191449 | alias | RED | RED | yes |
| `dtype_int8_matmul_promotion` | #191308 | metadata | RED | RED | yes |
| `distributions_validation_branch` | #194593 | graph | RED | RED | yes |
| `numerics_cpu_inductor_miscompile` | #190765 | numerics | GREEN | GREEN | yes |
| `alias_view_slice_scatter_copyback` | #195451 | alias | RED | RED | yes |
| `alias_diagonal_scatter_index_put_chain` | #195451 | alias | RED | RED | yes |

7 cases: 7 agree with the marker, 0 disagree, 0 could not be placed.
```

(The real run links every issue number; trimmed here for width.) This is
the same table CI appends to the job summary on every matrix cell — real,
executed output, not a description of one.

Each script compiles, so the seven of them cost about a minute. The verdicts
are cached in a JSON file under the system temporary directory, and CI's
job-summary step reuses what the `pytest` step in the same job already
measured; when it does, the tally line says so. An entry is reused only when
the torch build and hash, the Python version, the machine, the interpreter and
the case file's own bytes are all unchanged, so a torch upgrade or an edited
case is measured again rather than answered from the file. Point
`TORCH_COMPILE_CHECK_OBSERVATIONS` at another path to move it, or set it to an empty
value to switch it off:

```console
$ TORCH_COMPILE_CHECK_OBSERVATIONS= python -m cases.summary   # always re-run everything
$ python -m cases.summary --no-cache                          # the same thing, as a flag
```

The cache never stores a crash. A case that exits 2 -- a compile that failed
because the C++ toolchain is missing or misconfigured, a timeout, anything
that is not a clean RED or GREEN -- is `UNKNOWN`, and an `UNKNOWN` observation
is never written to the file; the tally line's "N of the M were reused" always
counts actual RED/GREEN verdicts, down to zero. This is what makes the cache
safe across an environment change: a broken box that cannot compile anything
would otherwise cache "every case is UNKNOWN" and keep answering that once the
box is fixed, since nothing about the case's own source changed, only the
machine's ability to run it. When the environment does change underneath this
file -- a new compiler, installed headers, an upgraded torch that was masked
by an unrelated failure -- clear `/tmp/torch-compile-check-observations-*.json`
or pass `--no-cache` (equivalently, `TORCH_COMPILE_CHECK_OBSERVATIONS=`) to be
sure the next run measures the box as it actually is now, not as the cache
last saw it.

A single twin through the actual CLI, for comparison:

```console
$ torch-compile-check cases/dtype_promotion.py
findings
  metadata  (1 fail)
    [fail] inductor output[0]
        dtype differs: eager torch.int8, inductor torch.int64
stage
  first diverges at inductor, which implicates inductor lowering/codegen
```

Three test modules read the corpus and ask three different questions:
`tests/test_corpus_twins.py` (does the tool's exit code and stage line agree
with the standalone script?), `tests/test_corpus_oracles.py` (did the right
*oracle* fire, graded against each case's own `check()`?), and
`tests/test_corpus_markers.py` (is the marker in `markers.py` still current,
as a warning rather than a failure?).

See [`cases/README.md`](../cases/README.md) for the walkthrough on adding a
case — the standalone script first, then the twin, then the marker and the
`tests/test_corpus_oracles.py` row, with a test that checks all three tables
agree.
