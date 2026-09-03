# cases

The regression corpus: one tiny model per known bug class, each carrying a
known-bad torch version marker. See PLAN.md "Regression corpus". This file
is the detailed reference, including the walkthrough for adding a case; for
the short version and what running the corpus looks like, see
[docs/corpus.md](../docs/corpus.md).

Two shapes of file live here, and the difference is deliberate.

- **Standalone RED/GREEN scripts**, one per case in the C-1 corpus
  (`alias_slice_scatter_copyback.py`, `alias_noop_view_identity.py`,
  `dtype_int8_matmul_promotion.py`, `distributions_validation_branch.py`,
  `numerics_cpu_inductor_miscompile.py`). Each runs itself, prints one
  `RED`/`GREEN` line with the torch version and build hash it measured, and
  exits non-zero when the bug reproduces. `FINDINGS.md` is the ground truth
  table they fill in.
- **torch-compile-check targets**, following the discovery convention of PLAN.md: a
  module-level `model` or `fn` plus `inputs` or `get_inputs()`, and nothing
  else, so `torch-compile-check cases/<file>.py` runs the case through the tool
  itself. `dtype_promotion.py` is the discovery-convention twin of
  `dtype_int8_matmul_promotion.py`, `alias_copyback.py` the twin of
  `alias_slice_scatter_copyback.py`, `alias_noop_view.py` the twin of
  `alias_noop_view_identity.py`, `distributions_binomial_kl.py` the twin of
  `distributions_validation_branch.py`, and `numerics_polyjuice_minmax.py`
  the twin of `numerics_cpu_inductor_miscompile.py`. `tests/test_corpus_twins.py`
  runs the standalone script and its twin together on every test run and
  asserts they agree, so the two files cannot drift apart silently.

Two modules here are not cases at all.

- `markers.py` is the known-bad version table: per case, the torch versions and
  build commits a RED was measured on, the fix PR and where it landed, and
  `expected_verdict(case, torch_version, git_version)`, which answers `RED`,
  `GREEN`, or `UNKNOWN`. It imports nothing from torch, so the arithmetic can be
  tested against version strings rather than against whatever this machine has.
- `summary.py` runs each standalone script in a subprocess and renders one
  Markdown table of observed against expected. `python -m cases.summary` from
  the repository root prints it; CI appends it to the job summary on every
  matrix cell.

Three test modules read all of this, and they ask three different questions.
`tests/test_corpus_twins.py` asks whether the tool's exit code and stage line
agree with the standalone script. `tests/test_corpus_oracles.py` asks whether
the right *oracle* fired, by running each case's `build()` through the runner
and grading the findings against the case's own `check()`.
`tests/test_corpus_markers.py` asks whether the marker in `markers.py` is still
current, and answers with a warning rather than a failure -- a nightly that
fixes a bug upstream must not turn this repository red.

## Adding a case

Write the standalone script first: reproduce the bug from the issue as
literally as you can (`build()` returning `(fn, example_inputs)`, `check()`
comparing eager against compiled, a `main()` that prints one `RED`/`GREEN`
line and exits accordingly), add its row to `FINDINGS.md`, and record a
version marker -- the torch build, git hash, and architecture the RED or
GREEN verdict was measured on, since a case can flip between them purely
because of which torch it runs against. Then write the discovery-convention
twin: the same reproducer, trimmed to a module-level `fn` (or `model`) plus
`inputs`, with nothing else at module scope, so `torch-compile-check
cases/<twin>.py` exercises it through the real tool rather than the
standalone script's own comparison. Add the pair to
`tests/test_corpus_twins.py`'s parametrize list, naming the backend the
stage verdict lands on when the standalone script is RED (deterministic
across torch builds, since it follows the tool's fixed lane order rather
than anything environment-specific -- explain why in the twin's docstring
if the first *diverging* backend and the bug's actual fix location differ,
the way `alias_noop_view.py` does). A twin whose bug gets fixed upstream is
not deleted: its docstring records the version marker for RED and the test
already handles the GREEN case, which is the point of anchoring the
assertion to the standalone script's live verdict rather than a hardcoded
exit code.

Then add the case to the two other tables that describe it, which a test
checks are all three in agreement: a `CaseMarker` in `markers.py` (issue,
oracle, the versions RED was measured on, and the fix point if there is one --
each field is a fact with a provenance, so do not write down an inference where
the record wants a measurement), and a row in `tests/test_corpus_oracles.py`'s
`CORPUS`, naming the adapter that hands your `check()` the arguments it reads
and the set of oracles a RED reaches the report through.
